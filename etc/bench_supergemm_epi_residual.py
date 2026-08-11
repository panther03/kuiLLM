"""Is SuperGEMM's residual gap to the reference specific to the epilogue?

TASK A follow-up. Comparing SuperGEMM against the reference with its vectorized
C read patched out isolates the cost of the port's scalar-only C read. Any gap
that REMAINS after that is not the C read, so it must be either Kuiper
extraction overhead or run-to-run noise.

This measures that residual at higher precision than the main benchmark
(interleaved repeats, mean +- stdev) and, crucially, measures the SAME residual
for the already-accepted NO-EPILOGUE port against ITS reference. If the two
residuals agree, the epilogue port introduces nothing: the gap is whatever the
accepted baseline already had.
"""
import statistics

import torch

from kuipy import run
from kuipy.benchmarking import warm_up
from etc.ref_flat_gemm import load as load_ref

TILE = dict(bm=128, bn=128, bk=32, wm=32, wn=64, skew=8)
CASES = [("lm_head", 256, 896, 151936), ("square4096", 4096, 4096, 4096)]
REPEATS, ITERS = 9, 100
ALPHA, BETA = 0.5, 2.0


def time_once(fn, args, iters=ITERS):
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def ab(label, contenders, args):
    """Interleave the contenders so drift hits both equally."""
    samples = {k: [] for k in contenders}
    for _ in range(REPEATS):
        for k, fn in contenders.items():
            samples[k].append(time_once(fn, args))
    print(f"\n=== {label} ===")
    base = None
    for k, v in samples.items():
        med = statistics.median(v)
        sd = statistics.stdev(v)
        if base is None:
            base = med
        print(f"  {k:28s} {med:9.2f} us  +-{sd:5.2f}  "
              f"({100 * (med / base - 1):+6.2f}%)")
    return samples


warm_up(500)
addmm = torch.ops.aten.addmm.default
mm = torch.ops.aten.mm.default
sg_epi = run(addmm, impl="supergemm")
sg_noepi = run(mm, impl="supergemm")

for name, M, K, N in CASES:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    ref_scalar = load_ref(stem="gemm_tc_flat_nosplitk_epi", group=1, **TILE,
                          cs_m=N, cs_n=1, force_scalar_c=True)
    ab(f"{name} EPILOGUE (dense C)", {
        "cuda_ref_g1_scalarc": lambda C, A, B: ref_scalar(C, A, B, ALPHA, BETA),
        "supergemm_epi": lambda C, A, B: sg_epi(C, A, B, alpha=ALPHA,
                                                beta=BETA),
    }, (c, a, bt.t()))

    ref_noepi = load_ref(stem="gemm_tc_flat_nosplitk_noepi", group=1, **TILE)
    ab(f"{name} NO-EPILOGUE (accepted baseline)", {
        "cuda_ref_noepi_g1": ref_noepi,
        "supergemm_noepi": sg_noepi,
    }, (a, bt.t()))


# The residual above is not the C read path (that is already patched out of
# `cuda_ref_g1_scalarc`). The extracted code differs from the reference in one
# further way visible in SASS: KaRaMeL leaves the drain's vector-group loop
# rolled, so 8 C loads are in flight where the reference's unrolled loop has
# 32. Rolling that one loop in the reference tests whether it accounts for the
# gap.
print("\n\n########## vector-group loop unrolling ##########")
for name, M, K, N in CASES:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    common = dict(stem="gemm_tc_flat_nosplitk_epi", group=1, **TILE,
                  cs_m=N, cs_n=1, force_scalar_c=True)
    unrolled = load_ref(**common)
    rolled = load_ref(**common, roll_v=True)
    ab(f"{name} drain unrolling (dense C, scalar C read)", {
        "ref_scalarc_unrolled_v": lambda C, A, B: unrolled(C, A, B, ALPHA,
                                                           BETA),
        "ref_scalarc_ROLLED_v": lambda C, A, B: rolled(C, A, B, ALPHA, BETA),
        "supergemm_epi": lambda C, A, B: sg_epi(C, A, B, alpha=ALPHA,
                                                beta=BETA),
    }, (c, a, bt.t()))
