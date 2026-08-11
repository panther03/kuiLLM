"""SuperGEMM epilogue (verified, Kuiper) vs its unverified CUDA reference.

``kuiops/common/supergemm/mm_epi`` is a port of
``kuipy/unverified/gemm_tc_flat_nosplitk_epi.cu``, which computes
``D = beta*C + alpha*(A @ B^T)``. Both are timed by the same driver at the SAME
tiling on the Qwen decode shapes; the existing broadcast-C Kuiper epilogues are
shown for scale.

C is read through a generic view, so the two C shapes the reference's CS_M/CS_N
select at compile time -- a dense (M, N) matrix and a length-N row bias -- are
benchmarked separately. Note the reference vectorizes its C read in both of
these configurations while the port always reads C scalarly, so this measures
the cost of that deliberate simplification.
"""
import sys

import torch

from kuipy import run
from kuipy.benchmarking import bench_matrix
from etc.ref_flat_gemm import load as load_ref

CASES = [
    ("qkv_proj", 256, 896, 896),
    ("mlp_up", 256, 896, 4864),
    ("mlp_down", 256, 4864, 896),
    ("lm_head", 256, 896, 151936),
    ("square4096", 4096, 4096, 4096),
]

TILE = dict(bm=128, bn=128, bk=32, wm=32, wn=64, skew=8)

ALPHA, BETA = 0.5, 2.0

BCAST = "--bcast" in sys.argv


def make_inputs(m, k, n):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    cshape = (n,) if BCAST else (m, n)
    c = torch.randn(*cshape, device="cuda", dtype=torch.bfloat16)
    return (c, a, bt.t()), dict(alpha=ALPHA, beta=BETA)


def ref_epi(m, k, n, group=8):
    """The unverified kernel, adapted to the driver's calling convention."""
    fn = load_ref(stem="gemm_tc_flat_nosplitk_epi", group=group, **TILE,
                  cs_m=0 if BCAST else n, cs_n=1)
    return lambda c, a, bt, alpha, beta: fn(c, a, bt, alpha, beta)


op = torch.ops.aten.addmm.default
kuiper = [
    ("supergemm", run(op, impl="supergemm")),
    ("tc2d_tn_bcast" if BCAST else "tc2d_to", run(
        op, impl="tc2d_tn_bcast" if BCAST else "tc2d_to")),
]


def impls_for(m, k, n):
    return kuiper + [
        ("cuda_ref", ref_epi(m, k, n)),
        # GROUP=1 degenerates the reference's L2 swizzle to the plain
        # row-major block decode SuperGEMM currently uses.
        ("cuda_ref_g1", ref_epi(m, k, n, group=1)),
    ]


import pandas as pd

frames = []
for case in CASES:
    frames.append(bench_matrix(
        cases=[case],
        make_inputs=make_inputs,
        case_columns=lambda m, k, n: (m, k, n),
        case_column_names=("M", "K", "N"),
        impls=impls_for(*case[1:]),
        reference=torch.addmm,
        flops=lambda m, k, n: 2 * m * k * n,
    ))
df = pd.concat(frames, ignore_index=True)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(f"C shape: {'(N,) row bias' if BCAST else '(M, N) dense'}, "
      f"alpha={ALPHA} beta={BETA}")
print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
