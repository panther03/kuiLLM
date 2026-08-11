"""Split-K SuperGEMM (verified, Kuiper) vs its unverified CUDA reference.

``kuiops/common/supergemm/mm_splitk`` is a port of
``kuipy/unverified/gemm_tc_flat_splitk_noepi.cu``, so those two are the honest
comparison. The non-split Kuiper kernel and its own CUDA reference are shown
alongside, because split-K only pays off where the non-split grid underfills
the GPU: the shapes below are the skinny-M / large-K ones the reference's own
measurements cite, each at the split count it cites.

The timing settings are deliberately generous: the split-K launcher blocks the
CPU on a stream sync, so the GPU idles between iterations and needs a much
longer settle than the back-to-back kernels do before its clocks stop climbing
(a 50ms settle read it 70% slow).

Both split-K contenders are pinned to the SAME tiling and split count per case
-- the Kuiper one by building its spec directly rather than going through
autotuning -- so this is a port-quality comparison and not a tuning contest.

Run with:  micromamba run -n kuillm env PYTHONPATH=. python etc/bench_supergemm_splitk.py
"""
import torch

from kuipy import run, kuiops
from kuipy.benchmarking import bench_matrix
from etc.ref_flat_gemm import load as load_ref
from etc.ref_flat_gemm_splitk import load as load_ref_splitk

aten = torch.ops.aten

# The four shapes the reference's own measurements cite, written there as
# M x N x K; here as (M, K, N). Each is paired with the split count the
# reference cites for it. K/splits must stay divisible by bk.
CASES = [
    ("down_proj_sp4",  256, 4864, 896),
    ("down_proj_m128_sp8", 128, 4864, 896),
    ("qkv_proj_sp2",   256, 896, 1152),
    ("mlp_up_sp2",     256, 896, 4864),
]

_TILE = dict(bm=128, bn=128, bk=32, wm=32, wn=64, skew=8)

PLANS = {
    (256, 4864, 896):  (_TILE, 4),
    (128, 4864, 896):  (_TILE, 8),
    (256, 896, 1152):  (_TILE, 2),
    (256, 896, 4864):  (_TILE, 2),
}

_GROUP = 1  # what the Kuiper kernels are instantiated at


def _plan(a, b):
    return PLANS[(int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))]


def make_inputs(m, k, n):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return (a, bt.t()), {}


_impl = kuiops.MmImpl()


def kuiper_splitk(a, b):
    """Kuiper split-K, pinned to the case's tiling and split count."""
    tile, splits = _plan(a, b)
    spec = _impl._spec("supergemm_splitk",
                       dict(**tile, group=_GROUP, splits=splits),
                       torch.bfloat16, torch.float32, torch.bfloat16)
    return _impl.run(spec, (a, b), {})


def kuiper_nosplit(a, b):
    """Kuiper non-split SuperGEMM at the same tiling."""
    tile, _ = _plan(a, b)
    spec = _impl._spec("supergemm", dict(**tile, group=_GROUP),
                       torch.bfloat16, torch.float32, torch.bfloat16)
    return _impl.run(spec, (a, b), {})


def cuda_ref_splitk(a, b):
    tile, splits = _plan(a, b)
    return load_ref_splitk(splits=splits, group=_GROUP, **tile)(a, b)


def cuda_ref_nosplit(a, b):
    tile, _ = _plan(a, b)
    return load_ref(group=_GROUP, **tile)(a, b)


impls = [
    ("kuiper_splitk", kuiper_splitk),
    ("cuda_ref_splitk", cuda_ref_splitk),
    ("kuiper_nosplit", kuiper_nosplit),
    ("cuda_ref_nosplit", cuda_ref_nosplit),
    ("tc2d_tn", run(aten.mm.default, impl="tc2d_tn")),
]

def run_matrix():
    return bench_matrix(
        cases=CASES,
        make_inputs=make_inputs,
        case_columns=lambda m, k, n: (m, k, n, PLANS[(m, k, n)][1]),
        case_column_names=("M", "K", "N", "splits"),
        impls=impls,
        reference=torch.mm,
        flops=lambda m, k, n: 2 * m * k * n,
        iters=200, warmup=50, warmup_ms=500, settle_ms=200,
    )


# The whole matrix is run twice and only the second is reported: the first
# measurement of a contender that blocks on a stream sync reads ~70% slow
# however long the settle is, because its duty cycle -- not the settle -- is
# what the clocks track.
run_matrix()
df = run_matrix()

import pandas as pd
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 60)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
