"""Split-K + epilogue SuperGEMM (verified, Kuiper) vs its unverified reference.

``kuiops/common/supergemm/mm_splitk_epi`` is a port of
``kuipy/unverified/gemm_tc_flat_splitk_epi.cu``, which computes
``D = beta*C + alpha*(A @ B^T)`` with k split SPLITS ways, so those two are the
honest comparison. The non-split epilogue kernel and the no-epi split-K kernel
are shown alongside to separate "what split-K costs" from "what the epilogue
costs".

C is read through a generic view, so the two C shapes the reference's
CS_M/CS_N select at compile time -- a dense (M, N) matrix and a length-N row
bias -- are benchmarked separately (``--bcast``). The reference vectorizes its
C read when the view allows it while the port always reads C scalarly, so
``cuda_ref_splitk_epi_sc`` is the reference patched onto its own scalar path:
that is the like-for-like column, and the difference between it and
``cuda_ref_splitk_epi`` is the cost of the deliberate simplification.

The timing settings are the ones ``bench_supergemm_splitk.py`` justifies: the
split-K launcher blocks the CPU on a stream sync, so the GPU idles between
iterations and its clocks need a long settle. The whole matrix is run twice and
only the second is reported.

Run with:
  micromamba run -n kuillm env PYTHONPATH=. python etc/bench_supergemm_splitk_epi.py
  micromamba run -n kuillm env PYTHONPATH=. python etc/bench_supergemm_splitk_epi.py --bcast
"""
import sys

import torch

from kuipy import run, kuiops
from kuipy.benchmarking import bench_matrix
from etc.ref_flat_gemm import load as load_ref
from etc.ref_flat_gemm_splitk import load as load_ref_splitk

aten = torch.ops.aten

# The shapes the reference's own measurements cite, written there as M x N x K;
# here as (M, K, N). Each is paired with the split count the reference cites.
# K/splits must stay divisible by bk.
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
ALPHA, BETA = 0.5, 2.0

BCAST = "--bcast" in sys.argv


def _plan(a, b):
    return PLANS[(int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))]


def make_inputs(m, k, n):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    cshape = (n,) if BCAST else (m, n)
    c = torch.randn(*cshape, device="cuda", dtype=torch.bfloat16)
    return (c, a, bt.t()), dict(alpha=ALPHA, beta=BETA)


_impl = kuiops.AddmmImpl()
_mm = kuiops.MmImpl()


def kuiper_splitk_epi(c, a, bt, alpha, beta):
    """Kuiper split-K + epilogue, pinned to the case's tiling and split count."""
    tile, splits = _plan(a, bt)
    spec = _impl._spec("supergemm_splitk_epi",
                       dict(**tile, group=_GROUP, splits=splits),
                       torch.bfloat16, torch.float32, torch.bfloat16,
                       cbcast=BCAST)
    return _impl.run(spec, (c, a, bt), dict(alpha=alpha, beta=beta))


def kuiper_epi_nosplit(c, a, bt, alpha, beta):
    """Kuiper non-split epilogue kernel at the same tiling."""
    tile, _ = _plan(a, bt)
    spec = _impl._spec("supergemm", dict(**tile, group=_GROUP),
                       torch.bfloat16, torch.float32, torch.bfloat16,
                       cbcast=BCAST)
    return _impl.run(spec, (c, a, bt), dict(alpha=alpha, beta=beta))


def _cs(n):
    return (0, 1) if BCAST else (n, 1)


def _ref(a, bt, force_scalar_c):
    tile, splits = _plan(a, bt)
    cs_m, cs_n = _cs(int(bt.shape[1]))
    return load_ref_splitk(stem="gemm_tc_flat_splitk_epi", splits=splits,
                           group=_GROUP, **tile, cs_m=cs_m, cs_n=cs_n,
                           force_scalar_c=force_scalar_c)


def cuda_ref_splitk_epi(c, a, bt, alpha, beta):
    return _ref(a, bt, False)(c, a, bt, alpha, beta)


def cuda_ref_splitk_epi_sc(c, a, bt, alpha, beta):
    """The reference forced onto its own scalar-C path: the like-for-like one."""
    return _ref(a, bt, True)(c, a, bt, alpha, beta)


def cuda_ref_epi_nosplit(c, a, bt, alpha, beta):
    tile, _ = _plan(a, bt)
    cs_m, cs_n = _cs(int(bt.shape[1]))
    fn = load_ref(stem="gemm_tc_flat_nosplitk_epi", group=_GROUP, **tile,
                  cs_m=cs_m, cs_n=cs_n)
    return fn(c, a, bt, alpha, beta)


def torch_ref(c, a, bt, alpha, beta):
    return torch.addmm(c, a, bt, alpha=alpha, beta=beta)


impls = [
    ("kuiper_splitk_epi", kuiper_splitk_epi),
    ("cuda_ref_splitk_epi_sc", cuda_ref_splitk_epi_sc),
    ("cuda_ref_splitk_epi", cuda_ref_splitk_epi),
    ("kuiper_epi_nosplit", kuiper_epi_nosplit),
    ("cuda_ref_epi_nosplit", cuda_ref_epi_nosplit),
]


def run_matrix():
    return bench_matrix(
        cases=CASES,
        make_inputs=make_inputs,
        case_columns=lambda m, k, n: (m, k, n, PLANS[(m, k, n)][1]),
        case_column_names=("M", "K", "N", "splits"),
        impls=impls,
        reference=torch_ref,
        flops=lambda m, k, n: 2 * m * k * n,
        iters=200, warmup=50, warmup_ms=500, settle_ms=200,
    )


run_matrix()
df = run_matrix()

import pandas as pd
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 60)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
