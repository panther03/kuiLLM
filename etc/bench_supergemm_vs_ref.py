"""SuperGEMM (verified, Kuiper) vs its unverified CUDA reference.

``kuiops/common/supergemm/mm`` is a port of
``kuipy/unverified/gemm_tc_flat_nosplitk_noepi.cu``, so the two should land
within noise of each other: any gap is Kuiper/extraction overhead, not
algorithm. Both are timed by the same driver, at the SAME tiling, on the Qwen
decode shapes. ``tc2d_tn`` (the previous Kuiper tensor-core mm) and cuBLAS are
shown for scale.
"""
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

# The tiling SuperGEMM's own candidate ordering picks first for every shape
# above (bm/bn 128, bk 32, 8 warps of 32x64, skew = chunk). The reference is a
# fixed specialization, so it must be compiled at the same point to be a
# like-for-like comparison rather than a tuning contest.
TILE = dict(bm=128, bn=128, bk=32, wm=32, wn=64, skew=8)


def make_inputs(m, k, n):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return (a, bt.t()), {}


op = torch.ops.aten.mm.default
impls = [
    ("supergemm", run(op, impl="supergemm")),
    ("cuda_ref", load_ref(**TILE)),
    # GROUP=1 degenerates the reference's L2 swizzle to the plain row-major
    # block decode SuperGEMM currently uses, isolating the swizzle's effect.
    ("cuda_ref_g1", load_ref(**TILE, group=1)),
    ("tc2d_tn", run(op, impl="tc2d_tn")),
]

df = bench_matrix(
    cases=CASES,
    make_inputs=make_inputs,
    case_columns=lambda m, k, n: (m, k, n),
    case_column_names=("M", "K", "N"),
    impls=impls,
    reference=torch.mm,
    flops=lambda m, k, n: 2 * m * k * n,
)

import pandas as pd
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
