"""GROUP sweep of the unverified reference's L2 block-index swizzle.

GROUP is pure scheduling -- any value is correct -- so the only way to pick the
default SuperGEMM compiles with is to measure it. Sweeping the reference rather
than SuperGEMM keeps this independent of the F* toolchain.
"""
import torch

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
GROUPS = (1, 2, 4, 8, 16)


def make_inputs(m, k, n):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return (a, bt.t()), {}


df = bench_matrix(
    cases=CASES,
    make_inputs=make_inputs,
    case_columns=lambda m, k, n: (m, k, n),
    case_column_names=("M", "K", "N"),
    impls=[(f"g{g}", load_ref(**TILE, group=g)) for g in GROUPS],
    reference=torch.mm,
    flops=lambda m, k, n: 2 * m * k * n,
)

import pandas as pd
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
