"""Benchmark the SuperGEMM backend against the other mm backends and PyTorch
on the Qwen2.5-0.5B decode shapes (D = A @ B^T, B stored row-major (n, k))."""
import torch

from kuipy import run
from kuipy.benchmarking import bench_matrix

CASES = [
    ("qkv_proj", 256, 896, 896),
    ("mlp_up", 256, 896, 4864),
    ("mlp_down", 256, 4864, 896),
    ("lm_head", 256, 896, 151936),
    ("square4096", 4096, 4096, 4096),
]


def make_inputs(m, k, n):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return (a, bt.t()), {}


op = torch.ops.aten.mm.default
impls = [(name, run(op, impl=name))
         for name in ("supergemm", "tc2d_tn")]

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
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
