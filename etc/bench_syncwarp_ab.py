"""A/B the epilogue's post-inner-loop __syncwarp on the extracted CUDA.

The per-store __syncwarp inside the drain loop is load-bearing: Kuiper's spec
for [store_matrix_sync] says the result is immediately visible, which only
holds if the warp reconverges right after, and there is no primitive that
models what the intrinsic actually does. The one that closes the *outer* body,
however, is issued immediately after the last per-store sync, so it is pure
overhead.

Kuipy caches the extracted .cu, so the variants are produced by patching that
file and dropping only the built artifacts for the instantiation. Variants are
alternated across repeats so clock drift cancels rather than biasing whichever
ran first.
"""
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch

import kuipy.compile as _compile
from kuipy import run
from kuipy.benchmarking import time_call, warm_up, rel_fro

MODULE = ("Kuiops.Mm.SuperGEMM.Bf16_F32_Bf16_P"
          "_bm128_bn128_bk32_wm32_wn64_skew8_group1")
CU = Path(".kuipy_cache/cu") / (MODULE.replace(".", "_") + ".cu")
PRISTINE = Path("/tmp/sg_pristine.cu")

CASES = [
    ("qkv_proj", 256, 896, 896),
    ("mlp_up", 256, 896, 4864),
    ("mlp_down", 256, 4864, 896),
    ("lm_head", 256, 896, 151936),
    ("square4096", 4096, 4096, 4096),
]


def patch(drop_trailing):
    """Rewrite the cached .cu, optionally dropping the redundant __syncwarp.

    The target is the __syncwarp that sits between the close of the inner
    store loop and the [blocksN] binding -- matched structurally rather than
    by line number so a re-extraction cannot silently shift it.
    """
    src = PRISTINE.read_text()
    if drop_trailing:
        needle = "        }\n        __syncwarp();\n        uint32_t blocksN"
        assert src.count(needle) == 1, "anchor not unique; .cu shape changed"
        src = src.replace(needle, "        }\n        uint32_t blocksN")
    CU.write_text(src)
    _compile.delete_kernel(MODULE)
    # delete_kernel also removes the .cu, so restore it afterwards
    CU.write_text(src)


def measure():
    mm = run(torch.ops.aten.mm.default, impl="supergemm")
    out = {}
    for name, m, k, n in CASES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        bt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        b = bt.t()
        warm_up(50)
        time_call(lambda: mm(a, b), iters=50, warmup=10)  # prime
        warm_up(50)
        t, got = time_call(lambda: mm(a, b), iters=50, warmup=10)
        out[name] = (t, rel_fro(got, torch.mm(a, b)))
    return out


if __name__ == "__main__":
    variant = sys.argv[1]
    patch(drop_trailing=(variant == "nosync"))
    res = measure()
    for name, (t, err) in res.items():
        print(f"{variant}\t{name}\t{t:.4f}\t{err:.6g}")
