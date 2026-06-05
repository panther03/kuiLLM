"""Correctness tests for kuiper_ext wrappers and the KuiperLinear module.

Run with:  cd /home/julien/work/kuiLLM && .venv/bin/python -m pytest tests/

(Or directly:  PATH=$PWD/.venv/bin:$PATH .venv/bin/python tests/test_kuiper_ops.py)
"""

import math
import sys

import torch
import torch.nn as nn

import kuiper_ext

_DEVICE = "cuda" if torch.cuda.is_available() else None


def _need_cuda():
    if _DEVICE is None:
        print("CUDA not available, skipping")
        sys.exit(0)


def _assert_close(name, got, ref, atol, rtol):
    err = (got - ref).abs().max().item()
    rerr = (got - ref).abs().div(ref.abs().clamp(min=1)).max().item()
    print(f"  {name:30s}  max_abs={err:.3e}  max_rel={rerr:.3e}")
    assert err <= atol or rerr <= rtol, f"{name}: error too large ({err}, {rerr})"


def test_mm_bf16():
    print("[test_mm_bf16]")
    torch.manual_seed(0)
    for M, K, N in [(64, 64, 64), (64, 128, 128), (192, 256, 64)]:
        Ar = torch.randn(M, K, device="cpu", dtype=torch.bfloat16)
        A =  Ar.to(device=_DEVICE)
        Br = torch.randn(K, N, device="cpu", dtype=torch.bfloat16)
        B =  Br.to(device=_DEVICE)

        res = kuiper_ext.mm_bf16xbf16_bf16(A, B)
        _assert_close(f"mm_bf16 {M}x{K}x{N}",
                      res.to(device="cpu"), Ar @ Br,
                      atol=1e-2, rtol=1e-3)

def test_bmm_f32():
    print("[test_bmm_f32]")
    torch.manual_seed(0)
    for batch, M, K, N in [(3, 64, 64, 64), (3, 431, 69, 50), (6, 100, 128, 100)]:
        Ar = torch.randn(batch, M, K, device="cpu", dtype=torch.float32)
        A =  Ar.to(device=_DEVICE)
        Br = torch.randn(batch, K, N, device="cpu", dtype=torch.float32)
        B =  Br.to(device=_DEVICE)

        res = kuiper_ext.bmm_f32xf32_f32(A, B)
        _assert_close(f"bmm_f32 {batch}x{M}x{K}x{N}",
                      res.to(device="cpu"), torch.bmm(Ar, Br),
                      atol=1e-3, rtol=1e-4)

def main():
    _need_cuda()
    print(f"Using device: {_DEVICE}")
    print(f"kuiper_ext available: {kuiper_ext.is_available()}")
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
