"""Correctness tests for kuiper_ext wrappers and the KuiperLinear module.

Run with:  cd /home/julien/work/kuiLLM && .venv/bin/python -m pytest tests/

(Or directly:  PATH=$PWD/.venv/bin:$PATH .venv/bin/python tests/test_kuiper_ops.py)
"""

import math
import sys

import torch
import torch.nn as nn

import kuiper_ext
from kuiper_ext.integration import KuiperLinear, enable_matmul_patch, disable_matmul_patch


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


def test_matmul_f32():
    print("[test_matmul_f32]")
    torch.manual_seed(0)
    for M, K, N in [(32, 32, 32), (64, 96, 128), (128, 64, 192)]:
        A = torch.randn(M, K, device=_DEVICE, dtype=torch.float32)
        B = torch.randn(K, N, device=_DEVICE, dtype=torch.float32)
        _assert_close(f"matmul_f32 {M}x{K}x{N}",
                      kuiper_ext.matmul_f32(A, B), A @ B,
                      atol=1e-3, rtol=1e-4)


def test_matmul_f16():
    print("[test_matmul_f16]")
    torch.manual_seed(0)
    for M, K, N in [(64, 64, 64), (128, 64, 256), (192, 128, 64)]:
        A = torch.randn(M, K, device=_DEVICE, dtype=torch.float16)
        B = torch.randn(K, N, device=_DEVICE, dtype=torch.float16)
        ref = (A.float() @ B.float()).half()
        _assert_close(f"matmul_f16 {M}x{K}x{N}",
                      kuiper_ext.matmul_f16(A, B), ref,
                      atol=0.1, rtol=0.01)


def test_gemm_f32_in_place():
    print("[test_gemm_f32_in_place]")
    torch.manual_seed(0)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device=_DEVICE, dtype=torch.float32)
    B = torch.randn(K, N, device=_DEVICE, dtype=torch.float32)
    C = torch.randn(M, N, device=_DEVICE, dtype=torch.float32)
    C0 = C.clone()
    kuiper_ext.gemm_f32_(2.0, 0.5, A, B, C)
    ref = 2.0 * (A @ B) + 0.5 * C0
    _assert_close("gemm_f32_", C, ref, atol=1e-3, rtol=1e-4)


def test_bmm_f32():
    print("[test_bmm_f32]")
    torch.manual_seed(0)
    A = torch.randn(3, 32, 64, device=_DEVICE, dtype=torch.float32)
    B = torch.randn(3, 64, 96, device=_DEVICE, dtype=torch.float32)
    _assert_close("bmm_f32", kuiper_ext.bmm_f32(A, B), torch.bmm(A, B),
                  atol=1e-3, rtol=1e-4)


def test_softmax_bounded():
    """Kuiper softmax has no max-shift; only test small bounded logits."""
    print("[test_softmax_bounded]")
    torch.manual_seed(0)
    x = torch.randn(8, 16, device=_DEVICE, dtype=torch.float32) * 0.3
    ref = torch.softmax(x, dim=-1)
    y = x.clone()
    kuiper_ext.softmax_last_f32_(y)
    _assert_close("softmax_f32 bounded", y, ref, atol=1e-5, rtol=1e-5)


def test_log_softmax_bounded():
    print("[test_log_softmax_bounded]")
    torch.manual_seed(0)
    x = torch.randn(4, 16, device=_DEVICE, dtype=torch.float32) * 0.2
    ref = torch.log_softmax(x, dim=-1)
    y = x.clone()
    kuiper_ext.log_softmax_last_f32_(y)
    _assert_close("log_softmax_f32 bounded", y, ref, atol=1e-4, rtol=1e-4)


def test_row_sum_last():
    print("[test_row_sum_last]")
    torch.manual_seed(0)
    x = torch.randn(4, 64, device=_DEVICE, dtype=torch.float32)
    _assert_close("row_sum_last", kuiper_ext.row_sum_last_f32(x, 32), x.sum(dim=-1),
                  atol=1e-4, rtol=1e-5)


def test_row_scale():
    print("[test_row_scale]")
    torch.manual_seed(0)
    m = torch.randn(8, 16, device=_DEVICE, dtype=torch.float32)
    s = torch.randn(8, device=_DEVICE, dtype=torch.float32)
    ref = m * s.unsqueeze(-1)
    mm = m.clone()
    kuiper_ext.row_scale_f32_(mm, s)
    _assert_close("row_scale", mm, ref, atol=0, rtol=0)


def test_kuiper_linear_f32():
    print("[test_kuiper_linear_f32]")
    torch.manual_seed(0)
    lin = nn.Linear(64, 96, bias=True).to(_DEVICE, dtype=torch.float32)
    x = torch.randn(32, 64, device=_DEVICE, dtype=torch.float32)
    klin = KuiperLinear.from_linear(lin)
    _assert_close("KuiperLinear f32", klin(x), lin(x), atol=1e-4, rtol=1e-5)


def test_matmul_bf16():
    print("[test_matmul_bf16]")
    torch.manual_seed(0)
    for M, K, N in [(64, 64, 64), (128, 64, 256), (192, 128, 64)]:
        A = torch.randn(M, K, device=_DEVICE, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=_DEVICE, dtype=torch.bfloat16)
        ref = (A.float() @ B.float())  # full-precision reference
        out_f32 = kuiper_ext.matmul_bf16_to_f32(A, B)
        # f32 acc + bf16 inputs: relative error should be in the bf16 ULP range.
        _assert_close(f"matmul_bf16_to_f32 {M}x{K}x{N}", out_f32, ref,
                      atol=0.1, rtol=0.02)


def test_kuiper_linear_bf16():
    print("[test_kuiper_linear_bf16]")
    torch.manual_seed(0)
    lin = nn.Linear(128, 256, bias=True).to(_DEVICE, dtype=torch.bfloat16)
    x = torch.randn(64, 128, device=_DEVICE, dtype=torch.bfloat16)
    klin = KuiperLinear.from_linear(lin)   # default cast_bf16_to_f16=False
    _assert_close("KuiperLinear bf16 (native)", klin(x), lin(x),
                  atol=0.5, rtol=0.05)


def test_kuiper_linear_f16():
    print("[test_kuiper_linear_f16]")
    torch.manual_seed(0)
    lin = nn.Linear(128, 256, bias=True).to(_DEVICE, dtype=torch.float16)
    x = torch.randn(64, 128, device=_DEVICE, dtype=torch.float16)
    klin = KuiperLinear.from_linear(lin)
    _assert_close("KuiperLinear f16", klin(x), lin(x), atol=0.5, rtol=0.05)


def test_kuiper_linear_fallback():
    print("[test_kuiper_linear_fallback]")
    torch.manual_seed(0)
    lin = nn.Linear(64, 96, bias=True).to(_DEVICE, dtype=torch.float32)
    # M=63 not multiple of 32 → fallback path
    x = torch.randn(63, 64, device=_DEVICE, dtype=torch.float32)
    klin = KuiperLinear.from_linear(lin)
    _assert_close("KuiperLinear fallback shape", klin(x), lin(x), atol=1e-4, rtol=1e-5)
    # bf16 with shape-incompatible M (not multiple of 64) → fallback
    lin = nn.Linear(64, 96, bias=True).to(_DEVICE, dtype=torch.bfloat16)
    x = torch.randn(32, 64, device=_DEVICE, dtype=torch.bfloat16)
    klin = KuiperLinear.from_linear(lin)
    _assert_close("KuiperLinear fallback bf16 shape", klin(x), lin(x), atol=0, rtol=0)


def test_matmul_patch():
    print("[test_matmul_patch]")
    torch.manual_seed(0)
    A = torch.randn(64, 96, device=_DEVICE, dtype=torch.float32)
    B = torch.randn(96, 128, device=_DEVICE, dtype=torch.float32)
    ref = torch.matmul(A, B)
    enable_matmul_patch()
    try:
        got = torch.matmul(A, B)
    finally:
        disable_matmul_patch()
    _assert_close("matmul patch", got, ref, atol=1e-3, rtol=1e-4)


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
