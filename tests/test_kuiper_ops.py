"""Correctness tests for the JIT Kuiper dispatcher (kuiper_ext.KuiperMode).

Every eligible aten op is intercepted and served by an on-demand extracted +
compiled Kuiper kernel. References are computed OUTSIDE the mode (stock
PyTorch). The first run of each instantiation compiles (slow); later runs are
cached.

Run with:  cd /home/julien/work/kuiLLM && KUIPER_JIT_STRICT=1 .venv/bin/python -m pytest tests/test_kuiper_ops.py -s
"""
import sys

import torch
import pytest

import kuiper_ext

_DEVICE = "cuda" if torch.cuda.is_available() else None


def _need_cuda():
    if _DEVICE is None:
        pytest.skip("CUDA not available")


def _rel(got, ref):
    return ((got.float() - ref.float()).abs().max()
            / ref.float().abs().clamp(min=1e-6).max()).item()


def test_is_available():
    assert kuiper_ext.is_available()


def test_mm_f32():
    _need_cuda()
    A = torch.randn(256, 64, device="cuda")
    B = torch.randn(64, 128, device="cuda")
    ref = A @ B
    with kuiper_ext.KuiperMode():
        out = torch.mm(A, B)
    assert _rel(out, ref) < 1e-3


def test_mm_bf16():
    _need_cuda()
    A = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    ref = A.float() @ B.float()
    with kuiper_ext.KuiperMode():
        out = torch.mm(A, B)
    assert out.dtype == torch.bfloat16
    assert _rel(out, ref) < 2e-2


def test_addmm_f32():
    _need_cuda()
    M, K, N = 128, 64, 128
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    bias = torch.randn(N, device="cuda")
    ref = torch.addmm(bias, A, B, beta=0.5, alpha=2.0)
    with kuiper_ext.KuiperMode():
        out = torch.addmm(bias, A, B, beta=0.5, alpha=2.0)
    assert _rel(out, ref) < 1e-3


def test_linear_bf16():
    _need_cuda()
    lin = torch.nn.Linear(64, 128, bias=True).cuda().to(torch.bfloat16)
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    ref = lin(x)
    with kuiper_ext.KuiperMode():
        out = lin(x)
    assert out.dtype == torch.bfloat16
    assert _rel(out, ref) < 5e-2


def test_bmm_f32():
    _need_cuda()
    A = torch.randn(3, 64, 32, device="cuda")
    B = torch.randn(3, 32, 16, device="cuda")
    ref = torch.bmm(A, B)
    with kuiper_ext.KuiperMode():
        out = torch.bmm(A, B)
    assert _rel(out, ref) < 1e-3


def test_elementwise():
    _need_cuda()
    xb = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    yb = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    xf = torch.rand(256, device="cuda") + 0.5
    yf = torch.randn(256, device="cuda")

    refs = {
        "silu": torch.nn.functional.silu(xb.float()),
        "neg": -xb.float(),
        "add_bf16": xb.float() + yb.float(),
        "mul_bf16": xb.float() * yb.float(),
        "rsqrt": torch.rsqrt(xf),
        "square": yf * yf,
        "sin": torch.sin(yf),
        "cos": torch.cos(yf),
        "mul_f32": xf * yf,
        "add_const": xf + 1e-3,
        "mul_const": xf * 0.5,
    }
    with kuiper_ext.KuiperMode():
        got = {
            "silu": torch.nn.functional.silu(xb),
            "neg": torch.neg(xb),
            "add_bf16": xb + yb,
            "mul_bf16": xb * yb,
            "rsqrt": torch.rsqrt(xf),
            "square": yf ** 2,
            "sin": torch.sin(yf),
            "cos": torch.cos(yf),
            "mul_f32": xf * yf,
            "add_const": xf + 1e-3,
            "mul_const": xf * 0.5,
        }
    for k, ref in refs.items():
        assert _rel(got[k], ref) < 2e-2, f"{k}: rel too large"


def test_mean_f32_lastdim():
    _need_cuda()
    x = torch.randn(64, 128, device="cuda")
    ref = x.mean(dim=-1, keepdim=True)
    with kuiper_ext.KuiperMode():
        out = torch.mean(x, dim=-1, keepdim=True)
    assert out.shape == ref.shape
    assert _rel(out, ref) < 1e-4


def test_cat_bf16_1d():
    _need_cuda()
    a = torch.randn(10, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(7, device="cuda", dtype=torch.bfloat16)
    ref = torch.cat([a, b])
    with kuiper_ext.KuiperMode():
        out = torch.cat([a, b])
    assert torch.equal(out, ref)


def test_cast():
    _need_cuda()
    xb = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    xf = torch.randn(64, device="cuda")
    with kuiper_ext.KuiperMode():
        to_f32 = xb.to(torch.float32)
        to_bf16 = xf.to(torch.bfloat16)
    assert torch.equal(to_f32, xb.float())
    assert torch.equal(to_bf16, xf.to(torch.bfloat16))


def test_arange_i64():
    _need_cuda()
    ref = torch.arange(16, device="cuda")
    with kuiper_ext.KuiperMode():
        out = torch.arange(16, device="cuda")
    assert out.dtype == torch.int64
    assert torch.equal(out, ref)


def test_gather_bf16():
    _need_cuda()
    src = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, 8, (5, 4), device="cuda")
    ref = torch.gather(src, 0, idx)
    with kuiper_ext.KuiperMode():
        out = torch.gather(src, 0, idx)
    assert torch.equal(out, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
