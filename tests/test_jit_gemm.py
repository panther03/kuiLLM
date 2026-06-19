"""Correctness + caching tests for the JIT Kuiper GEMM pipeline.

These extract, compile and run on-demand Kuiper kernels, so the FIRST run of a
new instantiation is slow (F* + nvcc, tens of seconds). Subsequent runs hit the
in-process / on-disk cache and are fast.

Run with:  cd /home/julien/work/kuiLLM && KUIPY_JIT_STRICT=1 .venv/bin/python -m pytest tests/test_jit_gemm.py -s
"""
import sys
import time

import torch

import pytest

from kuipy import gemm
from kuipy import compile as jit_compile

_DEVICE = "cuda" if torch.cuda.is_available() else None
aten = torch.ops.aten


def _need_cuda():
    if _DEVICE is None:
        pytest.skip("CUDA not available")


def _run(impl, func, args, kwargs):
    spec = impl.supported(func, args, kwargs)
    assert spec is not None, "expected a supported spec"
    return impl.run(spec, args, kwargs)


def test_mm_f32():
    _need_cuda()
    impl = gemm.GemmImpl()
    A = torch.randn(256, 64, device="cuda")
    B = torch.randn(64, 128, device="cuda")
    out = _run(impl, aten.mm.default, (A, B), {})
    assert torch.allclose(out, A @ B, atol=1e-3, rtol=1e-3)


def test_addmm_f32_with_alpha_beta():
    _need_cuda()
    impl = gemm.GemmImpl()
    M, N, K = 128, 128, 64
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    bias = torch.randn(N, device="cuda")
    ref = torch.addmm(bias, A, B, beta=0.5, alpha=2.0)
    out = _run(impl, aten.addmm.default, (bias, A, B), {"beta": 0.5, "alpha": 2.0})
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_mm_bf16_tensorcore():
    _need_cuda()
    impl = gemm.GemmImpl()
    A = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    out = _run(impl, aten.mm.default, (A, B), {})
    ref = A.float() @ B.float()
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert out.dtype == torch.bfloat16
    assert rel < 2e-2


def test_unsupported_falls_through():
    _need_cuda()
    impl = gemm.GemmImpl()
    # K=63 is not divisible by any valid block-K tile -> no plan.
    A = torch.randn(128, 63, device="cuda")
    B = torch.randn(63, 128, device="cuda")
    assert impl.supported(aten.mm.default, (A, B), {}) is None
    # CPU tensors are never supported.
    Ac = torch.randn(64, 64)
    Bc = torch.randn(64, 64)
    assert impl.supported(aten.mm.default, (Ac, Bc), {}) is None


def test_cache_is_hot_on_second_call():
    _need_cuda()
    impl = gemm.GemmImpl()
    A = torch.randn(64, 64, device="cuda")
    B = torch.randn(64, 64, device="cuda")
    # First call may compile; second must be served from the in-process cache.
    _run(impl, aten.mm.default, (A, B), {})
    ext_names_before = set(jit_compile._loaded.keys())
    t0 = time.time()
    _run(impl, aten.mm.default, (A, B), {})
    dt = time.time() - t0
    # No new extension compiled, and the call is fast (no nvcc/F*).
    assert set(jit_compile._loaded.keys()) == ext_names_before
    assert dt < 2.0, f"hot call took {dt:.2f}s, expected <2s"


def test_kuiper_mode_integration():
    _need_cuda()
    import kuipy
    A = torch.randn(128, 64, device="cuda")
    B = torch.randn(64, 128, device="cuda")
    bias = torch.randn(128, device="cuda")
    with kuipy.KuiperMode():
        o_mm = torch.mm(A, B)
        o_am = torch.addmm(bias, A, B)
    assert torch.allclose(o_mm, A @ B, atol=1e-3, rtol=1e-3)
    assert torch.allclose(o_am, torch.addmm(bias, A, B), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
