"""Correctness tests for the JIT-compiled Kuiper bmm / addmm / softmax / sdpa ops.

The first run of each new instantiation compiles a kernel (F* + nvcc, tens of
seconds); later runs hit the cache.

Run with:
  cd /home/julien/work/kuiLLM && .venv/bin/python -m pytest tests/test_jit_ops.py -s
"""
import math
import sys

import pytest
import torch
import torch.nn.functional as F

from kuipy import kuiops

aten = torch.ops.aten
_DEVICE = "cuda" if torch.cuda.is_available() else None


def _need_cuda():
    if _DEVICE is None:
        pytest.skip("CUDA not available")


def _run(impl, func, args, kwargs=None):
    kwargs = kwargs or {}
    spec = impl.supported(func, args, kwargs)
    assert spec is not None, "expected a supported spec"
    return impl.run(spec, args, kwargs)


def _assert_close(out, ref, dtype):
    """f32: elementwise allclose. bf16/f16: relative Frobenius norm, which is the
    meaningful metric for low-precision accumulation (elementwise rtol explodes
    on near-zero reference entries from cancellation)."""
    o, r = out.float(), ref.float()
    if dtype == torch.float32:
        assert torch.allclose(o, r, atol=1e-3, rtol=1e-3)
    else:
        rel = (o - r).norm() / (r.norm() + 1e-6)
        assert rel < 3e-2, f"relative norm {rel.item():.4f} too large"


# ---------------------------------------------------------------------------
# bmm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bmm(dtype):
    _need_cuda()
    impl = kuiops.BmmImpl({})
    torch.manual_seed(0)
    A = torch.randn(5, 17, 9, device="cuda", dtype=dtype)
    B = torch.randn(5, 9, 13, device="cuda", dtype=dtype)
    out = _run(impl, aten.bmm.default, (A, B))
    ref = torch.bmm(A, B)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_bmm_unsupported():
    _need_cuda()
    impl = kuiops.BmmImpl({})
    # mismatched batch dim
    A = torch.randn(2, 4, 4, device="cuda")
    B = torch.randn(3, 4, 4, device="cuda")
    assert impl.supported(aten.bmm.default, (A, B), {}) is None
    # 2D inputs are not bmm
    A2 = torch.randn(4, 4, device="cuda")
    B2 = torch.randn(4, 4, device="cuda")
    assert impl.supported(aten.bmm.default, (A2, B2), {}) is None


# ---------------------------------------------------------------------------
# addmm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addmm(dtype, alpha, beta):
    _need_cuda()
    impl = kuiops.AddmmImpl({})
    torch.manual_seed(0)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)  # broadcast 1D bias
    kw = dict(alpha=alpha, beta=beta)
    out = _run(impl, aten.addmm.default, (bias, A, B), kw)
    ref = torch.addmm(bias, A, B, alpha=alpha, beta=beta)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_addmm_rejects_f64():
    _need_cuda()
    impl = kuiops.AddmmImpl({})
    # BlockTiling2D needs has_vec_cpy -> no f64.
    A = torch.randn(32, 32, device="cuda", dtype=torch.float64)
    B = torch.randn(32, 32, device="cuda", dtype=torch.float64)
    bias = torch.randn(32, device="cuda", dtype=torch.float64)
    assert impl.supported(aten.addmm.default, (bias, A, B), {}) is None


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(128, 64), (8, 5, 33)])
def test_softmax(dtype, shape):
    _need_cuda()
    impl = kuiops.SoftmaxImpl({})
    torch.manual_seed(0)
    X = torch.randn(*shape, device="cuda", dtype=dtype)
    dim = X.dim() - 1
    out = _run(impl, aten._softmax.default, (X, dim, False))
    ref = torch.softmax(X, dim=dim)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)
    rowsum_atol = 1e-3 if dtype == torch.float32 else 3e-2
    assert torch.allclose(out.float().sum(dim=-1),
                          torch.ones(out.shape[:-1], device="cuda"), atol=rowsum_atol)


def test_softmax_non_last_dim_unsupported():
    _need_cuda()
    impl = kuiops.SoftmaxImpl({})
    X = torch.randn(16, 16, device="cuda")
    # Only the last dim is supported.
    assert impl.supported(aten._softmax.default, (X, 0, False), {}) is None
    # half_to_float is unsupported.
    assert impl.supported(aten._softmax.default, (X, 1, True), {}) is None


# ---------------------------------------------------------------------------
# sdpa (efficient attention)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa(dtype):
    _need_cuda()
    impl = kuiops.SdpaImpl({})
    torch.manual_seed(0)
    N, H, L, S, E, Ev = 2, 3, 8, 10, 16, 12
    Q = torch.randn(N, H, L, E, device="cuda", dtype=dtype)
    K = torch.randn(N, H, S, E, device="cuda", dtype=dtype)
    V = torch.randn(N, H, S, Ev, device="cuda", dtype=dtype)
    bias = torch.randn(N, H, L, S, device="cuda", dtype=dtype)
    scale = 0.3
    func = aten._scaled_dot_product_efficient_attention.default
    args = (Q, K, V, bias, True)
    out, lse, seed, off = _run(impl, func, args, {"scale": scale})
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=bias, scale=scale)
    assert out.shape == (N, H, L, Ev)
    assert lse.shape == (N, H, L) and lse.dtype == torch.float32
    _assert_close(out, ref, dtype)
    scores = (Q.float() @ K.float().transpose(-1, -2)) * scale + bias.float()
    lse_ref = torch.logsumexp(scores, dim=-1)
    _assert_close(lse, lse_ref, dtype)


def test_sdpa_causal_unsupported():
    _need_cuda()
    impl = kuiops.SdpaImpl({})
    N, H, L, S, E, Ev = 1, 1, 4, 4, 8, 8
    Q = torch.randn(N, H, L, E, device="cuda")
    K = torch.randn(N, H, S, E, device="cuda")
    V = torch.randn(N, H, S, Ev, device="cuda")
    bias = torch.randn(N, H, L, S, device="cuda")
    func = aten._scaled_dot_product_efficient_attention.default
    # is_causal and dropout are unsupported.
    assert impl.supported(func, (Q, K, V, bias, True, 0.0, True), {}) is None
    assert impl.supported(func, (Q, K, V, bias, True, 0.1, False), {}) is None
    # A missing (None) bias is unsupported.
    assert impl.supported(func, (Q, K, V, None, True), {}) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
