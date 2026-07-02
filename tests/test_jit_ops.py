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
@pytest.mark.skip(reason="broadcasting currently disabled")
def test_addmm_1d_bias(dtype, alpha, beta):
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addmm(dtype, alpha, beta):
    _need_cuda()
    impl = kuiops.AddmmImpl({})
    torch.manual_seed(0)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    bias = torch.randn(M, N, device="cuda", dtype=dtype)  # 2D bias/C matrix
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
    bias = torch.randn(32, 32, device="cuda", dtype=torch.float64)
    assert impl.supported(aten.addmm.default, (bias, A, B), {}) is None


def test_addmm_rejects_broadcast_bias():
    _need_cuda()
    impl = kuiops.AddmmImpl({})
    M, K, N = 32, 32, 32
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    bias_1d = torch.randn(N, device="cuda")
    bias_wrong = torch.randn(1, N, device="cuda")
    assert impl.supported(aten.addmm.default, (bias_1d, A, B), {}) is None
    assert impl.supported(aten.addmm.default, (bias_wrong, A, B), {}) is None


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
@pytest.mark.skip(reason="SDPA currently disabled")
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

@pytest.mark.skip(reason="SDPA currently disabled")
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


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,dim", [((8, 5), 1), ((8, 5), 0), ((4, 3, 6), 2)])
def test_gather(dtype, shape, dim):
    _need_cuda()
    impl = kuiops.GatherImpl({})
    torch.manual_seed(0)
    Inp = torch.randn(*shape, device="cuda", dtype=dtype)
    idx_shape = list(shape)
    idx_shape[dim] = max(1, shape[dim] - 1)  # index pointwise <= input
    Idx = torch.randint(0, shape[dim], idx_shape, device="cuda", dtype=torch.int64)
    out = _run(impl, aten.gather.default, (Inp, dim, Idx))
    ref = torch.gather(Inp, dim, Idx)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_gather_unsupported():
    _need_cuda()
    impl = kuiops.GatherImpl({})
    Inp = torch.randn(4, 4, device="cuda")
    # index must be int64
    Idx_f = torch.zeros(4, 4, device="cuda", dtype=torch.int32)
    assert impl.supported(aten.gather.default, (Inp, 1, Idx_f), {}) is None
    # index rank must match input rank
    Idx_r = torch.zeros(4, device="cuda", dtype=torch.int64)
    assert impl.supported(aten.gather.default, (Inp, 1, Idx_r), {}) is None
    # index must be pointwise <= input on every axis
    Idx_big = torch.zeros(4, 5, device="cuda", dtype=torch.int64)
    assert impl.supported(aten.gather.default, (Inp, 1, Idx_big), {}) is None


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,dim", [((8, 5), 1), ((8, 5), 0), ((4, 3, 6), 2)])
def test_scatter(dtype, shape, dim):
    _need_cuda()
    impl = kuiops.ScatterImpl({})
    torch.manual_seed(0)
    Self = torch.randn(*shape, device="cuda", dtype=dtype)
    Src = torch.randn(*shape, device="cuda", dtype=dtype)
    # A per-row permutation along `dim` makes the scatter injective (no two
    # source cells target the same output location), matching the kernel's
    # assumption.
    n = shape[dim]
    view = [1] * len(shape)
    view[dim] = n
    perm = torch.stack([torch.randperm(n, device="cuda")
                        for _ in range(Self.numel() // n)])
    Idx = perm.reshape([s for d, s in enumerate(shape) if d != dim] + [n])
    Idx = Idx.movedim(-1, dim).contiguous()
    out = _run(impl, aten.scatter.src, (Self, dim, Idx, Src))
    ref = Self.clone().scatter_(dim, Idx, Src)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_scatter_unsupported():
    _need_cuda()
    impl = kuiops.ScatterImpl({})
    Self = torch.randn(4, 4, device="cuda")
    Idx = torch.zeros(4, 4, device="cuda", dtype=torch.int64)
    Src = torch.randn(4, 4, device="cuda")
    # src dtype must match self dtype
    Src_bf = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    assert impl.supported(aten.scatter.src, (Self, 1, Idx, Src_bf), {}) is None
    # index must be int64
    Idx_i32 = torch.zeros(4, 4, device="cuda", dtype=torch.int32)
    assert impl.supported(aten.scatter.src, (Self, 1, Idx_i32, Src), {}) is None
    # src and index shapes must match
    Src_big = torch.randn(4, 5, device="cuda")
    Idx_big = torch.zeros(4, 4, device="cuda", dtype=torch.int64)
    assert impl.supported(aten.scatter.src, (Self, 1, Idx_big, Src_big), {}) is None


# ---------------------------------------------------------------------------
# cat (binary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_cat(dtype, dim):
    _need_cuda()
    impl = kuiops.CatImpl({})
    torch.manual_seed(0)
    a_shape = [4, 3, 6]
    b_shape = list(a_shape)
    b_shape[dim] = a_shape[dim] + 2  # differ only along `dim`
    A = torch.randn(*a_shape, device="cuda", dtype=dtype)
    B = torch.randn(*b_shape, device="cuda", dtype=dtype)
    out = _run(impl, aten.cat.default, ([A, B], dim))
    ref = torch.cat([A, B], dim=dim)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_cat_unsupported():
    _need_cuda()
    impl = kuiops.CatImpl({})
    A = torch.randn(4, 4, device="cuda")
    B = torch.randn(4, 4, device="cuda")
    # only binary cat is supported
    assert impl.supported(aten.cat.default, ([A, B, A], 0), {}) is None
    # non-`dim` axes must agree
    Bmis = torch.randn(4, 5, device="cuda")
    assert impl.supported(aten.cat.default, ([A, Bmis], 0), {}) is None
    # dtypes must match
    Bbf = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    assert impl.supported(aten.cat.default, ([A, Bbf], 0), {}) is None

# ---------------------------------------------------------------------------
# elementwise: binary arithmetic, bitwise, comparisons, ternary select
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("func,ref", [
    (aten.sub.Tensor, torch.sub),
    (aten.div.Tensor, torch.div),
])
def test_elem_binary(dtype, func, ref):
    _need_cuda()
    impl = kuiops.ElementwiseImpl({})
    torch.manual_seed(0)
    A = torch.randn(3, 5, 7, device="cuda", dtype=dtype)
    B = torch.randn(3, 5, 7, device="cuda", dtype=dtype).abs() + 0.5
    out = _run(impl, func, (A, B))
    _assert_close(out, ref(A, B), dtype)


@pytest.mark.parametrize("func,c,ref", [
    (aten.sub.Tensor, 1.5, lambda x: x - 1.5),
    (aten.div.Tensor, 2.0, lambda x: x / 2.0),
])
def test_elem_scalar(func, c, ref):
    _need_cuda()
    impl = kuiops.ElementwiseImpl({})
    torch.manual_seed(0)
    A = torch.randn(4, 9, device="cuda", dtype=torch.float32)
    out = _run(impl, func, (A, c))
    _assert_close(out, ref(A), torch.float32)


def test_elem_bitwise():
    _need_cuda()
    impl = kuiops.ElementwiseImpl({})
    torch.manual_seed(0)
    A = torch.randint(0, 2, (2, 6, 4), device="cuda", dtype=torch.bool)
    B = torch.randint(0, 2, (2, 6, 4), device="cuda", dtype=torch.bool)
    assert torch.equal(_run(impl, aten.bitwise_not.default, (A,)), ~A)
    assert torch.equal(_run(impl, aten.bitwise_and.Tensor, (A, B)), A & B)
    assert torch.equal(_run(impl, aten.bitwise_or.Tensor, (A, B)), A | B)


@pytest.mark.parametrize("func,c,ref", [
    (aten.le.Scalar, 0.0, lambda x: x <= 0.0),
    (aten.lt.Scalar, 0.5, lambda x: x < 0.5),
    (aten.eq.Scalar, 0.0, lambda x: x == 0.0),
])
def test_elem_compare_scalar(func, c, ref):
    _need_cuda()
    impl = kuiops.ElementwiseImpl({})
    torch.manual_seed(0)
    A = torch.randn(5, 8, device="cuda", dtype=torch.float32)
    out = _run(impl, func, (A, c))
    assert out.dtype == torch.bool
    assert torch.equal(out, ref(A))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_elem_where(dtype):
    _need_cuda()
    impl = kuiops.ElementwiseImpl({})
    torch.manual_seed(0)
    C = torch.randint(0, 2, (3, 4, 5), device="cuda", dtype=torch.bool)
    X = torch.randn(3, 4, 5, device="cuda", dtype=dtype)
    Y = torch.randn(3, 4, 5, device="cuda", dtype=dtype)
    out = _run(impl, aten.where.self, (C, X, Y))
    _assert_close(out, torch.where(C, X, Y), dtype)


def test_elem_unsupported():
    _need_cuda()
    impl = kuiops.ElementwiseImpl({})
    A = torch.randn(4, 4, device="cuda")
    B = torch.randn(4, 4, device="cuda")
    # Tensor comparisons (bool output) have no not-in-place binary map kernel.
    assert impl.supported(aten.le.Tensor, (A, B), {}) is None
    assert impl.supported(aten.lt.Tensor, (A, B), {}) is None
    # Bitwise requires bool operands.
    Ai = torch.ones(4, 4, device="cuda", dtype=torch.int64)
    assert impl.supported(aten.bitwise_and.Tensor, (Ai, Ai), {}) is None
    # where with mismatched shapes / dtypes is unsupported (no broadcasting).
    C = torch.randint(0, 2, (4, 4), device="cuda", dtype=torch.bool)
    Ymis = torch.randn(4, 3, device="cuda")
    assert impl.supported(aten.where.self, (C, A, Ymis), {}) is None
    Ybf = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    assert impl.supported(aten.where.self, (C, A, Ybf), {}) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))