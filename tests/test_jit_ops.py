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

import kuipy
from kuipy import kuiops

aten = torch.ops.aten
_DEVICE = "cuda" if torch.cuda.is_available() else None


def _need_cuda():
    if _DEVICE is None:
        pytest.skip("CUDA not available")


def _need_tensor_cores(dtype):
    _need_cuda()
    minimum = (8, 0) if dtype == torch.bfloat16 else (7, 0)
    if torch.cuda.get_device_capability() < minimum:
        pytest.skip(f"{dtype} tensor cores require sm_{minimum[0]}{minimum[1]}+")


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
# mm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_dtype,acc_dtype,out_dtype,backend", [
    (torch.float16, torch.float16, torch.float16, "tc2d_to"),
    (torch.float16, torch.float32, torch.float32, "tc2d_to"),
    (torch.bfloat16, torch.float32, torch.float32, "tc2d_to"),
    (torch.float16, torch.float32, torch.float16, "tc2d_to"),
])
def test_mm_tensorcore2d(in_dtype, acc_dtype, out_dtype, backend):
    """With a contiguous B, SuperGEMM does not apply and the cast-on-write
    TensorCore2D backend handles every valid fragment/accumulator pairing."""
    _need_tensor_cores(in_dtype)
    impl = kuiops.MmImpl()
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=in_dtype)
    B = torch.randn(64, 64, device="cuda", dtype=in_dtype)
    kwargs = {"acc_dtype": acc_dtype, "out_dtype": out_dtype}
    spec = impl.supported(aten.mm.default, (A, B), kwargs)
    assert spec is not None and spec["backend"] == backend
    out = impl.run(spec, (A, B), kwargs)
    ref = torch.mm(A.float(), B.float()).to(out_dtype)
    assert out.dtype == out_dtype
    _assert_close(out, ref, out_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_tensorcore2d_to(dtype):
    """bf16 has no wmma accumulator fragment, so a bf16 output can only be
    reached through the cast-on-write backend."""
    _need_tensor_cores(dtype)
    impl = kuiops.MmImpl()
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda", dtype=dtype)
    B = torch.randn(64, 64, device="cuda", dtype=dtype)
    spec = impl.supported(aten.mm.default, (A, B), {})
    assert spec is not None and spec["backend"] == "tc2d_to"
    out = impl.run(spec, (A, B), {})
    assert out.dtype == dtype
    _assert_close(out, torch.mm(A, B), dtype)


def test_mm_blocktiling2d():
    """f32 is not a tensor-core input type, so only BlockTiling2D is left."""
    _need_cuda()
    impl = kuiops.MmImpl()
    torch.manual_seed(0)
    A = torch.randn(64, 64, device="cuda")
    B = torch.randn(64, 64, device="cuda")
    spec = impl.supported(aten.mm.default, (A, B), {})
    assert spec is not None and spec["backend"] == "bt2d"
    out = impl.run(spec, (A, B), {})
    _assert_close(out, torch.mm(A, B), torch.float32)


def test_mm_backend_selection():
    _need_tensor_cores(torch.bfloat16)
    impl = kuiops.MmImpl()

    def backend(dtype, **kwargs):
        A = torch.randn(64, 64, device="cuda", dtype=dtype)
        B = torch.randn(64, 64, device="cuda", dtype=dtype)
        spec = impl.supported(aten.mm.default, (A, B), kwargs)
        return spec and spec["backend"]

    assert backend(torch.float16) == "tc2d_to"
    assert backend(torch.float16, acc_dtype=torch.float16) == "tc2d_to"
    assert backend(torch.bfloat16, out_dtype=torch.float32) == "tc2d_to"
    assert backend(torch.float32) == "bt2d"
    assert backend(torch.bfloat16) == "tc2d_to"
    # bf16 can only accumulate in f32 on a tensor core, so an explicit bf16
    # accumulator falls through to the block-tiled backend.
    assert backend(torch.bfloat16, acc_dtype=torch.bfloat16) == "bt2d"
    assert backend(torch.bfloat16, acc_dtype=torch.float16) is None
    assert backend(torch.float64) is None
    # An explicit `impl` rejects the call rather than falling to another backend.
    assert backend(torch.bfloat16, impl="tc2d_to") == "tc2d_to"
    assert backend(torch.bfloat16, impl="tc2d") is None
    assert backend(torch.float32, impl="tc2d_to") is None
    assert backend(torch.float32, impl="bt2d") == "bt2d"
    assert backend(torch.float32, impl="nonsense") is None


# ---------------------------------------------------------------------------
# bmm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bmm(dtype):
    _need_cuda()
    torch.manual_seed(0)
    # BlockTiling2D (now the batched GEMM) needs tile-divisible dims.
    A = torch.randn(5, 64, 32, device="cuda", dtype=dtype)
    B = torch.randn(5, 32, 64, device="cuda", dtype=dtype)
    out = kuipy.run(aten.bmm.default)(A, B)
    ref = torch.bmm(A, B)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_bmm_unsupported():
    _need_cuda()
    impl = kuiops.BmmImpl()
    # mismatched batch dim
    A = torch.randn(2, 32, 32, device="cuda")
    B = torch.randn(3, 32, 32, device="cuda")
    assert impl.supported(aten.bmm.default, (A, B), {}) is None
    # 2D inputs are not bmm
    A2 = torch.randn(32, 32, device="cuda")
    B2 = torch.randn(32, 32, device="cuda")
    assert impl.supported(aten.bmm.default, (A2, B2), {}) is None
    # dims that no BlockTiling2D tiling divides
    A3 = torch.randn(5, 17, 9, device="cuda")
    B3 = torch.randn(5, 9, 13, device="cuda")
    assert impl.supported(aten.bmm.default, (A3, B3), {}) is None


# ---------------------------------------------------------------------------
# addmm
# ---------------------------------------------------------------------------

# The broadcast bias is read through the TensorCore2D.To epilogue, so it is
# available exactly for the tensor-core input dtypes.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addmm_1d_bias(dtype, alpha, beta):
    _need_cuda()
    torch.manual_seed(0)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)  # broadcast 1D bias
    kw = dict(alpha=alpha, beta=beta)
    out = kuipy.run(aten.addmm.default)(bias, A, B, **kw)
    ref = torch.addmm(bias, A, B, alpha=alpha, beta=beta)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addmm(dtype, alpha, beta):
    _need_cuda()
    impl = kuiops.AddmmImpl()
    torch.manual_seed(0)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    bias = torch.randn(M, N, device="cuda", dtype=dtype)  # 2D bias/C matrix
    kw = dict(alpha=alpha, beta=beta)
    spec = impl.supported(aten.addmm.default, (bias, A, B), kw)
    expected = "bt2d"
    if (dtype in kuiops._TC_INPUT_DTYPES
            and kuiops._tc_device_supported(dtype, A.device)):
        # tc2d stores the accumulator fragment into C, so it needs the output
        # to be the (f32) accumulator type; a 16-bit output goes cast-on-write.
        expected = "tc2d_to"
    assert spec is not None and spec["backend"] == expected
    out = impl.run(spec, (bias, A, B), kw)
    ref = torch.addmm(bias, A, B, alpha=alpha, beta=beta)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 2.0)])
def test_addmm_tensorcore2d(alpha, beta):
    _need_tensor_cores(torch.bfloat16)
    impl = kuiops.AddmmImpl()
    torch.manual_seed(0)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(M, N, device="cuda", dtype=torch.float32)
    kw = dict(alpha=alpha, beta=beta, out_dtype=torch.float32)
    spec = impl.supported(aten.addmm.default, (bias, A, B), kw)
    assert spec is not None and spec["backend"] == "tc2d_to"
    out = impl.run(spec, (bias, A, B), kw)
    ref = alpha * (A.float() @ B.float()) + beta * bias
    assert out.dtype == torch.float32
    _assert_close(out, ref, torch.float32)


@pytest.mark.parametrize("cshape", ["dense", "bcast"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 2.0), (-1.5, 3.25)])
def test_addmm_supergemm(cshape, alpha, beta):
    """SuperGEMM reads C through a generic read-only view, so one verified
    kernel serves both an (M, N) matrix and a broadcast row bias."""
    _need_tensor_cores(torch.bfloat16)
    impl = kuiops.AddmmImpl()
    torch.manual_seed(0)
    M, K, N = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t()
    bias = torch.randn(*((M, N) if cshape == "dense" else (N,)),
                       device="cuda", dtype=torch.bfloat16)
    kw = dict(alpha=alpha, beta=beta, impl="supergemm")
    spec = impl.supported(aten.addmm.default, (bias, A, B), kw)
    assert spec is not None and spec["backend"] == "supergemm"
    assert spec["cbcast"] == (cshape == "bcast")
    out = impl.run(spec, (bias, A, B), kw)
    ref = torch.addmm(bias, A, B, alpha=alpha, beta=beta)
    assert out.shape == ref.shape
    _assert_close(out, ref, torch.bfloat16)


def test_addmm_supergemm_rejects_unaligned_a():
    _need_tensor_cores(torch.bfloat16)
    impl = kuiops.AddmmImpl()
    # The staging pipeline reads A with 128-bit loads, so a row that is not a
    # whole number of 16-byte chunks is out of reach.
    M, K, N = 128, 132, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t()
    bias = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    kw = dict(impl="supergemm")
    assert impl.supported(aten.addmm.default, (bias, A, B), kw) is None


def test_addmm_rejects_f64():
    _need_cuda()
    impl = kuiops.AddmmImpl()
    # BlockTiling2D needs has_vec_cpy -> no f64.
    A = torch.randn(32, 32, device="cuda", dtype=torch.float64)
    B = torch.randn(32, 32, device="cuda", dtype=torch.float64)
    bias = torch.randn(32, 32, device="cuda", dtype=torch.float64)
    assert impl.supported(aten.addmm.default, (bias, A, B), {}) is None


def test_addmm_rejects_broadcast_bias():
    _need_cuda()
    impl = kuiops.AddmmImpl()
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
@pytest.mark.skip(reason="RowSoftmax extraction takes ~6min per instantiation")
def test_softmax(dtype, shape):
    _need_cuda()
    torch.manual_seed(0)
    X = torch.randn(*shape, device="cuda", dtype=dtype)
    dim = X.dim() - 1
    out = kuipy.run(aten._softmax.default)(X, dim, False)
    ref = torch.softmax(X, dim=dim)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)
    rowsum_atol = 1e-3 if dtype == torch.float32 else 3e-2
    assert torch.allclose(out.float().sum(dim=-1),
                          torch.ones(out.shape[:-1], device="cuda"), atol=rowsum_atol)


def test_softmax_non_last_dim_unsupported():
    _need_cuda()
    impl = kuiops.SoftmaxImpl()
    X = torch.randn(16, 16, device="cuda")
    # Only the last dim is supported.
    assert impl.supported(aten._softmax.default, (X, 0, False), {}) is None
    # half_to_float is unsupported.
    assert impl.supported(aten._softmax.default, (X, 1, True), {}) is None


# ---------------------------------------------------------------------------
# exact integer reductions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func,dtype,kwargs", [
    (aten.sum.dim_IntList, torch.int32, {}),
    (aten.prod.dim_int, torch.int8, {"dtype": torch.int16}),
    (aten.all.dim, torch.int32, {}),
    (aten.any.dim, torch.uint8, {}),
])
def test_hreduce_poly(func, dtype, kwargs):
    _need_cuda()
    X = torch.tensor(
        [[[1, 2, 0, 3], [2, 1, 1, 2], [1, 1, 1, 1]],
         [[3, 1, 2, 1], [0, 2, 1, 3], [2, 2, 1, 1]]],
        device="cuda", dtype=dtype)
    args = (X, -1, False)
    out = kuipy.run(func)(*args, **kwargs)
    ref = func(*args, **kwargs)
    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    assert torch.equal(out, ref)


def test_hreduce_poly_approx_batched_float_sum():
    _need_cuda()
    X = torch.randn(2, 3, 512, device="cuda", dtype=torch.float32)
    args = (X, [-1], False)
    out = kuipy.run(aten.sum.dim_IntList)(*args)
    _assert_close(out, aten.sum.dim_IntList(*args), torch.float32)


def test_hreduce_poly_dtype_conversion():
    _need_cuda()
    X = torch.arange(8, device="cuda", dtype=torch.int8).reshape(2, 4)
    args = (X, [1], False)
    out = kuipy.run(aten.sum.dim_IntList)(*args, dtype=torch.uint16)
    ref = torch.tensor([6, 22], device="cuda", dtype=torch.uint16)
    assert torch.equal(out, ref)
    assert out.dtype == torch.uint16


def test_hreduce_poly_support_constraints():
    _need_cuda()
    impl = kuiops.HReducePolyImpl()
    ints = torch.ones(2, 4, device="cuda", dtype=torch.int32)
    floats = ints.float()
    assert impl.supported(
        aten.sum.dim_IntList, (ints, [0]), {}) is None
    # keepdim only changes the shape of the output allocation.
    assert impl.supported(
        aten.sum.dim_IntList, (ints, [1], True), {}) is not None
    rank3 = ints.reshape(1, 2, 4)
    assert impl.supported(
        aten.sum.dim_IntList, (rank3, [2]), {}) is not None
    rank5 = ints.reshape(1, 1, 1, 2, 4)
    assert impl.supported(
        aten.sum.dim_IntList, (rank5, [4]), {}) is None
    assert impl.supported(
        aten.sum.dim_IntList, (ints, [-1]), {"dtype": torch.float32}) is None
    assert impl.supported(
        aten.sum.dim_IntList, (floats, [-1]), {}) is not None
    assert impl.supported(
        aten.sum.dim_IntList, (floats, [-1]), {"dtype": torch.float64}) is None
    assert impl.supported(aten.all.dim, (floats, -1), {}) is None
    assert impl.supported(
        aten.sum.dim_IntList, (ints.transpose(0, 1), [-1]), {}) is None
    empty_rows = torch.empty(0, 4, device="cuda", dtype=torch.int32)
    assert impl.supported(
        aten.sum.dim_IntList, (empty_rows, [-1]), {}) is None
    empty_cols = torch.empty(2, 0, device="cuda", dtype=torch.int32)
    assert impl.supported(
        aten.sum.dim_IntList, (empty_cols, [-1]), {}) is None


# ---------------------------------------------------------------------------
# fused pre_map / post_map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pre,post,ref_fn", [
    ([aten.relu.default], [], lambda x: torch.relu(x)),
    ([], [(aten.mul.Scalar, 0.5)], lambda x: x),
    ([(aten.pow.Tensor_Scalar, 2)], [(aten.div.Tensor, 64)], lambda x: x * x),
    ([aten.relu.default, (aten.mul.Scalar, 2.0), aten.silu.default], [],
     lambda x: torch.nn.functional.silu(torch.relu(x) * 2.0)),
])
def test_hreduce_poly_maps_approx(pre, post, ref_fn):
    """Maps fold into the reduction over the reals: pre-maps run per element,
    post-maps on the reduced value."""
    _need_cuda()
    X = torch.randn(3, 64, device="cuda", dtype=torch.float32)
    out = kuipy.run(aten.sum.dim_IntList, pre_map=pre, post_map=post)(X, [-1])
    ref = ref_fn(X).sum(dim=-1)
    for method, c in kuiops.resolve_maps(post, torch.float32, True):
        ref = {"mul": lambda a, b: a * b, "div": lambda a, b: a / b}[method](ref, c)
    _assert_close(out, ref, torch.float32)


def test_hreduce_poly_maps_exact():
    _need_cuda()
    X = torch.randint(0, 5, (3, 16), device="cuda", dtype=torch.uint16)
    out = kuipy.run(aten.sum.dim_IntList,
                    pre_map=[(aten.mul.Scalar, 3)],
                    post_map=[(aten.add.Scalar, 7)])(X, [-1],
                                                     dtype=torch.uint32)
    ref = (X.to(torch.int64) * 3).sum(dim=-1) + 7
    assert out.dtype == torch.uint32
    assert torch.equal(out.to(torch.int64), ref)


@pytest.mark.parametrize("shape,dim,keepdim", [
    ((3, 64), -1, True), ((3, 64), 1, False), ((2, 3, 32), -1, True),
])
def test_mean_via_reduce(shape, dim, keepdim):
    """`mean` is the plain reduction with a divide-by-`cols` post-map; the
    divisor is the kernel's own `cols` argument, not a passed-in constant."""
    _need_cuda()
    torch.manual_seed(0)
    X = torch.randn(*shape, device="cuda", dtype=torch.float32)
    out = kuipy.run(aten.mean.dim)(X, [dim], keepdim)
    ref = X.mean(dim=dim, keepdim=keepdim)
    assert out.shape == ref.shape
    _assert_close(out, ref, torch.float32)


def test_mean_with_maps():
    """RMSNorm's reduction: square each element, average, then rsqrt."""
    _need_cuda()
    torch.manual_seed(0)
    X = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    out = kuipy.run(aten.mean.dim,
                    pre_map=[(aten.pow.Tensor_Scalar, 2)],
                    post_map=[(aten.add.Tensor, 1e-6), aten.rsqrt.default],
                    )(X, [-1], True)
    _assert_close(out, torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + 1e-6),
                  torch.float32)


def test_mean_support_constraints():
    _need_cuda()
    impl = kuiops.HReducePolyImpl()
    floats = torch.ones(2, 4, device="cuda", dtype=torch.float32)
    ints = torch.ones(2, 4, device="cuda", dtype=torch.int32)
    assert impl.supported(aten.mean.dim, (floats, [-1], True), {}) is not None
    # Dividing by the reduced length needs the kernel specified over the reals.
    assert impl.supported(aten.mean.dim, (ints, [-1], True), {}) is None
    # Only the innermost axis is reduced.
    assert impl.supported(aten.mean.dim, (floats, [0], True), {}) is None


def test_hreduce_poly_maps_support_constraints():
    _need_cuda()
    impl = kuiops.HReducePolyImpl()
    floats = torch.ones(2, 4, device="cuda", dtype=torch.float32)
    ints = torch.ones(2, 4, device="cuda", dtype=torch.int32)

    def sup(**kw):
        return impl.supported(aten.sum.dim_IntList, (floats, [-1]), kw)

    assert sup(pre_map=[aten.relu.default]) is not None
    # No real-valued model for these, so they cannot be fused into a kernel
    # specified over the reals.
    assert sup(post_map=[aten.sin.default]) is None
    assert sup(post_map=[aten.cos.default]) is None
    # A binary op with no constant, and a unary op given one.
    assert sup(pre_map=[aten.mul.Tensor]) is None
    assert sup(pre_map=[(aten.relu.default, 1.0)]) is None
    # Only x ** 2 is expressible.
    assert sup(pre_map=[(aten.pow.Tensor_Scalar, 3)]) is None
    # The divisor's real value must be known non-zero.
    assert sup(post_map=[(aten.div.Tensor, 2)]) is not None
    assert sup(post_map=[(aten.div.Tensor, 0.5)]) is None
    assert sup(post_map=[(aten.div.Tensor, 0)]) is None
    # Signed integers have no `scalar` instance, so no map applies to them.
    assert impl.supported(aten.sum.dim_IntList, (ints, [-1]),
                          {"pre_map": [aten.relu.default]}) is None


# ---------------------------------------------------------------------------
# sdpa (efficient attention)
# ---------------------------------------------------------------------------

def _sdpa_ref(Q, K, V, bias, scale, causal):
    """Reference attention in f32. Causal follows the kernel: query position
    ``qpos`` attends to keys ``kj <= qpos + (Sk - Sq)`` (bottom-right aligned)."""
    sq, sk = Q.shape[2], K.shape[2]
    group = Q.shape[1] // K.shape[1]
    Kb = K.repeat_interleave(group, dim=1)
    Vb = V.repeat_interleave(group, dim=1)
    logits = (Q.float() @ Kb.float().transpose(-1, -2)) * scale
    if bias is not None:
        logits = logits + bias.float()
    if causal:
        qpos = torch.arange(sq, device=Q.device).view(sq, 1)
        kj = torch.arange(sk, device=Q.device).view(1, sk)
        logits = logits.masked_fill(kj > qpos + (sk - sq), float("-inf"))
    return (torch.softmax(logits, dim=-1) @ Vb.float()).to(Q.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mask", ["none", "dense", "keys"])
def test_sdpa(dtype, causal, mask):
    _need_tensor_cores(dtype)
    torch.manual_seed(0)
    b, hq, hkv, sq, sk, d = 2, 4, 2, 3, 24, 32
    Q = torch.randn(b, hq, sq, d, device="cuda", dtype=dtype)
    K = torch.randn(b, hkv, sk, d, device="cuda", dtype=dtype)
    V = torch.randn(b, hkv, sk, d, device="cuda", dtype=dtype)
    # "keys" is HF's decode mask: broadcast over batch, head and query.
    bshape = {"dense": (b, hq, sq, sk), "keys": (1, 1, 1, sk)}.get(mask)
    bias = None if bshape is None else \
        torch.randn(*bshape, device="cuda", dtype=dtype)
    original_bias = None if bias is None else bias.clone()
    scale = 0.3
    func = aten._scaled_dot_product_efficient_attention.default
    args = (Q, K, V, bias, False, 0.0, causal)
    out, lse, seed, off = kuipy.run(func)(*args, scale=scale)
    assert out.shape == (b, hq, sq, d) and out.dtype == dtype
    assert lse.shape == (b, hq, 0) and lse.dtype == torch.float32
    assert seed.shape == off.shape == ()
    if bias is not None:
        assert torch.equal(bias, original_bias)
    ref_bias = None if bias is None else bias.expand(b, hq, sq, sk)
    _assert_close(out, _sdpa_ref(Q, K, V, ref_bias, scale, causal), dtype)


def test_sdpa_unsupported():
    _need_cuda()
    impl = kuiops.SdpaImpl()
    dtype = torch.bfloat16
    b, hq, hkv, sq, sk, d = 1, 2, 1, 4, 8, 16
    def mk(shape, dt=dtype):
        return torch.randn(*shape, device="cuda", dtype=dt)
    Q = mk((b, hq, sq, d))
    K = mk((b, hkv, sk, d))
    V = mk((b, hkv, sk, d))
    bias = mk((b, hq, sq, sk))
    func = aten._scaled_dot_product_efficient_attention.default
    # LSE computation and dropout have no kernel support.
    assert impl.supported(func, (Q, K, V, bias, True), {}) is None
    assert impl.supported(func, (Q, K, V, bias, False, 0.1, False), {}) is None
    # The mask is read through a rotensor, so it may be absent, dense, or
    # broadcast over batch/head/query as (1, 1, 1, sk). Any other shape, and
    # any expanded (non-contiguous) view, is rejected; it must share the input
    # dtype.
    assert impl.supported(func, (Q, K, V, None, False), {}) is not None
    assert impl.supported(func, (Q, K, V, mk((1, 1, 1, sk)), False), {}) is not None
    assert impl.supported(
        func, (Q, K, V, mk((b, 1, 1, sk)).expand(b, hq, sq, sk), False), {}) is None
    assert impl.supported(
        func, (Q, K, V, mk((1, 1, sq, sk)), False), {}) is None
    assert impl.supported(
        func, (Q, K, V, mk((b, hq, sq, sk), torch.float32), False), {}) is None
    # Tensor-core fragments admit only bf16/f16 inputs with an f32 accumulator.
    f32 = [t.float() for t in (Q, K, V, bias)]
    assert impl.supported(func, (*f32, False), {}) is None
    # head_dim must be a multiple of 16 and sq <= sk.
    assert impl.supported(
        func, (mk((b, hq, sq, 24)), mk((b, hkv, sk, 24)), mk((b, hkv, sk, 24)),
               bias, False), {}) is None
    assert impl.supported(
        func, (mk((b, hq, sk + 8, d)), K, V, mk((b, hq, sk + 8, sk)), False),
        {}) is None
    # Causal attention IS supported.
    assert impl.supported(func, (Q, K, V, bias, False, 0.0, True), {}) is not None


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,dim", [((8, 5), 1), ((8, 5), 0), ((4, 3, 6), 2)])
def test_gather(dtype, shape, dim):
    _need_cuda()
    torch.manual_seed(0)
    Inp = torch.randn(*shape, device="cuda", dtype=dtype)
    idx_shape = list(shape)
    idx_shape[dim] = max(1, shape[dim] - 1)  # index pointwise <= input
    Idx = torch.randint(0, shape[dim], idx_shape, device="cuda", dtype=torch.int64)
    out = kuipy.run(aten.gather.default)(Inp, dim, Idx)
    ref = torch.gather(Inp, dim, Idx)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_gather_unsupported():
    _need_cuda()
    impl = kuiops.GatherImpl()
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
    out = kuipy.run(aten.scatter.src)(Self, dim, Idx, Src)
    ref = Self.clone().scatter_(dim, Idx, Src)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_scatter_unsupported():
    _need_cuda()
    impl = kuiops.ScatterImpl()
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
    torch.manual_seed(0)
    a_shape = [4, 3, 6]
    b_shape = list(a_shape)
    b_shape[dim] = a_shape[dim] + 2  # differ only along `dim`
    A = torch.randn(*a_shape, device="cuda", dtype=dtype)
    B = torch.randn(*b_shape, device="cuda", dtype=dtype)
    out = kuipy.run(aten.cat.default)([A, B], dim)
    ref = torch.cat([A, B], dim=dim)
    assert out.shape == ref.shape
    _assert_close(out, ref, dtype)


def test_cat_unsupported():
    _need_cuda()
    impl = kuiops.CatImpl()
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
    torch.manual_seed(0)
    A = torch.randn(3, 5, 7, device="cuda", dtype=dtype)
    B = torch.randn(3, 5, 7, device="cuda", dtype=dtype).abs() + 0.5
    out = kuipy.run(func)(A, B)
    _assert_close(out, ref(A, B), dtype)


@pytest.mark.parametrize("func,c,ref", [
    (aten.sub.Tensor, 1.5, lambda x: x - 1.5),
    (aten.div.Tensor, 2.0, lambda x: x / 2.0),
])
def test_elem_scalar(func, c, ref):
    _need_cuda()
    torch.manual_seed(0)
    A = torch.randn(4, 9, device="cuda", dtype=torch.float32)
    out = kuipy.run(func)(A, c)
    _assert_close(out, ref(A), torch.float32)


def test_elem_bitwise():
    _need_cuda()
    torch.manual_seed(0)
    A = torch.randint(0, 2, (2, 6, 4), device="cuda", dtype=torch.bool)
    B = torch.randint(0, 2, (2, 6, 4), device="cuda", dtype=torch.bool)
    assert torch.equal(kuipy.run(aten.bitwise_not.default)(A), ~A)
    assert torch.equal(kuipy.run(aten.bitwise_and.Tensor)(A, B), A & B)
    assert torch.equal(kuipy.run(aten.bitwise_or.Tensor)(A, B), A | B)


@pytest.mark.parametrize("func,c,ref", [
    (aten.le.Scalar, 0.0, lambda x: x <= 0.0),
    (aten.lt.Scalar, 0.5, lambda x: x < 0.5),
    (aten.eq.Scalar, 0.0, lambda x: x == 0.0),
])
def test_elem_compare_scalar(func, c, ref):
    _need_cuda()
    torch.manual_seed(0)
    A = torch.randn(5, 8, device="cuda", dtype=torch.float32)
    out = kuipy.run(func)(A, c)
    assert out.dtype == torch.bool
    assert torch.equal(out, ref(A))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_elem_where(dtype):
    _need_cuda()
    torch.manual_seed(0)
    C = torch.randint(0, 2, (3, 4, 5), device="cuda", dtype=torch.bool)
    X = torch.randn(3, 4, 5, device="cuda", dtype=dtype)
    Y = torch.randn(3, 4, 5, device="cuda", dtype=dtype)
    out = kuipy.run(aten.where.self)(C, X, Y)
    _assert_close(out, torch.where(C, X, Y), dtype)


def test_elem_unsupported():
    _need_cuda()
    impl = kuiops.ElementwiseImpl()
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