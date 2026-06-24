"""
Python bindings for KuiOps kernels. These classes:
 a) check if a given aten op is supported by an instantiation of a KuiOps template
 b) extract the relevant kernel from the template, compile it, and call it.
"""
import sys
from . import compile as _compile
from .config import log

import torch
aten = torch.ops.aten

_MAX_NUMEL = 2097152 * 1024

def _scalar(x):
    
    if isinstance(x, (float, int)):
        return x
    if isinstance(x, torch.Tensor) and x.dim() == 0 and x.dtype == torch.float64:
        return float(x.item())
    return None


def _norm_dim(d, rank):
    return d + rank if d < 0 else d


class _Family:
    fst_template = None
    wrapper_template = None
    tune_key = ""
    tune_params = ()

    def __init__(self, all_tune_params):
        self.tune_params = all_tune_params.get(self.tune_key, self.tune_params)

    def _mod(self, module, fst_ctx, wrapper_ctx):
        return _compile.build_kernel(module,
            self.fst_template, fst_ctx,
            self.wrapper_template, wrapper_ctx)
    
    def tune(self, all_tune_params):
        all_tune_params[self.tune_key] = self.tune_params
        return all_tune_params

def torch_dtype_to_fstar(dt):
    return {
        torch.float16: "f16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.bfloat16: "bf16",
    }[dt]

def torch_dtype_to_ctype(dt):
    return {
        torch.float16: "__half",
        torch.float32: "float",
        torch.float64: "double",
        torch.bfloat16: "__nv_bfloat16",
    }[dt]

def torch_dtype_to_aten_scalar(dt):
    """libtorch ScalarType enum name, for allocating C++ output tensors."""
    return {
        torch.float16: "torch::kFloat16",
        torch.float32: "torch::kFloat32",
        torch.float64: "torch::kFloat64",
        torch.bfloat16: "torch::kBFloat16",
    }[dt]

# ---------------------------------------------------------------------------
# Elementwise (unary, binary, scalar-broadcast)
# ---------------------------------------------------------------------------

class ElementwiseImpl(_Family):
    fst_template = "elementwise/Kuiops.Elementwise.Inst.fst.j2"
    wrapper_template = "elementwise/wrapper_elementwise.cu.j2"
    tune_key = "ELEM"
    tune_params = ()
    
    def _aten_fn_to_fstar_impl(self,fn):
        impl = None
        try: 
            impl = { \
                aten.silu.default:      "silu",     \
                aten.neg.default:       "neg",      \
                aten.rsqrt.default:     "rsqrt",    \
                aten.cos.default:       "cos",      \
                aten.sin.default:       "sin",      \
                aten.pow.Tensor_Scalar: "pow", \
                aten.add.Tensor:        "add", \
                aten.add.Scalar:        "add", \
                aten.mul.Tensor:        "mul", \
                aten.mul.Scalar:        "mul",
            }[fn]
        except KeyError:
            print(f"[kuipy-jit] warning: elementwise function unsupported: {fn}",
                  file=sys.stderr)
        return impl

    def supported(self, func, args, kwargs):
    
        def unary(X):
            return (isinstance(X, torch.Tensor) and X.is_cuda
                    and (0 < X.numel() <= _MAX_NUMEL))

        def binary(A, B):
            return (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
                    and A.is_cuda and B.is_cuda and A.dtype == B.dtype
                    and tuple(A.shape) == tuple(B.shape) and 0 < A.numel() <= _MAX_NUMEL)

        impl = self._aten_fn_to_fstar_impl(func)
        if impl is not None:
            if len(args) == 1:
                if unary(args[0]):
                    return (impl, 1, [], [args[0]])
            elif len(args) == 2:
                constargs = []
                if (func is aten.add.Tensor or 
                    func is aten.add.Scalar) and kwargs.get("alpha", 1) != 1:
                    impl += "_alpha"
                    constargs += [_scalar(kwargs["alpha"])]

                if (func is aten.add.Scalar or 
                    func is aten.mul.Scalar or 
                    func is aten.pow.Tensor_Scalar) and unary(args[0]):
                    constargs += [_scalar(args[1])]
                    return (impl, 1, constargs, [args[0]])
                elif binary(args[0], args[1]):
                    return (impl, 2, constargs, [args[0], args[1]])
                
        return None

    def run(self, spec, args, kwargs):
        method, arity, constargs, call = spec
        dtype = torch_dtype_to_fstar(call[0].dtype)
        args = [f"(i{i}: {dtype})" for i in range(arity)]
        # TODO: integer constants
        # TODO: we should have a map_gpu that does not cast the constant to `et`,
        # but rather promotes the `et` tensor to the higher precision type and 
        # does the math and writes the result in that type. 
        # Or we can just do the math in that type and then cast it back,
        # this does not require a new kernel, but it is less precise 
        # and I don't think it reflects the behavior of PyTorch.
        all_args = [f"i{i}" for i in range(arity)] + [f"(fcast {c})" for c in constargs]

        module = f"Kuiops.Elementwise{arity}.{method.title()}.{dtype.title()}"
        name = f"elem{arity}_{method}_{dtype}_jit"
        fst_ctx = dict(
            module=module,
            name=name,
            fun=f"fun {' '.join(args)} -> {method} {' '.join(all_args)}",
            arity=arity,
            et=dtype
        )
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            arity=arity,
            cpp_et=torch_dtype_to_ctype(call[0].dtype),
        )
        return self._mod(module, fst_ctx, wrapper_ctx).run(*call)

# ---------------------------------------------------------------------------
# Matmul (no GEMM)
# ---------------------------------------------------------------------------

_SHMEM_BYTES = 101376
_MAX_THREADS = 1024
_WARP = 32

def _bt2d_tile(dtype, M, N, K):
    """BlockTiling2D tile params for ``dtype``. Returns (bm,bn,bk,tm,tn) or None.

    chunk and the shared-memory byte budget scale with the element size
    (``chunk et = 16/sizeof(et)``)."""
    itemsize = dtype.itemsize
    chunk = 16 // itemsize
    for bm in (128, 64, 32):
        if M % bm:
            continue
        for bn in (128, 64, 32):
            if N % bn or bn % chunk:
                continue
            for bk in (64, 32):
                if K % bk or bk % chunk:
                    continue
                if itemsize * bm * bk + itemsize * bk * bn > _SHMEM_BYTES:
                    continue
                for tm in (16, 8):
                    if bm % tm:
                        continue
                    for tn in (16, 8):
                        if bn % tn:
                            continue
                        threads = (bm // tm) * (bn // tn)
                        if threads > _MAX_THREADS:
                            continue
                        fill = chunk * threads
                        if (bm * bk) % fill or (bk * bn) % fill:
                            continue
                        return dict(bm=bm, bn=bn, bk=bk, tm=tm, tn=tn)
    log(f"no BT2D tile params found for M={M}, N={N}, K={K}")
    return None


def _tc2d_tile(dtype, M, N, K):
    """TensorCore2D (bf16->f32, chunk=8, tm=tn=tk=16). Returns tile dict or None."""
    chunk = 16 // (dtype.itemsize)
    tm = tn = tk = 16
    for bm in (128, 64):
        if M % bm or bm % tm:
            continue
        for bn in (128, 64):
            if N % bn or bn % chunk or bn % tn:
                continue
            for bk in (64, 32, 16):
                if K % bk or bk % chunk or bk % tk:
                    continue
                if 2 * bm * bk + 2 * bk * bn > _SHMEM_BYTES:
                    continue
                for wm in (16, 8, 4, 2):
                    if (bm % (tm * wm)) or (bm % (wm * tm)):
                        continue
                    for wn in (16, 8, 4, 2):
                        if bn % (tn * wn):
                            continue
                        warps = (bm // (wm * tm)) * (bn // (wn * tn))
                        if warps * _WARP > _MAX_THREADS:
                            continue
                        fill = chunk * warps * _WARP
                        if (bm * bk) % fill or (bk * bn) % fill:
                            continue
                        return dict(bm=bm, bn=bn, bk=bk, tm=tm, tn=tn, tk=tk, wm=wm, wn=wn)
    log(f"no TC2D tile params found for M={M}, N={N}, K={K}")
    return None


class MmImpl(_Family):
    fst_template = "mm/Kuiops.Mm.Inst.fst.j2"
    wrapper_template = "mm/wrapper_mm.cu.j2"
    tune_key = "MM"
    tune_params = {"impl": "tc2d"}

    def supported(self, func, args, kwargs):
        # supposedly mm supports an alpha parameter, we don't
        if len(args) != 2 or (kwargs.get("alpha", 1) != 1):
            return None
        A,B = args
        if not ((A.is_cuda and B.is_cuda) and
        # technically unecessary as this op does not support broadcast
            (A.dim() == 2 and B.dim() == 2) and
            (A.dtype == B.dtype)):
            return None
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            return None
        M, K, N = int(M), int(K), int(N)

        impl = self.tune_params["impl"]
        # tensorcores don't support f32 A, B matrices. fall back to BT2D.
        if (A.dtype not in (torch.bfloat16, torch.float16)):
            impl = "bt2d"
        
        if impl == "tc2d": 
            tile_params = _tc2d_tile(A.dtype,M,N,K)
            in_dtype = A.dtype
            # bfloat16 out not supported by TC2D
            out_dtype = torch.float32 if in_dtype == torch.bfloat16 else in_dtype

            if tile_params is not None:
                return (impl, tile_params, in_dtype, out_dtype, [A, B])
        elif impl == "bt2d":
            tile_params = _bt2d_tile(A.dtype, M, N, K)
            if tile_params is not None:
                return (impl, tile_params, A.dtype, A.dtype, [A, B])

        return None
        
    def run(self, spec, args, kwargs):
        impl, tile_params, in_dtype, out_dtype, call = spec
        in_dtype_fst = torch_dtype_to_fstar(in_dtype)
        out_dtype_fst = torch_dtype_to_fstar(out_dtype)

        tile_params_str = "_".join(f"{k}{v}" for k,v in tile_params.items())

        module = f"Kuiops.Mm.{impl.title()}.{in_dtype_fst.title()}.{out_dtype_fst.title()}.P_{tile_params_str}"
        # keeping it short for this one...
        name = "mm_jit"
        fst_ctx = dict(
            module=module,
            name=name,
            in_et=in_dtype_fst,
            out_et=out_dtype_fst,
            impl=impl,
            **tile_params
        )
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            cpp_in_et=torch_dtype_to_ctype(in_dtype),
            cpp_out_et=torch_dtype_to_ctype(out_dtype),
            out_scalar=torch_dtype_to_aten_scalar(out_dtype),
        )
        res = self._mod(module, fst_ctx, wrapper_ctx).run(*call)
        if out_dtype != in_dtype:
            res = res.to(in_dtype)
        return res

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)

# ---------------------------------------------------------------------------
# Batched matmul (bmm)
# ---------------------------------------------------------------------------

class BmmImpl(_Family):
    fst_template = "bmm/Kuiops.Bmm.Inst.fst.j2"
    wrapper_template = "bmm/wrapper_bmm.cu.j2"
    tune_key = "BMM"
    tune_params = ()

    def supported(self, func, args, kwargs):
        if len(args) != 2:
            return None
        A, B = args
        if not (A.is_cuda and B.is_cuda and A.dim() == 3 and B.dim() == 3 and
                A.dtype == B.dtype and A.dtype in _FLOAT_DTYPES):
            return None
        Bn, M, K = (int(x) for x in A.shape)
        Bn2, K2, N = (int(x) for x in B.shape)
        if Bn != Bn2 or K != K2:
            return None
        # bmmcomb_gpu_exact: rows*cols <= max_blocks*max_threads, plus the three
        # batch products must fit in a u32 index.
        if M * N > _MAX_NUMEL:
            return None
        if max(Bn * M * K, Bn * K * N, Bn * M * N) >= 2 ** 32:
            return None
        return (A.dtype, [A, B])

    def run(self, spec, args, kwargs):
        dtype, call = spec
        et = torch_dtype_to_fstar(dtype)
        module = f"Kuiops.Bmm.{et.title()}"
        name = "bmm_jit"
        fst_ctx = dict(module=module, name=name, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            cpp_et=torch_dtype_to_ctype(dtype),
        )
        return self._mod(module, fst_ctx, wrapper_ctx).run(*call)


_MAX_BLOCKS = 2097152

# ---------------------------------------------------------------------------
# addmm (GEMM with alpha/beta via BlockTiling2D)
# ---------------------------------------------------------------------------

class AddmmImpl(_Family):
    fst_template = "addmm/Kuiops.Addmm.Inst.fst.j2"
    wrapper_template = "addmm/wrapper_addmm.cu.j2"
    tune_key = "ADDMM"
    tune_params = ()

    def supported(self, func, args, kwargs):
        if len(args) != 3:
            return None
        Cin, A, B = args
        # BlockTiling2D needs scalar + has_vec_cpy: f32, f16, bf16 (no f64).
        if not (A.is_cuda and B.is_cuda and Cin.is_cuda and
                A.dim() == 2 and B.dim() == 2 and
                A.dtype == B.dtype == Cin.dtype and
                A.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            return None
        M, K = (int(x) for x in A.shape)
        K2, N = (int(x) for x in B.shape)
        if K != K2 or M * N > _MAX_BLOCKS:
            return None
        if tuple(torch.broadcast_shapes(tuple(Cin.shape), (M, N))) != (M, N):
            return None
        tile_params = _bt2d_tile(A.dtype, M, N, K)
        if tile_params is None:
            return None
        alpha = _scalar(kwargs.get("alpha", 1))
        beta = _scalar(kwargs.get("beta", 1))
        if alpha is None or beta is None:
            return None
        return (tile_params, A.dtype, M, N, [Cin, A, B], float(alpha), float(beta))

    def run(self, spec, args, kwargs):
        tile_params, dtype, M, N, call, alpha, beta = spec
        Cin, A, B = call
        et = torch_dtype_to_fstar(dtype)
        tile_params_str = "_".join(f"{k}{v}" for k, v in tile_params.items())
        module = f"Kuiops.Addmm.{et.title()}.P_{tile_params_str}"
        name = "addmm_jit"
        fst_ctx = dict(module=module, name=name, et=et, **tile_params)
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            cpp_et=torch_dtype_to_ctype(dtype),
        )
        C = torch.empty((M, N), dtype=dtype, device=A.device)
        C.copy_(Cin)
        return self._mod(module, fst_ctx, wrapper_ctx).run(C, A, B, alpha, beta)


# ---------------------------------------------------------------------------
# softmax (row-wise, last dim)
# ---------------------------------------------------------------------------

class SoftmaxImpl(_Family):
    fst_template = "softmax/Kuiops.Softmax.Inst.fst.j2"
    wrapper_template = "softmax/wrapper_softmax.cu.j2"
    tune_key = "SOFTMAX"
    tune_params = ()

    def supported(self, func, args, kwargs):
        # aten._softmax.default(self, dim, half_to_float)
        if len(args) != 3:
            return None
        X, dim, half_to_float = args
        if not (X.is_cuda and X.dtype in _FLOAT_DTYPES and not half_to_float):
            return None
        rank = X.dim()
        if rank < 1 or _norm_dim(dim, rank) != rank - 1:
            return None
        n = int(X.shape[-1])
        m = X.numel() // n if n else 0
        # RowSoftmax: m <= max_blocks, m*n <= max_blocks*max_threads.
        if m <= 0 or m > _MAX_BLOCKS or m * n > _MAX_NUMEL:
            return None
        return (X.dtype, m, n, [X])

    def run(self, spec, args, kwargs):
        dtype, m, n, call = spec
        X = call[0]
        et = torch_dtype_to_fstar(dtype)
        module = f"Kuiops.Softmax.{et.title()}"
        name = "softmax_jit"
        fst_ctx = dict(module=module, name=name, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            cpp_et=torch_dtype_to_ctype(dtype),
        )
        A = X.contiguous().reshape(m, n).clone()
        out = self._mod(module, fst_ctx, wrapper_ctx).run(A)
        return out.reshape(X.shape)


import math as _math

# ---------------------------------------------------------------------------
# Scaled dot-product attention (efficient_attention)
# ---------------------------------------------------------------------------

class SdpaImpl(_Family):
    fst_template = "sdpa/Kuiops.Sdpa.Inst.fst.j2"
    wrapper_template = "sdpa/wrapper_sdpa.cu.j2"
    tune_key = "SDPA"
    tune_params = ()

    def supported(self, func, args, kwargs):
        # _scaled_dot_product_efficient_attention(query, key, value, attn_bias,
        #   compute_log_sumexp, dropout_p=0.0, is_causal=False, *, scale=None)
        if len(args) < 5:
            return None
        Q, Kt, V, bias = args[0], args[1], args[2], args[3]
        dropout_p = args[5] if len(args) > 5 else kwargs.get("dropout_p", 0.0)
        is_causal = args[6] if len(args) > 6 else kwargs.get("is_causal", False)
        # The kernel needs a full additive bias and no masking / dropout.
        if bias is None or dropout_p != 0.0 or is_causal:
            return None
        if not all(isinstance(t, torch.Tensor) and t.is_cuda and t.dim() == 4
                   for t in (Q, Kt, V, bias)):
            return None
        if not (Q.dtype == Kt.dtype == V.dtype == bias.dtype and Q.dtype in _FLOAT_DTYPES):
            return None
        N, H, L, E = (int(x) for x in Q.shape)
        N2, H2, S, E2 = (int(x) for x in Kt.shape)
        N3, H3, S2, Ev = (int(x) for x in V.shape)
        if (N, H, E) != (N2, H2, E2) or (N, H, S) != (N3, H3, S2):
            return None
        if tuple(int(x) for x in bias.shape) != (N, H, L, S):
            return None
        # Kernel index/grid constraints.
        if (L * S > _MAX_NUMEL or L * Ev > _MAX_NUMEL or
                N * H * L > _MAX_BLOCKS or N * H * L * S > _MAX_NUMEL):
            return None
        if max(N * H * L * E, N * H * S * E, N * H * S * Ev,
                N * H * L * Ev, N * H * L * S) >= 2 ** 32:
            return None
        scale = kwargs.get("scale", None)
        if scale is None:
            scale = 1.0 / _math.sqrt(E)
        return (Q.dtype, float(scale), [Q, Kt, V, bias])

    def run(self, spec, args, kwargs):
        dtype, scale, call = spec
        et = torch_dtype_to_fstar(dtype)
        module = f"Kuiops.Sdpa.{et.title()}"
        name = "sdpa_jit"
        fst_ctx = dict(module=module, name=name, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            cpp_et=torch_dtype_to_ctype(dtype),
        )
        # PyTorch computes softmax(scale * Q@K^T + bias); the kernel computes
        # softmax(scale * (bias' + Q@K^T)). Pre-divide the bias so they agree.
        Q, Kt, V, bias = call
        bias = bias / scale
        out, lse = self._mod(module, fst_ctx, wrapper_ctx).run(Q, Kt, V, bias, scale)
        # aten returns (output, log_sumexp[f32], philox_seed, philox_offset).
        lse = lse.to(torch.float32)
        empty = torch.empty([], dtype=torch.int64)
        return (out, lse, empty, empty)
