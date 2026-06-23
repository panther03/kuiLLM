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
        torch.float32: "float",
        torch.float64: "double",
        torch.bfloat16: "__nv_bfloat16",
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
            return X.is_cuda and (0 < X.numel() <= _MAX_NUMEL)

        def binary(A, B):
            return (A.is_cuda and B.is_cuda and A.dtype == B.dtype
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

def _bt2d_tile(M, N, K):
    """BlockTiling2D (f32, chunk=4). Returns (bm,bn,bk,tm,tn) or None."""
    chunk = 4
    for bm in (128, 64, 32):
        if M % bm:
            continue
        for bn in (128, 64, 32):
            if N % bn or bn % chunk:
                continue
            for bk in (64, 32):
                if K % bk or bk % chunk:
                    continue
                if 4 * bm * bk + 4 * bk * bn > _SHMEM_BYTES:
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
            tile_params = _bt2d_tile(M,N,K)
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
            cpp_out_et=torch_dtype_to_ctype(out_dtype)
        )
        res = self._mod(module, fst_ctx, wrapper_ctx).run(*call)
        if out_dtype != in_dtype:
            res = res.to(in_dtype)
        return res


'''
# ---------------------------------------------------------------------------
# Reduce (mean along last dim, f32, keepdim)
# ---------------------------------------------------------------------------

class ReduceImpl(_Family):
    module = "Klas.Reduce"
    wrapper = "reduce.cu"

    def supported(self, func, args, kwargs):
        if not (len(args) >= 2):
            return None
        X, dim = args[0], args[1]
        keepdim = args[2] if len(args) >= 3 else kwargs.get("keepdim", False)
        dtype = args[3] if len(args) >= 4 else kwargs.get("dtype", None)
        if not (X.is_cuda and X.dtype == torch.float32 and X.is_contiguous()):
            return None
        if dtype is not None or keepdim is not True or X.dim() < 1:
            return None
        if isinstance(dim, (list, tuple)):
            if len(dim) != 1:
                return None
            dim = dim[0]
        if not isinstance(dim, int):
            return None
        if _norm_dim(dim, X.dim()) != X.dim() - 1:
            return None
        if X.numel() == 0 or X.shape[-1] == 0:
            return None
        rows = X.numel() // X.shape[-1]
        if not (0 < rows <= 2097152):
            return None
        return ("mean", X)

    def run(self, spec, args, kwargs):
        _, X = spec
        cols = X.shape[-1]
        out = self._mod().mean_f32_lastdim(X.reshape(-1, cols))
        return out.reshape(*X.shape[:-1], 1)


# ---------------------------------------------------------------------------
# CatCast (1D bf16 concat; bf16<->f32 casts)
# ---------------------------------------------------------------------------

class CatCastImpl(_Family):
    module = "Klas.CatCast"
    wrapper = "catcast.cu"

    def supported(self, func, args, kwargs):
        
        
        if func is aten.cat.default and len(args) >= 1:
            tensors = list(args[0])
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            if len(tensors) != 2:
                return None
            a, b = tensors
            if not (a.is_cuda and b.is_cuda):
                return None
            if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
                return None
            if a.dim() != 1 or b.dim() != 1 or a.numel() == 0 or b.numel() == 0:
                return None
            if not (a.is_contiguous() and b.is_contiguous()):
                return None
            if _norm_dim(dim, a.dim()) != 0:
                return None
            return ("cat2_bf16", [a, b])
        if func is aten._to_copy.default and len(args) == 1:
            x = args[0]
            dtype = kwargs.get("dtype", x.dtype)
            device = kwargs.get("device", None)
            if device is not None and str(device) != str(x.device):
                return None
            if not (x.is_cuda and x.is_contiguous() and x.numel() > 0):
                return None
            if x.dtype == torch.bfloat16 and dtype == torch.float32:
                return ("cast_bf16_to_f32", [x])
            if x.dtype == torch.float32 and dtype == torch.bfloat16:
                return ("cast_f32_to_bf16", [x])
            if x.dtype == torch.bfloat16 and dtype == torch.bfloat16:
                return ("cast_bf16_to_bf16", [x])
        return None

    def run(self, spec, args, kwargs):
        method, call = spec
        return getattr(self._mod(), method)(*call)


# ---------------------------------------------------------------------------
# Arange (i64, cuda)
# ---------------------------------------------------------------------------

class ArangeImpl(_Family):
    module = "Klas.Arange"
    wrapper = "arange.cu"

    def supported(self, func, args, kwargs):
        
        if len(args) != 1:
            return None
        n = args[0]
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", torch.int64)
        if isinstance(n, int) and device is not None and str(device).startswith("cuda") \
                and dtype is torch.int64 and n > 0:
            return ("arange", int(n))
        return None

    def run(self, spec, args, kwargs):
        return self._mod().arange_i64(spec[1], 0, 1)


# ---------------------------------------------------------------------------
# Gather (bf16 src, i64 idx, dim 0)
# ---------------------------------------------------------------------------

class GatherImpl(_Family):
    module = "Klas.Gather"
    wrapper = "gather.cu"

    def supported(self, func, args, kwargs):
        
        if len(args) != 3:
            return None
        src, dim, idx = args
        if not (src.is_cuda and idx.is_cuda):
            return None
        if src.dtype != torch.bfloat16 or idx.dtype != torch.int64:
            return None
        if not (src.is_contiguous() and idx.is_contiguous()):
            return None
        if src.dim() != idx.dim() or dim != 0:
            return None
        if src.dim() == 1:
            return ("gather", [src, idx])
        if src.dim() == 2 and idx.size(1) == src.size(1):
            return ("gather", [src, idx])
        return None

    def run(self, spec, args, kwargs):
        _, call = spec
        return self._mod().gather_bf16(*call)


# ---------------------------------------------------------------------------
# Batched GEMM (bmm, f32)
# ---------------------------------------------------------------------------

class BmmImpl(_Family):
    module = "Klas.GEMM.Batched"
    wrapper = "batched.cu"

    def supported(self, func, args, kwargs):
        
        if len(args) != 2:
            return None
        A, B = args
        if not (A.is_cuda and B.is_cuda):
            return None
        if A.dim() != 3 or B.dim() != 3:
            return None
        if A.dtype != torch.float32 or B.dtype != torch.float32:
            return None
        if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
            return None
        return ("bmm", [A, B])

    def run(self, spec, args, kwargs):
        _, call = spec
        return self._mod().bmm_f32(*call)
'''