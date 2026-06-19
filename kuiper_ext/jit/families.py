"""JIT implementations for the fixed Kuiper kernel families.

Each impl wraps one verified ``Klas.<Family>`` module (extracted on demand and
compiled with a static pybind wrapper in ``csrc/``) and exposes ``supported()``
+ ``run()``. ``supported()`` returns a small spec (or ``None``); ``run()`` calls
the compiled kernel. The eligibility checks mirror the families' C signatures.
"""
from . import compile as _compile

_MAX_NUMEL = 2097152 * 1024


def _scalar(x):
    import torch
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.Tensor) and x.dim() == 0 and x.dtype == torch.float64:
        return float(x.item())
    return None


def _norm_dim(d, rank):
    return d + rank if d < 0 else d


class _Family:
    module = None
    wrapper = None

    def _mod(self):
        return _compile.build_family(self.module, self.wrapper)


# ---------------------------------------------------------------------------
# Elementwise (unary, binary, scalar-broadcast)
# ---------------------------------------------------------------------------

class ElementwiseImpl(_Family):
    module = "Klas.Elementwise"
    wrapper = "elementwise.cu"

    def supported(self, func, args, kwargs):
        import torch
        aten = torch.ops.aten

        def unary(X, dt):
            return X.is_cuda and X.dtype == dt and 0 < X.numel() <= _MAX_NUMEL

        def binary(A, B, dt):
            return (A.is_cuda and B.is_cuda and A.dtype == dt and B.dtype == dt
                    and tuple(A.shape) == tuple(B.shape) and 0 < A.numel() <= _MAX_NUMEL)

        if func is aten.silu.default and len(args) == 1:
            if unary(args[0], torch.bfloat16):
                return ("silu_bf16", [args[0]])
        elif func is aten.neg.default and len(args) == 1:
            if unary(args[0], torch.bfloat16):
                return ("neg_bf16", [args[0]])
        elif func is aten.rsqrt.default and len(args) == 1:
            if unary(args[0], torch.float32):
                return ("rsqrt_f32", [args[0]])
        elif func is aten.cos.default and len(args) == 1:
            if unary(args[0], torch.float32):
                return ("cos_f32", [args[0]])
        elif func is aten.sin.default and len(args) == 1:
            if unary(args[0], torch.float32):
                return ("sin_f32", [args[0]])
        elif func is aten.pow.Tensor_Scalar and len(args) == 2:
            X, e = args
            if e in (2, 2.0) and unary(X, torch.float32):
                return ("square_f32", [X])
        elif func is aten.add.Tensor and len(args) >= 2:
            A, B = args[:2]
            if kwargs.get("alpha", 1) == 1:
                if isinstance(B, torch.Tensor) and binary(A, B, torch.bfloat16):
                    return ("add_bf16", [A, B])
                c = _scalar(B)
                if c is not None and unary(A, torch.float32):
                    return ("add_const_f32", [A, c])
        elif func is aten.add.Scalar and len(args) >= 2:
            A, B = args[:2]
            c = _scalar(B)
            if kwargs.get("alpha", 1) == 1 and c is not None and unary(A, torch.float32):
                return ("add_const_f32", [A, c])
        elif func is aten.mul.Tensor and len(args) >= 2:
            A, B = args[:2]
            if isinstance(B, torch.Tensor):
                if binary(A, B, torch.bfloat16):
                    return ("mul_bf16", [A, B])
                if binary(A, B, torch.float32):
                    return ("mul_f32", [A, B])
            c = _scalar(B)
            if c is not None and unary(A, torch.float32):
                return ("mul_const_f32", [A, c])
        elif func is aten.mul.Scalar and len(args) >= 2:
            A, B = args[:2]
            c = _scalar(B)
            if c is not None and unary(A, torch.float32):
                return ("mul_const_f32", [A, c])
        return None

    def run(self, spec, args, kwargs):
        method, call = spec
        return getattr(self._mod(), method)(*call)


# ---------------------------------------------------------------------------
# Reduce (mean along last dim, f32, keepdim)
# ---------------------------------------------------------------------------

class ReduceImpl(_Family):
    module = "Klas.Reduce"
    wrapper = "reduce.cu"

    def supported(self, func, args, kwargs):
        import torch
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
        import torch
        aten = torch.ops.aten
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
        import torch
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
        import torch
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
        import torch
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
