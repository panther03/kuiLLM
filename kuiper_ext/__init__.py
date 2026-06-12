import os
import sys
from pathlib import Path

try: 
    KUIPER_ROOT  = Path(os.environ["KUIPER_HOME"])
except KeyError:
    print("$KUIPER_HOME must be defined and point to the root of the kuiper repo!")
    
KUIPER_DIST  = KUIPER_ROOT / "dist"
KUIPER_INCS  = KUIPER_ROOT / "include"

_HERE        = Path(__file__).resolve().parent
_CSRC        = _HERE / "csrc"

# Kuiper .cu sources we need to link in (one per wrapped op family).
_KUIPER_SOURCES = [
    KUIPER_DIST / "Klas_GEMM_TensorCore2D.cu",
    KUIPER_DIST / "Klas_GEMM_BlockTiling2D.cu",
    KUIPER_DIST / "Klas_GEMM_Batched.cu",
    KUIPER_DIST / "Klas_Elementwise.cu",
    KUIPER_DIST / "Klas_Reduce.cu",
    KUIPER_DIST / "Klas_CatCast.cu",
    KUIPER_DIST / "Klas_Misc.cu",
]
_WRAPPER_SOURCES = [_CSRC / "ops.cu", _CSRC / "ops_elementwise.cu", _CSRC / "ops_reduce.cu", _CSRC / "ops_catcast.cu", _CSRC / "ops_misc.cu"]


_ext = None
_build_error = None


def _build():
    global _ext, _build_error
    if _ext is not None or _build_error is not None:
        return

    # torch.cpp_extension shells out to `ninja`; when this package is used with
    # an unactivated venv, the console script lives next to sys.executable but is
    # not necessarily on PATH.
    import shutil
    if shutil.which("ninja") is None:
        ninja = Path(sys.executable).parent / "ninja"
        if ninja.exists():
            os.environ["PATH"] = f"{ninja.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        import ninja
        os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    from torch.utils.cpp_extension import load
    # torch.cpp_extension shells out to `ninja` by name. When tests are run via
    # an absolute venv python, the venv's bin dir is not necessarily on PATH.
    os.environ["PATH"] = f"{Path(sys.prefix) / 'bin'}:{Path(sys.executable).resolve().parent}:{os.environ.get('PATH', '')}"

    sources = [str(p) for p in _WRAPPER_SOURCES + _KUIPER_SOURCES]
    extra_include_paths = [str(KUIPER_INCS), str(KUIPER_DIST)]
    # Build for the current device's compute capability. RTX 5070 Ti is sm_120.
    try:
        import torch
        cc = torch.cuda.get_device_capability(0)
        arch_flag = f"-gencode=arch=compute_{cc[0]}{cc[1]},code=sm_{cc[0]}{cc[1]}"
    except Exception:
        arch_flag = None
    # Torch's cpp_extension defines __CUDA_NO_HALF_CONVERSIONS__ et al by
    # default, which breaks Kuiper's TensorCore code (it uses brace-init
    # `(wmma::fragment<...>){0}` that needs the int->__half conversion).
    # Undefine them for our build.
    nvcc_flags = [
        "-O3", "--use_fast_math",
        "-Xcompiler", "-fPIC",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    if arch_flag:
        nvcc_flags.append(arch_flag)
    cxx_flags = ["-O3", "-std=c++17"]

    build_dir = _HERE / ".build"
    build_dir.mkdir(exist_ok=True)
    try:
        _ext = load(
            name="kuiper_ext_native",
            sources=sources,
            extra_include_paths=extra_include_paths,
            extra_cflags=cxx_flags,
            extra_cuda_cflags=nvcc_flags,
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as e:
        _build_error = e
        return


def _get():
    _build()
    if _ext is None:
        raise RuntimeError(f"kuiper_ext failed to build: {_build_error}")
    return _ext


def is_available() -> bool:
    _build()
    return _ext is not None


# Lazy attribute proxy: kuiper_ext.matmul_f32(...) lazy-builds, then dispatches.
def __getattr__(name):
    if name in {"_ext", "_build_error", "__path__"}:
        raise AttributeError(name)
    if name == "KuiperMode":
        return _KuiperMode_cls()
    return getattr(_get(), name)


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------
#
# We want the Kuiper kernels to be picked up *automatically* when PyTorch
# dispatches aten::mm / aten::addmm / aten::bmm on CUDA tensors of the right
# dtype + shape, with no model-side code changes beyond entering a context
# manager.
#
# We can't register at (aten::mm, CUDA) because that key already has a kernel,
# and re-registration is rejected by the dispatcher. PrivateUse1 would force a
# new device with its own alloc/copy/etc. Instead we hook in *above* the
# dispatcher with a TorchDispatchMode: the mode sees every aten call on the
# current thread, and we forward the matching ones to our typed C++ entry
# points. Everything else falls through to the regular CUDA kernel.

# =============================================================================
# addmm
# =============================================================================

def _supports_kuiper_addmm_common(bias, mat1, mat2, beta, alpha):
    if not (bias.is_cuda and mat1.is_cuda and mat2.is_cuda):
        return False
    if mat1.dim() != 2 or mat2.dim() != 2:
        return False
    # Bias may be 1D [N] (nn.Linear default) or 2D [M, N]. Anything else is
    # broadcastable in stock aten but we can't reshape it cleanly for Kuiper.
    if bias.dim() not in (1, 2):
        return False
    M, K = mat1.shape
    K2, N = mat2.shape
    if K != K2:
        return False
    if bias.dim() == 1:
        if bias.shape[0] != N:
            return False
    else:  # 2D
        if tuple(bias.shape) != (M, N):
            return False
    if not (isinstance(beta, (int, float)) and isinstance(alpha, (int, float))):
        return False
    return True

def _supports_kuiper_addmm_bf16(bias, mat1, mat2, beta, alpha):
    if (mat1.dtype != _torch().bfloat16 or mat2.dtype != _torch().bfloat16
            or bias.dtype != _torch().bfloat16):
        return False
    M, K = mat1.shape
    _,  N = mat2.shape
    return (M % 32 == 0) and (N % 32 == 0) and (K % 32 == 0)

# =============================================================================
# mm
# =============================================================================

def _supports_kuiper_mm_common(A, B):
    if not (A.is_cuda and B.is_cuda):
        return False
    if A.dim() != 2 or B.dim() != 2:
        return False
    return A.shape[1] == B.shape[0]

def _supports_kuiper_mm_bf16(A, B):
    if A.dtype != _torch().bfloat16 or B.dtype != _torch().bfloat16:
        return False
    M, K = A.shape
    K2, N = B.shape
    return (M % 64 == 0) and (N % 64 == 0) and (K % 64 == 0)

# =============================================================================
# bmm
# =============================================================================

def _supports_kuiper_bmm_common(A, B):
    if not (A.is_cuda and B.is_cuda):
        return False
    if A.dim() != 3 or B.dim() != 3:
        return False
    return A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]

def _supports_kuiper_bmm_f32(A, B):
    return A.dtype == _torch().float32 and B.dtype == _torch().float32

# =============================================================================
# elementwise
# =============================================================================

_MAX_ELEMENTWISE_NUMEL = 2097152 * 1024


def _supports_kuiper_unary(X, dtype):
    return (X.is_cuda and X.dtype == dtype and 0 < X.numel() <= _MAX_ELEMENTWISE_NUMEL)


def _supports_kuiper_binary(A, B, dtype):
    return (A.is_cuda and B.is_cuda and A.dtype == dtype and B.dtype == dtype
            and tuple(A.shape) == tuple(B.shape)
            and 0 < A.numel() <= _MAX_ELEMENTWISE_NUMEL)


def _scalar_value(x):
    if isinstance(x, (int, float)):
        return float(x)
    torch = _torch()
    if isinstance(x, torch.Tensor) and x.dim() == 0 and x.dtype == torch.float64:
        return float(x.item())
    return None


def _is_square_exponent(x):
    if isinstance(x, int):
        return x == 2
    if isinstance(x, float):
        return x == 2.0
    return False


# =============================================================================
# mean
# =============================================================================

def _normalize_single_dim(dim, ndim):
    if isinstance(dim, int):
        d = dim
    elif isinstance(dim, (list, tuple)) and len(dim) == 1:
        d = dim[0]
    else:
        return None
    if d < 0:
        d += ndim
    return d if 0 <= d < ndim else None

def _supports_kuiper_mean_f32_lastdim(X, dim, keepdim, dtype):
    if not (X.is_cuda and X.dtype == _torch().float32 and X.is_contiguous()):
        return False
    if dtype is not None or keepdim is not True or X.dim() < 1:
        return False
    if _normalize_single_dim(dim, X.dim()) != X.dim() - 1:
        return False
    if X.numel() == 0 or X.shape[-1] == 0:
        return False
    rows = X.numel() // X.shape[-1]
    return 0 < rows <= 2097152


# =============================================================================
# cat / cast
# =============================================================================

def _normalize_dim(dim, rank):
    return dim + rank if dim < 0 else dim

def _supports_kuiper_cat2_bf16_lastdim(tensors, dim):
    if len(tensors) != 2:
        return False
    a, b = tensors
    torch = _torch()
    if not (a.is_cuda and b.is_cuda):
        return False
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return False
    if a.dim() == 0 or b.dim() != a.dim():
        return False
    if tuple(a.shape) != tuple(b.shape):
        return False
    if not (a.is_contiguous() and b.is_contiguous()):
        return False
    return _normalize_dim(dim, a.dim()) == a.dim() - 1

def _supports_kuiper_to_copy(x, dtype):
    torch = _torch()
    return (x.is_cuda and x.is_contiguous() and x.numel() > 0 and
            ((x.dtype == torch.bfloat16 and dtype in (torch.float32, torch.bfloat16)) or
             (x.dtype == torch.float32 and dtype == torch.bfloat16)))


# =============================================================================
# misc: arange / gather
# =============================================================================

def _supports_kuiper_gather_bf16(src, dim, idx):
    import torch
    return (src.is_cuda and idx.is_cuda and src.dim() == 1 and idx.dim() == 1
            and dim == 0 and src.dtype == torch.bfloat16 and idx.dtype == torch.int64)

def _torch():
    import torch
    return torch


_KuiperMode_class = None


def _KuiperMode_cls():
    global _KuiperMode_class
    if _KuiperMode_class is not None:
        return _KuiperMode_class

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    ext = _get()  # force build so we fail loudly outside the mode

    aten = torch.ops.aten

    class KuiperMode(TorchDispatchMode):
        """Re-route eligible aten op calls to Kuiper kernels.

        Use as:
            with kuiper_ext.KuiperMode():
                model(...)
        """

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}

            # aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *,
            #             Scalar beta=1, Scalar alpha=1) -> Tensor
            # NOTE: __torch_dispatch__ does NOT fill in schema defaults, so
            # `beta`/`alpha` are absent from kwargs whenever the caller omits
            # them (the common case, including nn.Linear). Default to 1.
            if func is aten.addmm.default and len(args) == 3:
                bias, mat1, mat2 = args
                beta  = kwargs.get("beta",  1)
                alpha = kwargs.get("alpha", 1)
                if _supports_kuiper_addmm_common(bias, mat1, mat2, beta, alpha):
                    if _supports_kuiper_addmm_bf16(bias, mat1, mat2, beta, alpha):
                        # Kuiper's addmm wants a (M, N) bias buffer. nn.Linear
                        # ships a 1D [N] bias relying on broadcasting; expand
                        # and let the C++ wrapper clone it into a writable
                        # contiguous buffer.
                        M, N = mat1.shape[0], mat2.shape[1]
                        bias2d = bias.expand(M, N) if bias.dim() == 1 else bias
                        return ext.addmm_bf16xbf16xbf16_bf16(
                            mat1, mat2, bias2d, float(beta), float(alpha))

            # aten::mm(Tensor A, Tensor B) -> Tensor
            elif func is aten.mm.default and len(args) == 2:
                A, B = args
                if _supports_kuiper_mm_common(A, B):
                    if _supports_kuiper_mm_bf16(A, B):
                        return ext.mm_bf16xbf16_bf16(A, B)

            # aten::bmm(Tensor self, Tensor mat2) -> Tensor
            elif func is aten.bmm.default and len(args) == 2:
                A, B = args
                if _supports_kuiper_bmm_common(A, B):
                    if _supports_kuiper_bmm_f32(A, B):
                        return ext.bmm_f32xf32_f32(A, B)

            # unary elementwise
            elif func is aten.silu.default and len(args) == 1:
                (X,) = args
                if _supports_kuiper_unary(X, torch.bfloat16):
                    return ext.silu_bf16(X)
            elif func is aten.neg.default and len(args) == 1:
                (X,) = args
                if _supports_kuiper_unary(X, torch.bfloat16):
                    return ext.neg_bf16(X)
            elif func is aten.rsqrt.default and len(args) == 1:
                (X,) = args
                if _supports_kuiper_unary(X, torch.float32):
                    return ext.rsqrt_f32(X)
            elif func is aten.cos.default and len(args) == 1:
                (X,) = args
                if _supports_kuiper_unary(X, torch.float32):
                    return ext.cos_f32(X)
            elif func is aten.sin.default and len(args) == 1:
                (X,) = args
                if _supports_kuiper_unary(X, torch.float32):
                    return ext.sin_f32(X)
            elif func is aten.pow.Tensor_Scalar and len(args) == 2:
                X, exponent = args
                if _is_square_exponent(exponent) and _supports_kuiper_unary(X, torch.float32):
                    return ext.square_f32(X)

            # binary and scalar-broadcast elementwise
            elif func is aten.add.Tensor and len(args) >= 2:
                A, B = args[:2]
                alpha = kwargs.get("alpha", 1)
                if alpha == 1:
                    if isinstance(B, torch.Tensor) and _supports_kuiper_binary(A, B, torch.bfloat16):
                        return ext.add_bf16(A, B)
                    c = _scalar_value(B)
                    if c is not None and _supports_kuiper_unary(A, torch.float32):
                        return ext.add_const_f32(A, c)
            elif func is aten.add.Scalar and len(args) >= 2:
                A, B = args[:2]
                alpha = kwargs.get("alpha", 1)
                c = _scalar_value(B)
                if alpha == 1 and c is not None and _supports_kuiper_unary(A, torch.float32):
                    return ext.add_const_f32(A, c)
            elif func is aten.mul.Tensor and len(args) >= 2:
                A, B = args[:2]
                if isinstance(B, torch.Tensor):
                    if _supports_kuiper_binary(A, B, torch.bfloat16):
                        return ext.mul_bf16(A, B)
                    if _supports_kuiper_binary(A, B, torch.float32):
                        return ext.mul_f32(A, B)
                    c = _scalar_value(B)
                    if c is not None and _supports_kuiper_unary(A, torch.float32):
                        return ext.mul_const_f32(A, c)
            elif func is aten.mul.Scalar and len(args) >= 2:
                A, B = args[:2]
                c = _scalar_value(B)
                if c is not None and _supports_kuiper_unary(A, torch.float32):
                    return ext.mul_const_f32(A, c)

            # aten::mean.dim(Tensor self, int[]? dim, bool keepdim=False, *,
            #                ScalarType? dtype=None) -> Tensor
            elif func is aten.mean.dim and len(args) >= 2:
                X, dim = args[0], args[1]
                keepdim = args[2] if len(args) >= 3 else kwargs.get("keepdim", False)
                dtype = args[3] if len(args) >= 4 else kwargs.get("dtype", None)
                if _supports_kuiper_mean_f32_lastdim(X, dim, keepdim, dtype):
                    cols = X.shape[-1]
                    flat = X.reshape(-1, cols)
                    out = ext.mean_f32_lastdim(flat)
                    return out.reshape(*X.shape[:-1], 1)
            # aten::cat(Tensor[] tensors, int dim=0) -> Tensor
            elif func is aten.cat.default and len(args) >= 1:
                tensors = list(args[0])
                dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
                if _supports_kuiper_cat2_bf16_lastdim(tensors, dim):
                    return ext.cat2_bf16_lastdim(tensors[0], tensors[1])

            # torch Tensor.to(dtype) normally reaches aten::_to_copy from
            # __torch_dispatch__ when a real copy/cast is requested.
            elif func is aten._to_copy.default and len(args) == 1:
                x = args[0]
                dtype = kwargs.get("dtype", x.dtype)
                if _supports_kuiper_to_copy(x, dtype):
                    if x.dtype == torch.bfloat16 and dtype == torch.float32:
                        return ext.cast_bf16_to_f32(x)
                    if x.dtype == torch.float32 and dtype == torch.bfloat16:
                        return ext.cast_f32_to_bf16(x)
                    if x.dtype == torch.bfloat16 and dtype == torch.bfloat16:
                        return ext.cast_bf16_to_bf16(x)

            elif func is aten.arange.default and len(args) == 1:
                n = args[0]
                if isinstance(n, int) and kwargs.get("device", None) is not None:
                    device = kwargs.get("device")
                    dtype = kwargs.get("dtype", torch.int64)
                    if str(device).startswith("cuda") and dtype is torch.int64:
                        return ext.arange_i64(int(n))

            elif func is aten.gather.default and len(args) == 3:
                src, dim, idx = args
                if _supports_kuiper_gather_bf16(src, dim, idx):
                    return ext.gather_bf16(src, idx)

            return func(*args, **kwargs)

    _KuiperMode_class = KuiperMode
    return _KuiperMode_class
