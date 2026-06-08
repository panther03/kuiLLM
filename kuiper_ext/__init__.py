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
]
_WRAPPER_SOURCES = [_CSRC / "ops.cu"]


_ext = None
_build_error = None


def _build():
    global _ext, _build_error
    if _ext is not None or _build_error is not None:
        return

    from torch.utils.cpp_extension import load

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

            return func(*args, **kwargs)

    _KuiperMode_class = KuiperMode
    return _KuiperMode_class
