"""kuiper_ext — verified Kuiper GPU kernels exposed as PyTorch ops.

This module JIT-builds a CUDA extension on first import that links against a
curated subset of the verified kernels in /home/julien/work/kuiper/dist.

Public surface:

    kuiper_ext.matmul_f32(A, B)       -> Tensor       (f32, M,N,K % 32 == 0)
    kuiper_ext.gemm_f32_(a, b, A, B, C)               (C := a*A*B + b*C, in-place)
    kuiper_ext.matmul_f16(A, B)       -> Tensor       (f16, M,N,K % 64 == 0)
    kuiper_ext.bmm_f32(A, B)          -> Tensor       (rank-3, f32)

    kuiper_ext.softmax_last_f32_(x)   in-place        (NOT numerically stable)
    kuiper_ext.softmax_last_f16_(x)   in-place
    kuiper_ext.log_softmax_last_f32_(x) in-place

    kuiper_ext.row_sum_last_f32(x, nth)               (slow, mostly for tests)
    kuiper_ext.row_scale_f32_(mat, scales)            in-place

    kuiper_ext.is_available() -> bool                 (True iff build worked)

For high-level usage (replace nn.Linear, monkey-patch torch.matmul / softmax),
see `kuiper_ext.integration`.
"""

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
    KUIPER_DIST / "Klas_GEMM_BlockTiling1D.cu",
    KUIPER_DIST / "Klas_GEMM_SHMem.cu",
    KUIPER_DIST / "Klas_GEMM_TensorCore.cu",
    KUIPER_DIST / "Klas_GEMM_TensorCore2D.cu",
    KUIPER_DIST / "Klas_GEMM_Batched.cu",
    KUIPER_DIST / "Klas_GEMM_Naive3.cu",
    KUIPER_DIST / "Klas_Softmax.cu",
    KUIPER_DIST / "Klas_LogSoftmax.cu",
    KUIPER_DIST / "Klas_RowSoftmax.cu",
    KUIPER_DIST / "Klas_HReduce.cu",
    KUIPER_DIST / "Klas_RowScale.cu",
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
    return getattr(_get(), name)
