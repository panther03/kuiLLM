"""
A collection of unverified CUDA operator implementations, or otherwise not fully integrated Kuiper kernels.
These are mainly used as a reference for benchmarking (what is the equivalent CUDA implementation for some piece of code, 
i.e. what is the overhead of Kuiper itself). 

Conventions for kernels added here:
- <kernel>.cu: The main source/implementation of the kernel. 
   Should implement a `<kernel>_launch` function that is the main entry point into that kernel.
- kernels.h: A header file with all (or most) of the forward declarations for the `_launch` functions.
   The only exception is extracted Kuiper kernels; these generally come with a header file, and it's ok to put those in.
   But make sure you rename the file and the function to fit with the convention.
- wrapper.cpp: Where the PyTorch-facing implementation of the kernel should go. Should implement a C++ function called <kernel>
    and expose <kernel> as a Python function. As with Kuiops operators, the wrapper code should generally be minimal and not 
    call PyTorch implementations of kernels (such as elementwise casts), because that defeats the purpose.
    In the module definitions at the bottom of the file, try to give a descriptive comment for the kernel.

Make sure to <kernel> is a short, but descriptive name for what it implements. Mainly we want to know what kind of optimizations
it is doing.

The extension is built on first attribute access, so ``kuipy.unverified.<kernel>``
is the kernel's Python entry point. Kernels mirror the signature of the ATen op
they implement, so they are interchangeable with ``kuipy.run(op)`` in benchmarks.
"""
from pathlib import Path

from .. import config as C

_HERE = Path(__file__).resolve().parent
_SOURCES = ["wrapper.cpp", "gemm_hacky_epilogue.cu", "gemm_pipe.cu",
            "gemm_bcast_bias_epilogue.cu", "gemm_bcast_bias_epilogue2.cu",
            "flash_attn_fa1.cu", "flash_attn_fa2.cu",
            "flash_attn_manual_extract.cu"]

_mod = None


def module():
    """The compiled extension holding every unverified kernel."""
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load
        from .. import compile as _compile
        _mod = load(
            name="kuipy_unverified",
            sources=[str(_HERE / s) for s in _SOURCES],
            extra_include_paths=[str(_HERE), str(C.KUIPER_INCLUDE),
                                 str(C._REPO_ROOT / "include")],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=_compile._nvcc_flags(),
            verbose=(C.JIT_VERBOSITY > 0),
        )
    return _mod


def __getattr__(name):
    if name.startswith("__"):   # don't build for import-machinery lookups
        raise AttributeError(name)
    return getattr(module(), name)