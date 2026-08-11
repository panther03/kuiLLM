"""Build ``kuipy/unverified/gemm_tc_flat_*.cu`` as a torch extension.

The reference kernels are standalone: one specialization per translation unit,
tiling fixed by ``-D`` flags, so they are not in ``kuipy.unverified._SOURCES``.
This builds one variant on demand and hands back a ``(a, bt) -> d`` callable
with the same signature as ``kuipy.run(aten.mm)``, so the two can be dropped
into the same benchmark driver and timed identically.
"""
import hashlib
from pathlib import Path

import torch

from kuipy import config as C
from kuipy import compile as _compile

_HERE = Path(__file__).resolve().parent
_UNVERIFIED = C._REPO_ROOT / "kuipy" / "unverified"

# One wrapper per variant: `<stem>_launch` is the entry point by convention.
_WRAPPER = """
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if GEMM_BF16
using elem_t = __nv_bfloat16;
#define TORCH_ELEM at::kBFloat16
#else
using elem_t = __half;
#define TORCH_ELEM at::kHalf
#endif

extern "C" cudaError_t {stem}_launch(
    const elem_t* A, const elem_t* B, elem_t* D, int M, int N, int K,
    cudaStream_t stream);

// D = A @ B^T, with B passed as its transpose (i.e. `bt` is (k, n) with
// bt.t() row-major (n, k)) so the signature matches aten.mm(a, b.t()).
at::Tensor {stem}(at::Tensor A, at::Tensor Bt) {{
    TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(A.scalar_type() == TORCH_ELEM, "A dtype mismatch");
    TORCH_CHECK(Bt.scalar_type() == TORCH_ELEM, "B dtype mismatch");
    int M = A.size(0), K = A.size(1), N = Bt.size(1);
    // Bt is (k, n) whose transpose is the row-major (n, k) the kernel wants.
    at::Tensor B = Bt.t();
    TORCH_CHECK(A.is_contiguous(), "A must be row-major");
    TORCH_CHECK(B.is_contiguous(), "B^T must be row-major (n, k)");
    at::Tensor D = at::empty({{M, N}}, A.options());
    cudaError_t e = {stem}_launch(
        reinterpret_cast<const elem_t*>(A.data_ptr()),
        reinterpret_cast<const elem_t*>(B.data_ptr()),
        reinterpret_cast<elem_t*>(D.data_ptr()),
        M, N, K, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
    return D;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{stem}", &{stem}, "unverified flat tensor-core GEMM");
}}
"""

# The epilogue variant takes C, alpha and beta as well, and its C view is fixed
# at compile time by CS_M/CS_N (see "THE C VIEW" in the reference), so a given
# specialization serves exactly one C shape.
_WRAPPER_EPI = """
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if GEMM_BF16
using elem_t = __nv_bfloat16;
#define TORCH_ELEM at::kBFloat16
#else
using elem_t = __half;
#define TORCH_ELEM at::kHalf
#endif

extern "C" cudaError_t {stem}_launch(
    const elem_t* A, const elem_t* B, const elem_t* C, elem_t* D, int M,
    int N, int K, float alpha, float beta, cudaStream_t stream);

at::Tensor {stem}(at::Tensor Cin, at::Tensor A, at::Tensor Bt, double alpha,
                  double beta) {{
    TORCH_CHECK(A.is_cuda() && Bt.is_cuda() && Cin.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(A.scalar_type() == TORCH_ELEM, "A dtype mismatch");
    TORCH_CHECK(Bt.scalar_type() == TORCH_ELEM, "B dtype mismatch");
    TORCH_CHECK(Cin.scalar_type() == TORCH_ELEM, "C dtype mismatch");
    int M = A.size(0), K = A.size(1), N = Bt.size(1);
    at::Tensor B = Bt.t();
    TORCH_CHECK(A.is_contiguous(), "A must be row-major");
    TORCH_CHECK(B.is_contiguous(), "B^T must be row-major (n, k)");
    TORCH_CHECK(Cin.is_contiguous(), "C must be contiguous");
    at::Tensor D = at::empty({{M, N}}, A.options());
    cudaError_t e = {stem}_launch(
        reinterpret_cast<const elem_t*>(A.data_ptr()),
        reinterpret_cast<const elem_t*>(B.data_ptr()),
        reinterpret_cast<const elem_t*>(Cin.data_ptr()),
        reinterpret_cast<elem_t*>(D.data_ptr()),
        M, N, K, (float)alpha, (float)beta,
        at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
    return D;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{stem}", &{stem}, "unverified flat tensor-core GEMM (epilogue)");
}}
"""

_cache = {}


def load(stem="gemm_tc_flat_nosplitk_noepi", bf16=True,
         bm=128, bn=128, bk=32, wm=32, wn=64, skew=8, group=8,
         cs_m=None, cs_n=None):
    """Compile one tiling of ``stem`` and return its ``(a, bt) -> d`` callable.

    Passing ``cs_m``/``cs_n`` selects the epilogue variant's C view and the
    ``(c, a, bt, alpha, beta) -> d`` signature instead.
    """
    key = (stem, bf16, bm, bn, bk, wm, wn, skew, group, cs_m, cs_n)
    if key in _cache:
        return _cache[key]

    epi = cs_m is not None or cs_n is not None
    defines = [f"-DGEMM_BF16={1 if bf16 else 0}", f"-DBM={bm}", f"-DBN={bn}",
               f"-DBK={bk}", f"-DWM={wm}", f"-DWN={wn}", f"-DSKEW={skew}",
               f"-DGROUP={group}"]
    if epi:
        defines += [f"-DCS_M={cs_m}", f"-DCS_N={cs_n}"]
    tag = hashlib.sha1("".join(map(str, key)).encode()).hexdigest()[:10]
    name = f"ref_{stem}_{tag}"

    build_dir = C.KUIPY_CACHE / "build" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / "wrapper.cpp"
    src.write_text((_WRAPPER_EPI if epi else _WRAPPER).format(stem=stem))

    from torch.utils.cpp_extension import load as _load
    _compile._ensure_ninja_on_path()
    mod = _load(
        name=name,
        sources=[str(_UNVERIFIED / f"{stem}.cu"), str(src)],
        extra_include_paths=[str(_UNVERIFIED)],
        # The wrapper needs GEMM_BF16 too, to pick elem_t.
        extra_cflags=["-O2", "-std=c++17", f"-DGEMM_BF16={1 if bf16 else 0}"],
        extra_cuda_cflags=_compile._nvcc_flags() + defines,
        build_directory=str(build_dir),
        verbose=(C.JIT_VERBOSITY > 0),
    )
    fn = getattr(mod, stem)
    _cache[key] = fn
    return fn
