"""Build ``kuipy/unverified/gemm_tc_flat_splitk_noepi.cu`` as a torch extension.

Same idea as ``etc.ref_flat_gemm``, but the split-K launcher takes a caller-owned
fp32 ``(SPLITS, M, N)`` workspace, so it needs its own wrapper. The returned
callable has the same ``(a, bt) -> d`` signature as ``kuipy.run(aten.mm)``, and
allocates the workspace exactly where the Kuiper wrapper does, so the two are
timed on equal terms.
"""
import hashlib

import torch

from kuipy import config as C
from kuipy import compile as _compile

_UNVERIFIED = C._REPO_ROOT / "kuipy" / "unverified"

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
    const elem_t* A, const elem_t* B, elem_t* D, float* ws, int M, int N,
    int K, cudaStream_t stream);

at::Tensor {stem}(at::Tensor A, at::Tensor Bt) {{
    TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(A.scalar_type() == TORCH_ELEM, "A dtype mismatch");
    TORCH_CHECK(Bt.scalar_type() == TORCH_ELEM, "B dtype mismatch");
    int M = A.size(0), K = A.size(1), N = Bt.size(1);
    at::Tensor B = Bt.t();
    TORCH_CHECK(A.is_contiguous(), "A must be row-major");
    TORCH_CHECK(B.is_contiguous(), "B^T must be row-major (n, k)");
    at::Tensor D = at::empty({{M, N}}, A.options());
    at::Tensor W = at::empty({{(int64_t)SPLITS * M, N}},
                             A.options().dtype(at::kFloat));
    cudaError_t e = {stem}_launch(
        reinterpret_cast<const elem_t*>(A.data_ptr()),
        reinterpret_cast<const elem_t*>(B.data_ptr()),
        reinterpret_cast<elem_t*>(D.data_ptr()),
        reinterpret_cast<float*>(W.data_ptr()),
        M, N, K, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
    return D;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{stem}", &{stem}, "unverified split-K flat tensor-core GEMM");
}}
"""

_cache = {}


def load(splits=4, stem="gemm_tc_flat_splitk_noepi", bf16=True,
         bm=128, bn=128, bk=32, wm=32, wn=64, skew=8, group=8,
         reduce_threads=256):
    """Compile one (tiling, splits) of ``stem`` and return its callable."""
    key = (stem, bf16, splits, bm, bn, bk, wm, wn, skew, group, reduce_threads)
    if key in _cache:
        return _cache[key]

    defines = [f"-DGEMM_BF16={1 if bf16 else 0}", f"-DSPLITS={splits}",
               f"-DBM={bm}", f"-DBN={bn}", f"-DBK={bk}", f"-DWM={wm}",
               f"-DWN={wn}", f"-DSKEW={skew}", f"-DGROUP={group}",
               f"-DREDUCE_THREADS={reduce_threads}"]
    tag = hashlib.sha1("".join(map(str, key)).encode()).hexdigest()[:10]
    name = f"ref_{stem}_{tag}"

    build_dir = C.KUIPY_CACHE / "build" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / "wrapper.cpp"
    src.write_text(_WRAPPER.format(stem=stem))

    from torch.utils.cpp_extension import load as _load
    _compile._ensure_ninja_on_path()
    mod = _load(
        name=name,
        sources=[str(_UNVERIFIED / f"{stem}.cu"), str(src)],
        extra_include_paths=[str(_UNVERIFIED)],
        extra_cflags=["-O2", "-std=c++17", f"-DGEMM_BF16={1 if bf16 else 0}",
                      f"-DSPLITS={splits}"],
        extra_cuda_cflags=_compile._nvcc_flags() + defines,
        build_directory=str(build_dir),
        verbose=(C.JIT_VERBOSITY > 0),
    )
    fn = getattr(mod, stem)
    _cache[key] = fn
    return fn
