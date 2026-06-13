#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "Klas_CatCast.h"

namespace {

inline void sync_current_stream() {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture_status;
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "Kuiper kernels are not safe for CUDA graph capture");
    AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
}

template <typename T>
torch::Tensor wrap_kuiper_ptr(T* ptr, torch::IntArrayRef sizes, const torch::TensorOptions& opts) {
    TORCH_CHECK(ptr != nullptr, "Kuiper returned a null pointer");
    return torch::from_blob(ptr, sizes.vec(), [=](void* p) { cudaFree(p); }, opts);
}

}  // namespace

// Verified 1D concatenation: out = cat(a, b). Both inputs must be 1D bf16.
torch::Tensor kuiper_cat2_bf16(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "kuiper_cat2_bf16: tensors must be CUDA");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
                "kuiper_cat2_bf16: dtype must be bfloat16");
    TORCH_CHECK(a.dim() == 1 && b.dim() == 1, "kuiper_cat2_bf16: rank-1 inputs only");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "kuiper_cat2_bf16: inputs must be contiguous");
    TORCH_CHECK(a.size(0) > 0 && b.size(0) > 0,
                "kuiper_cat2_bf16: empty inputs unsupported");
    const int64_t lena = a.size(0), lenb = b.size(0);
    TORCH_CHECK(lena + lenb <= static_cast<int64_t>(2097152) * 1024,
                "kuiper_cat2_bf16: total length exceeds Kuiper grid size");

    at::cuda::CUDAGuard guard(a.device());
    sync_current_stream();

    __nv_bfloat16* p = Klas_CatCast_cat2_bf16(
        static_cast<uint32_t>(lena), static_cast<uint32_t>(lenb),
        reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(b.data_ptr()));
    return wrap_kuiper_ptr(p, {lena + lenb}, a.options());
}

torch::Tensor kuiper_cast_bf16_to_f32(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.numel() > 0 && x.numel() <= INT32_MAX);
    at::cuda::CUDAGuard guard(x.device());
    sync_current_stream();
    float* p = Klas_CatCast_cast_bf16_to_f32(static_cast<uint32_t>(x.numel()),
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()));
    return wrap_kuiper_ptr(p, x.sizes(), x.options().dtype(torch::kFloat32));
}

torch::Tensor kuiper_cast_f32_to_bf16(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.numel() > 0 && x.numel() <= INT32_MAX);
    at::cuda::CUDAGuard guard(x.device());
    sync_current_stream();
    __nv_bfloat16* p = Klas_CatCast_cast_f32_to_bf16(static_cast<uint32_t>(x.numel()),
        reinterpret_cast<float*>(x.data_ptr()));
    return wrap_kuiper_ptr(p, x.sizes(), x.options().dtype(torch::kBFloat16));
}

torch::Tensor kuiper_cast_bf16_to_bf16(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.numel() > 0 && x.numel() <= INT32_MAX);
    at::cuda::CUDAGuard guard(x.device());
    sync_current_stream();
    __nv_bfloat16* p = Klas_CatCast_cast_bf16_to_bf16(static_cast<uint32_t>(x.numel()),
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()));
    return wrap_kuiper_ptr(p, x.sizes(), x.options());
}

void register_catcast(pybind11::module& m) {
    m.def("cat2_bf16", &kuiper_cat2_bf16, "Klas_CatCast_cat2_bf16 (1D dim=0)");
    m.def("cast_bf16_to_f32", &kuiper_cast_bf16_to_f32, "bf16 to f32 cast");
    m.def("cast_f32_to_bf16", &kuiper_cast_f32_to_bf16, "f32 to bf16 cast");
    m.def("cast_bf16_to_bf16", &kuiper_cast_bf16_to_bf16, "bf16 copy cast");
}
