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

__global__ void cat2_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                 __nv_bfloat16* c, uint32_t outer, uint32_t inner) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row_width = inner * 2u;
    uint32_t total = outer * row_width;
    if (i < total) {
        uint32_t row = i / row_width;
        uint32_t col = i - row * row_width;
        uint32_t src = row * inner + (col < inner ? col : col - inner);
        c[i] = (col < inner) ? a[src] : b[src];
    }
}

}  // namespace

torch::Tensor kuiper_cat2_bf16_lastdim(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "kuiper_cat2_bf16_lastdim: tensors must be CUDA");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
                "kuiper_cat2_bf16_lastdim: dtype must be bfloat16");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "kuiper_cat2_bf16_lastdim: tensors must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "kuiper_cat2_bf16_lastdim: shapes must match");
    TORCH_CHECK(a.numel() > 0 && a.numel() <= INT32_MAX / 2, "kuiper_cat2_bf16_lastdim: unsupported size");

    auto out_sizes = a.sizes().vec();
    out_sizes.back() *= 2;
    auto out = torch::empty(out_sizes, a.options());

    at::cuda::CUDAGuard guard(a.device());
    sync_current_stream();
    const uint32_t inner = static_cast<uint32_t>(a.size(-1));
    const uint32_t outer = static_cast<uint32_t>(a.numel() / a.size(-1));
    const uint32_t threads = 256;
    const uint32_t blocks = (2 * static_cast<uint32_t>(a.numel()) + threads - 1) / threads;
    cat2_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), outer, inner);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    return out;
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
    m.def("cat2_bf16_lastdim", &kuiper_cat2_bf16_lastdim, "2-input bf16 cat along last dim");
    m.def("cast_bf16_to_f32", &kuiper_cast_bf16_to_f32, "bf16 to f32 cast");
    m.def("cast_f32_to_bf16", &kuiper_cast_f32_to_bf16, "f32 to bf16 cast");
    m.def("cast_bf16_to_bf16", &kuiper_cast_bf16_to_bf16, "bf16 copy cast");
}
