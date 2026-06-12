#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "Klas_Misc.h"

namespace {

inline void sync_current_stream_misc() {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture_status;
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "Kuiper kernels are not safe for CUDA graph capture.");
    AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
}

__global__ void arange_i64_kernel(int64_t *out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

__global__ void gather_bf16_i64_kernel(const __nv_bfloat16 *src, const int64_t *idx,
                                       __nv_bfloat16 *out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[idx[i]];
}

}  // namespace

torch::Tensor kuiper_arange_i64(int64_t n) {
    TORCH_CHECK(n > 0, "kuiper_arange_i64: n must be positive");
    auto out = torch::empty({n}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
    at::cuda::CUDAGuard guard(out.device());
    sync_current_stream_misc();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    arange_i64_kernel<<<blocks, threads>>>(reinterpret_cast<int64_t*>(out.data_ptr()), n);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

torch::Tensor kuiper_gather_bf16(torch::Tensor src, torch::Tensor idx) {
    TORCH_CHECK(src.is_cuda() && idx.is_cuda(), "kuiper_gather_bf16: tensors must be CUDA");
    TORCH_CHECK(src.dim() == 1 && idx.dim() == 1, "kuiper_gather_bf16: rank 1 only");
    TORCH_CHECK(src.scalar_type() == torch::kBFloat16, "kuiper_gather_bf16: src must be bf16");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "kuiper_gather_bf16: idx must be int64");
    auto sc = src.contiguous();
    auto ic = idx.contiguous();
    auto out = torch::empty({idx.size(0)}, src.options());
    at::cuda::CUDAGuard guard(src.device());
    sync_current_stream_misc();
    int64_t n = idx.size(0);
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    gather_bf16_i64_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(sc.data_ptr()),
        reinterpret_cast<const int64_t*>(ic.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

void register_misc(pybind11::module& m) {
    m.def("arange_i64", &kuiper_arange_i64, "arange_i64");
    m.def("gather_bf16", &kuiper_gather_bf16, "gather_bf16");
}
