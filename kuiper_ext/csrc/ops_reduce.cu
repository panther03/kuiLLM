// PyTorch wrappers around verified Kuiper reduction kernels.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "Klas_Reduce.h"

namespace {

inline void sync_current_stream() {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture_status;
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "Kuiper kernels are not safe for CUDA graph capture "
                "(they allocate, sync, and create streams).");
    AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
}

}  // namespace

torch::Tensor kuiper_mean_f32_lastdim(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "kuiper_mean_f32_lastdim: tensor must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
                "kuiper_mean_f32_lastdim: dtype must be float32");
    TORCH_CHECK(X.dim() == 2, "kuiper_mean_f32_lastdim: expected rank-2 [rows, cols]");
    TORCH_CHECK(X.is_contiguous(), "kuiper_mean_f32_lastdim: input must be contiguous");

    const int64_t rows = X.size(0);
    const int64_t cols = X.size(1);
    TORCH_CHECK(rows > 0 && cols > 0, "kuiper_mean_f32_lastdim: empty reductions unsupported");
    TORCH_CHECK(rows <= 2097152, "kuiper_mean_f32_lastdim: rows exceed Kuiper max_blocks");
    TORCH_CHECK(cols <= static_cast<int64_t>(UINT32_MAX) - 1024,
                "kuiper_mean_f32_lastdim: cols too large");

    auto Y = torch::empty({rows}, X.options());

    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();

    Klas_Reduce_mean_fw_f32_row(
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
        1.0f / static_cast<float>(cols), reinterpret_cast<float*>(X.data_ptr()),
        reinterpret_cast<float*>(Y.data_ptr()));

    return Y;
}

void register_reduce(pybind11::module& m) {
    m.def("mean_f32_lastdim", &kuiper_mean_f32_lastdim,
          "Klas_Reduce_mean_fw_f32_row");
}
