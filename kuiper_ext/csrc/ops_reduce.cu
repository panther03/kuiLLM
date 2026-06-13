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

// The verified Kuiper `Klas_Reduce_mean` is a 1D scalar reducer:
//   float Klas_Reduce_mean(uint32_t n, float *a)
// For the (rows, cols) reduce-along-last-dim case we loop in C++ and
// stage each scalar back into the output buffer.
torch::Tensor kuiper_mean_f32_lastdim(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "kuiper_mean_f32_lastdim: tensor must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
                "kuiper_mean_f32_lastdim: dtype must be float32");
    TORCH_CHECK(X.dim() == 2, "kuiper_mean_f32_lastdim: expected rank-2 [rows, cols]");
    TORCH_CHECK(X.is_contiguous(), "kuiper_mean_f32_lastdim: input must be contiguous");

    const int64_t rows = X.size(0);
    const int64_t cols = X.size(1);
    TORCH_CHECK(rows > 0 && cols > 0, "kuiper_mean_f32_lastdim: empty reductions unsupported");
    TORCH_CHECK(cols <= static_cast<int64_t>(UINT32_MAX) - 1024,
                "kuiper_mean_f32_lastdim: cols too large");

    auto Y = torch::empty({rows}, X.options());

    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();

    float* x_ptr = reinterpret_cast<float*>(X.data_ptr());
    float* y_ptr = reinterpret_cast<float*>(Y.data_ptr());
    for (int64_t r = 0; r < rows; ++r) {
        float scalar = Klas_Reduce_mean(static_cast<uint32_t>(cols), x_ptr + r * cols);
        AT_CUDA_CHECK(cudaMemcpy(y_ptr + r, &scalar, sizeof(float), cudaMemcpyHostToDevice));
    }

    return Y;
}

void register_reduce(pybind11::module& m) {
    m.def("mean_f32_lastdim", &kuiper_mean_f32_lastdim,
          "Klas_Reduce_mean (1D scalar mean, looped per row)");
}
