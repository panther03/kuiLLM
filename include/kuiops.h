// Shared trivial glue for JIT family wrappers (no proof content).
#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace kuiops {

inline void sync_current_stream() {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus cap;
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &cap));
    TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                "Kuiper JIT kernels are not safe for CUDA graph capture.");
    AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
}

inline torch::Tensor clone_in(const torch::Tensor& X) { return X.contiguous().clone(); }

} // namespace kuiops
