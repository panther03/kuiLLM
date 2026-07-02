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

// Narrow a contiguous int64 index tensor to uint32 (Kuiper's `size_t` extracts to
// uint32_t). Copies the low 4 bytes of each little-endian int64 word: gather /
// scatter indices are non-negative and bounded by a dimension size, so they
// always fit in 32 bits. This is pure data movement -- no aten compute / cast
// op. Issued async on the current stream; callers must `sync_current_stream()`
// (or otherwise stream-order) before the consuming kernel runs.
inline torch::Tensor index_to_u32(const torch::Tensor& Idx) {
    auto I = Idx.contiguous();
    auto U = torch::empty(I.sizes(), I.options().dtype(torch::kInt32));
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemcpy2DAsync(
        U.data_ptr(), 4, I.data_ptr(), 8, 4, (size_t)I.numel(),
        cudaMemcpyDeviceToDevice, stream.stream()));
    return U;
}

} // namespace kuiops
