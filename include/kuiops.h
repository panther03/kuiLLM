// Shared trivial glue for JIT family wrappers (no proof content).
#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Defined as an inline variable in kuiper.h (emitted by the kernel .cu
// translation unit). Declared here so the wrapper can point Kuiper launches at
// Torch's current stream.
extern cudaStream_t kpr_stream;

namespace kuiops {

// Route subsequent Kuiper kernel launches onto Torch's current CUDA stream.
// Must be called in each wrapper before invoking the kernel entry so the launch
// is ordered correctly and recorded into reduce-overhead CUDA graphs.
inline void use_current_stream() { kpr_stream = c10::cuda::getCurrentCUDAStream().stream(); }

inline torch::Tensor clone_in(const torch::Tensor& X) { return X.contiguous().clone(); }

// from_blob deleter for buffers a kernel allocated with cudaMalloc.
inline void cuda_free(void *p) { cudaFree(p); }

// Narrow a contiguous int64 index tensor to uint32 (Kuiper's `size_t` extracts to
// uint32_t). Copies the low 4 bytes of each little-endian int64 word: gather /
// scatter indices are non-negative and bounded by a dimension size, so they
// always fit in 32 bits. This is pure data movement -- no aten compute / cast
// op. Issued async on the current stream, so it is stream-ordered ahead of the
// consuming kernel (which also runs on the current stream).
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
