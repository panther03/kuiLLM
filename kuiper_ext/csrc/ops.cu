// PyTorch wrappers around verified Kuiper kernels.
//
// Each wrapper:
//   1. Validates dtype / contiguity / shape divisibility BEFORE calling
//      Kuiper (Kuiper's KPR_GUARD/MUST macros call abort()/exit() on a
//      mismatch, which would kill the Python interpreter).
//   2. Synchronises the torch current stream so producers complete before
//      Kuiper launches on its own fresh stream.
//   3. Calls the Kuiper top-level host function (which itself internally
//      does cudaDeviceSynchronize() after the launch).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Kuiper headers are C++ (they use templates via <kuiper.h>) so they must
// NOT be wrapped in extern "C".
#include "Klas_GEMM_BlockTiling1D.h"
#include "Klas_GEMM_SHMem.h"
#include "Klas_GEMM_TensorCore.h"
#include "Klas_GEMM_TensorCore2D.h"
#include "Klas_GEMM_Batched.h"
#include "Klas_GEMM_Naive3.h"
#include "Klas_Softmax.h"
#include "Klas_LogSoftmax.h"
#include "Klas_RowSoftmax.h"
#include "Klas_HReduce.h"
#include "Klas_RowScale.h"

namespace {

// Sync torch's current stream so any producers feeding `t` complete before
// Kuiper launches its own fresh stream. Kuiper internally syncs the device
// after every launch, so no post-sync is needed.
// TODO: still needed if switching to privateuse1 backend?
inline void sync_current_stream() {
    auto stream = c10::cuda::getCurrentCUDAStream();
    // Refuse to launch Kuiper kernels during CUDA graph capture: Kuiper
    // internally creates streams, allocates with cudaMalloc, and calls
    // cudaDeviceSynchronize — all hostile to capture.
    cudaStreamCaptureStatus capture_status;
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "Kuiper kernels are not safe for CUDA graph capture "
                "(they allocate, sync, and create streams).");
    AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
}

inline bool divisible(int64_t x, int64_t d) { return x % d == 0 && x > 0; }

}  // namespace

// =============================================================================
// mm
// =============================================================================

torch::Tensor kuiper_mm_bf16xbf16_bf16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16,
                "kuiper_mm_bf16xbf16_bf16: dtype must be bfloat16");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(divisible(M, 64) && divisible(N, 64) && divisible(K, 64),
                "kuiper_mm_bf16xbf16_bf16: M,N,K must all be divisible by 64 (got ",
                M, ",", N, ",", K, ")");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C_temp = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

    Klas_GEMM_TensorCore2D_g_gemm_bf16_f32_64x64x64_16x16x16_2x2(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<__nv_bfloat16*>(Ac.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()),
        reinterpret_cast<float*>(C_temp.data_ptr()));

    auto C = C_temp.to(torch::kBFloat16);
    return C;
}

torch::Tensor kuiper_mm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "kuiper_mm: tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "kuiper_mm: rank 2 only");
    TORCH_CHECK(A.size(1) == B.size(0), "kuiper_mm: contracting dim mismatch");

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    if (A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16 &&
        divisible(M, 64) && divisible(N, 64) && divisible(K, 64)) {
        return kuiper_mm_bf16xbf16_bf16(A, B);
    } else {
        return torch::mm(A, B);
    }
}

// =============================================================================
// bmm
// =============================================================================

torch::Tensor kuiper_bmm_f32xf32_f32(torch::Tensor A, torch::Tensor B) {
    
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "kuiper_bmm_f32xf32_f32: dtype must be float32");
    
    const int64_t Batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    float* pC = Klas_GEMM_Batched_batched_gemm_f32(
        (uint32_t)Batch, (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<float*>(Ac.data_ptr()),
        reinterpret_cast<float*>(Bc.data_ptr()));
    TORCH_CHECK(pC != NULL);

    auto C = torch::from_blob(pC, {Batch, M, N}, A.options()); // TODO: leak?
    return C;
}

torch::Tensor kuiper_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "kuiper_bmm: tensors must be CUDA");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "kuiper_bmm: rank 3 only");
    TORCH_CHECK(A.size(2) == B.size(1), "kuiper_bmm: contracting dim mismatch");
    TORCH_CHECK(A.size(0) == B.size(0), "kuiper_bmm: batch size mismatch");

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    if (A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32) {
        return kuiper_bmm_f32xf32_f32(A, B);
    } else {
        return torch::bmm(A, B);
    }
}

// =============================================================================
// expose raw ops for unit tests
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm",  &kuiper_mm, "C = A @ B");
    m.def("mm_bf16xbf16_bf16", &kuiper_mm_bf16xbf16_bf16, "Klas_GEMM_TensorCore2D_g_gemm_bf16_f32_64x64x64_16x16x16_2x2");
    m.def("bmm", &kuiper_bmm, "C = A @ B (batched)");
    m.def("bmm_f32xf32_f32", &kuiper_bmm_f32xf32_f32, "Klas_GEMM_Batched_batched_gemm_f32");
}

// =============================================================================
// registering in pytorch
// =============================================================================

// THIS DOESNT WORK : pytorch does not want TORCH_LIBRARY(aten ...) to be redefined with existing dispatch keys.
// we can add implementations for aten ops, but we have to use the PrviateUse1 dispatch key.
// This might cause problems because usually PrivateUse1 backends are for specialized devices;
// in our case, we just want to use a different set of GPU kernels, meaning we want to keep the "cuda" device
// implementations for memcpy, alloc, etc. Maybe it is not a big deal to just reimplement these and 
// call the CUDA runtime functions.
//     TODO

// TORCH_LIBRARY(aten, m) {
//   m.def("mm(Tensor A, Tensor B) -> Tensor");
//   m.def("bmm(Tensor A, Tensor B) -> Tensor");
//   m.impl("mm", torch::kCUDA, &kuiper_mm);
//   m.impl("bmm", torch::kCUDA, &kuiper_bmm);
// }