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
#include "Klas_GEMM_BlockTiling2D.h"
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
// TODO: probably not needed once all the kernels are Kuiper, but this is a minor detail
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
// addmm
// =============================================================================

torch::Tensor kuiper_addmm_bf16xbf16xbf16_bf16(torch::Tensor A, torch::Tensor B, torch::Tensor C,
        double beta, double alpha) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "kuiper_addmm: tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "kuiper_addmm: rank 2 only");
    TORCH_CHECK(A.size(1) == B.size(0) && A.size(0) == C.size(0) && B.size(1) == C.size(1), "kuiper_addmm: dim mismatch");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16
                && C.scalar_type() == torch::kBFloat16,
                "kuiper_addmm_bf16xbf16xbf16_bf16: dtype must be bfloat16");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(divisible(M, 32) && divisible(N, 32) && divisible(K, 32),
                "kuiper_addmm_bf16xbf16xbf16_bf16: M,N,K must all be divisible by 32 (got ",
                M, ",", N, ",", K, ")");

    const __nv_bfloat16 alpha_bf = __float2bfloat16(static_cast<float>(alpha));
    const __nv_bfloat16 beta_bf  = __float2bfloat16(static_cast<float>(beta));

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    // Kuiper kernel does an in-place GEMM-add: C <- alpha * (A @ B) + beta * C.
    // We must not stomp the caller's bias tensor, and the buffer must be
    // contiguous regardless of whether `C` was (e.g. an expanded 1D bias).
    auto Cout = C.contiguous().clone();

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x16(
        alpha_bf, beta_bf,
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<__nv_bfloat16*>(Ac.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Cout.data_ptr()));

    return Cout;
}

// =============================================================================
// mm
// =============================================================================

torch::Tensor kuiper_mm_bf16xbf16_bf16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "kuiper_mm: tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "kuiper_mm: rank 2 only");
    TORCH_CHECK(A.size(1) == B.size(0), "kuiper_mm: contracting dim mismatch");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16,
                "kuiper_mm_bf16xbf16_bf16: dtype must be bfloat16");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(divisible(M, 64) && divisible(N, 64) && divisible(K, 64),
                "kuiper_mm_bf16xbf16_bf16: M,N,K must all be divisible by 64 (got ",
                M, ",", N, ",", K, ")");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C_temp = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    Klas_GEMM_TensorCore2D_g_gemm_bf16_f32_64x64x64_16x16x16_2x2(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<__nv_bfloat16*>(Ac.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()),
        reinterpret_cast<float*>(C_temp.data_ptr()));

    auto C = C_temp.to(torch::kBFloat16);
    return C;
}

// =============================================================================
// bmm
// =============================================================================

torch::Tensor kuiper_bmm_f32xf32_f32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "kuiper_bmm: tensors must be CUDA");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "kuiper_bmm: rank 3 only");
    TORCH_CHECK(A.size(2) == B.size(1), "kuiper_bmm: contracting dim mismatch");
    TORCH_CHECK(A.size(0) == B.size(0), "kuiper_bmm: batch size mismatch");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "kuiper_bmm_f32xf32_f32: dtype must be float32");
    
    const int64_t Batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    float* pC = Klas_GEMM_Batched_batched_matmul_f32(
        (uint32_t)Batch, (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<float*>(Ac.data_ptr()),
        reinterpret_cast<float*>(Bc.data_ptr()));
    TORCH_CHECK(pC != NULL);

    auto C = torch::from_blob(pC, {Batch, M, N}, A.options()); // TODO: leak?
    return C;
}

// =============================================================================
// expose ops to python
// =============================================================================

void register_catcast(pybind11::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("addmm_bf16xbf16xbf16_bf16", &kuiper_addmm_bf16xbf16xbf16_bf16, "Klas_GEMM_BlockTiling2D_g_gemm_bf16_32x32x32_16x16");
    m.def("mm_bf16xbf16_bf16", &kuiper_mm_bf16xbf16_bf16, "Klas_GEMM_TensorCore2D_g_gemm_bf16_f32_64x64x64_16x16x16_2x2");
    m.def("bmm_f32xf32_f32", &kuiper_bmm_f32xf32_f32, "Klas_GEMM_Batched_batched_gemm_f32");
    register_catcast(m);
}

// note: integration with dispatcher happens in __init__.py