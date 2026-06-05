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
// MATMUL — float32
//
// PyTorch: C = A @ B with A:(M,K), B:(K,N), C:(M,N), all row-major.
// Kuiper:  Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr requires
//          M,N,K all divisible by 32.
// Returns a freshly-allocated f32 tensor.
// =============================================================================
torch::Tensor kuiper_matmul_f32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "kuiper_matmul_f32: tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "kuiper_matmul_f32: dtype must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "kuiper_matmul_f32: rank 2 only");
    TORCH_CHECK(A.size(1) == B.size(0), "kuiper_matmul_f32: contracting dim mismatch");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(divisible(M, 32) && divisible(N, 32) && divisible(K, 32),
                "kuiper_matmul_f32: M,N,K must all be divisible by 32 (got ",
                M, ",", N, ",", K, ")");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C = torch::empty({M, N}, A.options());

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(
        (uint32_t)M, (uint32_t)N, (uint32_t)K,
        Ac.data_ptr<float>(), Bc.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

// GEMM-style: C := alpha*A*B + beta*C, in-place on C.
void kuiper_gemm_f32_(double alpha, double beta,
                      torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
                "kuiper_gemm_f32_: tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32 &&
                    C.scalar_type() == torch::kFloat32,
                "kuiper_gemm_f32_: dtype must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2,
                "kuiper_gemm_f32_: rank 2 only");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "contracting dim mismatch");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");
    TORCH_CHECK(divisible(M, 32) && divisible(N, 32) && divisible(K, 32),
                "kuiper_gemm_f32_: M,N,K must all be divisible by 32 (got ",
                M, ",", N, ",", K, ")");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "kuiper_gemm_f32_: tensors must be contiguous");

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    Klas_GEMM_SHMem_g_gemm_f32_tile32_rrr(
        (float)alpha, (float)beta,
        (uint32_t)M, (uint32_t)N, (uint32_t)K,
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
}

// =============================================================================
// MATMUL — float16  (TensorCore, 64x64x64 tile)
//
// PyTorch: C = A @ B with A:(M,K), B:(K,N), C:(M,N), f16, row-major.
// Kuiper:  Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16 requires
//          M%64, N%64, K%64 == 0. It also DOES C += A*B (it loads C into
//          the WMMA accumulator), so we ZERO C before calling.
// Returns a freshly-allocated f16 tensor.
//
// Notes:
//   - Bf16 inputs are NOT supported here — Kuiper has no verified bf16 GEMM.
//     Callers may cast bf16 → f16 manually (with the precision loss this
//     entails). See API_MISMATCHES.md.
// =============================================================================
torch::Tensor kuiper_matmul_f16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "kuiper_matmul_f16: tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16 && B.scalar_type() == torch::kFloat16,
                "kuiper_matmul_f16: dtype must be float16 (cast bf16 manually)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "kuiper_matmul_f16: rank 2 only");
    TORCH_CHECK(A.size(1) == B.size(0), "kuiper_matmul_f16: contracting dim mismatch");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(divisible(M, 64) && divisible(N, 64) && divisible(K, 64),
                "kuiper_matmul_f16: M,N,K must all be divisible by 64 (got ",
                M, ",", N, ",", K, ")");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C = torch::empty({M, N}, A.options());

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<half*>(Ac.data_ptr()),
        reinterpret_cast<half*>(Bc.data_ptr()),
        reinterpret_cast<half*>(C.data_ptr()));
    return C;
}

// =============================================================================
// MATMUL — bfloat16 inputs, float32 output  (TensorCore2D, 64x64x64 tile)
//
// PyTorch: C = A @ B with A:(M,K), B:(K,N) in bf16; C returned as f32.
// Kuiper:  Klas_GEMM_TensorCore2D_g_gemm_bf16_f32_64x64x64_16x16x16_2x2
//          requires M,N,K divisible by 64. Accumulator is f32 (mandatory:
//          bf16 has no hardware accumulator variant). The kernel
//          fill_fragment-zeros the accumulator at the start of each tile,
//          so the output is C := A*B (NOT C += A*B). No zero-init of C
//          required.
//
// Caller is responsible for casting the f32 output to bf16 if they want a
// bf16 result. We expose the f32 output so the cast can be fused with any
// downstream consumer (e.g., a bias add or normalisation).
// =============================================================================
torch::Tensor kuiper_matmul_bf16_to_f32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(),
                "kuiper_matmul_bf16_to_f32: tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16
                && B.scalar_type() == torch::kBFloat16,
                "kuiper_matmul_bf16_to_f32: dtype must be bfloat16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "kuiper_matmul_bf16_to_f32: rank 2 only");
    TORCH_CHECK(A.size(1) == B.size(0),
                "kuiper_matmul_bf16_to_f32: contracting dim mismatch");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(divisible(M, 64) && divisible(N, 64) && divisible(K, 64),
                "kuiper_matmul_bf16_to_f32: M,N,K must all be divisible "
                "by 64 (got ", M, ",", N, ",", K, ")");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    // f32 output — the verified kernel's storage dtype.
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    // Signature: (rows=M, shared=K, cols=N, gA, gB, gC).
    Klas_GEMM_TensorCore2D_g_gemm_bf16_f32_64x64x64_16x16x16_2x2(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<__nv_bfloat16*>(Ac.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()),
        C.data_ptr<float>());
    return C;
}

torch::Tensor kuiper_matmul_bf16_to_bf16_no_tile(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(),
                "kuiper_matmul_bf16_to_f32: tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16
                && B.scalar_type() == torch::kBFloat16,
                "kuiper_matmul_bf16_to_f32: dtype must be bfloat16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "kuiper_matmul_bf16_to_f32: rank 2 only");
    TORCH_CHECK(A.size(1) == B.size(0),
                "kuiper_matmul_bf16_to_f32: contracting dim mismatch");
    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C = torch::empty({M, N}, A.options());
    auto Cc = C.contiguous();

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    Klas_GEMM_Naive3_g_matmul_bf16_rrr(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<__nv_bfloat16*>(Ac.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Cc.data_ptr()));
    return Cc;
}

// =============================================================================
// MATMUL — batched float32
//
// PyTorch: C[b] = A[b] @ B[b], A:(B,M,K), B:(B,K,N).
// Kuiper:  Klas_GEMM_Batched_batched_gemm_f32(batch,rows,shared,cols,a,b)
//          allocates a fresh output and returns its device pointer (!).
//          We can't safely consume that as a torch tensor (lifetime tied to
//          cudaMalloc, no torch allocator). So we loop using the
//          non-batched matmul instead.
// =============================================================================
torch::Tensor kuiper_bmm_f32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "kuiper_bmm_f32: tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "kuiper_bmm_f32: dtype must be float32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "kuiper_bmm_f32: rank 3 only");
    TORCH_CHECK(A.size(0) == B.size(0), "kuiper_bmm_f32: batch mismatch");
    TORCH_CHECK(A.size(2) == B.size(1), "kuiper_bmm_f32: contracting dim mismatch");
    const int64_t BS = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    TORCH_CHECK(divisible(M, 32) && divisible(N, 32) && divisible(K, 32),
                "kuiper_bmm_f32: M,N,K must be divisible by 32");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto C = torch::empty({BS, M, N}, A.options());

    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();

    float* Ap = Ac.data_ptr<float>();
    float* Bp = Bc.data_ptr<float>();
    float* Cp = C.data_ptr<float>();
    for (int64_t b = 0; b < BS; ++b) {
        Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr(
            (uint32_t)M, (uint32_t)N, (uint32_t)K,
            Ap + b*M*K, Bp + b*K*N, Cp + b*M*N);
    }
    return C;
}

// =============================================================================
// SOFTMAX (in-place, last dim only)
//
// PyTorch: F.softmax(x, dim=-1).  Kuiper does softmax over a flat 1D array,
// in-place. We flatten outer dims and launch one Kuiper call per row.
//
// NOTE on numerics: Kuiper's softmax computes `exp(x_i) / sum(exp(x_j))`
// directly WITHOUT subtracting the max. For attention logits that can be
// large (~ pre-scale ~ O(sqrt(d_head)) ~ several units), this WILL overflow
// to inf/NaN. Do not enable for attention softmax unscaled. The kernel is
// suitable for already-normalised inputs (e.g., bounded logits).
// =============================================================================
void kuiper_softmax_last_f32_(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() >= 1,
                "kuiper_softmax_last_f32_: f32 CUDA tensor, rank >=1");
    TORCH_CHECK(x.is_contiguous(), "kuiper_softmax_last_f32_: must be contiguous");
    const int64_t N = x.size(-1);
    const int64_t outer = x.numel() / N;
    TORCH_CHECK(N >= 1, "empty last dim");

    at::cuda::CUDAGuard guard(x.device());
    sync_current_stream();

    float* p = x.data_ptr<float>();
    Klas_RowSoftmax_row_softmax_rm_f32(outer, N, p);
}

void kuiper_softmax_last_f16_(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.dim() >= 1,
                "kuiper_softmax_last_f16_: f16 CUDA tensor, rank >=1");
    TORCH_CHECK(x.is_contiguous(), "kuiper_softmax_last_f16_: must be contiguous");
    const int64_t N = x.size(-1);
    const int64_t outer = x.numel() / N;
    TORCH_CHECK(N >= 1, "empty last dim");

    at::cuda::CUDAGuard guard(x.device());
    sync_current_stream();

    half* p = reinterpret_cast<half*>(x.data_ptr());
    for (int64_t i = 0; i < outer; ++i) {
        Klas_Softmax_softmax_f16((uint32_t)N, p + i*N);
    }
}

void kuiper_log_softmax_last_f32_(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() >= 1,
                "kuiper_log_softmax_last_f32_: f32 CUDA tensor, rank >=1");
    TORCH_CHECK(x.is_contiguous(), "kuiper_log_softmax_last_f32_: must be contiguous");
    const int64_t N = x.size(-1);
    const int64_t outer = x.numel() / N;
    TORCH_CHECK(N >= 1, "empty last dim");
    at::cuda::CUDAGuard guard(x.device());
    sync_current_stream();
    float* p = x.data_ptr<float>();
    for (int64_t i = 0; i < outer; ++i) {
        Klas_LogSoftmax_log_softmax_f32((uint32_t)N, p + i*N);
    }
}

// =============================================================================
// REDUCE-SUM along last dim (used as building block for mean / RMSNorm).
//
// Kuiper's HReduce returns a HOST scalar after sync. Doing this per row is
// extremely expensive (one device→host copy per row). Kept here for unit
// testing, NOT as a serious replacement for torch.sum/torch.mean.
// =============================================================================
torch::Tensor kuiper_row_sum_last_f32(torch::Tensor x, int64_t nth) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() >= 1,
                "kuiper_row_sum_last_f32: f32 CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "must be contiguous");
    const int64_t N = x.size(-1);
    const int64_t outer = x.numel() / N;
    TORCH_CHECK(nth > 0 && nth <= 1024, "nth must be in 1..1024");
    auto out_shape = x.sizes().vec();
    out_shape.pop_back();
    auto out = torch::empty(out_shape, x.options().device(torch::kCPU));

    at::cuda::CUDAGuard guard(x.device());
    sync_current_stream();

    float* p = x.data_ptr<float>();
    float* op = out.data_ptr<float>();
    for (int64_t i = 0; i < outer; ++i) {
        op[i] = Klas_HReduce_reduce_f32_plus((uint32_t)nth, (uint32_t)N, p + i*N);
    }
    return out.to(x.device());
}

// =============================================================================
// ROW SCALE  (in-place):  for r in [0,m): row[r] *= scales[r]
//
// PyTorch equivalent: x *= scales[:, None]   for x:(M,N), scales:(M,)
// =============================================================================
void kuiper_row_scale_f32_(torch::Tensor mat, torch::Tensor scales) {
    TORCH_CHECK(mat.is_cuda() && scales.is_cuda(),
                "kuiper_row_scale_f32_: tensors must be CUDA");
    TORCH_CHECK(mat.scalar_type() == torch::kFloat32 &&
                    scales.scalar_type() == torch::kFloat32,
                "kuiper_row_scale_f32_: dtype must be float32");
    TORCH_CHECK(mat.dim() == 2 && scales.dim() == 1, "shapes: mat (m,n), scales (m,)");
    const int64_t M = mat.size(0), N = mat.size(1);
    TORCH_CHECK(scales.size(0) == M, "scales length must match m");
    TORCH_CHECK(mat.is_contiguous() && scales.is_contiguous(), "must be contiguous");

    at::cuda::CUDAGuard guard(mat.device());
    sync_current_stream();

    Klas_RowScale_rowscale_f32_rowmajor(
        (uint32_t)M, (uint32_t)N,
        scales.data_ptr<float>(), mat.data_ptr<float>());
}

// =============================================================================
// pybind
// =============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_f32", &kuiper_matmul_f32,
          "C = A @ B  via Klas_GEMM_BlockTiling1D f32 (M,N,K %32==0)");
    m.def("gemm_f32_",  &kuiper_gemm_f32_,
          "C := alpha*A*B + beta*C  via Klas_GEMM_SHMem f32 tile32 (M,N,K %32==0)");
    m.def("matmul_f16", &kuiper_matmul_f16,
          "C = A @ B  via Klas_GEMM_TensorCore f16 (M,N,K %64==0)");
    m.def("matmul_bf16_to_f32", &kuiper_matmul_bf16_to_f32,
          "C(f32) = A(bf16) @ B(bf16) via Klas_GEMM_TensorCore2D bf16->f32 "
          "(64x64x64 2x2 tile, M,N,K %64==0). "
          "f32 accumulator is HW-mandated; caller may cast f32->bf16 after.");
    m.def("matmul_bf16_to_bf16_no_tile", &kuiper_matmul_bf16_to_bf16_no_tile,
          "Naive3");
    m.def("bmm_f32",    &kuiper_bmm_f32,
          "Batched f32 matmul (loops Klas_GEMM_BlockTiling1D over batch)");
    m.def("softmax_last_f32_",     &kuiper_softmax_last_f32_,
          "In-place softmax over last dim, f32. NOT numerically stable.");
    m.def("softmax_last_f16_",     &kuiper_softmax_last_f16_,
          "In-place softmax over last dim, f16. NOT numerically stable.");
    m.def("log_softmax_last_f32_", &kuiper_log_softmax_last_f32_,
          "In-place log_softmax over last dim, f32.");
    m.def("row_sum_last_f32",      &kuiper_row_sum_last_f32,
          "Sum over last dim, f32 (one HReduce call per row, slow).");
    m.def("row_scale_f32_",        &kuiper_row_scale_f32_,
          "In-place row scaling: row[i] *= scales[i] (rank-2, row-major).");
}
