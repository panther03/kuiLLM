// ARCHIVED naive reference (superseded by the optimized tc_linear.cu). Kept for
// correctness comparison; not compiled into the extension.
// bf16 tensor-core GEMM, drop-in for the F.linear / cuBLAS ampere_bf16_s*gemm
// kernels in the trace. Computes C = A @ W^T (+ bias) with fp32 accumulation.
//
// One block owns a 16(M) x 64(N) output tile and holds four warps, each warp
// computing a 16x16 sub-tile with a single WMMA accumulator. The 16xBK slab of
// A is loaded once per K-step and shared by all four warps (A reuse); tiles are
// zero-padded in shared memory so any M/N/K (not just multiples of 16) is
// handled. Un-tuned on purpose -- direct shared->WMMA, no double buffering.
#include "tc_kernels.h"
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define WARP 32
#define BM 16
#define BN 64
#define BK 16
#define NWARP (BN / 16)   // 4 warps, one 16-wide N stripe each

__global__ void tc_linear_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;                // 0..127
    const int warp = tid / WARP;                // 0..3
    const int lane = tid % WARP;

    __shared__ __nv_bfloat16 As[BM * BK];       // [16][16]
    __shared__ __nv_bfloat16 Bs[BK * BN];       // [16][64], Bs[k][n] = W[n0+n][k0+k]
    __shared__ float Cs[BM * BN];               // per-warp 16x16 accumulators

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += blockDim.x) {
            int i = idx / BK, j = idx % BK;
            bool ok = (m0 + i) < M && (k0 + j) < K;
            As[idx] = ok ? A[(int64_t)(m0 + i) * K + (k0 + j)] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
            int kk = idx / BN, n = idx % BN;
            bool ok = (n0 + n) < N && (k0 + kk) < K;
            Bs[idx] = ok ? W[(int64_t)(n0 + n) * K + (k0 + kk)] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::load_matrix_sync(a, As, BK);
        wmma::load_matrix_sync(b, Bs + warp * 16, BN);
        wmma::mma_sync(acc, a, b, acc);
        __syncthreads();
    }

    wmma::store_matrix_sync(Cs + warp * 16 * 16, acc, 16, wmma::mem_row_major);
    for (int e = lane; e < 16 * 16; e += WARP) {
        int i = e / 16, j = e % 16;
        int row = m0 + i, col = n0 + warp * 16 + j;
        if (row < M && col < N) {
            float val = Cs[warp * 16 * 16 + e];
            if (bias) val += __bfloat162float(bias[col]);
            C[(int64_t)row * N + col] = __float2bfloat16(val);
        }
    }
}

void tc_linear_launch(
    const __nv_bfloat16* A, const __nv_bfloat16* W, const __nv_bfloat16* bias,
    __nv_bfloat16* C, int M, int N, int K, cudaStream_t stream)
{
    if (M == 0 || N == 0) return;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    tc_linear_kernel<<<grid, NWARP * WARP, 0, stream>>>(A, W, bias, C, M, N, K);
}
