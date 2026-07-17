// bf16 tensor-core GEMM, drop-in for the F.linear / cuBLAS ampere_bf16_s*gemm
// kernels in the trace. Computes C = A @ W^T (+ bias) with fp32 accumulation.
//
// Tiled WMMA GEMM. Each block owns a 128(M) x 64(N) output tile; its 256
// threads (8 warps in a 4x2 grid) each own a 32x32 sub-tile in four 16x16 fp32
// accumulator fragments. Per K-step a 128x32 A slab and a 64x32 B slab (B kept
// row-major as W, transposed for the col-major B fragment) are staged into
// shared memory with vectorized 128-bit loads, then fed to the tensor cores.
//
// When the output tile grid is too small to fill the GPU but K is large (the
// down_proj shape: M=256, N=896, K=4864 -> only 14 tiles), gridDim.z splits K:
// each slice atomic-adds its partial into an fp32 `workspace`, and a finalize
// pass casts to bf16 (+bias). tc_linear_splitk(M,N,K) picks the split factor.
#include "tc_kernels.h"
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define WARP 32
#define BM TC_GEMM_BM      // 128
#define BN TC_GEMM_BN      // 64
#define BK TC_GEMM_BK      // 32
#define WARPS_M 4
#define WARPS_N 2
#define WM (BM / WARPS_M)  // 32 -> 2 M-fragments
#define WN (BN / WARPS_N)  // 32 -> 2 N-fragments
#define MF (WM / 16)       // 2
#define NF (WN / 16)       // 2
#define KF (BK / 16)       // 2
#define NTHREADS (WARPS_M * WARPS_N * WARP)  // 256

__global__ void tc_linear_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    float* __restrict__ workspace,
    int M, int N, int K, int split)
{
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    const int wr = warp / WARPS_N;
    const int wc = warp % WARPS_N;

    // K-slice owned by this z-slab (split-K); rounded to a BK multiple.
    const int kchunk = ((K + split - 1) / split + BK - 1) / BK * BK;
    const int kbeg = blockIdx.z * kchunk;
    int kend = kbeg + kchunk;
    if (kend > K) kend = K;

    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BN * BK];
    __shared__ float stage[WARPS_M * WARPS_N][16 * 16];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        for (int e = tid; e < (BM * BK) / 8; e += NTHREADS) {
            int i = e / (BK / 8);
            int k = (e % (BK / 8)) * 8;
            int gm = m0 + i, gk = k0 + k;
            __nv_bfloat16* dst = &As[i * BK + k];
            const __nv_bfloat16* src = A + (int64_t)gm * K + gk;
            if (gm < M && gk + 8 <= K) {
                *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
            } else {
#pragma unroll
                for (int t = 0; t < 8; ++t)
                    dst[t] = (gm < M && gk + t < K) ? src[t] : __float2bfloat16(0.0f);
            }
        }
        for (int e = tid; e < (BN * BK) / 8; e += NTHREADS) {
            int n = e / (BK / 8);
            int k = (e % (BK / 8)) * 8;
            int gn = n0 + n, gk = k0 + k;
            __nv_bfloat16* dst = &Bs[n * BK + k];
            const __nv_bfloat16* src = W + (int64_t)gn * K + gk;
            if (gn < N && gk + 8 <= K) {
                *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
            } else {
#pragma unroll
                for (int t = 0; t < 8; ++t)
                    dst[t] = (gn < N && gk + t < K) ? src[t] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < KF; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[MF];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[NF];
#pragma unroll
            for (int mi = 0; mi < MF; ++mi)
                wmma::load_matrix_sync(af[mi], As + (wr * WM + mi * 16) * BK + kk * 16, BK);
#pragma unroll
            for (int ni = 0; ni < NF; ++ni)
                wmma::load_matrix_sync(bf[ni], Bs + (wc * WN + ni * 16) * BK + kk * 16, BK);
#pragma unroll
            for (int mi = 0; mi < MF; ++mi)
#pragma unroll
                for (int ni = 0; ni < NF; ++ni)
                    wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < MF; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NF; ++ni) {
            wmma::store_matrix_sync(stage[warp], acc[mi][ni], 16, wmma::mem_row_major);
            int rbase = m0 + wr * WM + mi * 16;
            int cbase = n0 + wc * WN + ni * 16;
            for (int e = lane; e < 16 * 16; e += WARP) {
                int r = rbase + e / 16, c = cbase + e % 16;
                if (r >= M || c >= N) continue;
                float val = stage[warp][e];
                if (split > 1) {
                    atomicAdd(&workspace[(int64_t)r * N + c], val);
                } else {
                    if (bias) val += __bfloat162float(bias[c]);
                    C[(int64_t)r * N + c] = __float2bfloat16(val);
                }
            }
        }
    }
}

// Cast the split-K fp32 accumulator to bf16, adding bias.
__global__ void tc_linear_finalize_kernel(
    const float* __restrict__ workspace, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C, int M, int N)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)M * N;
    if (idx >= total) return;
    float val = workspace[idx];
    if (bias) val += __bfloat162float(bias[idx % N]);
    C[idx] = __float2bfloat16(val);
}

void tc_linear_launch(
    const __nv_bfloat16* A, const __nv_bfloat16* W, const __nv_bfloat16* bias,
    __nv_bfloat16* C, float* workspace, int M, int N, int K, bool no_splitk,
    cudaStream_t stream)
{
    if (M == 0 || N == 0) return;
    int split = no_splitk ? 1 : tc_linear_splitk(M, N, K);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, split);
    if (split > 1)
        cudaMemsetAsync(workspace, 0, (size_t)M * N * sizeof(float), stream);
    tc_linear_kernel<<<grid, NTHREADS, 0, stream>>>(A, W, bias, C, workspace, M, N, K, split);
    if (split > 1) {
        int threads = 256;
        int64_t blocks = ((int64_t)M * N + threads - 1) / threads;
        tc_linear_finalize_kernel<<<blocks, threads, 0, stream>>>(workspace, bias, C, M, N);
    }
}
