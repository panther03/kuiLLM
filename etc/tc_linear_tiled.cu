// SNAPSHOT: 64x64 tiled WMMA GEMM (no split-K). Within ~2-3x of cuBLAS on
// most shapes; kept as a checkpoint before the split-K optimization in
// tc_linear.cu. Not compiled into the extension.
// bf16 tensor-core GEMM, drop-in for the F.linear / cuBLAS ampere_bf16_s*gemm
// kernels in the trace. Computes C = A @ W^T (+ bias) with fp32 accumulation.
//
// Classic tiled WMMA GEMM. Each block owns a 128x128 output tile; its 256
// threads (8 warps in a 2x4 grid) each own a 64x32 sub-tile held in eight
// 16x16 fp32 accumulator fragments. Per K-step it stages a 128x32 A slab and a
// 128x32 B slab (B kept row-major as W, i.e. transposed for the col-major B
// fragment) into shared memory with vectorized 128-bit loads, then issues the
// WMMA MACs. Tiles are zero-padded so any M/N/K is handled.
#include "tc_kernels.h"
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define WARP 32
#define BM 64
#define BN 64
#define BK 32
#define WARPS_M 2          // warp grid rows
#define WARPS_N 2          // warp grid cols
#define WM (BM / WARPS_M)  // 64 -> 4 M-fragments
#define WN (BN / WARPS_N)  // 32 -> 2 N-fragments
#define MF (WM / 16)       // 4
#define NF (WN / 16)       // 2
#define KF (BK / 16)       // 2

__global__ void tc_linear_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    const int wr = warp / WARPS_N;   // 0..1
    const int wc = warp % WARPS_N;   // 0..3

    __shared__ __nv_bfloat16 As[BM * BK];   // [128][32] row-major (m, k)
    __shared__ __nv_bfloat16 Bs[BN * BK];   // [128][32] row-major (n, k) == W tile
    __shared__ float stage[WARPS_M * WARPS_N][16 * 16];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
    for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Stage A: As[i][k] = A[(m0+i)*K + k0+k]; float4 = 8 bf16.
        for (int e = tid; e < (BM * BK) / 8; e += blockDim.x) {
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
        // Stage B (== W tile): Bs[n][k] = W[(n0+n)*K + k0+k].
        for (int e = tid; e < (BN * BK) / 8; e += blockDim.x) {
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
                if (r < M && c < N) {
                    float val = stage[warp][e];
                    if (bias) val += __bfloat162float(bias[c]);
                    C[(int64_t)r * N + c] = __float2bfloat16(val);
                }
            }
        }
    }
}

void tc_linear_launch(
    const __nv_bfloat16* A, const __nv_bfloat16* W, const __nv_bfloat16* bias,
    __nv_bfloat16* C, int M, int N, int K, cudaStream_t stream)
{
    if (M == 0 || N == 0) return;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    tc_linear_kernel<<<grid, WARPS_M * WARPS_N * WARP, 0, stream>>>(A, W, bias, C, M, N, K);
}
