// TensorCore GEMM used as the linear/matmul reference in bench_tc_kernels.py.
//
// This started as the fp16xfp16->fp16 Kuiper TensorCore2D instantiation, but has
// been reworked into a mixed-precision GEMM:
//   * the accumulator type is decoupled from the input/output type (templated on
//     `AccT`); the shipped instantiation accumulates in fp32 while reading and
//     writing fp16, so it is numerically closer to cuBLAS than the old
//     accumulate-in-half kernel;
//   * it has a real GEMM epilogue, D = alpha*(A@B) + beta*C, applied in the
//     accumulator type and converted down to the fp16 output. tc2d_matmul_launch
//     keeps the plain matmul contract (alpha=1, beta=0); tc2d_gemm_launch exposes
//     the full GEMM.
//
// Layout is row-major, no transpose:
//   A : (M, K) half    B : (K, N) half    C : (M, N) half
// so to reproduce F.linear(x, W) the caller passes B = W^T (K, N).
//
// Tiling: 128x128x32 block tile, 16x16x16 WMMA, 2x2 warps (128 threads), so each
// warp owns a 64x64 output tile (4x4 fragments). The K-loop is double-buffered
// with cp.async so global loads overlap the tensor-core math, and the shared
// tiles are row-padded (SKEW) to keep the WMMA loads bank-conflict free. fp32
// accumulators cost twice the registers of fp16 ones, so the per-warp tile is
// sized to keep the accumulator register footprint in budget.
#include <kuiper.h>
#include "tc_kernels.h"
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

namespace {

constexpr int BM = 128;      // block tile rows
constexpr int BN = 128;      // block tile cols
constexpr int BK = 32;       // block tile depth
constexpr int WM = 16, WN = 16, WK = 16;   // WMMA fragment shape
constexpr int WARPS_M = 2;   // warp grid over the block tile (2x2 = 4 warps)
constexpr int WARPS_N = 2;
constexpr int WM_FRAGS = BM / (WARPS_M * WM);   // = 4  (fragments per warp, M)
constexpr int WN_FRAGS = BN / (WARPS_N * WN);   // = 4  (fragments per warp, N)
constexpr int THREADS = WARPS_M * WARPS_N * 32; // = 128
constexpr int KSTEPS = BK / WK;                 // = 2
constexpr int VEC = 8;       // halfs per 128-bit vectorized (cp.async) load
constexpr int SKEW = 8;      // shared-tile row padding to avoid bank conflicts
constexpr int LDA = BK + SKEW;   // sA row stride (ldm)
constexpr int LDB = BN + SKEW;   // sB row stride (ldm)

// Stream one 128x32 A tile + one 32x128 B tile from global into a shared buffer
// with cp.async (loads overlap the tensor-core math of the previous k-tile).
template <int BUFS>
__device__ __forceinline__ void load_tile_async(
        half (&sA)[BUFS][BM * LDA], half (&sB)[BUFS][BK * LDB], int buf,
        const half* __restrict__ gA, const half* __restrict__ gB,
        int block_row, int block_col, int kt, int K, int N) {
    #pragma unroll
    for (int i = threadIdx.x; i < (BM * BK) / VEC; i += THREADS) {
        int off = i * VEC, r = off / BK, c = off % BK;
        const half* src = gA + (size_t)(block_row * BM + r) * K + kt * BK + c;
        __pipeline_memcpy_async(&sA[buf][r * LDA + c], src, 16);
    }
    #pragma unroll
    for (int i = threadIdx.x; i < (BK * BN) / VEC; i += THREADS) {
        int off = i * VEC, r = off / BN, c = off % BN;
        const half* src = gB + (size_t)(kt * BK + r) * N + block_col * BN + c;
        __pipeline_memcpy_async(&sB[buf][r * LDB + c], src, 16);
    }
    __pipeline_commit();
}

template <typename AccT>
__global__ __launch_bounds__(THREADS)
void tc2d_gemm_kernel(const half* __restrict__ gA,
                      const half* __restrict__ gB,
                      half* __restrict__ gC,
                      int M, int N, int K,
                      float alpha, float beta) {
    __shared__ __align__(16) half sA[2][BM * LDA];   // double-buffered, ldm = LDA
    __shared__ __align__(16) half sB[2][BK * LDB];   // double-buffered, ldm = LDB

    const int num_n_blocks = N / BN;
    const int block_row = blockIdx.x / num_n_blocks;
    const int block_col = blockIdx.x % num_n_blocks;

    const int warp = threadIdx.x / 32;
    const int warp_row = warp / WARPS_N;   // 0..WARPS_M-1
    const int warp_col = warp % WARPS_N;   // 0..WARPS_N-1
    const int warp_row_base = warp_row * (WM_FRAGS * WM);
    const int warp_col_base = warp_col * (WN_FRAGS * WN);

    wmma::fragment<wmma::accumulator, WM, WN, WK, AccT> acc[WM_FRAGS][WN_FRAGS];
    #pragma unroll
    for (int i = 0; i < WM_FRAGS; i++)
        #pragma unroll
        for (int j = 0; j < WN_FRAGS; j++)
            wmma::fill_fragment(acc[i][j], AccT(0));

    const int num_k_tiles = K / BK;
    load_tile_async(sA, sB, 0, gA, gB, block_row, block_col, 0, K, N);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        __pipeline_wait_prior(0);
        __syncthreads();
        int buf = kt & 1;
        if (kt + 1 < num_k_tiles)
            load_tile_async(sA, sB, buf ^ 1, gA, gB,
                            block_row, block_col, kt + 1, K, N);

        #pragma unroll
        for (int ks = 0; ks < KSTEPS; ks++) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> aFrag[WM_FRAGS];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> bFrag[WN_FRAGS];
            #pragma unroll
            for (int i = 0; i < WM_FRAGS; i++)
                wmma::load_matrix_sync(aFrag[i],
                    &sA[buf][(warp_row_base + i * WM) * LDA + ks * WK], LDA);
            #pragma unroll
            for (int j = 0; j < WN_FRAGS; j++)
                wmma::load_matrix_sync(bFrag[j],
                    &sB[buf][(ks * WK) * LDB + warp_col_base + j * WN], LDB);
            #pragma unroll
            for (int i = 0; i < WM_FRAGS; i++)
                #pragma unroll
                for (int j = 0; j < WN_FRAGS; j++)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
    }

    // GEMM epilogue: D = alpha*acc + beta*C, evaluated in AccT, stored as half.
    #pragma unroll
    for (int i = 0; i < WM_FRAGS; i++) {
        #pragma unroll
        for (int j = 0; j < WN_FRAGS; j++) {
            int row = block_row * BM + warp_row_base + i * WM;
            int col = block_col * BN + warp_col_base + j * WN;
            half* dst = gC + (size_t)row * N + col;
            wmma::fragment<wmma::accumulator, WM, WN, WK, half> out;
            if (beta != 0.0f) {
                wmma::fragment<wmma::accumulator, WM, WN, WK, half> cfrag;
                wmma::load_matrix_sync(cfrag, dst, N, wmma::mem_row_major);
                #pragma unroll
                for (int t = 0; t < out.num_elements; t++)
                    out.x[t] = __float2half(alpha * (float)acc[i][j].x[t] +
                                            beta * __half2float(cfrag.x[t]));
            } else {
                #pragma unroll
                for (int t = 0; t < out.num_elements; t++)
                    out.x[t] = __float2half(alpha * (float)acc[i][j].x[t]);
            }
            wmma::store_matrix_sync(dst, out, N, wmma::mem_row_major);
        }
    }
}

} // namespace

// C = alpha*(A @ B) + beta*C, fp32-accumulate tensor-core GEMM. A:(M,K) B:(K,N)
// C:(M,N), all fp16 row-major. When beta != 0 the current contents of C are read
// back as the additive term (in-place GEMM). Requires M%128==0, N%128==0,
// K%32==0.
void tc2d_gemm_launch(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) {
    if (M == 0 || N == 0 || K == 0) return;
    KPR_GUARD(M % BM == 0);
    KPR_GUARD(N % BN == 0);
    KPR_GUARD(K % BK == 0);
    dim3 grid((M / BM) * (N / BN));
    tc2d_gemm_kernel<float><<<grid, THREADS>>>(
        A, B, C, M, N, K, alpha, beta);
    MUST(cudaGetLastError());
    MUST(cudaDeviceSynchronize());
}

// Plain matmul C = A @ B (alpha=1, beta=0): the bias-free F.linear drop-in.
void tc2d_matmul_launch(const half* A, const half* B, half* C,
                        int M, int N, int K) {
    tc2d_gemm_launch(A, B, C, M, N, K, 1.0f, 0.0f);
}
