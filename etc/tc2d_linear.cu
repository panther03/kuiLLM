// Kuiper TensorCore2D GEMM used as the drop-in replacement for tc_linear.cu in
// bench_tc_kernels.py. This is the fp16xfp16->fp16 `128x128x32_16x16x16_8x4`
// instantiation extracted verbatim from the Kuiper distribution
// ($KUIPER_HOME/dist/Klas_GEMM_TensorCore2D.{cu,h}); only the surrounding launch
// wrapper (tc2d_matmul_launch) is local glue.
//
// It computes a plain matmul C = A @ B (row-major, no bias / no transpose):
//   A : (M, K) half    B : (K, N) half    C : (M, N) half
// Tensor-core matmul, so unlike a GEMM there is no bias term; the caller must
// pass B already laid out as (K, N) (i.e. the transposed F.linear weight).
#include <kuiper.h>
#include "tc_kernels.h"

__global__
/**
  hoisted when extracting g_gemm_f16_f16_128x128x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   half *gA, half *gB, half *gC)
{
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, half, wmma::row_major),
                     8U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, half), 32U);
    uint32_t fi = 0U;
    for (; fi < 32U; fi++)
        wmma::fill_fragment(accFrags[fi], __float2half_rn(0.0f));
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf03 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        half *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 512U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i + threadIdx.x * 8U) / 128U;
            uint32_t col = (i + threadIdx.x * 8U) % 128U;
            vec_memcpy(local,
                       tileB + (cols * __anf03 * 32U + mcol * 128U +
                                cols * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf010 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 8U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 128U +
                                        __anf010 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf011 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf011 * 16U +
                                        threadIdx.x / 32U % 2U * 64U +
                                        i1 * 16U), 128U);
            uint32_t resIdxM = 0U;
            for (; resIdxM < 8U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }
    uint32_t i = 0U;
    for (; i < 8U; i++) {
        uint32_t j = 0U;
        for (; j < 4U; j++)
            wmma::store_matrix_sync(gC +
                                    (cols * (blockIdx.x / (cols / 128U)) *
                                     128U + blockIdx.x % (cols / 128U) * 128U +
                                     cols * (threadIdx.x / 32U / 2U) * 128U +
                                     threadIdx.x / 32U % 2U * 64U +
                                     cols * i * 16U + j * 16U),
                                    accFrags[i * 4U + j], cols,
                                    wmma::mem_row_major);
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              half *gA,
                                                              half *gB,
                                                              half *gC)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 16384U));
    KPR_KCALL(__hoisted_g_gemm_f16_f16_128x128x32_16x16x16_8x4_0, nblk, 64U,
              16384U, shared, cols, gA, gB, gC);
    MUST(cudaDeviceSynchronize());
}

void tc2d_matmul_launch(const half* A, const half* B, half* C,
                        int M, int N, int K) {
    if (M == 0 || N == 0 || K == 0) return;
    Klas_GEMM_TensorCore2D_g_gemm_f16_f16_128x128x32_16x16x16_8x4(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        const_cast<half*>(A), const_cast<half*>(B), C);
}
