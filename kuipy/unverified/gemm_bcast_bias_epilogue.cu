// Extracted instance of Kuiper TensorCore2D (`addmm`, f16 in / f32 accumulate /
// f16 out, 128x128x32 blocks, 16x16x16 fragments, 2x4 fragments per warp) with
// the epilogue rewritten by hand to take a *broadcast* C operand.
//
// Verbatim from .kuipy_cache/cu/Kuiops_Addmm_Tc2D_F16_F32_F16
// _P_bm128_bn128_bk32_tm16_tn16_tk16_wm2_wn4.cu apart from the epilogue; only
// the trailing KPR_STORE_COMB loop below is hand-written. Sketches what a
// broadcast-C `addmm` would look like in Kuiper, which currently cannot serve
// `nn.Linear`'s bias: the aten call passes C as a length-N row vector and
// Kuiper reads C through a tlayout, which is an injection, so the stride-0 row
// axis is inexpressible.
//
// The epilogue therefore materialises the broadcast instead of expressing it as
// a layout: the block's 128-column slice of the bias is staged into shared
// memory replicated over the 16 rows of a fragment, and the existing fragment
// load reads it back with a normal (non-zero) row stride. That keeps the global
// traffic for the bias at N elements per block rather than the M*N of a
// materialised C matrix, costs 4 KiB of shared memory (reused from the A stage,
// which is dead by then), and stays inside what Kuiper's tile abstraction can
// already express.
//
//   A : (M, K) f16    B : (K, N) f16    bias : (N,) f16    C : (M, N) f16
#include <kuiper.h>
#include <kuiops_compat.h>
#include "gemm_bcast_bias_epilogue.h"

__global__
/**
  hoisted when extracting addmm_jit
*/
static void
__hoisted_addmm_jit_0(float alpha,
                      float beta,
                      uint32_t cols,
                      uint32_t shared, half *gA, half *gB, half *gBias,
                      half *gC)
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
                     2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, half, wmma::row_major),
                     4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        uint32_t __anf03 = bkIdx;
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
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
        for (; i < 4096U; i += 2048U) {
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
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 32U +
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
            for (; resIdxM < 2U; resIdxM++) {
                uint32_t resIdxN = 0U;
                for (; resIdxN < 4U; resIdxN++) {
                    auto & acc_frag = accFrags[resIdxM * 4U + resIdxN];
                    wmma::mma_sync(acc_frag, aFrags[resIdxM], bFrags[resIdxN],
                                   acc_frag);
                }
            }
        }
    }

    // ---- hand-written broadcast-C epilogue ---------------------------------
    // Replicate this block's 128 bias columns over the 16 rows of a fragment,
    // in the (now dead) A stage. A fragment load at row stride 128 then yields
    // bias[col] for every row of the tile.
    half *sBias = sA;
    __syncthreads();
    for (uint32_t t = threadIdx.x; t < 16U * 128U; t += 256U)
        sBias[t] = gBias[mcol * 128U + t % 128U];
    __syncthreads();

    uint32_t warp_row = threadIdx.x / 32U / 2U;
    uint32_t warp_col = threadIdx.x / 32U % 2U;
    KPR_RETYPE_FRAG(accFrags[0], half) biasFrags[4U];
    for (uint32_t j = 0U; j < 4U; j++)
        wmma::load_matrix_sync(biasFrags[j], sBias + (warp_col * 64U + j * 16U),
                               128U, wmma::mem_row_major);

    for (uint32_t i = 0U; i < 2U; i++)
        for (uint32_t j = 0U; j < 4U; j++) {
            auto & acc = accFrags[i * 4U + j];
            KPR_RETYPE_FRAG(accFrags[0], half) out;
            for (uint32_t e = 0U; e < KPR_NELEM(acc); e++)
                KPR_FRAG_SET(out, e,
                             __float2half_rn(beta *
                                             __half2float(KPR_FRAG_GET
                                                          (biasFrags[j], e)) +
                                             alpha * KPR_FRAG_GET(acc, e)));
            wmma::store_matrix_sync(gC +
                                    (cols * mrow * 128U + mcol * 128U +
                                     cols * warp_row * 32U + warp_col * 64U +
                                     cols * i * 16U + j * 16U), out, cols,
                                    wmma::mem_row_major);
        }
}

void
gemm_bcast_bias_epilogue_launch(float alpha, float beta, uint32_t rows,
                                uint32_t cols, uint32_t shared, half *gA,
                                half *gB, half *gBias, half *gC, cudaStream_t s)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_SHMEM_FITS(16384U);
    MUST(cudaFuncSetAttribute(__hoisted_addmm_jit_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              16384U));
    KPR_KCALL(__hoisted_addmm_jit_0, nblk, 256U, 16384U, s, alpha, beta, cols,
              shared, gA, gB, gBias, gC);
}
