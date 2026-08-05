// Extracted instance of Kuiper TensorCore2D.To (`addmm`, f16 in / f32 accumulate
// / f16 out, 128x128x32 blocks, 16x16x16 fragments, 2x4 fragments per warp) with
// a *broadcast* C operand.
//
// Verbatim from .kuipy_cache/cu/Kuiops_Addmm_Tc2DTo_F16_F32_F16
// _P_bm128_bn128_bk32_tm16_tn16_tk16_wm2_wn4.cu apart from a single line: the
// epilogue's read of C drops its row term, so `gC` is indexed by column alone
// and is a length-N vector rather than an (M, N) matrix -- exactly the
// stride-0 row layout that `nn.Linear`'s bias needs and that a Kuiper tlayout,
// being an injection, cannot currently express.
//
// `To` is the out-of-place variant: it stages the accumulator through shared
// memory as f32 and then does a plain vectorised read of C and write of D,
// rather than the read-modify-write fragment store of TensorCore2D. That split
// is what makes the change a one-liner -- C's index function is already
// independent of D's, so relaxing it needs no new staging (cf.
// gemm_bcast_bias_epilogue.cu, which has to replicate the bias into shared to
// get it through a fragment load).
//
//   A : (M, K) f16    B : (K, N) f16    C : (N,) f16    D : (M, N) f16
#include <kuiper.h>
#include <kuiops_compat.h>
#include "gemm_bcast_bias_epilogue2.h"

__global__
/**
  hoisted when extracting addmm_jit
*/
static void
__hoisted_addmm_jit_0(float alpha,
                      float beta,
                      uint32_t cols,
                      uint32_t shared,
                      half *gA, half *gB, half *gC, half *gD, uint32_t nthr)
{
    KRML_MAYBE_UNUSED_VAR(nthr);
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    half *sA = (half *) KPR_SHMEM_AT(0U);
    half *sB = (half *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
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
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        half *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            half local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2half_rn(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
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
                       tileB + (cols * __anf0 * 32U + mcol * 128U + cols * row +
                                col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sB[row * 128U + col + k] = local[k];
        }
        __syncthreads();
        uint32_t dotIdx = 0U;
        for (; dotIdx < 2U; dotIdx++) {
            uint32_t __anf01 = dotIdx;
            half *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            half *tile_for_tc_b_tiles = sB;
            uint32_t i1 = 0U;
            for (; i1 < 4U; i1++)
                wmma::load_matrix_sync(bFrags[i1],
                                       tile_for_tc_b_tiles +
                                       (128U * __anf02 * 16U +
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
        KRML_HOST_IGNORE(__anf0 + 1U);
    }
    auto & accFrags0 = accFrags;
    uint32_t idx = 0U;
    for (; idx < 8U; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / 128U);
        uint32_t mcol2 = blockIdx.x % (cols / 128U);
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        wmma::store_matrix_sync(sTile + 16U * (threadIdx.x / 32U) * 16U,
                                accFrags0[idx], 16U, wmma::mem_row_major);
        __syncwarp();
        uint32_t __anf02 = idx;
        uint32_t vg = threadIdx.x % 32U;
        for (; vg < 32U; vg += 32U) {
            uint32_t flat = vg * 8U;
            uint32_t row = flat / 16U;
            uint32_t col = flat % 16U;
            uint32_t globalRow =
                mrow2 * 128U + threadIdx.x / 32U / 2U * 32U +
                __anf02 / 4U * 16U + row;
            uint32_t globalCol =
                mcol2 * 128U + threadIdx.x / 32U % 2U * 64U +
                __anf02 % 4U * 16U + col;
            half cbuf[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                cbuf[_i] = __float2half_rn(0.0f);
            vec_memcpy(cbuf, gC + globalCol);
            half obuf[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                obuf[_i] = __float2half_rn(0.0f);
            uint32_t k = 0U;
            for (; k < 8U; k++) {
                uint32_t vk = k;
                float av =
                    sTile[(16U * (threadIdx.x / 32U) + row) * 16U + col + vk];
                obuf[vk] =
                    __float2half_rn(beta * __half2float(cbuf[vk]) + alpha * av);
            }
            vec_memcpy(gD + (cols * globalRow + globalCol), obuf);
        }
    }
}

void
gemm_bcast_bias_epilogue2_launch
(float alpha, float beta, uint32_t rows, uint32_t cols, uint32_t shared,
half *gA, half *gB, half *gC, half *gD, cudaStream_t s)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute(__hoisted_addmm_jit_0,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              24576U));
    KPR_KCALL(__hoisted_addmm_jit_0,
              nblk,
              256U, 24576U, s, alpha, beta, cols, shared, gA, gB, gC, gD, 256U);
}
