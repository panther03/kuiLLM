#include <kuiper.h>
#include "tc_kernels.h"

typedef struct epilogue_dims_s {
    uint32_t bm;
    uint32_t bn;
    uint32_t bk;
    uint32_t tm;
    uint32_t tn;
    uint32_t tk;
    uint32_t wm;
    uint32_t wn;
    uint32_t nthr;
} epilogue_dims;

__global__
/**
  hoisted when extracting g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4
*/
static void
__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0(uint32_t shared,
                                                         uint32_t cols,
                                                         __nv_bfloat16 *gA,
                                                         __nv_bfloat16 *gB,
                                                         __nv_bfloat16 *gC,
                                                         __nv_bfloat16 *gD,
                                                         float alpha,
                                                         float beta,
                                                         uint32_t nthr)
{
    uint32_t num_n_tiles = cols / 128U;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    uint32_t num_k_tiles = shared / 32U;
    auto &
        aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 2U);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major), 4U);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float), 8U);
    uint32_t fi = 0U;
    for (; fi < 8U; fi++)
        wmma::fill_fragment(accFrags[fi], 0.0f);
    uint32_t bkIdx = 0U;
    for (; bkIdx < num_k_tiles; bkIdx++) {
        uint32_t __anf0 = bkIdx;
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        uint32_t i2 = 0U;
        for (; i2 < 4096U; i2 += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
            uint32_t row = (i2 + threadIdx.x * 8U) / 32U;
            uint32_t col = (i2 + threadIdx.x * 8U) % 32U;
            vec_memcpy(local,
                       tileA + (shared * mrow * 128U + __anf0 * 32U +
                                shared * row + col));
            uint32_t k = 0U;
            for (; k < 8U; k++)
                sA[row * 32U + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        uint32_t i = 0U;
        for (; i < 4096U; i += 2048U) {
            __nv_bfloat16 local[8U];
            for (uint32_t _i = 0U; _i < 8U; ++_i)
                local[_i] = __float2bfloat16(0.0f);
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
            __nv_bfloat16 *tile_for_tc_a_tiles = sA;
            uint32_t i0 = 0U;
            for (; i0 < 2U; i0++)
                wmma::load_matrix_sync(aFrags[i0],
                                       tile_for_tc_a_tiles +
                                       (32U * (threadIdx.x / 32U / 2U) * 32U +
                                        __anf01 * 16U + 32U * i0 * 16U), 32U);
            uint32_t __anf02 = dotIdx;
            __nv_bfloat16 *tile_for_tc_b_tiles = sB;
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
    epilogue_dims dims = {
        .bm = 128U,.bn = 128U,.bk = 32U,.tm = 16U,.tn = 16U,.tk = 16U,.wm =
            2U,.wn = 4U,
        .nthr = nthr
    };
    uint32_t bm = dims.bm;
    uint32_t bn = dims.bn;
    uint32_t tm = dims.tm;
    uint32_t tn = dims.tn;
    uint32_t wm = dims.wm;
    uint32_t wn = dims.wn;
    uint32_t idx = 0U;
    for (; idx < wm * wn; idx++) {
        uint32_t mrow2 = blockIdx.x / (cols / bn);
        uint32_t mcol2 = blockIdx.x % (cols / bn);
        uint32_t warpRow2 = threadIdx.x / 32U / (bn / (wn * tn));
        uint32_t warpCol2 = threadIdx.x / 32U % (bn / (wn * tn));
        float *sTile = (float *)KPR_SHMEM_AT(16384U);
        KPR_STORE_MATRIX_SYNC(sTile + tn * (threadIdx.x / 32U) * tm,
                              accFrags0[idx], tn, wmma::mem_row_major);
        uint32_t __anf02 = idx;
        uint32_t area = tm * tn;
        uint32_t flat = threadIdx.x % 32U;
        for (; flat < area; flat += 32U) {
            uint32_t __anf03 = flat;
            uint32_t row = __anf03 / tn;
            uint32_t col = __anf03 % tn;
            uint32_t globalRow =
                mrow2 * bm + warpRow2 * wm * tm + __anf02 / wn * tm + row;
            uint32_t globalCol =
                mcol2 * bn + warpCol2 * wn * tn + __anf02 % wn * tn + col;
            float av = sTile[(tm * (threadIdx.x / 32U) + row) * tn + col];
            gD[globalRow * cols + globalCol] =
                __float2bfloat16(beta *
                                 __bfloat162float(gC
                                                  [globalRow * cols +
                                                   globalCol]) + alpha * av);
        }
    }
}

void
Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4(uint32_t
                                                                       rows,
                                                                       uint32_t
                                                                       shared,
                                                                       uint32_t
                                                                       cols,
                                                                       __nv_bfloat16
                                                                       *gA,
                                                                       __nv_bfloat16
                                                                       *gB,
                                                                       __nv_bfloat16
                                                                       *gC,
                                                                       __nv_bfloat16
                                                                       *gD,
                                                                       float
                                                                       alpha,
                                                                       float
                                                                       beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    KPR_ASSERT(0U == 0U);
    KPR_ASSERT(0U == 0U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4_0, nblk,
              256U, 24576U, s, shared, cols, gA, gB, gC, gD, alpha, beta, 256U);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

void tc2d_gemm_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                      const __nv_bfloat16* C, __nv_bfloat16* D,
                      int M, int N, int K, float alpha, float beta) {
    if (M == 0 || N == 0 || K == 0) return;
    KPR_GUARD(beta == 0.0f || C != nullptr);
    if (beta == 0.0f) {
        MUST(cudaMemset(D, 0, (size_t)M * N * sizeof(*D)));
        C = D;
    }
    Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        const_cast<__nv_bfloat16*>(A), const_cast<__nv_bfloat16*>(B),
        const_cast<__nv_bfloat16*>(C), D, alpha, beta);
}

void tc2d_matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                        __nv_bfloat16* D, int M, int N, int K) {
    tc2d_gemm_launch(A, B, nullptr, D, M, N, K, 1.0f, 0.0f);
}
