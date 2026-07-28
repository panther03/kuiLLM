// Hand-written Kuiper TensorCore2D GEMM, benchmarked against the JIT-dispatched
// kuiops GEMM in bench_tc_kernels.py. The body is derived from the
// `128x128x32_16x16x16_8x4` instantiation extracted from the Kuiper
// distribution ($KUIPER_HOME/dist/Klas_GEMM_TensorCore2D.{cu,h}), retyped to
// bf16 in/out to match the rest of the benchmark; the
// shared-memory staging and the tensor-core inner loop are unchanged in spirit
// (single-buffered vec_memcpy loads, 32-deep k-tiles, no software pipelining).
//
// Two changes are layered on as a sketch of a planned Kuiper update:
//   * the accumulator type is decoupled from the bf16 inputs/outputs (`acc_t`)
//     and widened to fp32, so the reduction no longer rounds every partial sum
//     to bf16; and
//   * the trailing store is replaced by a real GEMM epilogue,
//     D = alpha*(A@B) + beta*C, evaluated in acc_t and converted down to bf16.
//
// Widening the accumulator forces one tiling parameter to move: with the
// original 8x4 fragments-per-warp / 64-thread layout, 32 fp32 accumulator
// fragments per warp overflow the register file and spill. The 128x128 tile is
// therefore spread over 8 warps (256 threads) so each warp owns an 8-fragment
// (2x4) output tile -- ~64 accumulator registers, no spill. Block/k tiling and
// the shared layout are otherwise identical.
//
//   A : (M, K) bf16    B : (K, N) bf16    C : (M, N) bf16   (row-major)
// The caller passes B already laid out as (K, N) (the transposed F.linear
// weight); when beta != 0 the current contents of C are the additive term.
#include <kuiper.h>
#include "tc_kernels.h"

// Accumulator type, decoupled from the bf16 in/out matrices.
using acc_t = float;

namespace {
constexpr uint32_t BM = 128, BN = 128, BK = 32;   // block tile
constexpr uint32_t FM = BM / 16, FN = BN / 16;     // 8x8 fragments per block
constexpr uint32_t WARPS_M = 4, WARPS_N = 2;       // 8 warps over the block tile
constexpr uint32_t MW = FM / WARPS_M;              // = 2  fragment rows / warp
constexpr uint32_t NW = FN / WARPS_N;              // = 4  fragment cols / warp
constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32;   // = 256
constexpr uint32_t LOAD_STEP = THREADS * 8;        // elements staged per load pass
}

__global__
/**
  hoisted when extracting g_gemm_bf16_bf16_128x128x32_16x16x16_8x4
*/
static void
__hoisted_g_gemm_bf16_bf16_128x128x32_16x16x16_8x4_0(uint32_t shared,
                                                   uint32_t cols,
                                                   __nv_bfloat16 *gA, __nv_bfloat16 *gB,
                                                   __nv_bfloat16 *gC,
                                                   float alpha, float beta)
{
    __nv_bfloat16 *sA = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
    __nv_bfloat16 *sB = (__nv_bfloat16 *) KPR_SHMEM_AT(8192U);
    // Dedicated fp32 epilogue scratch: a separately-allocated, float-typed shared
    // buffer (not an aliasing reinterpret of sA) so it is expressible in Kuiper,
    // where each shared buffer is typed and allocated at launch. Holds one 16x16
    // accumulator fragment per warp: 8 warps * 256 floats = 2048 floats = 8 KiB.
    float *sAcc = (float *) KPR_SHMEM_AT(16384U);
    uint32_t num_k_tiles = shared / BK;
    uint32_t num_n_tiles = cols / BN;
    uint32_t mrow = blockIdx.x / num_n_tiles;
    uint32_t mcol = blockIdx.x % num_n_tiles;

    uint32_t warp = threadIdx.x / 32U;
    uint32_t warp_row = warp / WARPS_N;   // 0..WARPS_M-1
    uint32_t warp_col = warp % WARPS_N;   // 0..WARPS_N-1

    auto & aFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16, wmma::row_major),
                     MW);
    auto & bFrags =
        KPR_INIT_ARR(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16, wmma::row_major),
                     NW);
    auto & accFrags =
        KPR_INIT_ARR(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, acc_t),
                     MW * NW);
    for (uint32_t fi = 0U; fi < MW * NW; fi++)
        wmma::fill_fragment(accFrags[fi], (acc_t) 0);

    for (uint32_t bkIdx = 0U; bkIdx < num_k_tiles; bkIdx++) {
        __syncthreads();
        __nv_bfloat16 *tileA = gA;
        for (uint32_t i2 = 0U; i2 < BM * BK; i2 += LOAD_STEP) {
            __nv_bfloat16 local[8U];
            uint32_t row = (i2 + threadIdx.x * 8U) / BK;
            uint32_t col = (i2 + threadIdx.x * 8U) % BK;
            vec_memcpy(local,
                       tileA + (shared * mrow * BM + bkIdx * BK +
                                shared * row + col));
            for (uint32_t k = 0U; k < 8U; k++)
                sA[row * BK + col + k] = local[k];
        }
        __nv_bfloat16 *tileB = gB;
        for (uint32_t i = 0U; i < BK * BN; i += LOAD_STEP) {
            __nv_bfloat16 local[8U];
            uint32_t row = (i + threadIdx.x * 8U) / BN;
            uint32_t col = (i + threadIdx.x * 8U) % BN;
            vec_memcpy(local,
                       tileB + (cols * bkIdx * BK + mcol * BN +
                                cols * row + col));
            for (uint32_t k = 0U; k < 8U; k++)
                sB[row * BN + col + k] = local[k];
        }
        __syncthreads();

        for (uint32_t dotIdx = 0U; dotIdx < BK / 16U; dotIdx++) {
            for (uint32_t mi = 0U; mi < MW; mi++) {
                uint32_t arow = warp_row * (MW * 16U) + mi * 16U;
                wmma::load_matrix_sync(aFrags[mi],
                                       sA + (arow * BK + dotIdx * 16U), BK);
            }
            for (uint32_t nj = 0U; nj < NW; nj++) {
                uint32_t bcol = warp_col * (NW * 16U) + nj * 16U;
                wmma::load_matrix_sync(bFrags[nj],
                                       sB + (dotIdx * 16U * BN + bcol), BN);
            }
            for (uint32_t mi = 0U; mi < MW; mi++)
                for (uint32_t nj = 0U; nj < NW; nj++) {
                    auto & acc_frag = accFrags[mi * NW + nj];
                    wmma::mma_sync(acc_frag, aFrags[mi], bFrags[nj], acc_frag);
                }
        }
    }

    // GEMM epilogue (bolt-on): D = alpha*acc + beta*C, evaluated in acc_t and
    // converted to bf16. Each acc_t (fp32) fragment is spilled to the dedicated
    // fp32 scratch sAcc via store_matrix_sync -- a *defined* row-major layout -- so
    // we never assume the fp32 and bf16 accumulator fragments share an (unspecified)
    // element layout. Each lane then reads back 8 contiguous columns of one row,
    // applies alpha/beta, and writes them to global as a single 128-bit vector.
    //
    // Synchronization uses __syncthreads only: __syncwarp is not modelled by Kuiper,
    // and the block barrier is free here -- the epilogue runs once, after the K-loop,
    // and is bound by the global output stores, so the barriers hide behind them.
    // sAcc is a dedicated buffer the K-loop never touches, so no leading barrier is
    // needed before the first store (the K-loop only reads/writes sA and sB).
    uint32_t lane = threadIdx.x % 32U;
    uint32_t lrow = lane / 2U;            // this lane owns row lrow, cols c0..c0+7
    uint32_t c0 = (lane % 2U) * 8U;
    for (uint32_t mi = 0U; mi < MW; mi++) {
        for (uint32_t nj = 0U; nj < NW; nj++) {
            auto & acc_frag = accFrags[mi * NW + nj];
            // store_matrix_sync reconverges the warp at entry, so the previous
            // iteration's reads of sAcc have retired before it is overwritten (no
            // pre-store barrier needed); the __syncthreads makes the collective
            // store visible before the differently-partitioned read-back below.
            wmma::store_matrix_sync(sAcc + warp * 256U, acc_frag, 16U,
                                    wmma::mem_row_major);
            __syncthreads();
            uint32_t row = mrow * BM + warp_row * (MW * 16U) + mi * 16U + lrow;
            uint32_t col = mcol * BN + warp_col * (NW * 16U) + nj * 16U + c0;
            __nv_bfloat16 *g = gC + row * cols + col;
            float *s = sAcc + warp * 256U + lrow * 16U + c0;
            __align__(16) __nv_bfloat16 out[8U];
            if (beta != 0.0f) {
                __align__(16) __nv_bfloat16 cin[8U];
                vec_memcpy(cin, g);
                for (uint32_t k = 0U; k < 8U; k++)
                    out[k] = __float2bfloat16(alpha * s[k] + beta * __bfloat162float(cin[k]));
            } else {
                for (uint32_t k = 0U; k < 8U; k++)
                    out[k] = __float2bfloat16(alpha * s[k]);
            }
            vec_memcpy(g, out);
        }
    }
}

void
Klas_GEMM_TensorCore2D_g_gemm_bf16_bf16_128x128x32_16x16x16_8x4(uint32_t rows,
                                                              uint32_t shared,
                                                              uint32_t cols,
                                                              __nv_bfloat16 *gA,
                                                              __nv_bfloat16 *gB,
                                                              __nv_bfloat16 *gC,
                                                              float alpha,
                                                              float beta)
{
    KPR_GUARD(rows % 128U == 0U);
    KPR_GUARD(shared % 32U == 0U);
    KPR_GUARD(cols % 128U == 0U);
    uint32_t nblk = rows / 128U * (cols / 128U);
    KPR_ASSERT(nblk <= 2097152U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(24576U);
    MUST(cudaFuncSetAttribute
         (__hoisted_g_gemm_bf16_bf16_128x128x32_16x16x16_8x4_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 24576U));
    KPR_KCALL(__hoisted_g_gemm_bf16_bf16_128x128x32_16x16x16_8x4_0, nblk, THREADS,
              24576U, s, shared, cols, gA, gB, gC, alpha, beta);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

// C = alpha*(A @ B) + beta*C, fp32-accumulate tensor-core GEMM. When beta != 0
// the current contents of C are read back as the additive term (in-place GEMM).
void tc2d_manual_gemm_launch(const __nv_bfloat16* A, const __nv_bfloat16* B,
                             __nv_bfloat16* C, int M, int N, int K,
                             float alpha, float beta) {
    if (M == 0 || N == 0 || K == 0) return;
    Klas_GEMM_TensorCore2D_g_gemm_bf16_bf16_128x128x32_16x16x16_8x4(
        (uint32_t)M, (uint32_t)K, (uint32_t)N,
        const_cast<__nv_bfloat16*>(A), const_cast<__nv_bfloat16*>(B), C,
        alpha, beta);
}
