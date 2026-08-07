// Templated Ampere tensor-core GEMM: D = alpha*(A@B) + beta*C.
//
// Written to stay inside the subset of CUDA that Kuiper can express, so it is a
// realistic upper bound on what a verified kernel could reach rather than an
// unreachable one: no inline PTX, no warp shuffles, no atomics and no reads or
// writes of individual WMMA fragment registers under an inferred layout. What
// is used is the plain `nvcuda::wmma` API, `cp.async` through the
// `__pipeline_*` intrinsics, and 128-bit vectorized loads/stores.
//
// Layouts are row-major throughout: A (M, K), B (K, N), C/D (M, N). Element
// type is half or __nv_bfloat16; accumulation is always fp32.
//
// Tiling. A block owns a BM x BN output tile and walks K in BK steps; the block
// tile is cut into WM x WN warp tiles, each of which is a (WM/16) x (WN/16)
// grid of 16x16x16 fragments held in registers for the whole k-loop. The A/B
// shared tiles are STAGES-deep and filled with cp.async so global traffic
// overlaps the math. Rows are padded by SKEW elements: the resulting row stride
// is odd in units of the 32 shared banks, which is what keeps both the cp.async
// fills and the fragment loads conflict-free (a plain BK/BN stride would put
// every row of a fragment in the same bank).
//
// Epilogue. Accumulators never leave the fragment abstraction: a 16-row band of
// the warp tile is `store_matrix_sync`d into shared scratch and read back with
// vector loads, so alpha/beta and the down-conversion are ordinary arithmetic
// on floats and the global writes are 128-bit and coalesced. This is also what
// makes the C operand free-form -- EPI_VEC broadcasts a length-N bias, which a
// fragment-level epilogue cannot do without replicating it into shared memory.
//
// Split-k. Some decode shapes (M=256, K=4864) do not fill the GPU: 56 blocks on
// 84 SMs. Splitting k across `splits` blocks in gridDim.z fixes that, but
// without atomics the partials have to be reduced by a second kernel, so each
// slice writes fp32 into a workspace and `gemm_tc_reduce_kernel` sums it and
// applies the epilogue.
//
// A caveat that is not ours to fix: CUDA lowers the bf16 `load_matrix_sync`
// through a different builtin than the half one, and that builtin does not use
// `ldmatrix` -- it emits four generic loads plus a register transpose per
// fragment instead of one LDSM. bf16 therefore runs ~15% behind fp16 here for
// reasons that have nothing to do with this kernel.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace gemm_tc {

using namespace nvcuda;

// Epilogue selector. NONE skips the read of C entirely (a plain matmul pays
// nothing for the GEMM's epilogue), MAT reads a dense (M, N) C, VEC broadcasts
// a length-N bias vector over the rows as nn.Linear's addmm needs.
enum Epilogue { EPI_NONE = 0, EPI_MAT = 1, EPI_VEC = 2 };

__device__ __forceinline__ half from_float(float v, half) {
    return __float2half_rn(v);
}
__device__ __forceinline__ __nv_bfloat16 from_float(float v, __nv_bfloat16) {
    return __float2bfloat16_rn(v);
}
__device__ __forceinline__ float to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

struct Params {
    const void* A;
    const void* B;
    const void* C;     // null when epi == EPI_NONE
    void* D;
    float* workspace;  // null unless splits > 1
    int M, N, K;
    float alpha, beta;
    int splits;
    int group;         // block-swizzle group height, in block tiles
    int epi;           // Epilogue
};

// Shared-tile geometry. Kept in one place so the launcher can size the dynamic
// allocation exactly as the kernel indexes it.
template <typename T, int BM, int BN, int BK, int WM, int WN, int STAGES,
          int SKEW>
struct Tile {
    static constexpr int WARPS = (BM / WM) * (BN / WN);
    static constexpr int THREADS = WARPS * 32;
    static constexpr int LDA = BK + SKEW;
    static constexpr int LDB = BN + SKEW;
    static constexpr int A_ELEMS = BM * LDA;
    static constexpr int B_ELEMS = BK * LDB;
    static constexpr int PIPE_BYTES =
        STAGES * (A_ELEMS + B_ELEMS) * (int)sizeof(T);
    // Epilogue scratch: one 16-row band of every warp's tile, in fp32, padded
    // by ESKEW for the same bank-conflict reason as the A/B tiles.
    static constexpr int ESKEW = 4;
    static constexpr int LDE = WN + ESKEW;
    static constexpr int EPI_BYTES = WARPS * 16 * LDE * (int)sizeof(float);
    static constexpr int SMEM_BYTES =
        PIPE_BYTES > EPI_BYTES ? PIPE_BYTES : EPI_BYTES;
    // 128-bit granule, in elements.
    static constexpr int VEC = 16 / (int)sizeof(T);
    static constexpr int A_VECS = BM * BK / VEC;
    static constexpr int B_VECS = BK * BN / VEC;
};

template <typename T, int BM, int BN, int BK, int WM, int WN, int STAGES,
          int SKEW>
__global__ __launch_bounds__(Tile<T, BM, BN, BK, WM, WN, STAGES, SKEW>::THREADS)
void gemm_tc_kernel(__grid_constant__ const Params p) {
    using Tl = Tile<T, BM, BN, BK, WM, WN, STAGES, SKEW>;
    constexpr int MFRAG = WM / 16, NFRAG = WN / 16;
    constexpr int KSTEP = BK / 16;

    extern __shared__ __align__(16) char smem_raw[];
    T* sA = reinterpret_cast<T*>(smem_raw);
    T* sB = sA + STAGES * Tl::A_ELEMS;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int warp_m = warp / (BN / WN);
    const int warp_n = warp % (BN / WN);

    // Swizzle the linear block index into `group` consecutive rows of block
    // tiles at a time: neighbouring blocks then share A rows and B columns, and
    // the working set of a wave stays in L2. Straight row-major ordering
    // streams a whole row of B per block row instead.
    const int nblk_n = p.N / BN;
    const int nblk_m = p.M / BM;
    int block_row, block_col;
    {
        const int per_group = p.group * nblk_n;
        const int gid = blockIdx.x / per_group;
        const int rem = blockIdx.x - gid * per_group;
        const int rows = min(p.group, nblk_m - gid * p.group);
        block_row = gid * p.group + rem % rows;
        block_col = rem / rows;
    }

    // k-range of this split, balanced so that no split is empty (an empty one
    // would leave its workspace slice uninitialised for the reduction to read).
    // Slice boundaries are BK-aligned so the inner loop never has a partial tile.
    const int ktiles_all = p.K / BK;
    const int kt0 = (int)((long long)ktiles_all * blockIdx.z / p.splits);
    const int kt1 = (int)((long long)ktiles_all * (blockIdx.z + 1) / p.splits);

    const T* __restrict__ gA = reinterpret_cast<const T*>(p.A);
    const T* __restrict__ gB = reinterpret_cast<const T*>(p.B);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MFRAG][NFRAG];
    #pragma unroll
    for (int i = 0; i < MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Staging addresses. A thread copies the same cells of every k-tile, so its
    // source pointers and shared destinations are loop-invariant apart from a
    // fixed k-stride: hoisting them out leaves the steady-state copy as a
    // straight run of cp.async plus one pointer bump, instead of the branchy
    // 64-bit index arithmetic a `for (i = tid; i < VECS; i += THREADS)` loop
    // re-derives on every k-tile. That arithmetic, not bandwidth, is what the
    // k-loop is otherwise dominated by.
    static_assert(Tl::A_VECS % Tl::THREADS == 0 &&
                  Tl::B_VECS % Tl::THREADS == 0,
                  "the staging copy must divide evenly over the block");
    constexpr int A_ITERS = Tl::A_VECS / Tl::THREADS;
    constexpr int B_ITERS = Tl::B_VECS / Tl::THREADS;
    // Rows of the tile a full sweep of the block covers.
    constexpr int A_STEP = Tl::THREADS * Tl::VEC / BK;
    constexpr int B_STEP = Tl::THREADS * Tl::VEC / BN;
    static_assert(Tl::THREADS * Tl::VEC % BK == 0 &&
                  Tl::THREADS * Tl::VEC % BN == 0,
                  "a block sweep must cover whole tile rows");

    const int a_r = (tid * Tl::VEC) / BK, a_c = (tid * Tl::VEC) % BK;
    const int b_r = (tid * Tl::VEC) / BN, b_c = (tid * Tl::VEC) % BN;
    const T* aSrc = gA + (size_t)(block_row * BM + a_r) * p.K +
                    (size_t)kt0 * BK + a_c;
    const T* bSrc = gB + (size_t)(kt0 * BK + b_r) * p.N +
                    (size_t)block_col * BN + b_c;
    const int aDst = a_r * Tl::LDA + a_c;
    const int bDst = b_r * Tl::LDB + b_c;

    auto stage_in = [&](int buf) {
        T* dstA = sA + buf * Tl::A_ELEMS + aDst;
        #pragma unroll
        for (int n = 0; n < A_ITERS; n++)
            __pipeline_memcpy_async(dstA + n * A_STEP * Tl::LDA,
                                    aSrc + (size_t)n * A_STEP * p.K, 16);
        T* dstB = sB + buf * Tl::B_ELEMS + bDst;
        #pragma unroll
        for (int n = 0; n < B_ITERS; n++)
            __pipeline_memcpy_async(dstB + n * B_STEP * Tl::LDB,
                                    bSrc + (size_t)n * B_STEP * p.N, 16);
        __pipeline_commit();
    };

    const int ktiles = kt1 - kt0;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktiles) {
            stage_in(s);
            aSrc += BK;
            bSrc += (size_t)BK * p.N;
        } else {
            __pipeline_commit();
        }
    }

    for (int kt = 0; kt < ktiles; kt++) {
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();
        const int buf = kt % STAGES;
        const T* bufA = sA + buf * Tl::A_ELEMS;
        const T* bufB = sB + buf * Tl::B_ELEMS;

        // Refill the buffer freed by the previous k-tile, STAGES-1 tiles ahead.
        // Issued before the math rather than after it so the copy is in flight
        // while this tile's mmas run; the __syncthreads() above is what makes
        // the buffer safe to overwrite (its last reader was iteration kt-1).
        if (kt + STAGES - 1 < ktiles) {
            stage_in((kt + STAGES - 1) % STAGES);
            aSrc += BK;
            bSrc += (size_t)BK * p.N;
        } else {
            __pipeline_commit();
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af[MFRAG];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> bf[NFRAG];
        #pragma unroll
        for (int ks = 0; ks < KSTEP; ks++) {
            #pragma unroll
            for (int i = 0; i < MFRAG; i++)
                wmma::load_matrix_sync(
                    af[i], bufA + (warp_m * WM + i * 16) * Tl::LDA + ks * 16,
                    Tl::LDA);
            #pragma unroll
            for (int j = 0; j < NFRAG; j++)
                wmma::load_matrix_sync(
                    bf[j], bufB + (ks * 16) * Tl::LDB + warp_n * WN + j * 16,
                    Tl::LDB);
            #pragma unroll
            for (int i = 0; i < MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < NFRAG; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }

    // ---------------------------------------------------------------- epilogue
    __syncthreads();
    float* scratch = reinterpret_cast<float*>(smem_raw) + warp * 16 * Tl::LDE;

    const int lane = tid % 32;
    constexpr int EVEC = Tl::VEC;                 // output elements per store
    constexpr int BAND_VECS = 16 * WN / EVEC;     // vector stores per band
    const int row_base = block_row * BM + warp_m * WM;
    const int col_base = block_col * BN + warp_n * WN;

    #pragma unroll
    for (int i = 0; i < MFRAG; i++) {
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            wmma::store_matrix_sync(scratch + j * 16, acc[i][j], Tl::LDE,
                                    wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int v = lane; v < BAND_VECS; v += 32) {
            const int r = (v * EVEC) / WN, c = (v * EVEC) % WN;
            const int gr = row_base + i * 16 + r, gc = col_base + c;
            float val[EVEC];
            #pragma unroll
            for (int e = 0; e < EVEC; e++)
                val[e] = scratch[r * Tl::LDE + c + e];

            if (p.splits > 1) {
                float* dst = p.workspace +
                             (size_t)blockIdx.z * p.M * p.N +
                             (size_t)gr * p.N + gc;
                #pragma unroll
                for (int e = 0; e < EVEC; e += 4)
                    *reinterpret_cast<float4*>(dst + e) =
                        make_float4(val[e], val[e + 1], val[e + 2], val[e + 3]);
                continue;
            }

            T out[EVEC];
            if (p.epi == EPI_NONE) {
                #pragma unroll
                for (int e = 0; e < EVEC; e++)
                    out[e] = from_float(p.alpha * val[e], T());
            } else {
                const T* cp = reinterpret_cast<const T*>(p.C);
                const T* src = (p.epi == EPI_VEC) ? cp + gc
                                                  : cp + (size_t)gr * p.N + gc;
                T cv[EVEC];
                *reinterpret_cast<int4*>(cv) =
                    *reinterpret_cast<const int4*>(src);
                #pragma unroll
                for (int e = 0; e < EVEC; e++)
                    out[e] = from_float(
                        p.alpha * val[e] + p.beta * to_float(cv[e]), T());
            }
            T* dst = reinterpret_cast<T*>(p.D) + (size_t)gr * p.N + gc;
            *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(out);
        }
    }
}

// Sums the `splits` fp32 partials and applies the epilogue. One thread per
// 128-bit output granule; the partials are read as float4.
template <typename T>
__global__ void gemm_tc_reduce_kernel(const float* __restrict__ ws,
                                      const void* __restrict__ Cp,
                                      void* __restrict__ Dp, int M, int N,
                                      float alpha, float beta, int splits,
                                      int epi) {
    constexpr int VEC = 16 / (int)sizeof(T);
    const size_t total = (size_t)M * N / VEC;
    const size_t stride = (size_t)M * N;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t off = i * VEC;
        float val[VEC];
        #pragma unroll
        for (int e = 0; e < VEC; e++) val[e] = 0.0f;
        for (int s = 0; s < splits; s++) {
            const float* src = ws + s * stride + off;
            #pragma unroll
            for (int e = 0; e < VEC; e += 4) {
                const float4 v = *reinterpret_cast<const float4*>(src + e);
                val[e] += v.x; val[e + 1] += v.y;
                val[e + 2] += v.z; val[e + 3] += v.w;
            }
        }
        T out[VEC];
        if (epi == EPI_NONE) {
            #pragma unroll
            for (int e = 0; e < VEC; e++)
                out[e] = from_float(alpha * val[e], T());
        } else {
            const T* cp = reinterpret_cast<const T*>(Cp);
            const T* src = (epi == EPI_VEC) ? cp + (off % N) : cp + off;
            T cv[VEC];
            *reinterpret_cast<int4*>(cv) = *reinterpret_cast<const int4*>(src);
            #pragma unroll
            for (int e = 0; e < VEC; e++)
                out[e] = from_float(alpha * val[e] + beta * to_float(cv[e]),
                                    T());
        }
        *reinterpret_cast<int4*>(reinterpret_cast<T*>(Dp) + off) =
            *reinterpret_cast<const int4*>(out);
    }
}

}  // namespace gemm_tc
