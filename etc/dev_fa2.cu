// Standalone dev harness for the FlashAttention-2 style bf16 tensor-core kernel.
// Correctness is checked against a naive fp32 GPU reference; timing is relative
// (final cuDNN comparison lives in bench_tc_kernels.py). Tile params are macros
// so an autotuning script can sweep them with nvcc -D.
//
//   nvcc -arch=sm_86 -O3 -DWARPS_M=4 -DKSTEP=2 dev_fa2.cu -o /tmp/fa2 && /tmp/fa2 <mode>
//
// Design: FA2 split-Q. One block owns BLOCK_M = WARPS_M*16 query rows of one
// (batch, q head). Each warp owns 16 rows and runs a FULLY INDEPENDENT online
// softmax over all keys -- it loads its own K/V subtiles into a per-warp shared
// buffer (only __syncwarp, never a block-wide __syncthreads in the inner loop),
// keeps the O accumulator resident in WMMA registers, and normalizes once at the
// end (delayed softmax division). The K/V re-reads across warps hit L2.
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

using namespace nvcuda;

#ifndef WARPS_M
#define WARPS_M 4          // query-row warps: BLOCK_M = WARPS_M*16 rows/block
#endif
#ifndef KSTEP
#define KSTEP 2            // key subtiles (of 16) processed per inner step
#endif
#define HD 64              // head dim (fixed for Qwen2.5)
#define WARP 32
#define BLOCK_M (WARPS_M * 16)
#define OD (HD / 16)       // output col fragments = 4
#define KW (KSTEP * 16)    // key columns per inner step

// element index i (0..7) of a 16x16 fp32 wmma accumulator held by `lane`
// maps to (row,col) of the stored tile (verified with probe_wmma_layout.cu):
//   row = lane/4 + (i&2 ? 8 : 0),  col = 2*(lane&3) + (i&1) + (i&4 ? 8 : 0)
__device__ __forceinline__ int frag_row(int lane, int i) { return (lane >> 2) + ((i & 2) ? 8 : 0); }
__device__ __forceinline__ int frag_col(int lane, int i) { return ((lane & 3) << 1) + (i & 1) + ((i & 4) ? 8 : 0); }

// reduce a per-row value across the 4 lanes of a group (they hold different
// columns of the same row). Groups are lanes sharing lane>>2.
__device__ __forceinline__ float grpmax(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}
__device__ __forceinline__ float grpsum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

__global__ void fa2_prefill(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ out,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal,
    int64_t ms_b, int64_t ms_h, int64_t ms_q, int64_t ms_k)
{
    const int group = Hq / Hkv;
    const int qt = blockIdx.x, qh = blockIdx.y, bi = blockIdx.z;
    const int kvh = qh / group;
    const int w = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int wrow0 = qt * BLOCK_M + w * 16;    // first query row this warp owns

    const __nv_bfloat16* kbase = k + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    const __nv_bfloat16* vbase = v + (((int64_t)bi * Hkv + kvh) * Sk) * D;

    // block-shared K/V chunk (loaded once, reused by every warp), per-warp P/O
    __shared__ __nv_bfloat16 Qsh[BLOCK_M * HD];
    __shared__ __nv_bfloat16 Ksh[KW * HD];
    __shared__ __nv_bfloat16 Vsh[KW * HD];
    __shared__ __nv_bfloat16 Psh[WARPS_M][16 * 16];
    __shared__ float Osh[WARPS_M][16 * 16];

    for (int idx = threadIdx.x; idx < BLOCK_M * HD; idx += blockDim.x) {
        int i = idx / HD, d = idx % HD, qpos = qt * BLOCK_M + i;
        Qsh[idx] = (qpos < Sq) ? q[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d]
                               : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf[OD];
    for (int dc = 0; dc < OD; ++dc)
        wmma::load_matrix_sync(qf[dc], Qsh + w * 16 * HD + dc * 16, HD);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[OD];
    for (int od = 0; od < OD; ++od) wmma::fill_fragment(oacc[od], 0.0f);
    float m_[2] = {-INFINITY, -INFINITY};   // running max: [0]=top row, [1]=bottom row
    float l_[2] = {0.0f, 0.0f};

    int kmax = Sk;
    if (causal) {
        int lastq = min(Sq - 1, qt * BLOCK_M + BLOCK_M - 1);
        kmax = min(Sk, lastq + (Sk - Sq) + 1);
    }

    for (int k0 = 0; k0 < kmax; k0 += KW) {
        for (int idx = threadIdx.x; idx < KW * HD; idx += blockDim.x) {
            int j = idx / HD, d = idx % HD;
            int kk = min(k0 + j, Sk - 1);   // clamp; OOB keys neutralised by mask
            Ksh[idx] = kbase[(int64_t)kk * D + d];
            Vsh[idx] = vbase[(int64_t)kk * D + d];
        }
        __syncthreads();

        // process each 16-wide key subtile with an online-softmax update (only
        // one S fragment live at a time -> low register pressure regardless of KW)
        for (int kn = 0; kn < KSTEP; ++kn) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
            wmma::fill_fragment(sacc, 0.0f);
            for (int dc = 0; dc < OD; ++dc) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, Ksh + kn * 16 * HD + dc * 16, HD);
                wmma::mma_sync(sacc, qf[dc], kf, sacc);
            }

            float rmax[2] = {-INFINITY, -INFINITY};
            for (int i = 0; i < 8; ++i) {
                int row = frag_row(lane, i), col = frag_col(lane, i);
                int qpos = wrow0 + row, kpos = k0 + kn * 16 + col;
                float s = sacc.x[i] * scale;
                bool valid = (qpos < Sq) && (kpos < Sk);
                if (causal && kpos > qpos + (Sk - Sq)) valid = false;
                if (valid) {
                    if (mask)
                        s += __bfloat162float(mask[(int64_t)bi * ms_b + (int64_t)qh * ms_h +
                                                   (int64_t)qpos * ms_q + (int64_t)kpos * ms_k]);
                } else {
                    s = -INFINITY;   // masking sentinel; never enters a matmul (see integration file)
                }
                sacc.x[i] = s;
                rmax[(i & 2) ? 1 : 0] = fmaxf(rmax[(i & 2) ? 1 : 0], s);
            }
            rmax[0] = grpmax(rmax[0]);
            rmax[1] = grpmax(rmax[1]);
            float mnew[2] = {fmaxf(m_[0], rmax[0]), fmaxf(m_[1], rmax[1])};
            float corr[2];
            for (int t = 0; t < 2; ++t) {
                corr[t] = __expf(m_[t] - mnew[t]);
                if (!isfinite(corr[t])) corr[t] = 0.0f;
            }
            for (int od = 0; od < OD; ++od)
                for (int i = 0; i < 8; ++i)
                    oacc[od].x[i] *= corr[(i & 2) ? 1 : 0];

            float rsum[2] = {0.0f, 0.0f};
            for (int i = 0; i < 8; ++i) {
                int row = frag_row(lane, i), col = frag_col(lane, i);
                int ridx = (i & 2) ? 1 : 0;
                float s = sacc.x[i];
                float p = (s == -INFINITY) ? 0.0f : __expf(s - mnew[ridx]);
                rsum[ridx] += p;
                Psh[w][row * 16 + col] = __float2bfloat16(p);
            }
            rsum[0] = grpsum(rsum[0]);
            rsum[1] = grpsum(rsum[1]);
            l_[0] = l_[0] * corr[0] + rsum[0];
            l_[1] = l_[1] * corr[1] + rsum[1];
            m_[0] = mnew[0]; m_[1] = mnew[1];
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
            wmma::load_matrix_sync(pf, Psh[w], 16);
            for (int od = 0; od < OD; ++od) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
                wmma::load_matrix_sync(vf, Vsh + kn * 16 * HD + od * 16, HD);
                wmma::mma_sync(oacc[od], pf, vf, oacc[od]);
            }
            __syncwarp();
        }
        __syncthreads();   // Ksh/Vsh reused by next chunk
    }

    float inv[2];
    inv[0] = (l_[0] > 0.0f) ? 1.0f / l_[0] : 0.0f;
    inv[1] = (l_[1] > 0.0f) ? 1.0f / l_[1] : 0.0f;
    for (int od = 0; od < OD; ++od) {
        for (int i = 0; i < 8; ++i)
            oacc[od].x[i] *= inv[(i & 2) ? 1 : 0];
        wmma::store_matrix_sync(Osh[w], oacc[od], 16, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * 16; idx += WARP) {
            int row = idx / 16, c = idx % 16, qpos = wrow0 + row, d = od * 16 + c;
            if (qpos < Sq)
                out[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d] =
                    __float2bfloat16(Osh[w][row * 16 + c]);
        }
        __syncwarp();
    }
}

// FA2 decode: Sq==1, memory-bound. One block owns a single (batch, kv head) and
// batches the whole GQA group's query heads into <=16 rows so K/V are read once
// per (batch, kv head). WARPS_D warps split the key range; each keeps a register-
// resident O accumulator + online-softmax state and they merge at the end. Tiny
// shared footprint -> many blocks resident -> the K/V streaming saturates DRAM.
#ifndef WARPS_D
#define WARPS_D 4
#endif

// cp.async multi-stage decode: single warp per block, but a NSTAGE-deep software
// pipeline keeps many K/V loads in flight so one warp saturates DRAM without
// needing high occupancy. Correctness for out-of-range keys relies on the
// softmax mask (stale/garbage K/V for masked keys never affects the result), so
// the async copies need no bounds handling beyond skipping OOB 16B chunks.
#ifndef NSTAGE
#define NSTAGE 2
#endif
#include <cuda_pipeline.h>

__device__ __forceinline__ void cp_tile(__nv_bfloat16* dst, const __nv_bfloat16* base,
                                         int k0, int Sk, int lane) {
    // 16 keys x HD bf16 = 128 chunks of 8 bf16 (16 bytes); 32 lanes -> 4 each.
    for (int c = lane; c < 16 * (HD / 8); c += WARP) {
        int j = c / (HD / 8), off = (c % (HD / 8)) * 8;
        if (k0 + j < Sk)
            __pipeline_memcpy_async(dst + j * HD + off,
                                    base + (int64_t)(k0 + j) * HD + off, 16);
    }
}

__global__ void fa2_decode_cp(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ out,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal,
    int64_t ms_b, int64_t ms_h, int64_t ms_q, int64_t ms_k)
{
    const int group = Hq / Hkv;
    const int R = group * Sq;
    const int rt = blockIdx.x, kvh = blockIdx.y, bi = blockIdx.z;
    const int lane = threadIdx.x;
    const int r0 = rt * 16;

    const __nv_bfloat16* kbase = k + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    const __nv_bfloat16* vbase = v + (((int64_t)bi * Hkv + kvh) * Sk) * D;

    __shared__ __nv_bfloat16 Qsh[16 * HD];
    __shared__ __nv_bfloat16 Ksh[NSTAGE][16 * HD];
    __shared__ __nv_bfloat16 Vsh[NSTAGE][16 * HD];
    __shared__ __nv_bfloat16 Psh[16 * 16];
    __shared__ float Osh[16 * 16];

    for (int idx = lane; idx < 16 * HD; idx += WARP) {
        int i = idx / HD, d = idx % HD, r = r0 + i;
        Qsh[idx] = (r < R) ? q[((((int64_t)bi * Hq + (kvh * group + r / Sq)) * Sq) + r % Sq) * D + d]
                           : __float2bfloat16(0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf[OD];
    for (int dc = 0; dc < OD; ++dc)
        wmma::load_matrix_sync(qf[dc], Qsh + dc * 16, HD);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[OD];
    for (int od = 0; od < OD; ++od) wmma::fill_fragment(oacc[od], 0.0f);
    float m_[2] = {-INFINITY, -INFINITY}, l_[2] = {0.0f, 0.0f};

    int kmax = Sk;
    if (causal) {
        int maxpos = -1;
        for (int i = 0; i < 16; ++i) { int r = r0 + i; if (r < R) maxpos = max(maxpos, r % Sq); }
        kmax = min(Sk, maxpos + (Sk - Sq) + 1);
    }
    const int ntiles = (kmax + 15) / 16;

    // prime the pipeline
    for (int s = 0; s < NSTAGE - 1 && s < ntiles; ++s) {
        cp_tile(Ksh[s], kbase, s * 16, Sk, lane);
        cp_tile(Vsh[s], vbase, s * 16, Sk, lane);
        __pipeline_commit();
    }

    for (int t = 0; t < ntiles; ++t) {
        int buf = t % NSTAGE;
        int pf = t + NSTAGE - 1;
        if (pf < ntiles) {
            cp_tile(Ksh[pf % NSTAGE], kbase, pf * 16, Sk, lane);
            cp_tile(Vsh[pf % NSTAGE], vbase, pf * 16, Sk, lane);
        }
        __pipeline_commit();
        __pipeline_wait_prior(NSTAGE - 1);
        __syncwarp();

        int k0 = t * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
        wmma::fill_fragment(sacc, 0.0f);
        for (int dc = 0; dc < OD; ++dc) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::load_matrix_sync(kf, Ksh[buf] + dc * 16, HD);
            wmma::mma_sync(sacc, qf[dc], kf, sacc);
        }

        float rmax[2] = {-INFINITY, -INFINITY};
        for (int i = 0; i < 8; ++i) {
            int row = frag_row(lane, i), col = frag_col(lane, i);
            int r = r0 + row, kpos = k0 + col;
            float s = sacc.x[i] * scale;
            bool valid = (r < R) && (kpos < Sk);
            int qh = 0, qpos = 0;
            if (r < R) { qh = kvh * group + r / Sq; qpos = r % Sq; }
            if (causal && kpos > qpos + (Sk - Sq)) valid = false;
            if (valid) {
                if (mask)
                    s += __bfloat162float(mask[(int64_t)bi * ms_b + (int64_t)qh * ms_h +
                                               (int64_t)qpos * ms_q + (int64_t)kpos * ms_k]);
            } else s = -INFINITY;
            sacc.x[i] = s;
            rmax[(i & 2) ? 1 : 0] = fmaxf(rmax[(i & 2) ? 1 : 0], s);
        }
        rmax[0] = grpmax(rmax[0]); rmax[1] = grpmax(rmax[1]);
        float mnew[2] = {fmaxf(m_[0], rmax[0]), fmaxf(m_[1], rmax[1])};
        float corr[2];
        for (int tt = 0; tt < 2; ++tt) { corr[tt] = __expf(m_[tt] - mnew[tt]); if (!isfinite(corr[tt])) corr[tt] = 0.0f; }
        for (int od = 0; od < OD; ++od)
            for (int i = 0; i < 8; ++i) oacc[od].x[i] *= corr[(i & 2) ? 1 : 0];

        float rsum[2] = {0.0f, 0.0f};
        for (int i = 0; i < 8; ++i) {
            int row = frag_row(lane, i), col = frag_col(lane, i), ridx = (i & 2) ? 1 : 0;
            float s = sacc.x[i];
            float p = (s == -INFINITY) ? 0.0f : __expf(s - mnew[ridx]);
            rsum[ridx] += p;
            Psh[row * 16 + col] = __float2bfloat16(p);
        }
        rsum[0] = grpsum(rsum[0]); rsum[1] = grpsum(rsum[1]);
        l_[0] = l_[0] * corr[0] + rsum[0]; l_[1] = l_[1] * corr[1] + rsum[1];
        m_[0] = mnew[0]; m_[1] = mnew[1];
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf2;
        wmma::load_matrix_sync(pf2, Psh, 16);
        for (int od = 0; od < OD; ++od) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
            wmma::load_matrix_sync(vf, Vsh[buf] + od * 16, HD);
            wmma::mma_sync(oacc[od], pf2, vf, oacc[od]);
        }
        __syncwarp();
    }

    float inv[2];
    inv[0] = (l_[0] > 0.0f) ? 1.0f / l_[0] : 0.0f;
    inv[1] = (l_[1] > 0.0f) ? 1.0f / l_[1] : 0.0f;
    for (int od = 0; od < OD; ++od) {
        for (int i = 0; i < 8; ++i) oacc[od].x[i] *= inv[(i & 2) ? 1 : 0];
        wmma::store_matrix_sync(Osh, oacc[od], 16, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * 16; idx += WARP) {
            int row = idx / 16, c = idx % 16, r = r0 + row, d = od * 16 + c;
            if (r < R)
                out[((((int64_t)bi * Hq + (kvh * group + r / Sq)) * Sq) + r % Sq) * D + d] =
                    __float2bfloat16(Osh[row * 16 + c]);
        }
        __syncwarp();
    }
}

__global__ void fa2_decode(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ out,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal,
    int64_t ms_b, int64_t ms_h, int64_t ms_q, int64_t ms_k)
{
    const int group = Hq / Hkv;
    const int kvh = blockIdx.y, bi = blockIdx.z;
    const int w = threadIdx.x / WARP, lane = threadIdx.x % WARP;

    const __nv_bfloat16* kbase = k + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    const __nv_bfloat16* vbase = v + (((int64_t)bi * Hkv + kvh) * Sk) * D;

    __shared__ __nv_bfloat16 Qsh[16 * HD];
    __shared__ __nv_bfloat16 Ksh[WARPS_D][16 * HD];
    __shared__ __nv_bfloat16 Vsh[WARPS_D][16 * HD];
    __shared__ __nv_bfloat16 Psh[WARPS_D][16 * 16];
    __shared__ float Ocomb[WARPS_D][16 * HD];    // per-warp O for the merge
    __shared__ float Msh[WARPS_D][16], Lsh[WARPS_D][16];

    // row r -> q head kvh*group + r (valid r < group), q position 0
    for (int idx = threadIdx.x; idx < 16 * HD; idx += blockDim.x) {
        int r = idx / HD, d = idx % HD;
        Qsh[idx] = (r < group) ? q[(((int64_t)bi * Hq + (kvh * group + r)) * Sq) * D + d]
                               : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf[OD];
    for (int dc = 0; dc < OD; ++dc)
        wmma::load_matrix_sync(qf[dc], Qsh + dc * 16, HD);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[OD];
    for (int od = 0; od < OD; ++od) wmma::fill_fragment(oacc[od], 0.0f);
    float m_[2] = {-INFINITY, -INFINITY};
    float l_[2] = {0.0f, 0.0f};

    for (int k0 = w * 16; k0 < Sk; k0 += WARPS_D * 16) {
        for (int idx = lane; idx < 16 * HD; idx += WARP) {
            int j = idx / HD, d = idx % HD;
            int kk = min(k0 + j, Sk - 1);
            Ksh[w][idx] = kbase[(int64_t)kk * D + d];
            Vsh[w][idx] = vbase[(int64_t)kk * D + d];
        }
        __syncwarp();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
        wmma::fill_fragment(sacc, 0.0f);
        for (int dc = 0; dc < OD; ++dc) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::load_matrix_sync(kf, Ksh[w] + dc * 16, HD);
            wmma::mma_sync(sacc, qf[dc], kf, sacc);
        }

        float rmax[2] = {-INFINITY, -INFINITY};
        for (int i = 0; i < 8; ++i) {
            int row = frag_row(lane, i), col = frag_col(lane, i);
            int qh = kvh * group + row, kpos = k0 + col;
            float s = sacc.x[i] * scale;
            bool valid = (row < group) && (kpos < Sk);
            if (valid) {
                if (mask)
                    s += __bfloat162float(mask[(int64_t)bi * ms_b + (int64_t)qh * ms_h +
                                               (int64_t)kpos * ms_k]);
            } else {
                s = -INFINITY;   // masking sentinel; never enters a matmul (see integration file)
            }
            sacc.x[i] = s;
            rmax[(i & 2) ? 1 : 0] = fmaxf(rmax[(i & 2) ? 1 : 0], s);
        }
        rmax[0] = grpmax(rmax[0]);
        rmax[1] = grpmax(rmax[1]);
        float mnew[2] = {fmaxf(m_[0], rmax[0]), fmaxf(m_[1], rmax[1])};
        float corr[2];
        for (int t = 0; t < 2; ++t) {
            corr[t] = __expf(m_[t] - mnew[t]);
            if (!isfinite(corr[t])) corr[t] = 0.0f;
        }
        for (int od = 0; od < OD; ++od)
            for (int i = 0; i < 8; ++i)
                oacc[od].x[i] *= corr[(i & 2) ? 1 : 0];

        float rsum[2] = {0.0f, 0.0f};
        for (int i = 0; i < 8; ++i) {
            int row = frag_row(lane, i), col = frag_col(lane, i);
            int ridx = (i & 2) ? 1 : 0;
            float s = sacc.x[i];
            float p = (s == -INFINITY) ? 0.0f : __expf(s - mnew[ridx]);
            rsum[ridx] += p;
            Psh[w][row * 16 + col] = __float2bfloat16(p);
        }
        rsum[0] = grpsum(rsum[0]);
        rsum[1] = grpsum(rsum[1]);
        l_[0] = l_[0] * corr[0] + rsum[0];
        l_[1] = l_[1] * corr[1] + rsum[1];
        m_[0] = mnew[0]; m_[1] = mnew[1];
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
        wmma::load_matrix_sync(pf, Psh[w], 16);
        for (int od = 0; od < OD; ++od) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
            wmma::load_matrix_sync(vf, Vsh[w] + od * 16, HD);
            wmma::mma_sync(oacc[od], pf, vf, oacc[od]);
        }
        __syncwarp();
    }

    // publish each warp's partials for the merge (all 32 lanes: lane>>2 covers
    // rows 0..7 for the top half and +8 for the bottom half)
    for (int od = 0; od < OD; ++od)
        wmma::store_matrix_sync(Ocomb[w] + od * 16, oacc[od], HD, wmma::mem_row_major);
    {
        int rtop = lane >> 2, rbot = rtop + 8;
        Msh[w][rtop] = m_[0]; Lsh[w][rtop] = l_[0];
        Msh[w][rbot] = m_[1]; Lsh[w][rbot] = l_[1];
    }
    __syncthreads();

    // warp 0 merges all warps' partials and writes the output
    if (w == 0) {
        for (int r = lane; r < group; r += WARP) {
            float gm = -INFINITY;
            for (int ww = 0; ww < WARPS_D; ++ww) gm = fmaxf(gm, Msh[ww][r]);
            float gl = 0.0f, acc[HD];
            for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
            for (int ww = 0; ww < WARPS_D; ++ww) {
                float sc = __expf(Msh[ww][r] - gm);
                if (!isfinite(sc)) sc = 0.0f;
                gl += sc * Lsh[ww][r];
                for (int d = 0; d < HD; ++d) acc[d] += sc * Ocomb[ww][r * HD + d];
            }
            float inv = (gl > 0.0f) ? 1.0f / gl : 0.0f;
            int qh = kvh * group + r;
            for (int d = 0; d < HD; ++d)
                out[(((int64_t)bi * Hq + qh) * Sq) * D + d] = __float2bfloat16(acc[d] * inv);
        }
    }
}

// ------------------------- reference + harness -------------------------------
__global__ void naive_attn(const __nv_bfloat16* q, const __nv_bfloat16* k,
                           const __nv_bfloat16* v, __nv_bfloat16* out,
                           int B, int Hq, int Hkv, int Sq, int Sk, int D,
                           float scale, bool causal) {
    int bi = blockIdx.z, qh = blockIdx.y, qpos = blockIdx.x * blockDim.x + threadIdx.x;
    if (qpos >= Sq) return;
    int group = Hq / Hkv, kvh = qh / group;
    const __nv_bfloat16* qp = q + ((((int64_t)bi * Hq + qh) * Sq) + qpos) * D;
    const __nv_bfloat16* kb = k + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    const __nv_bfloat16* vb = v + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    float m = -INFINITY, l = 0.0f, acc[HD];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    for (int kp = 0; kp < Sk; ++kp) {
        if (causal && kp > qpos + (Sk - Sq)) break;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += __bfloat162float(qp[d]) * __bfloat162float(kb[(int64_t)kp * D + d]);
        s *= scale;
        float mn = fmaxf(m, s), corr = __expf(m - mn), p = __expf(s - mn);
        l = l * corr + p;
        for (int d = 0; d < D; ++d) acc[d] = acc[d] * corr + p * __bfloat162float(vb[(int64_t)kp * D + d]);
        m = mn;
    }
    float inv = (l > 0) ? 1.0f / l : 0.0f;
    __nv_bfloat16* op = out + ((((int64_t)bi * Hq + qh) * Sq) + qpos) * D;
    for (int d = 0; d < D; ++d) op[d] = __float2bfloat16(acc[d] * inv);
}

static __nv_bfloat16* rnd(int64_t n) {
    std::vector<__nv_bfloat16> h(n);
    for (int64_t i = 0; i < n; ++i) h[i] = __float2bfloat16((float)(rand() / (double)RAND_MAX * 2 - 1) * 0.5f);
    __nv_bfloat16* d; cudaMalloc(&d, n * sizeof(__nv_bfloat16));
    cudaMemcpy(d, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    return d;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "prefill";
    int B = 256, Hq = 14, Hkv = 2, D = 64;
    int Sq, Sk; bool causal;
    bool decmode = !strcmp(mode, "decode") || !strcmp(mode, "decodecp");
    if (decmode) { Sq = 1; Sk = (argc > 2) ? atoi(argv[2]) : 1024; causal = false; }
    else { Sq = (argc > 2) ? atoi(argv[2]) : 512; Sk = Sq; causal = true; }
    int group = Hq / Hkv;
    float scale = 1.0f / sqrtf((float)D);

    srand(0);
    __nv_bfloat16 *q = rnd((int64_t)B * Hq * Sq * D), *k = rnd((int64_t)B * Hkv * Sk * D),
                  *v = rnd((int64_t)B * Hkv * Sk * D), *out, *ref;
    cudaMalloc(&out, (int64_t)B * Hq * Sq * D * sizeof(__nv_bfloat16));
    cudaMalloc(&ref, (int64_t)B * Hq * Sq * D * sizeof(__nv_bfloat16));

    naive_attn<<<dim3((Sq + 63) / 64, Hq, B), 64>>>(q, k, v, ref, B, Hq, Hkv, Sq, Sk, D, scale, causal);

    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, Hq, B);
    bool dec = !strcmp(mode, "decode");
    bool deccp = !strcmp(mode, "decodecp");
    auto run = [&]{
        if (deccp)
            fa2_decode_cp<<<dim3((group * Sq + 15) / 16, Hkv, B), WARP>>>(q, k, v, nullptr, out, B, Hq, Hkv, Sq, Sk, D, scale, causal, 0, 0, 0, 0);
        else if (dec)
            fa2_decode<<<dim3(1, Hkv, B), WARPS_D * WARP>>>(q, k, v, nullptr, out, B, Hq, Hkv, Sq, Sk, D, scale, causal, 0, 0, 0, 0);
        else
            fa2_prefill<<<grid, WARPS_M * WARP>>>(q, k, v, nullptr, out, B, Hq, Hkv, Sq, Sk, D, scale, causal, 0, 0, 0, 0);
    };
    run();
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("launch error: %s\n", cudaGetErrorString(e)); return 1; }

    int64_t N = (int64_t)B * Hq * Sq * D;
    std::vector<__nv_bfloat16> ho(N), hr(N);
    cudaMemcpy(ho.data(), out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), ref, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int64_t i = 0; i < N; ++i) {
        double a = __bfloat162float(ho[i]), b = __bfloat162float(hr[i]);
        num += (a - b) * (a - b); den += b * b;
    }
    double relerr = sqrt(num) / (sqrt(den) + 1e-9);

    cudaEvent_t s, en; cudaEventCreate(&s); cudaEventCreate(&en);
    for (int i = 0; i < 10; ++i) run();
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    int iters = 50;
    for (int i = 0; i < iters; ++i) run();
    cudaEventRecord(en); cudaEventSynchronize(en);
    float ms; cudaEventElapsedTime(&ms, s, en);
    printf("mode=%s WARPS_M=%d KSTEP=%d WARPS_D=%d | rel-err %.3e | %.1f us/call\n",
           mode, WARPS_M, KSTEP, WARPS_D, relerr, ms / iters * 1e3);
    return relerr < 5e-2 ? 0 : 2;
}
