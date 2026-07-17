// FlashAttention-2 style forward with bf16 tensor cores (WMMA 16x16x16, fp32
// accumulate), a drop-in replacement for the cuDNN flash_fprop SDPA in the
// infer_golden.py trace (F.scaled_dot_product_attention). See tc_kernels.h for
// the operator shape.
//
// The two regimes of the trace are handled by two kernels, both FA2-style:
// register-resident O accumulator (kept in WMMA fragments across the whole key
// loop -- no shared-memory O traffic), online softmax, and a single delayed
// normalization at the end.
//
//   * DECODE (Sq==1, the trace-dominant, memory-bound path): one warp per block
//     owns a 16-row query tile that BATCHES THE WHOLE GQA GROUP (row r -> q head
//     kvh*group + r/Sq, q pos r%Sq) so K/V are streamed from DRAM exactly once
//     per (batch, kv head). K/V tiles are prefetched with a DEC_STAGES-deep
//     cp.async software pipeline so a single warp keeps enough memory requests in
//     flight to saturate DRAM bandwidth (the bottleneck here) without needing
//     high occupancy. The tiny shared footprint lets many blocks stay resident.
//     This kernel also handles general Sq (used by force_decode_kernel).
//   * PREFILL (Sq>1, compute-bound): FA2 split-Q. One block owns BLOCK_M query
//     rows; each of PM_WARPS warps owns 16 rows and iterates all keys. K/V tiles
//     are loaded once into shared and reused by every warp; each 16-wide key
//     subtile is folded in with an online-softmax update (one score fragment
//     live at a time -> low register pressure).
//
// The register-resident O trick relies on the WMMA fp32 accumulator layout
// (verified empirically, see probe_wmma_layout.cu): each lane's 8 elements map
// to exactly two output rows -- row = lane/4 for elements {0,1,4,5} and
// row = lane/4 + 8 for {2,3,6,7} -- so a per-row softmax correction is applied
// by scaling those element groups, and per-row max/sum are finished with a
// shuffle reduction across the 4 lanes of a group (which hold different columns
// of the same row).
//
// ---------------------------------------------------------------------------
// VERIFICATION INVARIANT -- DO NOT "FIX" WITH EXTENDED REALS. Read before
// touching the causal / masking logic.
//
// The eventual verified Kuiper port of this flash-attention kernel launches a
// tensor-core matmul primitive, mma_sync' (specified by `emma` in
// Kuiper.TensorCore.Base.fsti), whose precondition requires its fragment
// operands to approximate *real* matrices. Kuiper has no extended reals -- it
// cannot say what real value a float +/-inf approximates. This kernel is
// deliberately structured so causal masking never triggers that requirement,
// and it can be verified with ordinary reals exactly like kfonline_softmax
// (Kuiper.Kernel.OnlineSoftmax.fst). The invariant:
//
//   1. -INFINITY NEVER ENTERS AN mma_sync' OPERAND. Both tensor-core matmuls
//      take only finite fragments: Q@K^T sees finite Q,K; P@V sees P whose
//      masked entries are the literal float 0.0 (see below). Masking happens
//      strictly *between* the two matmuls, on the scalar score, never folded
//      into an mma_sync' accumulate or a -inf additive bias matrix.
//
//   2. A masked probability is produced by a SELECT-TO-ZERO branch, not by
//      exp(-inf). We set the masked score to the sentinel -INFINITY, but the
//      probability is `p = (score == -INFINITY) ? 0.0f : exp(...)`. So p is the
//      real 0 by construction; we never evaluate exp at -inf. The real
//      reference softmax is defined over the *valid* key set only (masked keys
//      have probability 0 by the mask's definition), so p @! 0 holds without
//      any extended-real reasoning.
//
//   3. The only other use of -inf is `fmax(running_max, score)`. This is the
//      same skirt kfonline_softmax uses for its initial max: fmax(-inf, x) = x,
//      so the approximation relation on the max is only asserted once it has
//      absorbed >=1 finite score. Under is_causal the diagonal key kj == qpos
//      is always valid, so every emitted row has a finite max; a fully-masked
//      row (max stays -inf) cannot occur for causal, and is anyway guarded by
//      `inv = (l > 0) ? 1/l : 0` -> a defined finite 0 output for which we
//      simply claim no real-softmax approximation.
//
// Consequence: keep masking as a post-QK^T select. If you ever fold a -inf bias
// into the score matmul or compute exp(score + (-inf)) without the branch, you
// reintroduce the extended-reals obligation and this kernel becomes
// unverifiable in Kuiper. (This argument is about the is_causal path, which
// derives -inf from an index comparison. A caller-supplied additive mask tensor
// that itself contains -inf is a separate obligation -- the golden's causal
// generation uses is_causal with no mask tensor, so the clean path matters.)
// ---------------------------------------------------------------------------
#include "tc_kernels.h"
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <math.h>

using namespace nvcuda;

#define WARP 32
#define HD 64            // head dim (Qwen2.5 HEAD_DIM), D <= 64, D % 16 == 0
#define OD (HD / 16)     // output/head-dim fragments = 4
#define DEC_STAGES 2     // decode cp.async pipeline depth (swept: 2 is best on A6000)

// WMMA fp32 accumulator layout (verified with probe_wmma_layout.cu):
//   row = lane/4 + (i&2 ? 8 : 0),  col = 2*(lane&3) + (i&1) + (i&4 ? 8 : 0)
__device__ __forceinline__ int frag_row(int lane, int i) { return (lane >> 2) + ((i & 2) ? 8 : 0); }
__device__ __forceinline__ int frag_col(int lane, int i) { return ((lane & 3) << 1) + (i & 1) + ((i & 4) ? 8 : 0); }

// finish a per-row reduction across the 4 lanes of a group (they hold different
// columns of the same row).
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

// cp.async-stage a 16-key x HD bf16 tile into shared. Each 16-byte (8 bf16)
// chunk with an in-bounds key row is issued as an async copy; OOB chunks are
// skipped, leaving whatever the ring buffer held before -- those scores are
// masked to the -INFINITY sentinel below, so stale K/V never affects the result
// (and never enters a matmul as a real -inf; see header invariant).
__device__ __forceinline__ void cp_tile(__nv_bfloat16* dst, const __nv_bfloat16* base,
                                        int k0, int Sk, int lane) {
    for (int c = lane; c < 16 * (HD / 8); c += WARP) {
        int j = c / (HD / 8), off = (c % (HD / 8)) * 8;
        if (k0 + j < Sk)
            __pipeline_memcpy_async(dst + j * HD + off,
                                    base + (int64_t)(k0 + j) * HD + off, 16);
    }
}

// -------------------------------- decode -------------------------------------
// One warp per block; a 16-row query tile of R = group*Sq batched query rows for
// one (batch, kv head). Handles Sq==1 (hot path) and general Sq.
__global__ void tc_flash_attn_decode_kernel(
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
    __shared__ __nv_bfloat16 Ksh[DEC_STAGES][16 * HD];
    __shared__ __nv_bfloat16 Vsh[DEC_STAGES][16 * HD];
    __shared__ __nv_bfloat16 Psh[16 * 16];
    __shared__ float Osh[16 * 16];

    for (int idx = lane; idx < 16 * HD; idx += WARP) {
        int i = idx / HD, d = idx % HD, r = r0 + i;
        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (r < R) {
            int qh = kvh * group + r / Sq, qpos = r % Sq;
            val = q[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d];
        }
        Qsh[idx] = val;
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf[OD];
    for (int dc = 0; dc < OD; ++dc)
        wmma::load_matrix_sync(qf[dc], Qsh + dc * 16, HD);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[OD];
    for (int od = 0; od < OD; ++od) wmma::fill_fragment(oacc[od], 0.0f);
    float m_[2] = {-INFINITY, -INFINITY};
    float l_[2] = {0.0f, 0.0f};

    int kmax = Sk;
    if (causal) {
        int maxpos = -1;
        for (int i = 0; i < 16; ++i) { int r = r0 + i; if (r < R) maxpos = max(maxpos, r % Sq); }
        kmax = min(Sk, maxpos + (Sk - Sq) + 1);
    }
    const int ntiles = (kmax + 15) / 16;

    // prime the software pipeline (DEC_STAGES-1 tiles in flight)
    for (int s = 0; s < DEC_STAGES - 1 && s < ntiles; ++s) {
        cp_tile(Ksh[s], kbase, s * 16, Sk, lane);
        cp_tile(Vsh[s], vbase, s * 16, Sk, lane);
        __pipeline_commit();
    }

    for (int t = 0; t < ntiles; ++t) {
        int buf = t % DEC_STAGES;
        int pft = t + DEC_STAGES - 1;          // tile to prefetch this step
        if (pft < ntiles) {
            cp_tile(Ksh[pft % DEC_STAGES], kbase, pft * 16, Sk, lane);
            cp_tile(Vsh[pft % DEC_STAGES], vbase, pft * 16, Sk, lane);
        }
        __pipeline_commit();
        __pipeline_wait_prior(DEC_STAGES - 1);
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
            } else {
                s = -INFINITY;   // masking sentinel; never enters a matmul (see header invariant)
            }
            sacc.x[i] = s;
            rmax[(i & 2) ? 1 : 0] = fmaxf(rmax[(i & 2) ? 1 : 0], s);   // fmax(-inf,x)=x
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
            // select-to-zero: masked prob is the literal 0.0, NOT exp(-inf), so
            // P@V takes a finite operand (see header invariant).
            float p = (s == -INFINITY) ? 0.0f : __expf(s - mnew[ridx]);
            rsum[ridx] += p;
            Psh[row * 16 + col] = __float2bfloat16(p);
        }
        rsum[0] = grpsum(rsum[0]);
        rsum[1] = grpsum(rsum[1]);
        l_[0] = l_[0] * corr[0] + rsum[0];
        l_[1] = l_[1] * corr[1] + rsum[1];
        m_[0] = mnew[0]; m_[1] = mnew[1];
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
        wmma::load_matrix_sync(pf, Psh, 16);
        for (int od = 0; od < OD; ++od) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
            wmma::load_matrix_sync(vf, Vsh[buf] + od * 16, HD);
            wmma::mma_sync(oacc[od], pf, vf, oacc[od]);
        }
        __syncwarp();
    }

    float inv[2];
    inv[0] = (l_[0] > 0.0f) ? 1.0f / l_[0] : 0.0f;
    inv[1] = (l_[1] > 0.0f) ? 1.0f / l_[1] : 0.0f;
    for (int od = 0; od < OD; ++od) {
        for (int i = 0; i < 8; ++i)
            oacc[od].x[i] *= inv[(i & 2) ? 1 : 0];
        wmma::store_matrix_sync(Osh, oacc[od], 16, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * 16; idx += WARP) {
            int row = idx / 16, c = idx % 16, r = r0 + row, d = od * 16 + c;
            if (r < R) {
                int qh = kvh * group + r / Sq, qpos = r % Sq;
                out[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d] =
                    __float2bfloat16(Osh[row * 16 + c]);
            }
        }
        __syncwarp();
    }
}

// -------------------------------- prefill ------------------------------------
#define PM_WARPS 8                 // query-row warps
#define PM (PM_WARPS * 16)         // BLOCK_M query rows per block
#define PKN 2                      // key subtiles (of 16) per shared chunk
#define PKW (PKN * 16)

__global__ void tc_flash_attn_prefill_kernel(
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
    const int wrow0 = qt * PM + w * 16;    // first query row this warp owns

    const __nv_bfloat16* kbase = k + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    const __nv_bfloat16* vbase = v + (((int64_t)bi * Hkv + kvh) * Sk) * D;

    __shared__ __nv_bfloat16 Qsh[PM * HD];
    __shared__ __nv_bfloat16 Ksh[PKW * HD];
    __shared__ __nv_bfloat16 Vsh[PKW * HD];
    __shared__ __nv_bfloat16 Psh[PM_WARPS][16 * 16];
    __shared__ float Osh[PM_WARPS][16 * 16];

    for (int idx = threadIdx.x; idx < PM * HD; idx += blockDim.x) {
        int i = idx / HD, d = idx % HD, qpos = qt * PM + i;
        Qsh[idx] = (qpos < Sq) ? q[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d]
                               : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf[OD];
    for (int dc = 0; dc < OD; ++dc)
        wmma::load_matrix_sync(qf[dc], Qsh + w * 16 * HD + dc * 16, HD);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[OD];
    for (int od = 0; od < OD; ++od) wmma::fill_fragment(oacc[od], 0.0f);
    float m_[2] = {-INFINITY, -INFINITY};
    float l_[2] = {0.0f, 0.0f};

    int kmax = Sk;
    if (causal) {
        int lastq = min(Sq - 1, qt * PM + PM - 1);
        kmax = min(Sk, lastq + (Sk - Sq) + 1);
    }

    for (int k0 = 0; k0 < kmax; k0 += PKW) {
        for (int idx = threadIdx.x; idx < PKW * HD; idx += blockDim.x) {
            int j = idx / HD, d = idx % HD;
            int kk = min(k0 + j, Sk - 1);   // clamp; OOB keys neutralised by mask below
            Ksh[idx] = kbase[(int64_t)kk * D + d];
            Vsh[idx] = vbase[(int64_t)kk * D + d];
        }
        __syncthreads();

        for (int kn = 0; kn < PKN; ++kn) {
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
                    s = -INFINITY;   // masking sentinel; never enters a matmul (see header invariant)
                }
                sacc.x[i] = s;
                rmax[(i & 2) ? 1 : 0] = fmaxf(rmax[(i & 2) ? 1 : 0], s);   // fmax(-inf,x)=x
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
                // select-to-zero: masked prob is the literal 0.0 (see header invariant).
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

void tc_flash_attn_launch(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* mask, __nv_bfloat16* out,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal,
    int64_t ms_b, int64_t ms_h, int64_t ms_q, int64_t ms_k,
    bool force_decode_kernel,
    cudaStream_t stream)
{
    if (B == 0 || Sq == 0) return;
    if (Sq > 1 && !force_decode_kernel) {
        dim3 grid((Sq + PM - 1) / PM, Hq, B);
        tc_flash_attn_prefill_kernel<<<grid, PM_WARPS * WARP, 0, stream>>>(
            q, k, v, mask, out, B, Hq, Hkv, Sq, Sk, D, scale, causal,
            ms_b, ms_h, ms_q, ms_k);
        return;
    }
    const int group = Hq / Hkv;
    const int R = group * Sq;
    dim3 grid((R + 15) / 16, Hkv, B);
    tc_flash_attn_decode_kernel<<<grid, WARP, 0, stream>>>(
        q, k, v, mask, out, B, Hq, Hkv, Sq, Sk, D, scale, causal,
        ms_b, ms_h, ms_q, ms_k);
}
