// FlashAttention forward with bf16 tensor cores, drop-in for the cuDNN
// flash_fprop SDPA in the trace (F.scaled_dot_product_attention). See
// tc_kernels.h for the operator shape.
//
// One block owns a 16-row query tile for a single (batch, KV head) and holds
// NWARPS warps that split the KEY dimension. Crucially the 16 rows batch the
// whole GQA group together: row r maps to (q_head_in_group = r / Sq,
// q_pos = r % Sq), so the `group` query heads that share this KV head read K/V
// exactly once instead of `group` times -- the win that makes the memory-bound
// decode path (Sq=1, one row per head) fast. Each warp keeps its own online-
// softmax partial (running max `Msh[w]`, denominator `Lsh[w]`, accumulator
// `Osh[w]`); a final combine merges the NWARPS partials (m = max, rescale by
// exp(m_w - m)). The two matmuls run on tensor cores: S = Q@K^T (contracted
// over D in 16-wide chunks) and P@V (per 16-wide output chunk).
//
// ---------------------------------------------------------------------------
// VERIFICATION INVARIANT -- DO NOT "FIX" WITH EXTENDED REALS. Read before
// touching the causal / masking logic.
//
// The eventual verified Kuiper port of *this* flash-attention kernel must
// discharge the precondition of the tensor-core matmul primitive it launches:
// mma_sync' (specified by `emma` in Kuiper.TensorCore.Base.fsti) requires its
// fragment operands to approximate *real* matrices. Kuiper has no extended
// reals -- it cannot say what real value a float +/-inf approximates. This
// kernel is deliberately structured so that requirement is never triggered by
// causal masking, and it can be verified with ordinary reals exactly like
// kfonline_softmax (Kuiper.Kernel.OnlineSoftmax.fst). The invariant:
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
//      is always valid, so every emitted row has a finite max; the fully-masked
//      row (max stays -inf) cannot occur for causal, and is anyway guarded by
//      `inv = (l > 0) ? 1/l : 0` -> a defined finite 0 output for which we
//      simply claim no real-softmax approximation.
//
// Consequence: keep masking as a post-QK^T select. If you ever fold a -inf bias
// into the score matmul or compute exp(score + (-inf)) without the branch, you
// reintroduce the extended-reals obligation and this kernel becomes
// unverifiable in Kuiper. (Note: this argument is about the is_causal path,
// which derives -inf from an index comparison. A caller-supplied additive mask
// tensor that itself contains -inf is a separate obligation -- the golden's
// causal generation uses is_causal with no mask tensor, so the clean path is
// the one that matters.)
// ---------------------------------------------------------------------------
#include "tc_kernels.h"
#include <mma.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace nvcuda;

#define WARP 32
#define NWARPS 4
#define BM 16
#define BN 16
#define HD 64        // shared tile width (head_dim <= 64)

__global__ void tc_flash_attn_kernel(
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
    const int R = group * Sq;                 // batched query rows for this KV head
    const int rt = blockIdx.x, kvh = blockIdx.y, bi = blockIdx.z;
    const int tid = threadIdx.x, w = tid / WARP, lane = tid % WARP;
    const int r0 = rt * BM;

    const __nv_bfloat16* kbase = k + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    const __nv_bfloat16* vbase = v + (((int64_t)bi * Hkv + kvh) * Sk) * D;

    __shared__ __nv_bfloat16 Qs[BM * HD];
    __shared__ __nv_bfloat16 Ksh[NWARPS][BN * HD];
    __shared__ __nv_bfloat16 Vsh[NWARPS][BN * HD];
    __shared__ float Ssh[NWARPS][BM * BN];
    __shared__ __nv_bfloat16 Psh[NWARPS][BM * BN];
    __shared__ float Osh[NWARPS][BM * HD];
    __shared__ float PVc[NWARPS][BM * 16];
    __shared__ float Msh[NWARPS][BM], Lsh[NWARPS][BM], cw[NWARPS][BM];
    __shared__ float gm_sh[BM], gl_sh[BM], scale_sh[NWARPS][BM];

    // row i -> (q head, q position); q_head = kvh*group + i_head.
    for (int idx = tid; idx < BM * D; idx += blockDim.x) {
        int i = idx / D, d = idx % D, r = r0 + i;
        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (r < R) {
            int qh = kvh * group + r / Sq, qpos = r % Sq;
            val = q[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d];
        }
        Qs[i * HD + d] = val;
    }
    if (lane < BM) { Msh[w][lane] = -INFINITY; Lsh[w][lane] = 0.0f; }
    for (int idx = lane; idx < BM * HD; idx += WARP) Osh[w][idx] = 0.0f;
    __syncthreads();

    // Causal early-exit: this tile's rows never attend past key `kmax`.
    int kmax = Sk;
    if (causal) {
        int maxpos = -1;
        for (int i = 0; i < BM; ++i) {
            int r = r0 + i;
            if (r < R) maxpos = max(maxpos, r % Sq);
        }
        kmax = min(Sk, maxpos + (Sk - Sq) + 1);
    }
    const int nkt = (kmax + BN - 1) / BN;
    for (int jt = w; jt < nkt; jt += NWARPS) {
        const int k0 = jt * BN;
        for (int idx = lane; idx < BN * D; idx += WARP) {
            int j = idx / D, d = idx % D;
            bool ok = (k0 + j) < Sk;
            Ksh[w][j * HD + d] = ok ? kbase[(int64_t)(k0 + j) * D + d] : __float2bfloat16(0.0f);
            Vsh[w][j * HD + d] = ok ? vbase[(int64_t)(k0 + j) * D + d] : __float2bfloat16(0.0f);
        }
        __syncwarp();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
        wmma::fill_fragment(sacc, 0.0f);
        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::load_matrix_sync(qf, Qs + dc, HD);
            wmma::load_matrix_sync(kf, Ksh[w] + dc, HD);
            wmma::mma_sync(sacc, qf, kf, sacc);
        }
        wmma::store_matrix_sync(Ssh[w], sacc, BN, wmma::mem_row_major);
        __syncwarp();

        if (lane < BM) {
            int i = lane, r = r0 + i;
            int qh = kvh * group + (r < R ? r / Sq : 0);
            int qpos = (r < R) ? r % Sq : 0;
            float rowmax = -INFINITY;
            for (int j = 0; j < BN; ++j) {
                int kj = k0 + j;
                bool valid = (r < R) && kj < Sk;
                if (causal && kj > qpos + (Sk - Sq)) valid = false;
                float s;
                if (valid) {
                    s = Ssh[w][i * BN + j] * scale;
                    if (mask)
                        s += __bfloat162float(mask[(int64_t)bi * ms_b + (int64_t)qh * ms_h +
                                                   (int64_t)qpos * ms_q + (int64_t)kj * ms_k]);
                } else {
                    s = -INFINITY;   // masked: sentinel only; never fed to a matmul (see header invariant)
                }
                Ssh[w][i * BN + j] = s;
                rowmax = fmaxf(rowmax, s);   // fmax(-inf,x)=x -- skirtable like kfonline_softmax
            }
            float mnew = fmaxf(Msh[w][i], rowmax);
            float corr = __expf(Msh[w][i] - mnew);
            if (!isfinite(corr)) corr = 0.0f;
            float rowsum = 0.0f;
            for (int j = 0; j < BN; ++j) {
                float sv = Ssh[w][i * BN + j];
                // Select-to-zero: masked prob is the literal 0.0, NOT exp(-inf).
                // This is what lets P@V take a finite operand and keeps -inf out
                // of the reals (see header invariant).
                float p = (sv == -INFINITY) ? 0.0f : __expf(sv - mnew);
                Psh[w][i * BN + j] = __float2bfloat16(p);
                rowsum += p;
            }
            Lsh[w][i] = Lsh[w][i] * corr + rowsum;
            cw[w][i] = corr;
            Msh[w][i] = mnew;
        }
        __syncwarp();

        for (int idx = lane; idx < BM * HD; idx += WARP)
            Osh[w][idx] *= cw[w][idx / HD];
        __syncwarp();

        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pvacc;
            wmma::fill_fragment(pvacc, 0.0f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
            wmma::load_matrix_sync(pf, Psh[w], BN);
            wmma::load_matrix_sync(vf, Vsh[w] + dc, HD);
            wmma::mma_sync(pvacc, pf, vf, pvacc);
            wmma::store_matrix_sync(PVc[w], pvacc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < BM * 16; idx += WARP) {
                int i = idx / 16, jj = idx % 16;
                Osh[w][i * HD + dc + jj] += PVc[w][idx];
            }
            __syncwarp();
        }
    }
    __syncthreads();

    // Combine the NWARPS key-split partials for each query row.
    if (w == 0 && lane < BM) {
        int i = lane;
        float gm = -INFINITY;
#pragma unroll
        for (int ww = 0; ww < NWARPS; ++ww) gm = fmaxf(gm, Msh[ww][i]);
        float gl = 0.0f;
#pragma unroll
        for (int ww = 0; ww < NWARPS; ++ww) {
            float sc = __expf(Msh[ww][i] - gm);
            if (!isfinite(sc)) sc = 0.0f;
            scale_sh[ww][i] = sc;
            gl += sc * Lsh[ww][i];
        }
        gm_sh[i] = gm;
        gl_sh[i] = gl;
    }
    __syncthreads();

    if (w == 0) {
        for (int idx = lane; idx < BM * D; idx += WARP) {
            int i = idx / D, d = idx % D, r = r0 + i;
            if (r >= R) continue;
            int qh = kvh * group + r / Sq, qpos = r % Sq;
            float acc = 0.0f;
#pragma unroll
            for (int ww = 0; ww < NWARPS; ++ww)
                acc += scale_sh[ww][i] * Osh[ww][i * HD + d];
            float inv = (gl_sh[i] > 0.0f) ? (1.0f / gl_sh[i]) : 0.0f;
            out[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d] = __float2bfloat16(acc * inv);
        }
    }
}

// Prefill path (Sq > 1). Here every query row does real work and all queries in
// a block share the same keys, so we invert the strategy: one block owns a
// P_QROWS-row query tile for a single (batch, q head) and each of the P_NWARPS
// warps owns 16 of those rows. K/V tiles are streamed once into shared memory
// and reused by every warp (no `group`-fold or key-split redundancy), and each
// warp runs an independent online softmax over all keys -- so there is no
// cross-warp combine. Causal early-exit skips key tiles beyond the tile's rows.
#define P_NWARPS 4
#define P_QROWS (P_NWARPS * 16)   // 64 query rows per block

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
    const int tid = threadIdx.x, w = tid / WARP, lane = tid % WARP;
    const int q0 = qt * P_QROWS;

    const __nv_bfloat16* kbase = k + (((int64_t)bi * Hkv + kvh) * Sk) * D;
    const __nv_bfloat16* vbase = v + (((int64_t)bi * Hkv + kvh) * Sk) * D;

    __shared__ __nv_bfloat16 Qs[P_QROWS * HD];
    __shared__ __nv_bfloat16 Ksh[BN * HD];
    __shared__ __nv_bfloat16 Vsh[BN * HD];
    __shared__ float Ssh[P_NWARPS][16 * BN];
    __shared__ __nv_bfloat16 Psh[P_NWARPS][16 * BN];
    __shared__ float Osh[P_QROWS * HD];
    __shared__ float PVc[P_NWARPS][16 * 16];
    __shared__ float Msh[P_QROWS], Lsh[P_QROWS], cwsh[P_QROWS];

    for (int idx = tid; idx < P_QROWS * D; idx += blockDim.x) {
        int i = idx / D, d = idx % D, qpos = q0 + i;
        __nv_bfloat16 val = __float2bfloat16(0.0f);
        if (qpos < Sq)
            val = q[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d];
        Qs[i * HD + d] = val;
    }
    for (int i = tid; i < P_QROWS; i += blockDim.x) { Msh[i] = -INFINITY; Lsh[i] = 0.0f; }
    for (int idx = tid; idx < P_QROWS * HD; idx += blockDim.x) Osh[idx] = 0.0f;
    __syncthreads();

    int kmax = Sk;
    if (causal) {
        int maxpos = min(Sq - 1, q0 + P_QROWS - 1);
        kmax = min(Sk, maxpos + (Sk - Sq) + 1);
    }
    const int nkt = (kmax + BN - 1) / BN;
    for (int jt = 0; jt < nkt; ++jt) {
        const int k0 = jt * BN;
        for (int idx = tid; idx < BN * D; idx += blockDim.x) {
            int j = idx / D, d = idx % D;
            bool ok = (k0 + j) < Sk;
            Ksh[j * HD + d] = ok ? kbase[(int64_t)(k0 + j) * D + d] : __float2bfloat16(0.0f);
            Vsh[j * HD + d] = ok ? vbase[(int64_t)(k0 + j) * D + d] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
        wmma::fill_fragment(sacc, 0.0f);
        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::load_matrix_sync(qf, Qs + w * 16 * HD + dc, HD);
            wmma::load_matrix_sync(kf, Ksh + dc, HD);
            wmma::mma_sync(sacc, qf, kf, sacc);
        }
        wmma::store_matrix_sync(Ssh[w], sacc, BN, wmma::mem_row_major);
        __syncwarp();

        if (lane < 16) {
            int i = w * 16 + lane, qpos = q0 + i;
            float rowmax = -INFINITY;
            for (int j = 0; j < BN; ++j) {
                int kj = k0 + j;
                bool valid = (qpos < Sq) && kj < Sk;
                if (causal && kj > qpos + (Sk - Sq)) valid = false;
                float s;
                if (valid) {
                    s = Ssh[w][lane * BN + j] * scale;
                    if (mask)
                        s += __bfloat162float(mask[(int64_t)bi * ms_b + (int64_t)qh * ms_h +
                                                   (int64_t)qpos * ms_q + (int64_t)kj * ms_k]);
                } else {
                    s = -INFINITY;   // masked: sentinel only; never fed to a matmul (see header invariant)
                }
                Ssh[w][lane * BN + j] = s;
                rowmax = fmaxf(rowmax, s);   // fmax(-inf,x)=x -- skirtable like kfonline_softmax
            }
            float mnew = fmaxf(Msh[i], rowmax);
            float corr = __expf(Msh[i] - mnew);
            if (!isfinite(corr)) corr = 0.0f;
            float rowsum = 0.0f;
            for (int j = 0; j < BN; ++j) {
                float sv = Ssh[w][lane * BN + j];
                // Select-to-zero: masked prob is the literal 0.0, NOT exp(-inf),
                // so P@V takes a finite operand (see header invariant).
                float p = (sv == -INFINITY) ? 0.0f : __expf(sv - mnew);
                Psh[w][lane * BN + j] = __float2bfloat16(p);
                rowsum += p;
            }
            Lsh[i] = Lsh[i] * corr + rowsum;
            cwsh[i] = corr;
            Msh[i] = mnew;
        }
        __syncwarp();

        for (int idx = lane; idx < 16 * HD; idx += WARP)
            Osh[(w * 16 + idx / HD) * HD + idx % HD] *= cwsh[w * 16 + idx / HD];
        __syncwarp();

        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pvacc;
            wmma::fill_fragment(pvacc, 0.0f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
            wmma::load_matrix_sync(pf, Psh[w], BN);
            wmma::load_matrix_sync(vf, Vsh + dc, HD);
            wmma::mma_sync(pvacc, pf, vf, pvacc);
            wmma::store_matrix_sync(PVc[w], pvacc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16 * 16; idx += WARP) {
                int i = idx / 16, jj = idx % 16;
                Osh[(w * 16 + i) * HD + dc + jj] += PVc[w][idx];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    if (lane < 16) {
        int i = w * 16 + lane, qpos = q0 + i;
        if (qpos < Sq) {
            float inv = (Lsh[i] > 0.0f) ? (1.0f / Lsh[i]) : 0.0f;
            for (int d = 0; d < D; ++d)
                out[((((int64_t)bi * Hq + qh) * Sq) + qpos) * D + d] =
                    __float2bfloat16(Osh[i * HD + d] * inv);
        }
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
        dim3 grid((Sq + P_QROWS - 1) / P_QROWS, Hq, B);
        tc_flash_attn_prefill_kernel<<<grid, P_NWARPS * WARP, 0, stream>>>(
            q, k, v, mask, out, B, Hq, Hkv, Sq, Sk, D, scale, causal,
            ms_b, ms_h, ms_q, ms_k);
        return;
    }
    const int group = Hq / Hkv;
    const int R = group * Sq;
    dim3 grid((R + BM - 1) / BM, Hkv, B);
    tc_flash_attn_kernel<<<grid, NWARPS * WARP, 0, stream>>>(
        q, k, v, mask, out, B, Hq, Hkv, Sq, Sk, D, scale, causal,
        ms_b, ms_h, ms_q, ms_k);
}
