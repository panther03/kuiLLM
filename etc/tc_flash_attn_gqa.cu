// SNAPSHOT (not compiled): GQA-batched key-split flash attention. Decode 2.3-3.0x
// of cuDNN; prefill ~15-17x. Kept as a reference point.
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
                    s = -INFINITY;
                }
                Ssh[w][i * BN + j] = s;
                rowmax = fmaxf(rowmax, s);
            }
            float mnew = fmaxf(Msh[w][i], rowmax);
            float corr = __expf(Msh[w][i] - mnew);
            if (!isfinite(corr)) corr = 0.0f;
            float rowsum = 0.0f;
            for (int j = 0; j < BN; ++j) {
                float sv = Ssh[w][i * BN + j];
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

void tc_flash_attn_launch(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* mask, __nv_bfloat16* out,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal,
    int64_t ms_b, int64_t ms_h, int64_t ms_q, int64_t ms_k,
    cudaStream_t stream)
{
    if (B == 0 || Sq == 0) return;
    const int group = Hq / Hkv;
    const int R = group * Sq;
    dim3 grid((R + BM - 1) / BM, Hkv, B);
    tc_flash_attn_kernel<<<grid, NWARPS * WARP, 0, stream>>>(
        q, k, v, mask, out, B, Hq, Hkv, Sq, Sk, D, scale, causal,
        ms_b, ms_h, ms_q, ms_k);
}
