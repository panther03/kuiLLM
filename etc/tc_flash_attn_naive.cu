// ARCHIVED naive reference (superseded by the optimized tc_flash_attn.cu). Kept for
// correctness comparison; not compiled into the extension.
// FlashAttention forward with bf16 tensor cores, drop-in for the cuDNN
// flash_fprop SDPA in the trace (F.scaled_dot_product_attention). See
// tc_kernels.h for the operator shape.
//
// One block = one warp and owns a 16-row query tile for a single (batch, query
// head). It streams the keys/values in 16-row tiles, keeping the online-softmax
// state (running max `m`, denominator `l`, output accumulator `Osh`) in shared
// memory. The two matmuls run on tensor cores: S = Q@K^T (16x16 per key tile,
// contracted over D in 16-wide chunks) and P@V (accumulated into Osh). The
// softmax reduction/rescale happens in shared memory between the two matmuls,
// which sidesteps the opaque WMMA fragment element layout. Un-tuned reference.
#include "tc_kernels.h"
#include <mma.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace nvcuda;

#define WARP 32
#define BM 16
#define BN 16
#define HDMAX 64                 // max head_dim (Qwen2.5 uses 64)

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
    const int qt = blockIdx.x;                 // query tile
    const int hq = blockIdx.y;                 // query head
    const int bi = blockIdx.z;                 // batch
    const int lane = threadIdx.x;              // 0..31
    const int q0 = qt * BM;
    const int hk = hq / (Hq / Hkv);            // GQA-mapped kv head

    const __nv_bfloat16* qbase = q + (((int64_t)bi * Hq + hq) * Sq) * D;
    const __nv_bfloat16* kbase = k + (((int64_t)bi * Hkv + hk) * Sk) * D;
    const __nv_bfloat16* vbase = v + (((int64_t)bi * Hkv + hk) * Sk) * D;
    __nv_bfloat16* obase = out + (((int64_t)bi * Hq + hq) * Sq) * D;

    __shared__ __nv_bfloat16 Qs[BM * HDMAX];
    __shared__ __nv_bfloat16 Ks[BN * HDMAX];
    __shared__ __nv_bfloat16 Vs[BN * HDMAX];
    __shared__ float Ss[BM * BN];
    __shared__ __nv_bfloat16 Ps[BM * BN];
    __shared__ float PV[BM * HDMAX];
    __shared__ float Osh[BM * HDMAX];
    __shared__ float m_[BM], l_[BM], corr_[BM];

    for (int idx = lane; idx < BM * D; idx += WARP) {
        int i = idx / D, d = idx % D;
        Qs[i * HDMAX + d] = (q0 + i < Sq) ? qbase[(int64_t)(q0 + i) * D + d]
                                          : __float2bfloat16(0.0f);
    }
    for (int idx = lane; idx < BM * HDMAX; idx += WARP) Osh[idx] = 0.0f;
    if (lane < BM) { m_[lane] = -INFINITY; l_[lane] = 0.0f; }
    __syncthreads();

    const int nkt = (Sk + BN - 1) / BN;
    for (int jt = 0; jt < nkt; ++jt) {
        const int k0 = jt * BN;
        for (int idx = lane; idx < BN * D; idx += WARP) {
            int j = idx / D, d = idx % D;
            bool ok = (k0 + j) < Sk;
            Ks[j * HDMAX + d] = ok ? kbase[(int64_t)(k0 + j) * D + d] : __float2bfloat16(0.0f);
            Vs[j * HDMAX + d] = ok ? vbase[(int64_t)(k0 + j) * D + d] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // S = Q @ K^T  (unscaled), contracted over D in 16-wide chunks.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
        wmma::fill_fragment(sacc, 0.0f);
        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::load_matrix_sync(qf, Qs + dc, HDMAX);
            wmma::load_matrix_sync(kf, Ks + dc, HDMAX);
            wmma::mma_sync(sacc, qf, kf, sacc);
        }
        wmma::store_matrix_sync(Ss, sacc, BN, wmma::mem_row_major);
        __syncthreads();

        // Online softmax update (one lane per query row).
        if (lane < BM) {
            int i = lane;
            int qi = q0 + i;
            float rowmax = -INFINITY;
            for (int j = 0; j < BN; ++j) {
                int kj = k0 + j;
                bool valid = kj < Sk;
                if (causal && kj > qi + (Sk - Sq)) valid = false;
                float s;
                if (valid) {
                    s = Ss[i * BN + j] * scale;
                    if (mask)
                        s += __bfloat162float(mask[(int64_t)bi * ms_b + (int64_t)hq * ms_h +
                                                   (int64_t)qi * ms_q + (int64_t)kj * ms_k]);
                } else {
                    s = -INFINITY;
                }
                Ss[i * BN + j] = s;
                rowmax = fmaxf(rowmax, s);
            }
            float mnew = fmaxf(m_[i], rowmax);
            float corr = __expf(m_[i] - mnew);
            if (!isfinite(corr)) corr = 0.0f;
            float rowsum = 0.0f;
            for (int j = 0; j < BN; ++j) {
                float sv = Ss[i * BN + j];
                float p = (sv == -INFINITY) ? 0.0f : __expf(sv - mnew);
                Ps[i * BN + j] = __float2bfloat16(p);
                rowsum += p;
            }
            l_[i] = l_[i] * corr + rowsum;
            corr_[i] = corr;
            m_[i] = mnew;
        }
        __syncthreads();

        for (int idx = lane; idx < BM * D; idx += WARP) {
            int i = idx / D, d = idx % D;
            Osh[i * HDMAX + d] *= corr_[i];
        }

        // PV = P @ V, accumulated into the rescaled Osh.
        for (int dc = 0; dc < D; dc += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pvacc;
            wmma::fill_fragment(pvacc, 0.0f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
            wmma::load_matrix_sync(pf, Ps, BN);
            wmma::load_matrix_sync(vf, Vs + dc, HDMAX);
            wmma::mma_sync(pvacc, pf, vf, pvacc);
            wmma::store_matrix_sync(PV + dc, pvacc, HDMAX, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = lane; idx < BM * D; idx += WARP) {
            int i = idx / D, d = idx % D;
            Osh[i * HDMAX + d] += PV[i * HDMAX + d];
        }
        __syncthreads();
    }

    if (lane < BM) {
        int i = lane;
        int qi = q0 + i;
        if (qi < Sq) {
            float inv = (l_[i] > 0.0f) ? (1.0f / l_[i]) : 0.0f;
            for (int d = 0; d < D; ++d)
                obase[(int64_t)qi * D + d] = __float2bfloat16(Osh[i * HDMAX + d] * inv);
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
    dim3 grid((Sq + BM - 1) / BM, Hq, B);
    tc_flash_attn_kernel<<<grid, WARP, 0, stream>>>(
        q, k, v, mask, out, B, Hq, Hkv, Sq, Sk, D, scale, causal,
        ms_b, ms_h, ms_q, ms_k);
}
