// CUDA FlashAttention forward (varlen + GQA + causal). See flash_attn.h for the
// operator shape. One warp computes one (query token, query head): the warp
// streams the key/value rows of that query's sequence, keeping the running
// max/sum/output accumulator (online softmax) in registers striped across lanes.
#include "flash_attn.h"
#include <cuda_runtime.h>
#include <math.h>

#define WARP 32
#define MAX_HEAD_DIM 256
#define ACC_PER_LANE (MAX_HEAD_DIM / WARP)  // registers per lane (head_dim/32)

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Largest b with cu[b] <= qg (i.e. the sequence owning packed query row qg).
__device__ __forceinline__ int find_seq(const int32_t* cu, int batch, int qg) {
    int lo = 0, hi = batch;  // answer in [0, batch)
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (cu[mid] <= qg) lo = mid; else hi = mid;
    }
    return lo;
}

__global__ void flash_attn_varlen_fwd_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int batch, int total_q, int n_heads_q, int n_heads_kv, int head_dim,
    float softmax_scale, bool causal)
{
    const int lane = threadIdx.x;              // 0..31
    const int qg   = blockIdx.x;               // packed query row
    const int h    = blockIdx.y;               // query head
    if (qg >= total_q) return;

    const int b = find_seq(cu_seqlens_q, batch, qg);
    const int q_start = cu_seqlens_q[b];
    const int seqlen_q = cu_seqlens_q[b + 1] - q_start;
    const int k_start = cu_seqlens_k[b];
    const int seqlen_k = cu_seqlens_k[b + 1] - k_start;

    const int qi = qg - q_start;                       // query pos in its seq
    const int group = n_heads_q / n_heads_kv;          // GQA fan-out
    const int hk = h / group;                          // mapped kv head

    // Highest key index this query may attend to (bottom-right causal align).
    int jmax = seqlen_k - 1;
    if (causal) {
        int bound = qi + (seqlen_k - seqlen_q);
        if (bound < jmax) jmax = bound;
    }

    // Load this query's head_dim vector, striped over the warp.
    const __nv_bfloat16* qrow = q + ((int64_t)qg * n_heads_q + h) * head_dim;
    float qreg[ACC_PER_LANE];
    float acc[ACC_PER_LANE];
#pragma unroll
    for (int t = 0; t < ACC_PER_LANE; ++t) {
        int d = lane + t * WARP;
        qreg[t] = (d < head_dim) ? __bfloat162float(qrow[d]) : 0.0f;
        acc[t]  = 0.0f;
    }

    float m = -INFINITY;   // running max
    float l = 0.0f;        // running denominator

    for (int j = 0; j <= jmax; ++j) {
        const int krow = k_start + j;
        const __nv_bfloat16* kptr = k + ((int64_t)krow * n_heads_kv + hk) * head_dim;
        float partial = 0.0f;
#pragma unroll
        for (int t = 0; t < ACC_PER_LANE; ++t) {
            int d = lane + t * WARP;
            if (d < head_dim) partial += qreg[t] * __bfloat162float(kptr[d]);
        }
        float s = warp_reduce_sum(partial) * softmax_scale;

        float m_new = fmaxf(m, s);
        float corr  = __expf(m - m_new);   // exp(-inf - -inf) handled as 0 below
        if (!isfinite(corr)) corr = 0.0f;  // first iter: m == -inf
        float p = __expf(s - m_new);

        const __nv_bfloat16* vptr = v + ((int64_t)krow * n_heads_kv + hk) * head_dim;
        l = l * corr + p;
#pragma unroll
        for (int t = 0; t < ACC_PER_LANE; ++t) {
            int d = lane + t * WARP;
            float vv = (d < head_dim) ? __bfloat162float(vptr[d]) : 0.0f;
            acc[t] = acc[t] * corr + p * vv;
        }
        m = m_new;
    }

    __nv_bfloat16* orow = out + ((int64_t)qg * n_heads_q + h) * head_dim;
    const bool masked = (jmax < 0) || (l == 0.0f);
    const float inv_l = masked ? 0.0f : (1.0f / l);
#pragma unroll
    for (int t = 0; t < ACC_PER_LANE; ++t) {
        int d = lane + t * WARP;
        if (d < head_dim) orow[d] = __float2bfloat16(acc[t] * inv_l);
    }
    if (lane == 0)
        lse[(int64_t)h * total_q + qg] = masked ? -INFINITY : (m + logf(l));
}

void flash_attn_varlen_fwd_launch(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const int32_t* cu_seqlens_q, const int32_t* cu_seqlens_k,
    __nv_bfloat16* out, float* lse,
    int batch, int total_q, int n_heads_q, int n_heads_kv, int head_dim,
    float softmax_scale, bool causal, cudaStream_t stream)
{
    if (total_q == 0) return;
    dim3 grid(total_q, n_heads_q);
    flash_attn_varlen_fwd_kernel<<<grid, WARP, 0, stream>>>(
        q, k, v, cu_seqlens_q, cu_seqlens_k, out, lse,
        batch, total_q, n_heads_q, n_heads_kv, head_dim, softmax_scale, causal);
}
