// Plain-CUDA FlashAttention (varlen, GQA, causal) shaped to match
// flash_attn._flash_attn_varlen_forward as observed in kernel_call.log:
//
//   q   : (total_q, n_heads_q,  head_dim)  bf16
//   k,v : (total_k, n_heads_kv, head_dim)  bf16   (GQA: n_heads_q % n_heads_kv == 0)
//   cu_seqlens_q / cu_seqlens_k : (batch+1,) int32   (packed / ragged layout)
//   out : (total_q, n_heads_q, head_dim)   bf16
//   lse : (n_heads_q, total_q)             f32   (softmax log-sum-exp)
//
// Accumulation is done in fp32; scores are scaled by `softmax_scale`. Causal
// masking uses the FlashAttention bottom-right alignment convention.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

// Launch the forward kernel on `stream`. All tensor pointers are device
// pointers; the caller owns allocation/lifetime. `head_dim <= 256`.
void flash_attn_varlen_fwd_launch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const int32_t* cu_seqlens_q,
    const int32_t* cu_seqlens_k,
    __nv_bfloat16* out,
    float* lse,
    int batch,
    int total_q,
    int n_heads_q,
    int n_heads_kv,
    int head_dim,
    float softmax_scale,
    bool causal,
    cudaStream_t stream);
