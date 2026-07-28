// Reference CUDA implementations of the two kernel families that dominate the
// infer_golden.py nsys trace, written as drop-in replacements for the stock
// kernels PyTorch dispatches to:
//
//   * the bf16 tensor-core GEMMs (F.linear -> cuBLAS ampere_bf16_s*gemm_*), and
//   * the fused cuDNN FlashAttention forward
//     (F.scaled_dot_product_attention -> cudnn_*_sdpa_*_flash_fprop_wmma_*).
//
// Both use bf16 Ampere tensor cores (16x16x16 WMMA, fp32 accumulate). They are
// deliberately un-tuned (fixed small tiles, no swizzle/latest-arch features) --
// the goal is a correct, tensor-core, flash-style reference to later verify a
// Kuiper implementation against, not to match cuDNN/cuBLAS throughput.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// Hand-written mixed-precision tensor-core GEMM (see tc2d_linear_manual.cu):
// bf16 in/out, fp32 accumulate, with a real epilogue C = alpha*(A @ B) + beta*C
// applied in place. Row-major, no transpose, so to reproduce F.linear(x, W) the
// caller passes B = W^T (K, N).
//   A : (M, K)  bf16  row-major
//   B : (K, N)  bf16  row-major
//   C : (M, N)  bf16  row-major, read as the additive term when beta != 0 and
//                     overwritten with the result
// Requires M % 128 == 0, N % 128 == 0, K % 32 == 0.
void tc2d_manual_gemm_launch(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    float alpha, float beta);

// FlashAttention forward matching F.scaled_dot_product_attention (bf16, GQA,
// optional additive mask, optional causal). Layout is the dense 4D SDPA layout:
//   q   : (B, Hq,  Sq, D)  bf16
//   k,v : (B, Hkv, Sk, D)  bf16   (GQA: Hq % Hkv == 0)
//   out : (B, Hq,  Sq, D)  bf16
// `mask` (nullable) is an additive bias broadcast over (B, Hq, Sq, Sk) via the
// four element strides `ms_*` (a stride of 0 broadcasts that dim). Online
// softmax keeps the running max / denominator / output accumulator in shared
// memory; the two matmuls (QK^T and P@V) run on tensor cores. D <= 64, D % 16 == 0.
// Sq > 1 (prefill) uses a dedicated query-tiled kernel; pass force_decode_kernel
// =true to always use the GQA-batched key-split decode kernel instead.
void tc_flash_attn_launch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* mask,   // nullable
    __nv_bfloat16* out,
    int B, int Hq, int Hkv, int Sq, int Sk, int D,
    float scale, bool causal,
    int64_t ms_b, int64_t ms_h, int64_t ms_q, int64_t ms_k,
    bool force_decode_kernel,
    cudaStream_t stream);
