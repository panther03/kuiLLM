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
#include <cstdint>

// C = A @ W^T (+ bias), i.e. torch.nn.functional.linear semantics.
//   A    : (M, K)        bf16   row-major
//   W    : (N, K)        bf16   row-major (the transposed weight, as in F.linear)
//   bias : (N,)          bf16   or nullptr
//   C    : (M, N)        bf16   row-major
// Contraction accumulates in fp32; the output is rounded to bf16. When the
// output tile grid is too small to fill the GPU but K is large (e.g. down_proj),
// the launcher splits K across gridDim.z: each slice atomic-adds its partial
// product into `workspace` (M*N fp32), which a finalize pass casts to bf16 (+bias).
// `workspace` may be null iff tc_linear_splitk(M,N,K) == 1. Pass no_splitk=true
// to force the single-pass (split=1) kernel regardless of the heuristic.
void tc_linear_launch(
    const __nv_bfloat16* A,
    const __nv_bfloat16* W,
    const __nv_bfloat16* bias,   // nullable
    __nv_bfloat16* C,
    float* workspace,            // nullable when splitk == 1
    int M, int N, int K,
    bool no_splitk,
    cudaStream_t stream);

// GEMM block tile (kept in sync with tc_linear.cu) and the split-K heuristic,
// shared so the wrapper can size/zero the fp32 workspace.
#define TC_GEMM_BM 128
#define TC_GEMM_BN 64
#define TC_GEMM_BK 32

inline int tc_linear_splitk(int M, int N, int K) {
    long base = (long)((M + TC_GEMM_BM - 1) / TC_GEMM_BM) *
                ((N + TC_GEMM_BN - 1) / TC_GEMM_BN);
    if (base <= 0) return 1;
    if (base >= 84) return 1;                 // enough blocks to fill the SMs
    int s = (int)((168 + base - 1) / base);    // aim for ~2 waves of blocks
    int maxs = K / TC_GEMM_BK;
    if (maxs < 1) maxs = 1;
    if (s > maxs) s = maxs;
    if (s > 8) s = 8;
    if (s < 1) s = 1;
    return s;
}

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
