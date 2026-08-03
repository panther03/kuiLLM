#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

void gemm_hacky_epilogue_launch(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    float alpha, float beta);

void gemm_pipe_launch(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

void flash_attn_fa1_launch(
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

void flash_attn_fa2_launch(
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