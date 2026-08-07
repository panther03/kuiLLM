#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

void gemm_hacky_epilogue_launch(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

void gemm_pipe_launch(const half* A, const half* B, half* C, int M, int N, int K,
                      float alpha, float beta, cudaStream_t stream);

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
// ---------------------------------------------------------------------------
// Templated tensor-core GEMM (gemm_tc.cuh / gemm_tc.cu). Tilings are registered
// in a table; `config` indexes it and `gemm_tc_config_info` publishes each
// entry as {bm, bn, bk, wm, wn, stages, skew, warps, smem_bytes} so a caller
// can tell which tilings a shape admits.
// ---------------------------------------------------------------------------
int gemm_tc_num_configs();
void gemm_tc_config_info(int index, int* out /* [9] */);
void gemm_tc_launch(bool bf16, int config, const void* A, const void* B,
                    const void* C, void* D, float* workspace, int M, int N,
                    int K, float alpha, float beta, int splits, int group,
                    int epi, cudaStream_t stream);
