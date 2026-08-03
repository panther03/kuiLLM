
#ifndef FLASH_ATTN_MANUAL_EXTRACT_H
#define FLASH_ATTN_MANUAL_EXTRACT_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

void
flash_attn_manual_extract_launch(uint32_t nblk,
                                           uint32_t nw,
                                           uint32_t nthr,
                                           uint32_t b,
                                           uint32_t hq,
                                           uint32_t hkv,
                                           uint32_t group,
                                           uint32_t sq,
                                           uint32_t rows,
                                           uint32_t tiles,
                                           uint32_t sk,
                                           uint32_t d,
                                           __nv_bfloat16 * gQ,
                                           __nv_bfloat16 * gK,
                                           __nv_bfloat16 * gV,
                                           __nv_bfloat16 * gmask,
                                           __nv_bfloat16 * gout,
                                           bool causal,
                                           float scale, cudaStream_t s);

#define FLASH_ATTN_MANUAL_EXTRACT_H_DEFINED
#endif                          /* FLASH_ATTN_MANUAL_EXTRACT_H */
