
#ifndef Kuiops_Sdpa_Flash_Inst_H
#define Kuiops_Sdpa_Flash_Inst_H

#include <kuiper.h>
#include <kuiops_compat.h>

void
Kuiops_Sdpa_Flash_Inst_sdpa_flash_bf16_f32(uint32_t nblk,
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

#define Kuiops_Sdpa_Flash_Inst_H_DEFINED
#endif                          /* Kuiops_Sdpa_Flash_Inst_H */
