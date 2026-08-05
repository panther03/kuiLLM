#ifndef GEMM_BCAST_BIAS_EPILOGUE_H
#define GEMM_BCAST_BIAS_EPILOGUE_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// gC := beta*gBias[col] + alpha*(gA @ gB), with gBias a length-`cols` row vector.
void
gemm_bcast_bias_epilogue_launch(float alpha, float beta, uint32_t rows,
                                uint32_t cols, uint32_t shared, half *gA,
                                half *gB, half *gBias, half *gC, cudaStream_t s);

#define GEMM_BCAST_BIAS_EPILOGUE_H_DEFINED
#endif                          /* GEMM_BCAST_BIAS_EPILOGUE_H */
