#ifndef GEMM_BCAST_BIAS_EPILOGUE2_H
#define GEMM_BCAST_BIAS_EPILOGUE2_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// gD := beta*gC[col] + alpha*(gA @ gB), with gC a length-`cols` row vector.
void
gemm_bcast_bias_epilogue2_launch(float alpha, float beta, uint32_t rows,
                                 uint32_t cols, uint32_t shared, half *gA,
                                 half *gB, half *gC, half *gD, cudaStream_t s);

#define GEMM_BCAST_BIAS_EPILOGUE2_H_DEFINED
#endif                          /* GEMM_BCAST_BIAS_EPILOGUE2_H */
