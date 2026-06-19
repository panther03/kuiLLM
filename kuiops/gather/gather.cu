// JIT family wrapper: verified Kuiper gather (Klas.Gather).
#include <cstdint>
#include <cuda_bf16.h>
#include "common.h"
#include "Klas_Gather.h"

using kuiper_jit::sync_current_stream;

// 2D gather along dim 0 (1D is treated as cols=1 by the caller).
torch::Tensor gather_bf16(torch::Tensor src, torch::Tensor idx) {
    const int64_t cols = src.dim() == 1 ? 1 : src.size(1);
    const int64_t rows_src = src.size(0);
    const int64_t rows_out = idx.size(0);
    auto out = torch::empty(idx.sizes(), src.options());
    at::cuda::CUDAGuard g(src.device());
    sync_current_stream();
    Klas_Gather_gather_bf16_u64_2d(
        (uint32_t)cols, (uint32_t)(rows_src * cols), (uint32_t)(rows_out * cols),
        reinterpret_cast<__nv_bfloat16*>(src.data_ptr()),
        reinterpret_cast<uint64_t*>(idx.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_bf16", &gather_bf16);
}
