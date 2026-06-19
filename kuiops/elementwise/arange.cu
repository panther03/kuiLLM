// JIT family wrapper: verified Kuiper arange (Klas.Arange).
#include <cstdint>
#include "common.h"
#include "Klas_Arange.h"

using kuiper_jit::sync_current_stream;

torch::Tensor arange_i64(int64_t n, int64_t start, int64_t step) {
    auto out = torch::empty({n},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
    at::cuda::CUDAGuard g(out.device());
    sync_current_stream();
    uint64_t* p = Klas_Arange_arange_i64((uint32_t)n, (uint64_t)start, (uint64_t)step);
    TORCH_CHECK(p != nullptr, "Klas_Arange_arange_i64 returned null");
    AT_CUDA_CHECK(cudaMemcpy(out.data_ptr(), p, n * sizeof(int64_t), cudaMemcpyDeviceToDevice));
    AT_CUDA_CHECK(cudaFree(p));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("arange_i64", &arange_i64);
}
