// JIT family wrapper: verified Kuiper mean reducer (Klas.Reduce).
#include <cstdint>
#include "common.h"
#include "Klas_Reduce.h"

using kuiper_jit::sync_current_stream;

// Klas_Reduce_mean is a 1D scalar reducer: float mean(uint32_t n, float* a).
// For reduce-along-last-dim of a [rows, cols] tensor we loop per row.
torch::Tensor mean_f32_lastdim(torch::Tensor X) {
    const int64_t rows = X.size(0), cols = X.size(1);
    auto Y = torch::empty({rows}, X.options());
    at::cuda::CUDAGuard g(X.device());
    sync_current_stream();
    float* x = reinterpret_cast<float*>(X.data_ptr());
    float* y = reinterpret_cast<float*>(Y.data_ptr());
    for (int64_t r = 0; r < rows; ++r) {
        float s = Klas_Reduce_mean((uint32_t)cols, x + r * cols);
        AT_CUDA_CHECK(cudaMemcpy(y + r, &s, sizeof(float), cudaMemcpyHostToDevice));
    }
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_f32_lastdim", &mean_f32_lastdim);
}
