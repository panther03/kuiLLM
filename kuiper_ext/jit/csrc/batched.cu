// JIT family wrapper: verified Kuiper batched GEMM (Klas.GEMM.Batched).
#include <cstdint>
#include "common.h"
#include "Klas_GEMM_Batched.h"

using kuiper_jit::sync_current_stream;

torch::Tensor bmm_f32(torch::Tensor A, torch::Tensor B) {
    const int64_t batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    at::cuda::CUDAGuard g(A.device());
    sync_current_stream();
    float* p = Klas_GEMM_Batched_batched_matmul_f32(
        (uint32_t)batch, (uint32_t)M, (uint32_t)K, (uint32_t)N,
        reinterpret_cast<float*>(Ac.data_ptr()),
        reinterpret_cast<float*>(Bc.data_ptr()));
    TORCH_CHECK(p != nullptr, "Klas_GEMM_Batched_batched_matmul_f32 returned null");
    return torch::from_blob(p, {batch, M, N},
        [=](void* q) { cudaFree(q); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_f32", &bmm_f32);
}
