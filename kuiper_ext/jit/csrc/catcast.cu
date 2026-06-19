// JIT family wrapper: verified Kuiper concat + cast kernels (Klas.CatCast).
#include <cuda_bf16.h>
#include "common.h"
#include "Klas_CatCast.h"

using kuiper_jit::sync_current_stream;

namespace {
template <typename T>
torch::Tensor wrap(T* p, torch::IntArrayRef sizes, const torch::TensorOptions& opts) {
    TORCH_CHECK(p != nullptr, "Kuiper returned a null pointer");
    return torch::from_blob(p, sizes.vec(), [=](void* q) { cudaFree(q); }, opts);
}
}  // namespace

torch::Tensor cat2_bf16(torch::Tensor a, torch::Tensor b) {
    const int64_t la = a.size(0), lb = b.size(0);
    at::cuda::CUDAGuard g(a.device()); sync_current_stream();
    __nv_bfloat16* p = Klas_CatCast_cat2_bf16((uint32_t)la, (uint32_t)lb,
        reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(b.data_ptr()));
    return wrap(p, {la + lb}, a.options());
}
torch::Tensor cast_bf16_to_f32(torch::Tensor x) {
    at::cuda::CUDAGuard g(x.device()); sync_current_stream();
    float* p = Klas_CatCast_cast_bf16_to_f32((uint32_t)x.numel(),
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()));
    return wrap(p, x.sizes(), x.options().dtype(torch::kFloat32));
}
torch::Tensor cast_f32_to_bf16(torch::Tensor x) {
    at::cuda::CUDAGuard g(x.device()); sync_current_stream();
    __nv_bfloat16* p = Klas_CatCast_cast_f32_to_bf16((uint32_t)x.numel(),
        reinterpret_cast<float*>(x.data_ptr()));
    return wrap(p, x.sizes(), x.options().dtype(torch::kBFloat16));
}
torch::Tensor cast_bf16_to_bf16(torch::Tensor x) {
    at::cuda::CUDAGuard g(x.device()); sync_current_stream();
    __nv_bfloat16* p = Klas_CatCast_cast_bf16_to_bf16((uint32_t)x.numel(),
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()));
    return wrap(p, x.sizes(), x.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cat2_bf16", &cat2_bf16);
    m.def("cast_bf16_to_f32", &cast_bf16_to_f32);
    m.def("cast_f32_to_bf16", &cast_f32_to_bf16);
    m.def("cast_bf16_to_bf16", &cast_bf16_to_bf16);
}
