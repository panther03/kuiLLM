// JIT family wrapper: verified Kuiper elementwise kernels (Klas.Elementwise).
#include <cuda_bf16.h>
#include "common.h"
#include "Klas_Elementwise.h"

using kuiper_jit::sync_current_stream;

namespace {
constexpr int64_t kMaxNumel = 2097152LL * 1024LL;

inline torch::Tensor clone_in(const torch::Tensor& X) { return X.contiguous().clone(); }
}  // namespace

torch::Tensor silu_bf16(torch::Tensor X) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_silu_fw_bf16((uint32_t)Y.numel(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()));
    return Y;
}
torch::Tensor neg_bf16(torch::Tensor X) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_neg_fw_bf16((uint32_t)Y.numel(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()));
    return Y;
}
torch::Tensor rsqrt_f32(torch::Tensor X) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_rsqrt_fw_f32((uint32_t)Y.numel(), reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}
torch::Tensor square_f32(torch::Tensor X) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_square_fw_f32((uint32_t)Y.numel(), reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}
torch::Tensor cos_f32(torch::Tensor X) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_cos_fw_f32((uint32_t)Y.numel(), reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}
torch::Tensor sin_f32(torch::Tensor X) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_sin_fw_f32((uint32_t)Y.numel(), reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}
torch::Tensor add_bf16(torch::Tensor A, torch::Tensor B) {
    auto Y = clone_in(A); auto Bc = B.contiguous();
    at::cuda::CUDAGuard g(A.device()); sync_current_stream();
    Klas_Elementwise_add_fw_bf16((uint32_t)Y.numel(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()));
    return Y;
}
torch::Tensor mul_bf16(torch::Tensor A, torch::Tensor B) {
    auto Y = clone_in(A); auto Bc = B.contiguous();
    at::cuda::CUDAGuard g(A.device()); sync_current_stream();
    Klas_Elementwise_mul_fw_bf16((uint32_t)Y.numel(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()));
    return Y;
}
torch::Tensor mul_f32(torch::Tensor A, torch::Tensor B) {
    auto Y = clone_in(A); auto Bc = B.contiguous();
    at::cuda::CUDAGuard g(A.device()); sync_current_stream();
    Klas_Elementwise_mul_fw_f32((uint32_t)Y.numel(),
        reinterpret_cast<float*>(Y.data_ptr()), reinterpret_cast<float*>(Bc.data_ptr()));
    return Y;
}
torch::Tensor add_const_f32(torch::Tensor X, double c) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_add_const_fw_f32((float)c, (uint32_t)Y.numel(),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}
torch::Tensor mul_const_f32(torch::Tensor X, double c) {
    auto Y = clone_in(X);
    at::cuda::CUDAGuard g(X.device()); sync_current_stream();
    Klas_Elementwise_mul_const_fw_f32((float)c, (uint32_t)Y.numel(),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_bf16", &silu_bf16);
    m.def("neg_bf16", &neg_bf16);
    m.def("rsqrt_f32", &rsqrt_f32);
    m.def("square_f32", &square_f32);
    m.def("cos_f32", &cos_f32);
    m.def("sin_f32", &sin_f32);
    m.def("add_bf16", &add_bf16);
    m.def("mul_bf16", &mul_bf16);
    m.def("mul_f32", &mul_f32);
    m.def("add_const_f32", &add_const_f32);
    m.def("mul_const_f32", &mul_const_f32);
}
