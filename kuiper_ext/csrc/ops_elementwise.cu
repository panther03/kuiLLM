// PyTorch wrappers around verified Kuiper elementwise kernels.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "Klas_Elementwise.h"

namespace {

constexpr int64_t kMaxElementwiseNumel = 2097152LL * 1024LL;

inline void sync_current_stream() {
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus capture_status;
    AT_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "Kuiper kernels are not safe for CUDA graph capture");
    AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
}

inline void check_unary(const torch::Tensor& X, c10::ScalarType dtype, const char* name) {
    TORCH_CHECK(X.is_cuda(), name, ": tensor must be CUDA");
    TORCH_CHECK(X.scalar_type() == dtype, name, ": wrong dtype");
    TORCH_CHECK(X.numel() > 0 && X.numel() <= kMaxElementwiseNumel,
                name, ": numel must be in (0, ", kMaxElementwiseNumel, "]");
}

inline void check_binary(const torch::Tensor& A, const torch::Tensor& B, c10::ScalarType dtype,
                         const char* name) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), name, ": tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == dtype && B.scalar_type() == dtype, name, ": wrong dtype");
    TORCH_CHECK(A.sizes() == B.sizes(), name, ": shapes must match");
    TORCH_CHECK(A.numel() > 0 && A.numel() <= kMaxElementwiseNumel,
                name, ": numel must be in (0, ", kMaxElementwiseNumel, "]");
}

} // namespace

torch::Tensor kuiper_silu_bf16(torch::Tensor X) {
    check_unary(X, torch::kBFloat16, "kuiper_silu_bf16");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_silu_fw_bf16(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()));
    return Y;
}

torch::Tensor kuiper_neg_bf16(torch::Tensor X) {
    check_unary(X, torch::kBFloat16, "kuiper_neg_bf16");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_neg_fw_bf16(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()));
    return Y;
}

torch::Tensor kuiper_rsqrt_f32(torch::Tensor X) {
    check_unary(X, torch::kFloat32, "kuiper_rsqrt_f32");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_rsqrt_fw_f32(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}

torch::Tensor kuiper_square_f32(torch::Tensor X) {
    check_unary(X, torch::kFloat32, "kuiper_square_f32");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_square_fw_f32(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}

torch::Tensor kuiper_cos_f32(torch::Tensor X) {
    check_unary(X, torch::kFloat32, "kuiper_cos_f32");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_cos_fw_f32(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}

torch::Tensor kuiper_sin_f32(torch::Tensor X) {
    check_unary(X, torch::kFloat32, "kuiper_sin_f32");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_sin_fw_f32(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}

torch::Tensor kuiper_add_bf16(torch::Tensor A, torch::Tensor B) {
    check_binary(A, B, torch::kBFloat16, "kuiper_add_bf16");
    auto Y = A.contiguous().clone();
    auto Bc = B.contiguous();
    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();
    Klas_Elementwise_add_fw_bf16(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()));
    return Y;
}

torch::Tensor kuiper_mul_bf16(torch::Tensor A, torch::Tensor B) {
    check_binary(A, B, torch::kBFloat16, "kuiper_mul_bf16");
    auto Y = A.contiguous().clone();
    auto Bc = B.contiguous();
    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();
    Klas_Elementwise_mul_fw_bf16(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Bc.data_ptr()));
    return Y;
}

torch::Tensor kuiper_mul_f32(torch::Tensor A, torch::Tensor B) {
    check_binary(A, B, torch::kFloat32, "kuiper_mul_f32");
    auto Y = A.contiguous().clone();
    auto Bc = B.contiguous();
    at::cuda::CUDAGuard guard(A.device());
    sync_current_stream();
    Klas_Elementwise_mul_fw_f32(static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<float*>(Y.data_ptr()), reinterpret_cast<float*>(Bc.data_ptr()));
    return Y;
}

torch::Tensor kuiper_add_const_f32(torch::Tensor X, double c) {
    check_unary(X, torch::kFloat32, "kuiper_add_const_f32");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_add_const_fw_f32(static_cast<float>(c), static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}

torch::Tensor kuiper_mul_const_f32(torch::Tensor X, double c) {
    check_unary(X, torch::kFloat32, "kuiper_mul_const_f32");
    auto Y = X.contiguous().clone();
    at::cuda::CUDAGuard guard(X.device());
    sync_current_stream();
    Klas_Elementwise_mul_const_fw_f32(static_cast<float>(c), static_cast<uint32_t>(Y.numel()),
        reinterpret_cast<float*>(Y.data_ptr()));
    return Y;
}

void register_elementwise(pybind11::module& m) {
    m.def("silu_bf16", &kuiper_silu_bf16, "Klas_Elementwise_silu_fw_bf16");
    m.def("neg_bf16", &kuiper_neg_bf16, "Klas_Elementwise_neg_fw_bf16");
    m.def("rsqrt_f32", &kuiper_rsqrt_f32, "Klas_Elementwise_rsqrt_fw_f32");
    m.def("square_f32", &kuiper_square_f32, "Klas_Elementwise_square_fw_f32");
    m.def("cos_f32", &kuiper_cos_f32, "Klas_Elementwise_cos_fw_f32");
    m.def("sin_f32", &kuiper_sin_f32, "Klas_Elementwise_sin_fw_f32");
    m.def("add_bf16", &kuiper_add_bf16, "Klas_Elementwise_add_fw_bf16");
    m.def("mul_bf16", &kuiper_mul_bf16, "Klas_Elementwise_mul_fw_bf16");
    m.def("mul_f32", &kuiper_mul_f32, "Klas_Elementwise_mul_fw_f32");
    m.def("add_const_f32", &kuiper_add_const_f32, "Klas_Elementwise_add_const_fw_f32");
    m.def("mul_const_f32", &kuiper_mul_const_f32, "Klas_Elementwise_mul_const_fw_f32");
}
