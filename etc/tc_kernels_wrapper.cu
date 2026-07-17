// PyTorch glue exposing the two tensor-core reference kernels as a single
// extension: `linear` (F.linear drop-in) and `sdpa`
// (F.scaled_dot_product_attention drop-in). Inputs are made contiguous, outputs
// are allocated in bf16, and the current CUDA stream is used so the calls slot
// straight into the golden pipeline.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "tc_kernels.h"

static inline const __nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());
}
static inline __nv_bfloat16* bf16_ptr_mut(torch::Tensor& t) {
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

static inline const half* half_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const half*>(t.data_ptr());
}
static inline half* half_ptr_mut(torch::Tensor& t) {
    return reinterpret_cast<half*>(t.data_ptr());
}

// C = A @ B, a plain tensor-core matmul (no bias, no transpose) backed by the
// Kuiper TensorCore2D kernel. A:(M,K) B:(K,N) C:(M,N), all fp16 row-major. To
// reproduce torch.nn.functional.linear(A, W) the caller passes B = W^T (K,N).
torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "tc2d matmul expects fp16 A/B");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.size(1) must equal B.size(0)");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    const int64_t M = Ac.size(0);
    const int64_t K = Ac.size(1);
    const int64_t N = Bc.size(1);
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0,
                "tc2d matmul requires M%128==0, N%128==0, K%32==0");

    auto C = torch::empty({M, N}, Ac.options());

    at::cuda::CUDAGuard g(A.device());
    tc2d_matmul_launch(half_ptr(Ac), half_ptr(Bc), half_ptr_mut(C),
                       (int)M, (int)N, (int)K);
    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

// FlashAttention forward. q:(B,Hq,Sq,D) k/v:(B,Hkv,Sk,D); optional 4D additive
// mask broadcast over (B,Hq,Sq,Sk). Mirrors F.scaled_dot_product_attention with
// enable_gqa=True.
torch::Tensor sdpa(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                   c10::optional<torch::Tensor> mask, double scale, bool causal,
                   bool force_decode_kernel) {
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16 &&
                    v.dtype() == torch::kBFloat16, "tc sdpa expects bf16 q/k/v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");

    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    const int64_t B = qc.size(0), Hq = qc.size(1), Sq = qc.size(2), D = qc.size(3);
    const int64_t Hkv = kc.size(1), Sk = kc.size(2);
    TORCH_CHECK(D <= 64 && D % 16 == 0, "tc sdpa supports head_dim <= 64, multiple of 16");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv (GQA)");
    TORCH_CHECK(kc.size(3) == D && vc.size(3) == D, "k/v head_dim must match q");

    auto out = torch::empty({B, Hq, Sq, D}, qc.options());

    const __nv_bfloat16* mptr = nullptr;
    int64_t ms_b = 0, ms_h = 0, ms_q = 0, ms_k = 0;
    torch::Tensor mc;
    if (mask.has_value() && mask->defined()) {
        mc = mask->contiguous();
        TORCH_CHECK(mc.dtype() == torch::kBFloat16, "mask must be bf16");
        TORCH_CHECK(mc.dim() == 4, "mask must be 4D (broadcast over B,Hq,Sq,Sk)");
        auto bc = [&](int d, int64_t want) -> int64_t {
            TORCH_CHECK(mc.size(d) == want || mc.size(d) == 1, "mask dim not broadcastable");
            return mc.size(d) == 1 ? 0 : mc.stride(d);
        };
        ms_b = bc(0, B); ms_h = bc(1, Hq); ms_q = bc(2, Sq); ms_k = bc(3, Sk);
        mptr = bf16_ptr(mc);
    }

    at::cuda::CUDAGuard g(q.device());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    tc_flash_attn_launch(bf16_ptr(qc), bf16_ptr(kc), bf16_ptr(vc), mptr,
                         bf16_ptr_mut(out), (int)B, (int)Hq, (int)Hkv, (int)Sq,
                         (int)Sk, (int)D, (float)scale, causal,
                         ms_b, ms_h, ms_q, ms_k, force_decode_kernel, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "fp16 Kuiper TensorCore2D matmul (CUDA)",
          py::arg("A"), py::arg("B"));
    m.def("sdpa", &sdpa, "bf16 tensor-core flash attention (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask") = c10::nullopt,
          py::arg("scale"), py::arg("causal"), py::arg("force_decode_kernel") = false);
}
