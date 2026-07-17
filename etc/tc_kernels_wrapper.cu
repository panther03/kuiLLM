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
#include "tc_kernels.h"

static inline const __nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());
}
static inline __nv_bfloat16* bf16_ptr_mut(torch::Tensor& t) {
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

// C = A @ W^T (+ bias). A may have any leading dims; only the last is the
// contraction. Mirrors torch.nn.functional.linear.
torch::Tensor linear(torch::Tensor A, torch::Tensor W, c10::optional<torch::Tensor> bias,
                     bool no_splitk) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && W.dtype() == torch::kBFloat16,
                "tc linear expects bf16 A/W");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (N, K)");
    TORCH_CHECK(A.size(-1) == W.size(1), "A last dim must equal W.size(1)");

    auto Ac = A.contiguous();
    auto Wc = W.contiguous();
    const int64_t K = Wc.size(1);
    const int64_t N = Wc.size(0);
    const int64_t M = Ac.numel() / K;

    auto out_shape = A.sizes().vec();
    out_shape.back() = N;
    auto C = torch::empty(out_shape, Ac.options());

    const __nv_bfloat16* bptr = nullptr;
    torch::Tensor bc;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->dtype() == torch::kBFloat16, "bias must be bf16");
        TORCH_CHECK(bias->numel() == N, "bias must have N elements");
        bc = bias->contiguous();
        bptr = bf16_ptr(bc);
    }

    at::cuda::CUDAGuard g(A.device());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    torch::Tensor ws;
    float* wsptr = nullptr;
    if (!no_splitk && tc_linear_splitk((int)M, (int)N, (int)K) > 1) {
        ws = torch::empty({M * N}, Ac.options().dtype(torch::kFloat32));
        wsptr = ws.data_ptr<float>();
    }
    tc_linear_launch(bf16_ptr(Ac), bf16_ptr(Wc), bptr, bf16_ptr_mut(C), wsptr,
                     (int)M, (int)N, (int)K, no_splitk, stream);
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
    m.def("linear", &linear, "bf16 tensor-core linear (CUDA)",
          py::arg("A"), py::arg("W"), py::arg("bias") = c10::nullopt,
          py::arg("no_splitk") = false);
    m.def("sdpa", &sdpa, "bf16 tensor-core flash attention (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask") = c10::nullopt,
          py::arg("scale"), py::arg("causal"), py::arg("force_decode_kernel") = false);
}
