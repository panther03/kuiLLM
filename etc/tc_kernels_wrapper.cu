// PyTorch glue exposing the tensor-core reference kernels as a single
// extension: `gemm_manual` (the hand-written TensorCore2D GEMM) and `sdpa`
// (F.scaled_dot_product_attention drop-in). Inputs are made contiguous, outputs
// are allocated in bf16. SDPA uses the current CUDA stream; the extracted Kuiper
// GEMM manages and synchronizes its own stream.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "tc_kernels.h"
#include "Kuiops_Sdpa_Flash_Inst.h"

static inline const __nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());
}
static inline __nv_bfloat16* bf16_ptr_mut(torch::Tensor& t) {
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr());
}

// C = alpha*(A @ B) + beta*C_in, the hand-written tensor-core GEMM epilogue
// (fp32 accumulate, bf16 in/out) from tc2d_linear_manual.cu. A:(M,K) B:(K,N)
// row-major; the kernel works in place, so C_in is copied into the output
// (broadcast rules are the caller's responsibility -- C_in must be (M,N)).
torch::Tensor gemm_manual(torch::Tensor A, torch::Tensor B,
                          c10::optional<torch::Tensor> C_in,
                          double alpha, double beta) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16,
                "tc2d manual gemm expects bf16 A/B");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.size(1) must equal B.size(0)");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    const int64_t M = Ac.size(0);
    const int64_t K = Ac.size(1);
    const int64_t N = Bc.size(1);
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0,
                "tc2d manual gemm requires M%128==0, N%128==0, K%32==0");

    torch::Tensor C;
    if (beta != 0.0) {
        TORCH_CHECK(C_in.has_value() && C_in->defined(),
                    "beta != 0 requires a C tensor");
        TORCH_CHECK(C_in->dtype() == torch::kBFloat16, "C must be bf16");
        TORCH_CHECK(C_in->dim() == 2 && C_in->size(0) == M && C_in->size(1) == N,
                    "C must be (M, N)");
        C = C_in->contiguous().clone();
    } else {
        C = torch::empty({M, N}, Ac.options());
    }

    at::cuda::CUDAGuard g(A.device());
    tc2d_manual_gemm_launch(bf16_ptr(Ac), bf16_ptr(Bc), bf16_ptr_mut(C),
                            (int)M, (int)N, (int)K, (float)alpha, (float)beta);
    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

// FlashAttention forward. q:(B,Hq,Sq,D) k/v:(B,Hkv,Sk,D); optional 4D additive
// mask broadcast over (B,Hq,Sq,Sk). Mirrors F.scaled_dot_product_attention with
// enable_gqa=True.
static torch::Tensor sdpa_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                               c10::optional<torch::Tensor> mask, double scale,
                               bool causal, bool force_decode_kernel, bool fa1) {
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
    auto launch = fa1 ? tc_flash_attn_fa1_launch : tc_flash_attn_launch;
    launch(bf16_ptr(qc), bf16_ptr(kc), bf16_ptr(vc), mptr,
           bf16_ptr_mut(out), (int)B, (int)Hq, (int)Hkv, (int)Sq,
           (int)Sk, (int)D, (float)scale, causal,
           ms_b, ms_h, ms_q, ms_k, force_decode_kernel, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor sdpa(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                   c10::optional<torch::Tensor> mask, double scale, bool causal,
                   bool force_decode_kernel) {
    return sdpa_impl(q, k, v, mask, scale, causal, force_decode_kernel, false);
}

// Same operator via the FA1-style reference kernel the Kuiper port targets.
torch::Tensor sdpa_fa1(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                       c10::optional<torch::Tensor> mask, double scale,
                       bool causal, bool force_decode_kernel) {
    return sdpa_impl(q, k, v, mask, scale, causal, force_decode_kernel, true);
}

// Same operator, but dispatched to the *verified* Kuiper FlashAttention decode
// kernel extracted from Kuiops.Sdpa.Flash.Inst. The Kuiper kernel is generic in
// the tensor layouts but this instance fixes row-major (B,Hq,Sq,D) q/out,
// (B,Hkv,Sk,D) k/v and a dense row-major (B,Hq,Sq,Sk) additive mask -- it has no
// broadcast layout (a tlayout must be injective), so the mask is required and
// must be materialised by the caller.
torch::Tensor sdpa_kuiper(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor mask, double scale, bool causal,
                          int64_t nwarps) {
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16 &&
                    v.dtype() == torch::kBFloat16 && mask.dtype() == torch::kBFloat16,
                "kuiper sdpa expects bf16 q/k/v/mask");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && mask.dim() == 4,
                "q/k/v/mask must be 4D");

    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous(),
         mc = mask.contiguous();
    const int64_t B = qc.size(0), Hq = qc.size(1), Sq = qc.size(2), D = qc.size(3);
    const int64_t Hkv = kc.size(1), Sk = kc.size(2);
    TORCH_CHECK(D % 16 == 0 && D > 0, "head_dim must be a positive multiple of 16");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be a multiple of Hkv (GQA)");
    TORCH_CHECK(kc.size(3) == D && vc.size(3) == D, "k/v head_dim must match q");
    TORCH_CHECK(mc.size(0) == B && mc.size(1) == Hq && mc.size(2) == Sq &&
                mc.size(3) == Sk, "mask must be dense (B,Hq,Sq,Sk)");
    TORCH_CHECK(Sq <= Sk, "Sq must be <= Sk");

    const int64_t group = Hq / Hkv;
    const int64_t rows = group * Sq;
    const int64_t tiles = (rows + 15) / 16;
    const int64_t nblk = B * Hkv * tiles;

    auto out = torch::empty({B, Hq, Sq, D}, qc.options());

    at::cuda::CUDAGuard g(q.device());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    Kuiops_Sdpa_Flash_Inst_sdpa_flash_bf16_f32(
        (uint32_t)nblk, (uint32_t)nwarps, (uint32_t)(nwarps * 32), (uint32_t)B,
        (uint32_t)Hq, (uint32_t)Hkv, (uint32_t)group, (uint32_t)Sq,
        (uint32_t)rows, (uint32_t)tiles, (uint32_t)Sk, (uint32_t)D,
        const_cast<__nv_bfloat16*>(bf16_ptr(qc)),
        const_cast<__nv_bfloat16*>(bf16_ptr(kc)),
        const_cast<__nv_bfloat16*>(bf16_ptr(vc)),
        const_cast<__nv_bfloat16*>(bf16_ptr(mc)),
        bf16_ptr_mut(out), causal, (float)scale, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_manual", &gemm_manual,
          "hand-written bf16 in/out, fp32-accumulate tensor-core GEMM "
          "D = alpha*(A@B) + beta*C (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("C") = c10::nullopt,
          py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    m.def("sdpa", &sdpa, "bf16 tensor-core flash attention (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask") = c10::nullopt,
          py::arg("scale"), py::arg("causal"), py::arg("force_decode_kernel") = false);
    m.def("sdpa_fa1", &sdpa_fa1, "bf16 tensor-core flash attention, FA1 reference (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask") = c10::nullopt,
          py::arg("scale"), py::arg("causal"), py::arg("force_decode_kernel") = false);
    m.def("sdpa_kuiper", &sdpa_kuiper,
          "verified Kuiper bf16 tensor-core flash attention (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask"),
          py::arg("scale"), py::arg("causal"), py::arg("nwarps") = 4);
}
