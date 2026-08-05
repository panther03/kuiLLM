// N.B: Try to compile this file with g++/clang instead of nvcc. At one point in time
// compiling torch/extension.h with nvcc was super slow, and my impression is it should be avoidable
// because we aren't launching any kernels here so we don't need the syntactic features of CUDA. If 
// there is truly a need to compile this file with nvcc, then go ahead and rename it to wrapper.cu. 
// Either way, once you figure it out, remove this comment.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "kernels.h"
#include "flash_attn_manual_extract.h"
#include "gemm_bcast_bias_epilogue.h"
#include "gemm_bcast_bias_epilogue2.h"

template <typename T>
static inline const T* cptr(const torch::Tensor& t) {
    return reinterpret_cast<const T*>(t.data_ptr());
}
template <typename T>
static inline T* mptr(torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

// ---------------------------------------------------------------------------
// GEMM: torch.addmm(input, mat1, mat2, *, beta, alpha) = beta*input + alpha*(mat1@mat2)
// ---------------------------------------------------------------------------

// The kernels accumulate into their output buffer in place, so the epilogue term
// is supplied by seeding the output with `input`.
static torch::Tensor gemm_out(const torch::Tensor& input, int64_t M, int64_t N,
                              double beta, torch::ScalarType dtype) {
    TORCH_CHECK(input.dtype() == dtype, "input dtype must match mat1/mat2");
    TORCH_CHECK(input.dim() == 2 && input.size(0) == M && input.size(1) == N,
                "input must be (M, N); broadcasting is not supported");
    return beta != 0.0 ? input.contiguous().clone()
                       : torch::empty({M, N}, input.options());
}

static void gemm_check(const torch::Tensor& A, const torch::Tensor& B,
                       torch::ScalarType dtype) {
    TORCH_CHECK(A.dtype() == dtype && B.dtype() == dtype, "unsupported dtype");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "mat1/mat2 must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "mat1.size(1) must equal mat2.size(0)");
    TORCH_CHECK(A.size(0) % 128 == 0 && B.size(1) % 128 == 0 && A.size(1) % 32 == 0,
                "requires M%128==0, N%128==0, K%32==0");
}

static torch::Tensor gemm_hacky_epilogue(torch::Tensor input, torch::Tensor A,
                                         torch::Tensor B, double beta, double alpha) {
    gemm_check(A, B, torch::kBFloat16);
    auto Ac = A.contiguous(), Bc = B.contiguous();
    const int64_t M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto out = gemm_out(input, M, N, beta, torch::kBFloat16);

    at::cuda::CUDAGuard g(A.device());
    gemm_hacky_epilogue_launch(cptr<__nv_bfloat16>(Ac), cptr<__nv_bfloat16>(Bc),
                               mptr<__nv_bfloat16>(out), (int)M, (int)N, (int)K,
                               (float)alpha, (float)beta,
                               c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

static torch::Tensor gemm_pipe(torch::Tensor input, torch::Tensor A,
                               torch::Tensor B, double beta, double alpha) {
    gemm_check(A, B, torch::kFloat16);
    auto Ac = A.contiguous(), Bc = B.contiguous();
    const int64_t M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto out = gemm_out(input, M, N, beta, torch::kFloat16);

    at::cuda::CUDAGuard g(A.device());
    gemm_pipe_launch(cptr<half>(Ac), cptr<half>(Bc), mptr<half>(out),
                     (int)M, (int)N, (int)K, (float)alpha, (float)beta,
                     c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

// The epilogue takes the bias as a length-N row vector and broadcasts it over
// the rows, so unlike the kernels above there is nothing to seed the output
// with: it is written, not accumulated into.
static torch::Tensor gemm_bcast_bias_epilogue(torch::Tensor input, torch::Tensor A,
                                              torch::Tensor B, double beta,
                                              double alpha) {
    gemm_check(A, B, torch::kFloat16);
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input dtype must match mat1/mat2");
    TORCH_CHECK(input.dim() == 1 && input.size(0) == B.size(1),
                "input must be a length-N row vector");
    auto Ac = A.contiguous(), Bc = B.contiguous(), bias = input.contiguous();
    const int64_t M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto out = torch::empty({M, N}, Ac.options());

    at::cuda::CUDAGuard g(A.device());
    gemm_bcast_bias_epilogue_launch((float)alpha, (float)beta, (uint32_t)M,
                                    (uint32_t)N, (uint32_t)K, mptr<half>(Ac),
                                    mptr<half>(Bc), mptr<half>(bias),
                                    mptr<half>(out),
                                    c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

static torch::Tensor gemm_bcast_bias_epilogue2(torch::Tensor input, torch::Tensor A,
                                               torch::Tensor B, double beta,
                                               double alpha) {
    gemm_check(A, B, torch::kFloat16);
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input dtype must match mat1/mat2");
    TORCH_CHECK(input.dim() == 1 && input.size(0) == B.size(1),
                "input must be a length-N row vector");
    auto Ac = A.contiguous(), Bc = B.contiguous(), bias = input.contiguous();
    const int64_t M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto out = torch::empty({M, N}, Ac.options());

    at::cuda::CUDAGuard g(A.device());
    gemm_bcast_bias_epilogue2_launch((float)alpha, (float)beta, (uint32_t)M,
                                     (uint32_t)N, (uint32_t)K, mptr<half>(Ac),
                                     mptr<half>(Bc), mptr<half>(bias),
                                     mptr<half>(out),
                                     c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
// SDPA: F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p,
//                                      is_causal, scale)
// ---------------------------------------------------------------------------

struct SdpaShape {
    int64_t B, Hq, Hkv, Sq, Sk, D;
    torch::Tensor q, k, v;
    double scale;
};

static SdpaShape sdpa_shape(const torch::Tensor& q, const torch::Tensor& k,
                            const torch::Tensor& v, double dropout_p,
                            c10::optional<double> scale) {
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16 &&
                    v.dtype() == torch::kBFloat16, "expects bf16 query/key/value");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "query/key/value must be 4D");
    TORCH_CHECK(dropout_p == 0.0, "dropout is not implemented");
    SdpaShape s;
    s.q = q.contiguous(); s.k = k.contiguous(); s.v = v.contiguous();
    s.B = s.q.size(0); s.Hq = s.q.size(1); s.Sq = s.q.size(2); s.D = s.q.size(3);
    s.Hkv = s.k.size(1); s.Sk = s.k.size(2);
    TORCH_CHECK(s.Hkv > 0 && s.Hq % s.Hkv == 0, "Hq must be a multiple of Hkv (GQA)");
    TORCH_CHECK(s.k.size(3) == s.D && s.v.size(3) == s.D,
                "key/value head_dim must match query");
    s.scale = scale.has_value() ? *scale : 1.0 / std::sqrt((double)s.D);
    return s;
}

template <typename Launch>
static torch::Tensor flash_attn_fa(Launch launch, torch::Tensor q, torch::Tensor k,
                                   torch::Tensor v,
                                   c10::optional<torch::Tensor> attn_mask,
                                   double dropout_p, bool is_causal,
                                   c10::optional<double> scale,
                                   bool force_decode_kernel) {
    auto s = sdpa_shape(q, k, v, dropout_p, scale);
    TORCH_CHECK(s.D <= 64 && s.D % 16 == 0, "supports head_dim <= 64, multiple of 16");
    auto out = torch::empty({s.B, s.Hq, s.Sq, s.D}, s.q.options());

    const __nv_bfloat16* mask_ptr = nullptr;
    int64_t ms_b = 0, ms_h = 0, ms_q = 0, ms_k = 0;
    torch::Tensor mc;
    if (attn_mask.has_value() && attn_mask->defined()) {
        mc = attn_mask->contiguous();
        TORCH_CHECK(mc.dtype() == torch::kBFloat16, "attn_mask must be bf16");
        TORCH_CHECK(mc.dim() == 4, "attn_mask must be 4D (broadcast over B,Hq,Sq,Sk)");
        auto bc = [&](int d, int64_t want) -> int64_t {
            TORCH_CHECK(mc.size(d) == want || mc.size(d) == 1,
                        "attn_mask dim not broadcastable");
            return mc.size(d) == 1 ? 0 : mc.stride(d);
        };
        ms_b = bc(0, s.B); ms_h = bc(1, s.Hq); ms_q = bc(2, s.Sq); ms_k = bc(3, s.Sk);
        mask_ptr = cptr<__nv_bfloat16>(mc);
    }

    at::cuda::CUDAGuard g(q.device());
    launch(cptr<__nv_bfloat16>(s.q), cptr<__nv_bfloat16>(s.k), cptr<__nv_bfloat16>(s.v),
           mask_ptr, mptr<__nv_bfloat16>(out), (int)s.B, (int)s.Hq, (int)s.Hkv,
           (int)s.Sq, (int)s.Sk, (int)s.D, (float)s.scale, is_causal,
           ms_b, ms_h, ms_q, ms_k, force_decode_kernel,
           c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

static torch::Tensor flash_attn_fa1(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    c10::optional<torch::Tensor> attn_mask,
                                    double dropout_p, bool is_causal,
                                    c10::optional<double> scale,
                                    bool force_decode_kernel) {
    return flash_attn_fa(flash_attn_fa1_launch, q, k, v, attn_mask, dropout_p,
                         is_causal, scale, force_decode_kernel);
}

static torch::Tensor flash_attn_fa2(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    c10::optional<torch::Tensor> attn_mask,
                                    double dropout_p, bool is_causal,
                                    c10::optional<double> scale,
                                    bool force_decode_kernel) {
    return flash_attn_fa(flash_attn_fa2_launch, q, k, v, attn_mask, dropout_p,
                         is_causal, scale, force_decode_kernel);
}

// The extracted kernel reads the mask through a Kuiper tlayout, which is an
// injection, so the mask must be dense (B,Hq,Sq,Sk) rather than broadcast.
static torch::Tensor flash_attn_manual_extract(torch::Tensor q, torch::Tensor k,
                                               torch::Tensor v, torch::Tensor attn_mask,
                                               double dropout_p, bool is_causal,
                                               c10::optional<double> scale,
                                               int64_t nwarps) {
    auto s = sdpa_shape(q, k, v, dropout_p, scale);
    TORCH_CHECK(s.D % 16 == 0 && s.D > 0, "head_dim must be a positive multiple of 16");
    TORCH_CHECK(s.Sq <= s.Sk, "Sq must be <= Sk");
    auto mc = attn_mask.contiguous();
    TORCH_CHECK(mc.dtype() == torch::kBFloat16, "attn_mask must be bf16");
    TORCH_CHECK(mc.dim() == 4 && mc.size(0) == s.B && mc.size(1) == s.Hq &&
                mc.size(2) == s.Sq && mc.size(3) == s.Sk,
                "attn_mask must be dense (B,Hq,Sq,Sk)");

    const int64_t group = s.Hq / s.Hkv;
    const int64_t rows = group * s.Sq;
    const int64_t tiles = (rows + 15) / 16;
    const int64_t nblk = s.B * s.Hkv * tiles;

    auto out = torch::empty({s.B, s.Hq, s.Sq, s.D}, s.q.options());

    at::cuda::CUDAGuard g(q.device());
    flash_attn_manual_extract_launch(
        (uint32_t)nblk, (uint32_t)nwarps, (uint32_t)(nwarps * 32), (uint32_t)s.B,
        (uint32_t)s.Hq, (uint32_t)s.Hkv, (uint32_t)group, (uint32_t)s.Sq,
        (uint32_t)rows, (uint32_t)tiles, (uint32_t)s.Sk, (uint32_t)s.D,
        const_cast<__nv_bfloat16*>(cptr<__nv_bfloat16>(s.q)),
        const_cast<__nv_bfloat16*>(cptr<__nv_bfloat16>(s.k)),
        const_cast<__nv_bfloat16*>(cptr<__nv_bfloat16>(s.v)),
        const_cast<__nv_bfloat16*>(cptr<__nv_bfloat16>(mc)),
        mptr<__nv_bfloat16>(out), is_causal, (float)s.scale,
        c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_hacky_epilogue", &gemm_hacky_epilogue,
          "Extracted instance of Kuiper TensorCore2D with a hacked-on epilogue (bf16).",
          py::arg("input"), py::arg("mat1"), py::arg("mat2"),
          py::kw_only(), py::arg("beta") = 1.0, py::arg("alpha") = 1.0);
    m.def("gemm_pipe", &gemm_pipe,
          "GEMM with software pipelining (fp16 in/out, fp32 accumulate).",
          py::arg("input"), py::arg("mat1"), py::arg("mat2"),
          py::kw_only(), py::arg("beta") = 1.0, py::arg("alpha") = 1.0);
    m.def("gemm_bcast_bias_epilogue", &gemm_bcast_bias_epilogue,
          "Extracted instance of Kuiper TensorCore2D (fp16 in/out, fp32 "
          "accumulate) whose epilogue broadcasts a length-N bias vector over "
          "the rows, as nn.Linear's addmm needs.",
          py::arg("input"), py::arg("mat1"), py::arg("mat2"),
          py::kw_only(), py::arg("beta") = 1.0, py::arg("alpha") = 1.0);
    m.def("gemm_bcast_bias_epilogue2", &gemm_bcast_bias_epilogue2,
          "As gemm_bcast_bias_epilogue, but derived from TensorCore2D.To, whose "
          "out-of-place epilogue makes the broadcast a one-line change.",
          py::arg("input"), py::arg("mat1"), py::arg("mat2"),
          py::kw_only(), py::arg("beta") = 1.0, py::arg("alpha") = 1.0);
    m.def("flash_attn_fa1", &flash_attn_fa1,
          "semi-fast FlashAttention using tensor cores.",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("attn_mask") = c10::nullopt, py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false, py::arg("scale") = c10::nullopt,
          py::arg("force_decode_kernel") = false);
    m.def("flash_attn_fa2", &flash_attn_fa2,
          "FlashAttention2-like impl using software pipelining, register caching, etc.",
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("attn_mask") = c10::nullopt, py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false, py::arg("scale") = c10::nullopt,
          py::arg("force_decode_kernel") = false);
    m.def("flash_attn_manual_extract", &flash_attn_manual_extract,
          "Extracted instance of Kuiops.Sdpa.Flash.",
          py::arg("query"), py::arg("key"), py::arg("value"), py::arg("attn_mask"),
          py::arg("dropout_p") = 0.0, py::arg("is_causal") = false,
          py::arg("scale") = c10::nullopt, py::arg("nwarps") = 4);
}
