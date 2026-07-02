// PyTorch glue for the plain-CUDA FlashAttention kernel, mirroring the JIT
// family wrappers (kuiops/*/wrapper_*.cu.j2): contiguous the inputs, allocate
// outputs with the *kernel's* output dtype, guard the device + sync the current
// stream, then launch. Exposes `run` returning the same 4-tuple as
// flash_attn._flash_attn_varlen_forward: (out, softmax_lse, S_dmask, rng_state).
#include <cuda_bf16.h>
#include "kuiops.h"        // include/kuiops.h: CUDAGuard + sync_current_stream
#include "flash_attn.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
op_flash_attn_varlen(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,        // unused by the kernel (bounds only)
    int64_t max_seqlen_k,        // unused by the kernel (bounds only)
    double dropout_p,
    double softmax_scale,
    bool causal)
{
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16 &&
                    v.dtype() == torch::kBFloat16,
                "flash_attn kernel expects bf16 q/k/v");
    TORCH_CHECK(dropout_p == 0.0, "flash_attn kernel does not support dropout");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "expected packed (total, heads, head_dim) tensors");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 &&
                    cu_seqlens_k.dtype() == torch::kInt32,
                "cu_seqlens must be int32");

    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
    auto cuq = cu_seqlens_q.contiguous(), cuk = cu_seqlens_k.contiguous();

    const int64_t total_q   = qc.size(0);
    const int64_t n_heads_q = qc.size(1);
    const int64_t head_dim  = qc.size(2);
    const int64_t n_heads_kv = kc.size(1);
    const int64_t batch = cuq.size(0) - 1;

    TORCH_CHECK(head_dim <= 256, "flash_attn kernel supports head_dim <= 256");
    TORCH_CHECK(n_heads_kv > 0 && n_heads_q % n_heads_kv == 0,
                "n_heads_q must be a multiple of n_heads_kv (GQA)");
    TORCH_CHECK(kc.size(2) == head_dim && vc.size(2) == head_dim,
                "k/v head_dim must match q");

    auto out = torch::empty({total_q, n_heads_q, head_dim}, qc.options());
    // LSE is (n_heads_q, total_q) in fp32, matching flash_attn's softmax_lse.
    auto lse = torch::empty({n_heads_q, total_q},
                            qc.options().dtype(torch::kFloat32));

    at::cuda::CUDAGuard g(q.device());
    kuiops::sync_current_stream();
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    flash_attn_varlen_fwd_launch(
        reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(vc.data_ptr()),
        cuq.data_ptr<int32_t>(),
        cuk.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        lse.data_ptr<float>(),
        (int)batch, (int)total_q, (int)n_heads_q, (int)n_heads_kv, (int)head_dim,
        (float)softmax_scale, causal, stream);
    AT_CUDA_CHECK(cudaGetLastError());

    // Trailing outputs mirror the aten signature but are unused here:
    //   S_dmask is empty (no returned dropout mask), rng_state is a 2-elt stub.
    auto s_dmask = torch::empty({0}, qc.options());
    auto rng_state = torch::empty({2}, q.options().dtype(torch::kInt64));
    return std::make_tuple(out, lse, s_dmask, rng_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &op_flash_attn_varlen, "flash_attn varlen forward (CUDA)");
}
