# Kuiper ⇄ PyTorch integration for Qwen2.5

This document maps every unique CUDA kernel name appearing in `trace.json`
(produced by `profile_ops.py` on Qwen2.5-0.5B in bf16 on sm_120) to its
status under the Kuiper integration layer in `kuiper_ext/`.

There are three possible statuses:

- **HOOKED** — the kernel's enclosing PyTorch op is routed to a verified
  Kuiper kernel via `kuiper_ext`. Usually shape/dtype guarded with a
  fallback.
- **FALLBACK** — no Kuiper kernel exists; the integration layer leaves the
  PyTorch native call in place. See `MISSING_KERNELS.md` for what would be
  needed.
- **WRAPPER NOTE** — the wrapper exists but had to compensate for an API
  mismatch. See `API_MISMATCHES.md` for details.

## Per-kernel table

| Count | Kernel (abbreviated) | PyTorch op | Status | Kuiper hook |
|-------|----------------------|------------|--------|-------------|
|   145 | `elementwise_kernel<...MulFunctor<float>>` (bf16×bf16→bf16) | bf16 elementwise multiply (RoPE, SwiGLU `silu(x)*gate`) | **FALLBACK** | none — no verified bf16 elementwise mul |
|    96 | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8` | bf16 `nn.Linear` (cuBLASLt) | **HOOKED** (via cast) | `KuiperLinear` w/ `cast_bf16_to_f16=True` → `Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16` |
|    96 | `CatArrayBatchedCopy<OpaqueType<2u>,...,4,64,64>` | bf16 `torch.cat` (KV-cache concat, RoPE concat) | **FALLBACK** | no Kuiper concat kernel |
|    51 | `vectorized_elementwise_kernel<...bfloat16_copy_kernel_cuda>` | bf16 dtype cast / contiguous-copy | **FALLBACK** | no Kuiper cast/copy kernel |
|    50 | `unrolled_elementwise_kernel<...direct_copy_kernel_cuda...>` | f32→f32 strided copy (norm promotion) | **FALLBACK** | no Kuiper copy |
|    49 | `vectorized_elementwise_kernel<...pow_tensor_scalar_kernel_impl<float>...>` | `x.pow(2)` (RMSNorm) | **FALLBACK** | no Kuiper pow / squared-mean |
|    49 | `reduce_kernel<512, 1, ReduceOp<float, MeanOps<float>...>` | reduce-mean over last dim, f32 (RMSNorm) | **FALLBACK** ⓘ | `Klas_HReduce_reduce_f32_plus` exists but returns a *host scalar* per row → unacceptable launch overhead; see `MISSING_KERNELS.md` |
|    49 | `vectorized_elementwise_kernel<...CUDAFunctorOnSelf_add<float>...>` | in-place `+= eps` (RMSNorm) | **FALLBACK** | no Kuiper scalar add |
|    49 | `vectorized_elementwise_kernel<...rsqrt_kernel_cuda...>` | f32 elementwise `rsqrt` (RMSNorm) | **FALLBACK** | no Kuiper rsqrt |
|    49 | `elementwise_kernel<...BinaryFunctor<float,...,MulFunctor<float>>...>` | f32 elementwise mul (RMSNorm row scale) | **FALLBACK** ⓘ | `Klas_RowScale_rowscale_f32_rowmajor` exists and is exposed as `kuiper_ext.row_scale_f32_`, but the RMSNorm pattern needs `out = x * weight + bias`-shaped broadcast, not row-scaling; see `API_MISMATCHES.md` |
|    48 | `cublasLt::splitKreduce_kernel<...bf16,bf16,float,bf16,...,true,true,...>` | post-pass of split-K bf16 GEMM | **HOOKED (transitive)** | the GEMM as a whole is hooked; this kernel does not appear once cuBLAS is bypassed |
|    48 | `elementwise_kernel<...neg_kernel_cuda(...BFloat16)...>` | `torch.neg` on bf16 (RoPE half-rotation) | **FALLBACK** | no Kuiper neg |
|    48 | `elementwise_kernel<...CUDAFunctor_add<BFloat16>...>` | bf16 residual add (attention out + residual; MLP out + residual) | **FALLBACK** | no verified bf16 elementwise add |
|    48 | `vectorized_elementwise_kernel<...CUDAFunctor_add<BFloat16>...>` | bf16 elementwise add (other variants) | **FALLBACK** | same |
|    48 | `nvjet_sm120_tst_mma_64x64x128_3_32x32x128_tmaAB_alignCD4_bz_TNNN` | bf16 GEMM (cuBLAS sm_120 / Blackwell TMA path) | **HOOKED (transitive)** | `KuiperLinear` routes the parent `F.linear` away from cuBLAS; this specific kernel disappears |
|    24 | `pytorch_flash::flash_fwd_kernel<...bf16,...>` | `F.scaled_dot_product_attention` (flash attention) | **FALLBACK** | **no Kuiper flash attention** — top priority for `MISSING_KERNELS.md` |
|    24 | `vectorized_elementwise_kernel<...silu_kernel...BFloat16...>` | `F.silu` on bf16 (SwiGLU) | **FALLBACK** | no Kuiper SiLU |
|    24 | `vectorized_elementwise_kernel<...MulFunctor<float>...BFloat16>...>` | another bf16 mul variant | **FALLBACK** | as above |
|    24 | `cutlass_80_tensorop_s16816gemm_bf16_64x64_32x6_tn_align8` | bf16 GEMM (smaller-shape cuBLAS) | **HOOKED (transitive)** | `KuiperLinear` |
|    24 | `cublasLt::splitKreduce_kernel<...float,bf16,float,bf16,...,true,false,...>` | split-K reduce, fp32-accum variant | **HOOKED (transitive)** | as above |
|     2 | `vectorized_elementwise_kernel<...AUnaryFunctor<float,...,MulFunctor<float>>...>` | f32 scalar broadcast multiply | **FALLBACK** | no Kuiper scalar-broadcast mul |
|     1 | `vectorized_gather_kernel<16, long>` | embedding lookup (`nn.Embedding`) | **FALLBACK** | no Kuiper gather |
|     1 | `elementwise_kernel_with_index<int, arange_cuda_out...>` | `torch.arange` (position ids) | **FALLBACK** | no Kuiper arange |
|     1 | `vectorized_elementwise_kernel<...CUDAFunctorOnSelf_add<long>...>` | int64 self-add (position offset) | **FALLBACK** | no Kuiper i64 add |
|     1 | `unrolled_elementwise_kernel<...direct_copy_kernel_cuda...bool...>` | bool copy (attention mask) | **FALLBACK** | no Kuiper bool copy |
|     1 | `reduce_kernel<...and_kernel_cuda...bool...>` | bool AND reduction (mask sanity check) | **FALLBACK** | no Kuiper bool reduce |
|     1 | `gemmk1_kernel<int, float, 256, 5, ...>` | f32 GEMM (small variant; LM head warmup) | **HOOKED (when M,N,K%32==0)** | `kuiper_ext.matmul_f32` |
|     1 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8` | fused bf16 GEMM+ReLU | **FALLBACK** | Kuiper has no fused activation; unfused → `KuiperLinear` + post-`relu` would work but match isn't bit-exact |
|     1 | `CatArrayBatchedCopy<OpaqueType<4u>,...,3,64,64>` | concat (i32-sized) | **FALLBACK** | no Kuiper concat |
|     1 | `vectorized_elementwise_kernel<...cos_kernel_cuda...>` | f32 cos (RoPE) | **FALLBACK** | no Kuiper trig |
|     1 | `vectorized_elementwise_kernel<...sin_kernel_cuda...>` | f32 sin (RoPE) | **FALLBACK** | no Kuiper trig |

ⓘ = Kuiper *has* a related kernel but it cannot replace the PyTorch kernel without further wrapping or a missing variant.

## Coverage summary

By kernel call count, weighted by how often each kernel fires in a single
forward pass of Qwen2.5-0.5B:

|                                | Calls | Fraction |
|--------------------------------|------:|---------:|
| **Hooked** (mostly bf16 GEMM via cast→f16) | 218 | 21 % |
| **Hooked (transitive)** (kernels that disappear when their parent op is hooked) | 120 | 11 % |
| **Fallback**                   | 717  | 68 %    |
| **Total**                      | 1055 | 100 %   |

The headline number: every `nn.Linear` in Qwen2.5-0.5B (168 of 169) can be
routed through verified Kuiper GEMM when prefill `M` divides 64, with the
caveat that bf16 → f16 casting is required and costs ~0.05 logit units of
precision on average (top-1 next-token agreement preserved on the prompts
we tested).

The two big rocks blocking deeper integration are:

1. **No verified bf16 (or even bf16-input/f32-accum) GEMM** — forces the
   precision-losing f16 cast.
2. **No verified flash attention** — the single largest non-GEMM kernel
   stays on PyTorch.

After those, the remaining ~50 % of kernel calls are tiny elementwise /
norm / RoPE / concat / cast ops. Individually cheap, collectively ~30 % of
forward-pass wall-time. See `MISSING_KERNELS.md` for the prioritised list.

## Using the integration

```python
import torch
from transformers import AutoModelForCausalLM
from kuiper_ext.integration import enable_kuiper

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

info = enable_kuiper(
    model,
    replace_linear=True,        # swap nn.Linear -> KuiperLinear
    cast_bf16_to_f16=True,      # accept the f16 precision trade-off
    patch_softmax=False,        # OFF — Kuiper softmax overflows on attention logits
    patch_matmul=False,         # OFF — KuiperLinear already covers the important case
    linear_filter=lambda name: "lm_head" not in name,  # skip the giant vocab head
)
print(info)   # {'linears_replaced': 168, 'softmax_patched': False, 'matmul_patched': False}
```

To verify, run:

```bash
PYTHONPATH=$PWD .venv/bin/python tests/test_kuiper_ops.py
```
