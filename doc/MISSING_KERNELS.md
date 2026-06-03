# Kuiper kernels needed to fully replace PyTorch ops in Qwen2.5

Generated from analysis of `trace.json` against the kernels currently in
`/home/julien/work/kuiper/dist/`. Sorted roughly by impact-per-engineering-
effort.

Each entry uses the following structure:

- **Why** — what PyTorch op (and which Qwen2.5 subcomponent) this kernel
  serves.
- **Signature** — a proposed F* / extracted-C signature.
- **Notes** — caveats, related Kuiper machinery, tiles to consider.

## 1. Verified bf16 GEMM  (highest impact)

The single biggest gap. Qwen2.5 is bf16 end-to-end; every weight matmul
currently flows through cuBLAS bf16 GEMM (1055 traced calls → 168 bf16
GEMMs). The existing `KuiperLinear` wrapper has to cast bf16 → f16 before
calling `Klas_GEMM_TensorCore_g_gemm_f16_f16_*`, which loses ~3 bits of
mantissa precision (f16 has 10, bf16 has 7 — but f16's exponent range is
much smaller than bf16's, hence the *real* concern: overflow/underflow on
attention scores and large activations).

- **Why** Replace every `cutlass_80_..._bf16_...gemm_*`,
  `nvjet_sm120_tst_mma_64x64x128_..._TNNN`, `cublasLt::splitKreduce_kernel<...bf16,bf16,...>`
  call (≈ 218 calls per forward).
- **Signature (proposed):**
  ```c
  // bf16 in, fp32 accum, bf16 out — what cuBLAS does for Qwen.
  void Klas_GEMM_TensorCore_g_gemm_bf16_bf16_<BM>x<BN>x<BK>_16x16x16(
      uint32_t rows, uint32_t shared, uint32_t cols,
      __nv_bfloat16 *gA, __nv_bfloat16 *gB, __nv_bfloat16 *gC);
  // Optional: alpha/beta variant for fused linear+residual.
  ```
- **Notes** WMMA on sm_120 supports `nv_bfloat16` directly via
  `wmma::fragment<..., __nv_bfloat16, ...>`. The existing
  `Klas_GEMM_TensorCore` extraction patterns should port one-to-one.
  Mirror the same tile menu as the f16 version (16x16x16, 32x32x32,
  64x64x64, 32x8x16, 8x32x16). Highest-priority single shape: **64x64x64
  with f32 accumulator**, since Qwen prefill prefers that.

## 2. Verified flash attention forward  (second-largest)

24 calls of `pytorch_flash::flash_fwd_kernel<bf16,128,128,128,4>` per
forward pass, one per layer's attention. This is the single biggest
non-GEMM kernel; replacing it would unlock most of the remaining
fall-back time.

- **Why** Replace `F.scaled_dot_product_attention(...)` (which dispatches
  to flash-attention-2 by default).
- **Signature (proposed):**
  ```c
  // Q,K,V: (B, H, S, Dh) bf16, row-major. O: same. Causal optional.
  void Klas_FlashAttention_fwd_bf16(
      uint32_t B, uint32_t H, uint32_t S, uint32_t Dh,
      __nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V,
      __nv_bfloat16 *O, float scale, bool causal);
  ```
- **Notes** Kuiper already has `Kuiper_Example_OnlineSoftmax` which is the
  numerical heart of flash attention (max-shifted streaming softmax). It
  also has `Klas_GEMM_TensorCore2D` which is the right shape for tiled
  Q·Kᵀ and (P·V) inside the flash kernel. A reasonable plan: compose
  these in a new `Klas_FlashAttention` module, parameterising over block
  sizes `Br`, `Bc`. Variable-length / paged-KV / GQA can come later.
  Even a non-causal/no-GQA `Br=64, Bc=64, Dh=64` instantiation would
  cover Qwen2.5-0.5B's `Dh=64` and 14-head GQA after head-broadcasting.

## 3. Verified RMSNorm (and its building blocks)

Three of the top-5 most-frequent non-GEMM kernels are RMSNorm components:
`pow_tensor_scalar`, `reduce_kernel<MeanOps>`, `rsqrt_kernel`,
`CUDAFunctorOnSelf_add` (+ eps), `MulFunctor<float>` (× scale).
49 calls of each → 245 RMSNorm-related launches per forward.

A fused kernel is by far the right answer (avoids 5× kernel-launch
overhead and 4× round-trips to GMEM for the same tensor).

- **Why** Replace the 5-launch composite emitted for each `RMSNorm`.
- **Signature (proposed):**
  ```c
  // x: (B*S, D) row-major, weight: (D,), eps: scalar. In-place.
  void Klas_RMSNorm_rmsnorm_inplace_<dt>(
      uint32_t rows, uint32_t cols,
      <dt> *x, <dt> *weight, float eps);
  // Variant: not-in-place
  void Klas_RMSNorm_rmsnorm_<dt>(
      uint32_t rows, uint32_t cols,
      <dt> *x, <dt> *weight, <dt> *out, float eps);
  ```
- **Notes** The existing `Klas_HReduce` and `Klas_RowScale` cover the
  reduce-mean and per-row multiply phases respectively, but combining
  them in Python through `kuiper_ext` would be *slower* than the current
  PyTorch path because of the device-host scalar round-trip in HReduce.
  A genuine fused-RMSNorm Kuiper kernel is what's needed. Dtype menu
  should include bf16 and f16 with f32 accumulation.

## 4. Verified RoPE  (rotary position embedding)

Per forward we see: `cos_kernel_cuda` (1), `sin_kernel_cuda` (1),
`neg_kernel_cuda<BFloat16>` (48), `MulFunctor<...,BFloat16>` (many of the
145 mul calls), `CatArrayBatchedCopy<OpaqueType<2u>>` (24 of the 96).
Together: ~190 launches per forward.

- **Why** Replace the RoPE rotation `q_out = q*cos + rotate_half(q)*sin`
  (and similarly for k).
- **Signature (proposed):**
  ```c
  // q: (B, H, S, Dh), cos,sin: (S, Dh/2) or (S, Dh), bf16.
  void Klas_RoPE_apply_bf16(
      uint32_t B, uint32_t H, uint32_t S, uint32_t Dh,
      __nv_bfloat16 *q, __nv_bfloat16 *k,
      __nv_bfloat16 *cos, __nv_bfloat16 *sin);
  ```
- **Notes** Should also expose a precompute kernel for the cos/sin
  tables. Block-tiled with one tile per (batch, head, seq-chunk) is
  natural; pure elementwise.

## 5. Verified SwiGLU  (MLP activation)

Per forward: `silu_kernel<BFloat16>` (24) + an elementwise mul (one of the
145) = `out = silu(gate(x)) * up(x)`. The gate and up projections are
covered by `KuiperLinear`; only the elementwise activation+mul remains.

- **Why** Replace the two-kernel composite at the end of each MLP block.
- **Signature (proposed):**
  ```c
  // a, b, out: (N,) bf16. out[i] = silu(a[i]) * b[i].
  void Klas_Activation_swiglu_bf16(
      uint32_t n, __nv_bfloat16 *gate, __nv_bfloat16 *up, __nv_bfloat16 *out);
  ```

## 6. Concat / KV-cache append

`CatArrayBatchedCopy` fires 96 times per forward (24 layers × 4 concats:
KV append k, KV append v, RoPE concat q-half, RoPE concat k-half).

- **Why** Replace `torch.cat(...)` for known-rank, last-or-second-to-last
  dim concats.
- **Signature (proposed):**
  ```c
  // Concatenate a and b along last dim into out (sizes summed on last dim).
  void Klas_Concat_concat_last_<dt>(
      uint32_t outer, uint32_t na, uint32_t nb,
      <dt> *a, <dt> *b, <dt> *out);
  ```

## 7. Embedding lookup (gather)

`vectorized_gather_kernel<16, long>` — one call per forward, but the only
op in the Qwen2.5 forward pass that uses `int64` indexing.

- **Why** Replace `nn.Embedding` lookup.
- **Signature (proposed):**
  ```c
  void Klas_Embedding_gather_bf16(
      uint32_t batch, uint32_t seq, uint32_t hidden,
      __nv_bfloat16 *table,   // (vocab, hidden)
      uint64_t *ids,          // (batch, seq)
      __nv_bfloat16 *out);    // (batch, seq, hidden)
  ```

## 8. Small / one-shot kernels (lower priority)

These each fire 1–2 times per forward and exist only on the cold path:

- `arange` (long) — position-id generation; could be CPU-side then memcpy.
- `direct_copy_kernel_cuda(bool)` — mask conversion.
- `and_kernel_cuda(bool)` — mask sanity check.
- `CUDAFunctorOnSelf_add<long>` — position-offset bump.
- `bfloat16_copy_kernel_cuda` — bf16 → f32 promotion before RMSNorm.

These are not worth a custom Kuiper kernel each, but a small generic set
of typed copy / arange / boolean-reduction helpers would clean them up.

## 9. Verified GEMM input/output type variants (smaller polish)

The current Kuiper GEMM menu is:
- `f32 / f64 / u32 / u64` (row-major, multiple tiles).
- `f16` with `f16` and `f32` accumulator (only one variant with f32
  accumulator: `Klas_GEMM_TensorCore_g_gemm_f16_f32_32x32x32_16x16x16`).
- No `bf16` at all.
- No transposed-B variant. PyTorch `F.linear` does `x @ wᵀ` and our
  `KuiperLinear` pays a one-time transpose at module-swap time. A
  native `*_rrc` (B already row-major-treated-as-transposed) GEMM
  would let us keep the original `nn.Linear.weight` layout.

## 10. Numerically-stable softmax  (correctness fix, not perf)

The current `Klas_Softmax_softmax_*` family computes
`exp(x_i) / sum(exp(x_j))` directly — no max-shift. This will produce
`inf/nan` on attention logits. The existing
`Kuiper_Example_OnlineSoftmax` does have the max-shifted streaming
implementation; promoting it from `Example_` to `Klas_` (i.e., out of
the examples namespace into the public kernel namespace) and giving it
the same dim/dtype menu as `Klas_Softmax` would be a one-line
substitution in `kuiper_ext`. After that we could safely turn on
`enable_softmax_patch()` for any softmax that doesn't go through the
flash-attention path.

---

## Quick reference — which Kuiper kernels we already wrap

| Kuiper kernel                             | Exposed as                       | Used in `KuiperLinear`? |
|-------------------------------------------|----------------------------------|--------------------------|
| `Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr` | `kuiper_ext.matmul_f32`          | yes (f32 path)           |
| `Klas_GEMM_SHMem_g_gemm_f32_tile32_rrr`   | `kuiper_ext.gemm_f32_`           | no                       |
| `Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16` | `kuiper_ext.matmul_f16` | yes (f16, bf16-via-cast) |
| `Klas_GEMM_BlockTiling1D_*` (looped)      | `kuiper_ext.bmm_f32`             | no                       |
| `Klas_Softmax_softmax_f32`                | `kuiper_ext.softmax_last_f32_`   | no (unsafe by default)   |
| `Klas_Softmax_softmax_f16`                | `kuiper_ext.softmax_last_f16_`   | no                       |
| `Klas_LogSoftmax_log_softmax_f32`         | `kuiper_ext.log_softmax_last_f32_` | no                     |
| `Klas_HReduce_reduce_f32_plus`            | `kuiper_ext.row_sum_last_f32`    | no (test-only)           |
| `Klas_RowScale_rowscale_f32_rowmajor`     | `kuiper_ext.row_scale_f32_`      | no                       |
