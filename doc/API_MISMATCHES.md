# API mismatches between Kuiper kernels and PyTorch ops

Compensating C/C++ (in `kuiper_ext/csrc/ops.cu`) and Python
(`kuiper_ext/integration.py`) glue handles each mismatch below. Items
marked **upstream-fix-desirable** are the ones I'd address in Kuiper itself
to get a cleaner wrapper.

## 1. Stream model

- **Kuiper:** every top-level extracted host function does
  `cudaStreamCreate(&fresh)` inside `KPR_KCALL`, launches on `fresh`,
  destroys the stream, then `cudaDeviceSynchronize()`. The fresh stream
  is independent of any stream the caller might be on.
- **PyTorch:** every op runs on the current `at::cuda::CUDAStream`,
  which has its own ordering relative to prior tensor producers/
  consumers. Launching on a *fresh* stream without an explicit wait
  (e.g., a `cudaStreamWaitEvent`) can race with prior torch ops on
  asynchronous APIs.
- **Wrapper compensation:** before every Kuiper call we
  `cudaStreamSynchronize(at::cuda::getCurrentCUDAStream())` to flush
  any pending torch producers. Kuiper's post-call
  `cudaDeviceSynchronize()` then provides the post-condition. This is
  conservative — two host-side syncs per Kuiper call — but always
  correct.
- **Upstream-fix-desirable:** add a stream parameter to the extracted
  host functions, or expose a `kuiper_set_default_stream(stream)`
  thread-local. Then we'd just pass torch's current stream and avoid
  both syncs.

## 2. CUDA graph capture is forbidden

- **Kuiper:** uses `cudaMalloc`, `cudaMemcpy`, `cudaFree`,
  `cudaStreamCreate/Destroy`, `cudaDeviceSynchronize` — all of which
  are illegal during graph capture.
- **PyTorch:** Qwen2.5 inference doesn't capture in eager mode, but
  `torch.compile(..., mode="reduce-overhead")` or
  `torch.cuda.graph(...)` would.
- **Wrapper compensation:** every wrapper begins with
  `cudaStreamIsCapturing(stream, &status)`; if `status !=
  cudaStreamCaptureStatusNone` we raise a `TORCH_CHECK` so
  the user gets a clear error rather than CUDA's cryptic
  `cudaErrorStreamCaptureInvalidated`.

## 3. Hard `abort()` / `exit(1)` on guard failure

- **Kuiper:** `KPR_GUARD(...)` calls `abort()`. `MUST(...)` calls
  `exit(1)`. So a wrong shape kills the Python interpreter — extremely
  hostile to a dispatcher that wants to *fall back*.
- **Wrapper compensation:** every wrapper mirrors every relevant
  `KPR_GUARD` (e.g., M,N,K divisibility, dtype, contiguity) and uses
  `TORCH_CHECK(...)` instead, which throws a recoverable
  `c10::Error` that Python sees as `RuntimeError`. The
  high-level dispatcher (`KuiperLinear.forward`,
  `_patched_matmul`, `_patched_softmax`) catches the error and falls
  back to the original PyTorch op.
- **Upstream-fix-desirable:** an opt-in mode where guards return an
  error code instead of aborting. Could be a `-DKUIPER_NO_ABORT` macro
  selected at extraction time.

## 4. TensorCore GEMM is C := C + A·B, not C := A·B

- **Kuiper:** `Klas_GEMM_TensorCore_g_gemm_f16_f16_*` (and all the
  TensorCore2D variants) call `wmma::load_matrix_sync(accumFrag, gC,
  ...)` before the MMA loop. Net effect: `C += A·B` with no `alpha`/
  `beta` exposed at the API surface.
- **PyTorch:** `torch.matmul` and `F.linear` expect `C := A·B` (no
  accumulation).
- **Wrapper compensation:** `kuiper_matmul_f16` allocates `C` with
  `torch::zeros(...)`, not `torch::empty(...)`, so the load-into-
  accumulator becomes a no-op. Costs one extra full-tensor write per
  matmul.
- **Upstream-fix-desirable:** expose explicit `alpha`/`beta` (zero
  initialised when desired) like the f32 `g_gemm_*` variants already
  do, OR provide a `g_matmul_*` (no-load) f16 TensorCore variant
  matching the `g_matmul` family of the other GEMM modules.

## 5. Tile-divisibility is hard, not soft

- **Kuiper:** `Klas_GEMM_BlockTiling{1,2}D_g_matmul_*` and
  TensorCore variants require `M`, `N`, `K` to be multiples of the
  specific tile. There is no boundary handling — out-of-tile rows
  silently get garbage (BlockTiling1D) or hit a `KPR_GUARD` (newer
  variants).
- **PyTorch:** sizes are arbitrary. Qwen2.5-0.5B has `Dhead=64`,
  `Dintermediate=4864`, `Dhidden=896`. Prefill `M` equals the prompt
  length (anything ≥1); decode `M=1`.
- **Wrapper compensation:** `KuiperLinear` only takes the Kuiper path
  when the *flattened* `M = numel(x)/in_features` divides 64 (for f16)
  or 32 (for f32). Otherwise falls back to `F.linear`.
- **Consequence:** during single-prompt autoregressive decoding,
  `M=1` so **no** Kuiper GEMM ever fires. Only batched-prefill
  benefits. Workarounds:
  - Pad `M` up to the nearest multiple of 64 and slice the output (not
    yet implemented; adds memory traffic).
  - Verify a small-M tile in Kuiper (`16x64x64`?).

## 6. `F.linear` does `x @ wᵀ`; Kuiper does `A @ B`

- **PyTorch:** `nn.Linear.weight` has shape `(out_features,
  in_features)`. `F.linear(x, w, b)` computes `x @ w.t() + b`.
- **Kuiper:** `Klas_GEMM_*` take `A, B, C` all row-major-row-major-row-
  major (`_rrr`). No fused transpose.
- **Wrapper compensation:** `KuiperLinear` stores its weight as
  `weight_t` with shape `(in_features, out_features)` (i.e., already
  transposed and contiguous) so that the forward call is exactly
  `x_flat @ weight_t`. The one-time `lin.weight.t().contiguous()`
  inside `from_linear()` pays the transpose at module-swap time, not
  per-call.
- **Upstream-fix-desirable:** add `*_rrc` (or `*_rcc`) GEMM variants
  that consume B as if column-major (i.e., its row-major data is
  treated as transposed).

## 7. `F.linear` has a bias; Kuiper GEMM doesn't

- **Kuiper:** `Klas_GEMM_*` is pure C := alpha·A·B + beta·C; no bias
  broadcast.
- **Wrapper compensation:** `KuiperLinear.forward` does the bias add
  in PyTorch (`out + self.bias`). This adds one extra elementwise
  kernel per linear (which then appears in the trace as a
  `vectorized_elementwise_kernel` — itself unhooked, see
  `MISSING_KERNELS.md` §1).
- **Upstream-fix-desirable:** fused `g_gemm_bias_<dt>(... bias)`
  variant.

## 8. No bf16 anywhere in Kuiper; Qwen2.5 is bf16 everywhere

- **Kuiper:** f32, f64, u32, u64, f16 only.
- **PyTorch model:** Qwen2.5 weights, activations, residuals — all
  bf16. f32 is used only for the RMSNorm internal computation.
- **Wrapper compensation:** `KuiperLinear(cast_bf16_to_f16=True)`
  downcasts bf16 → f16 before the GEMM and re-upcasts the result. f16
  has 10 mantissa bits vs bf16's 7, but *crucially* a much smaller
  exponent range — large weights or activations risk overflow. On the
  prompts we tested, top-1 next-token agreement holds; max per-logit
  error ~0.05 % of logit magnitude.
- **Upstream-fix-desirable:** see `MISSING_KERNELS.md` §1 — verified
  bf16 GEMM is the single highest-impact addition.

## 9. Kuiper softmax is 1-D and not numerically stable

- **Kuiper:** `Klas_Softmax_softmax_f32(uint32_t lena, float *a)`
  takes a flat 1-D array and operates **in-place** as
  `a[i] = exp(a[i]) / sum_j exp(a[j])`. There is no `dim=`
  parameter and no max-shift.
- **PyTorch:** `F.softmax(x, dim=-1)` accepts any-rank tensors, picks
  a reduction dim, allocates a fresh output, and subtracts the max
  along that dim first to avoid overflow.
- **Wrapper compensation:** `kuiper_softmax_last_f32_(x)` flattens
  the outer dims, loops over rows, calls the Kuiper kernel per row.
  This makes `outer` kernel launches per call — fine for tiny
  tensors, terrible for attention scores `[B, H, S, S]` where outer =
  `B*H*S`. *And* the no-max-shift issue is unfixed at the wrapper
  layer. `enable_softmax_patch()` is therefore **off by default** and
  documented as experimental.
- **Upstream-fix-desirable:** the existing
  `Kuiper_Example_OnlineSoftmax` IS the max-shifted streaming softmax;
  promote it out of the `Example_` namespace into a real `Klas_`
  module with `(stride, rows, cols, *in, *out)` semantics so it can
  reduce along any axis. The current 1D API is unusable in practice.

## 10. `Klas_HReduce` returns a host scalar

- **Kuiper:** `Klas_HReduce_reduce_f32_plus(nth, lena, *a) -> float`
  reduces a flat 1-D array on device, then does a
  `cudaMemcpyDeviceToHost` of the single result and returns it. Sync.
- **PyTorch:** `torch.sum(x, dim=-1)` returns a *device* tensor with
  shape `x.shape[:-1]` that downstream ops can consume on-device.
- **Wrapper compensation:** `kuiper_row_sum_last_f32` loops over
  outer dims calling the reduction per row, accumulates results into
  a *host* tensor, and copies the whole thing back to GPU at the end.
  This is one D→H copy per row, which is unusable in production.
  Kept only as a correctness test of the Kuiper reduction primitive.
- **Upstream-fix-desirable:** a device-output variant
  `Klas_HReduce_reduce_f32_plus_into(uint32_t nth, uint32_t lena,
  float *a, float *out)` that writes the result to a GPU scalar
  (or, even better, a batched variant
  `(uint32_t rows, uint32_t cols, float *a, float *out)` that does
  the loop on-device).

## 11. `Klas_RowScale` is row-vec × matrix; PyTorch typically wants
       `matrix × col-vec`

- **Kuiper:** `rowscale_f32_rowmajor(m, n, a, b)` computes
  `b[r,c] := a[r] * b[r,c]` — `a` is a column of length `m`, applied
  to each row of `b`. So semantically this is **column-scaling viewed
  as row-by-row multiply**.
- **PyTorch:** `x * scales[:, None]` (with `scales` broadcasting
  along the *non-reduction* dim — i.e., applied across rows but
  varies per row) matches Kuiper's behaviour. So this part lines up.
- **What does NOT line up:** RMSNorm's final step is
  `x_normalized = x * weight[None, :]` — a *column-vector* (one
  scalar per *column*, i.e., per feature) broadcast over rows. That's
  the opposite axis. Kuiper has no row-broadcast (per-column) scaling
  primitive.
- **Wrapper compensation:** `kuiper_ext.row_scale_f32_` exposes only
  the same-axis case. Not used in any wrapper path; RMSNorm stays on
  fallback.
- **Upstream-fix-desirable:** add `colscale_*_rowmajor(m, n, *a:[n], *b:[m,n])`.

## 12. Batched matmul allocates with `cudaMalloc` and returns a raw pointer

- **Kuiper:** `Klas_GEMM_Batched_batched_gemm_f32(batch, rows, shared,
  cols, a, b)` calls `cudaMalloc` for the output and **returns the raw
  pointer**. There is no way to integrate that with PyTorch's caching
  allocator: ownership is unclear and the memory cannot be tracked or
  freed by torch.
- **Wrapper compensation:** `kuiper_bmm_f32` ignores the batched
  Kuiper variant entirely and **loops** over the batch calling
  `Klas_GEMM_BlockTiling1D_g_matmul_f32_tile32_rrr` per item, with
  output pre-allocated by torch.
- **Upstream-fix-desirable:** out-parameter style — accept a
  pre-allocated output buffer instead of returning a fresh `cudaMalloc`-
  ed pointer.

---

## Summary: glue layer responsibilities

The C++ wrapper in `ops.cu` is responsible for:

1. Pre-call: `at::cuda::CUDAGuard`, `cudaStreamIsCapturing` check,
   `cudaStreamSynchronize` on the current torch stream.
2. Validate every KPR_GUARD condition with `TORCH_CHECK` (recoverable).
3. Allocate or zero-initialise outputs as the kernel requires.
4. Call the Kuiper top-level host function.
5. (No post-call work — Kuiper syncs the device internally.)

The Python wrapper in `integration.py` is responsible for:

6. Dispatching by dtype (f32 / f16 / bf16-with-cast / fallback).
7. Flattening N-D tensors to (M, K) when the underlying Kuiper kernel
   is 2-D.
8. Catching `RuntimeError` from `TORCH_CHECK` and falling back to the
   original PyTorch op rather than propagating the error.
9. Pre-transposing `nn.Linear.weight` once at module-swap time (point
   §6 above).
10. Adding the bias outside the GEMM (point §7).
