# kuiLLM performance

End-to-end Qwen2.5-0.5B-Instruct inference with verified Kuiper GPU kernels,
measured against the same pipeline running on stock PyTorch.

Reproduce with:

```bash
python3 infer.py --batch 256                 # Kuiper kernels
python3 infer.py --batch 256 --no-kuiper     # stock Inductor / Triton / cuBLAS / cuDNN
```

## Setup

| | |
| --- | --- |
| GPU | NVIDIA RTX A6000 (GA102, 48 GB, 300 W), 1 device |
| CUDA | driver runtime 13.0, nvcc 13.3 |
| PyTorch | 2.12.1 |
| Model | Qwen/Qwen2.5-0.5B-Instruct, bf16 |
| Model shape | 24 layers, 14 query heads, 2 KV heads, head dim 64, hidden 896, FFN 4864, vocab 151936 |
| Compile mode | `torch.compile(mode="reduce-overhead")` — the decode step replays a CUDA graph |
| kuiLLM commit | `f7627bc` |
| Kuiper commit | `2ee02cc3` |

## Run parameters

| | |
| --- | --- |
| Batch size | 256 sequences |
| Prompt length | 45 tokens/seq (11520 tokens total) |
| Generated | 256 tokens/seq (65536 tokens total) |
| Sampling | greedy (`--temperature 0`) |
| KV cache | preallocated to 309 positions, `(B, 2, 309, 64)` per layer per tensor |
| Warm-up | 5 decode steps before timing |

Prefill is one batched forward over the 45-token prompt; decode is 255 CUDA
graph replays of a single-token step.

## End-to-end throughput

Three consecutive runs of each configuration, same process invocation, GPU
otherwise idle.

| Configuration | Prefill (ms) | Prompt tok/s | Decode (ms) | Generate tok/s | tok/s/seq |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kuiper | 117.7 | 97872 | 2854.7 | 22868 | 89.3 |
| Kuiper | 118.4 | 97300 | 2845.0 | 22946 | 89.6 |
| Kuiper | 118.4 | 97300 | 2842.4 | 22966 | 89.7 |
| **Kuiper (median)** | **118.4** | **97300** | **2845.0** | **22946** | **89.6** |
| stock | 117.9 | 97700 | 1614.5 | 40435 | 157.9 |
| stock | 118.0 | 97666 | 1614.8 | 40425 | 157.9 |
| stock | 118.7 | 97046 | 1616.6 | 40381 | 157.7 |
| **stock (median)** | **118.0** | **97666** | **1614.8** | **40425** | **157.9** |

Prefill is identical because the prefill forward is not compiled and does not
go through the Kuiper pass. Decode on Kuiper kernels is **1.76× slower** than
stock.

Run-to-run spread is under 0.5% in both configurations.

## Operators offloaded

Taken from `python3 infer.py --dump-kernels`, which traces the compiled decode
graph. Of **36 distinct operator signatures** in the graph, **4 distinct
operators** run on Kuiper kernels (deduplicated: one entry per operator, not
per shape).

| Operator | Kuiper kernel | Calls/step |
| --- | --- | ---: |
| `aten::mm` | `Kuiops.Mm.SuperGEMM` / `Kuiops.Mm.SuperGEMMSplitK` | 97 |
| `aten::addmm` | `Kuiops.Addmm.SuperGEMM` (broadcast bias) | 24 |
| `aten::_scaled_dot_product_cudnn_attention` | `Kuiops.Sdpa.Flash` | 24 |
| `kuiperjit::hreduce_poly` | `Kuiops.Reduce` (RMS-norm mean, emitted by the fusion pass) | 49 |

Every `mm` and `addmm` in the model is served by a SuperGEMM backend; neither
`bt2d` nor `tc2d_to` is reachable on this model, because every GEMM here has a
transposed (row-major `(N, K)`) right operand.

## Device time

`torch.profiler` over the measured decode region, 127 steps, batch 256. Times
are self device time.

### Kuiper

Total decode device time **1210.0 ms** (9.53 ms/step). Kuiper kernels account
for **1089.8 ms — 90.1%** of it.

| Kernel | Total (ms) | Share | Calls | µs/call |
| --- | ---: | ---: | ---: | ---: |
| `sdpa_flash_jit` | 485.2 | 40.1% | 3048 | 159.2 |
| `mm_jit_supergemm` | 372.5 | 30.8% | 9271 | 40.2 |
| `mm_jit_supergemm_splitk` (partials) | 117.0 | 9.7% | 3048 | 38.4 |
| `addmm_jit_supergemm` | 51.9 | 4.3% | 3048 | 17.0 |
| `hreduce_poly_jit` | 50.0 | 4.1% | 6223 | 8.0 |
| `mm_jit_supergemm_splitk` (reduce) | 13.3 | 1.1% | 3048 | 4.3 |
| *(Kuiper subtotal)* | *1089.8* | *90.1%* | | |
| Triton (SiLU, RoPE, RMS-norm scale, argmax, cache writes) | 98.3 | 8.1% | | |
| `multi_tensor_apply` | 21.9 | 1.8% | | |

### Stock

Total decode device time **719.3 ms** (5.66 ms/step).

| Kernel | Total (ms) | Share | Calls |
| --- | ---: | ---: | ---: |
| `ampere_bf16_s1688gemm_128x128_tn` | 217.9 | 30.3% | 6096 |
| `cudnn ... sdpa_sm80_flash_fprop_wmma` | 143.1 | 19.9% | 3048 |
| `ampere_bf16_s16816gemm_256x128_tn` (lm_head) | 90.8 | 12.6% | 127 |
| `ampere_bf16_s16816gemm_128x128_tn` | 82.4 | 11.5% | 3048 |
| `ampere_bf16_s16816gemm_64x64_sliced1x2_relu_tn` | 32.7 | 4.5% | 3048 |
| `ampere_bf16_s16816gemm_64x64_tn` | 26.6 | 3.7% | 3048 |
| `cublasLt::splitKreduce_kernel` | 12.6 | 1.8% | 3048 |
| *(cuBLAS + cuDNN subtotal)* | *606.1* | *84.3%* | |

### Where the gap is

| Workload | Kuiper (ms) | stock (ms) | ratio |
| --- | ---: | ---: | ---: |
| Attention | 485.2 | 143.1 | **3.39×** |
| GEMM (mm + addmm, incl. split-K reduce) | 554.7 | 463.0 | **1.20×** |
| Kuiper `hreduce_poly` vs its fused stock equivalent | 50.0 | n/a | n/a |
| Everything else (Triton, argmax, cache, RMS-norm) | 120.2 | 113.2 | 1.06× |

The RMS-norm mean is the one operator with no stock counterpart to compare
against: Kuiper runs it as a standalone `hreduce_poly` (50.0 ms), while stock
Inductor fuses the reduction into its norm kernel, so it does not appear as a
separate line. It is included in "everything else" on the Kuiper side.

The verified GEMMs are within 20% of cuBLAS across the whole model, which is
the headline result: SuperGEMM's software-pipelined tensor-core inner loop is
competitive with vendor kernels on every shape the model uses.

The remaining end-to-end gap is dominated by **flash attention**, which is
3.4× slower than cuDNN's and, at 40% of decode device time, is now the single
largest item in the profile. Batch-256 single-token decode attention is
memory-bound — it streams roughly 39 MB of KV cache per layer per step, which
is about 51 µs at A6000 bandwidth against the 159 µs measured — so the kernel
is running at roughly a third of the achievable rate. That is the highest-value
target for further work, well ahead of any GEMM tuning.

## GEMM shapes and tuned configurations

`tune_params.json`, selected by `KUIPY_AUTOTUNE=1` on this GPU at batch 256.
`M = 256` (batch) for every decode GEMM.

| Site | Op | K | N | Backend | Tile |
| --- | --- | ---: | ---: | --- | --- |
| `qkv_proj` | `addmm` (broadcast bias) | 896 | 1152 | supergemm | bm128 bn64 bk64 wm64 wn32 |
| `o_proj` | `mm` | 896 | 896 | supergemm | bm64 bn64 bk64 wm16 wn64 |
| `gate_proj`, `up_proj` | `mm` | 896 | 4864 | supergemm | bm128 bn128 bk64 wm64 wn32 |
| `down_proj` | `mm` | 4864 | 896 | supergemm_splitk | bm128 bn128 bk64 wm64 wn32, 4 splits |
| `lm_head` | `mm` | 896 | 151936 | supergemm | bm128 bn128 bk64 wm64 wn32 |

`down_proj` is the one K-heavy shape (K = 4864, N = 896), so it has too few
output tiles to fill the GPU; split-K across 4 partials wins there by a wide
margin — forcing it back to the plain kernel costs 15% end to end (22.9k →
19.5k tok/s). Split-K is CUDA-graph-safe, so it participates in graph capture
like any other backend.

`group` (the L2 output swizzle) is pinned to 1 pending a fix to the ownership
proof, so the wide-N GEMMs (`lm_head` especially) are leaving some L2 reuse on
the table.

## Why SuperGEMM trails the non-pipelined references

On a square compute-bound shape (4096³, fp16 in / fp32 accumulate / fp16 out)
SuperGEMM reaches 70.8 TFLOP/s where the unverified `gemm_tc` reaches 90.0 and
upstream `TensorCore2D` reaches ~88. A software-pipelined kernel losing to
non-pipelined ones is surprising, so the gap was profiled directly. It is not
the k-loop: at a matched tile the two kernels emit the same mainloop work (64
`HMMA.16816`, 16 `LDGSTS.E.BYPASS.128` each), neither spills, and both are
limited to 8 warps/SM at their respective tuned configurations. Three
independent causes account for it, in descending order.

**1. The epilogue scratch is not aliased with the pipeline buffers.**
`gemm_tc` sizes its dynamic allocation `max(pipe, epilogue)`; SuperGEMM must
sum them, because Kuiper's `SHMem` has no overlay descriptor and so cannot
express lifetime-based retyping of one allocation (`Mm.Shared.fsti`). The extra
band is what decides occupancy:

| bm128 bn128 bk32, fp16 | shared bytes | blocks/SM | TFLOP/s |
| --- | ---: | ---: | ---: |
| SuperGEMM `wm64 wn64` | 58368 (40960 pipe + 17408 band) | 1 | 56.3 |
| SuperGEMM `wm128 wn32` | 50176 (40960 + 9216) | 2 | 70.8 |
| SuperGEMM `wm32 wn64 bk64` | 108544 | — | exceeds the 101376 limit |
| `gemm_tc`, same tile, aliased | 37888 | 2 | 80.3 |

The square warp tile — the one `gemm_tc` prefers — drops SuperGEMM to one block
per SM, which is why autotuning walks away from it to the lopsided `wm128 wn32`
whose band is small enough to keep two. Aliasing would make the square tile cost
`max(40960, 17408) = 40960` and fit two blocks. It also currently costs whole
regions of the tile space outright, as the third row shows.

**2. The L2 swizzle is disabled.** `gemm_tc` takes `group` at run time, so the
swizzle can be priced on an otherwise identical kernel: 90.0 TFLOP/s at
`group=8` against 79.3 at `group=1`, or 13%. `Mm.Swizzle.fsti` already provides
the proven permutation and its bijection bridge; only the ownership reindex in
`Mm.Kernel.fst` is outstanding.

**3. SuperGEMM has two pipeline stages, not three.** `gemm_tc` at `STAGES=3`
beats itself at `STAGES=2` by 6.6% (90.0 against 84.4, both at `group=8`).
SuperGEMM's flip-flop barrier is fixed at two buffers.

So the honest apples-to-apples baseline is `gemm_tc` at two stages and
`group=1`, which is 80.3 TFLOP/s; SuperGEMM's 70.8 sits about 12% under it, and
that 12% is the shared-memory band consuming the occupancy headroom. The
remaining distance to 90 is the swizzle and the third stage.

## Notes on measurement

* All numbers are from CUDA graph replay, so host launch overhead is out of the
  loop in both configurations.
* The committed `tune_params.json` these numbers were taken with was produced
  by the eager timing loop. Autotuning now times candidates the same way the
  model runs them — recorded into a CUDA graph and replayed — because a Python
  dispatch (tens of µs) costs more than a decode-shaped GEMM does on the device,
  so eager timing picks winners out of host noise. Re-tuning under graph timing
  already changes the `o_proj` winner (19.9 µs against the 20.9 µs tile eager
  timing chose), so these figures are a floor, not a ceiling.
* The profiled runs use 128 generated tokens rather than 256 to keep the trace
  small; the per-step device time is unchanged.
