# Kuiper Inductor backend — design & plan

**Status:** proposal / not implemented. Target: `torch 2.12.1 +cu13`, A6000 (sm86).

## 1. Thesis

The project replaces PyTorch's kernels — **cuBLAS, cuDNN, and JIT'd Triton** —
with verified Kuiper kernels. On the eager path, `KuiperMode` (a
`TorchDispatchMode`) does this per ATen op. But eager can't fuse, and the fused
Triton kernels that make `torch.compile` fast **bypass the dispatcher entirely**,
so `KuiperMode` can never touch them.

The answer is to make **Kuiper a first-class `torch.compile` (Inductor) codegen
backend** that serves *every node it can* with a verified kernel, leaving
PyTorch's kernels to cover only the remainder:

- **Plug into a fixed shape** — a `mm`/`bmm`/`addmm` node → a Kuiper GEMM;
  `scaled_dot_product_attention` → a Kuiper attention kernel. *(Replaces
  cuBLAS/cuDNN.)*
- **Generic reduce+map fusion** — Inductor's fused pointwise/reduction groups →
  a JIT-specialized Kuiper `map`/`reduce`. *(Replaces the `triton_*fused_*`
  kernels.)*
- **Epilogue/prologue fusion into Kuiper templates** — e.g. `relu(mm(x,w)+b)` as a
  *single* Kuiper GEMM whose epilogue is a fused `map`. This can even expose
  fusions Inductor's default path leaves on the table (§6.2).
- **Fallback** — anything Kuiper can't (yet) express degrades to Triton / cuBLAS /
  cuDNN, so the compiled artifact is always correct and CUDA-graphed. Coverage
  grows monotonically; we never risk the whole model.

The uniform enabler is that **every Kuiper kernel is a verified *polymorphic
higher-order* function**, specialized at JIT time by a rendered body: `map` takes
an arbitrary pointwise function, `reduce` an arbitrary associative combiner + a
pre-map, and a GEMM can take an epilogue `map`. So the *same*
instantiate-and-cache machinery (`kuipy/compile.py`) drives plug-in replacement,
generic fusion, and template-epilogue fusion alike.

`torch.compile` additionally gives us, for free, **CUDA-graph capture**
(`reduce-overhead`) and **one hook where all of the above coexist** — the
model-wide equivalent of what `KuiperMode` does per op.

## 2. Where we are

- `etc/infer_golden_compiled.py` (Triton steelman) ≈ **340 ms**; the interceptable
  manual graph `etc/infer_golden.py` ≈ **375 ms** (batch 256, 64 tok, bf16 decode).
- `kuipy/graph.py` registers a `kuiper` Dynamo/AOT backend that runs `boxed_nop`:
  it captures the graph then replays every ATen op eagerly so `KuiperMode`
  intercepts each one — but it **never calls Inductor**, so nothing fuses (only a
  hand-coded `mm+add→addmm` rule) and there are no CUDA graphs. That is exactly
  why it still shows standalone `rsqrt`/`mean` kernels and is slow (`KERNELS.md`).

This plan **supersedes the `boxed_nop` backend** with a real Inductor codegen
backend that keeps its "everything visible / Kuiper-first" spirit but gains
Inductor's fusion grouping, scheduling, memory planning, and CUDA graphs.

## 3. The uniform mechanism (higher-order kernels ↔ Inductor IR)

Kuiper's instantiation templates substitute an **arbitrary F\* body** into a
once-and-for-all verified kernel:

- **map** (`kuiops/elementwise/Kuiops.Elementwise.Inst.fst.j2`):
  `map_gpu #et (fun i0 i1 -> <body>) lena …`. `<body>` is any composition of the
  verified primitives in `Kuiops.Elementwise.fsti` (`add mul div rsqrt silu square
  neg sin cos bwhere` …). Fusing a pointwise chain = *composing* that body. No new
  proofs — `map_gpu` is proven for any pure function.
- **reduce** (`kuiops/reduce/…`, `Kuiops.HReducePoly.Exact`):
  `reduce #et (<combiner>) (<pre_map>) threads lena a` — polymorphic in **both** an
  associative combiner **and** a pre-map, so `reduce∘map` is direct. (`mean` is the
  one-block-per-row, last-axis specialization.)
- **GEMM** (`kuiops/mm|addmm|bmm/…`): a verified matmul (TensorCore2D / BlockTiling2D)
  selected by `MmImpl/AddmmImpl/BmmImpl`. A GEMM parameterized by an **epilogue
  `map`** turns `matmul → (bias/act) → out` into one kernel (§6.2; upstream if the
  epilogue-carrying GEMM variant doesn't yet exist in Kuiper).
- **attention** (`kuiops/sdpa/…`, `SdpaImpl`).

Inductor's IR is the same algebra, and — crucially — **Inductor has already done
the hard part** by the time a backend sees a node: its decompositions have broken
RMSNorm/RoPE/etc. into primitive `Pointwise`/`Reduction` IR, and its scheduler has
proven each fusion legal and grouped the nodes. We only translate.

| Inductor (`torch/_inductor`)                                   | Kuiper                                                |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| `ir.Pointwise.inner_fn` (an `ops.*` expression)                | `map_gpu` body `fun i… -> …`                           |
| `ir.Reduction` = `inner_fn` + `reduction_type`                 | `reduce (combiner) (pre_map) …`                       |
| `ops.mul/add/sub/div/neg/rsqrt/sin/cos/where/eq/…`             | `Kuiops.Elementwise.fsti` primitives                  |
| `ops.to_dtype` (f32 up/downcast around reductions)             | `Kuiper.Float.Casts.fcast`                            |
| `ops.constant`                                                 | `cast_constarg`                                       |
| `mm/addmm/bmm` template/extern node                            | Kuiper GEMM (TC2D/BT2D) via `Mm/Addmm/BmmImpl`        |
| matmul node + pointwise `epilogue_nodes`                       | Kuiper GEMM + epilogue `map` (one kernel)             |
| `scaled_dot_product_attention` extern                          | Kuiper attention (`SdpaImpl` / flash)                 |

## 4. Architecture — one backend, every node kind

Inductor supports coexisting per-device codegen backends, and the **CUTLASS
backend is the exact precedent**: `torch/_inductor/codegen/cuda_combined_scheduling.py`
defines `CUDACombinedScheduling(BaseScheduling)`, which holds a `TritonScheduling`
plus specialized schedulings, routes each node via `choose_node_backend` /
`is_<x>_template`, fuses epilogues into templates via `codegen_template(...,
epilogue_nodes, prologue_nodes)`, and **falls back to `_triton_scheduling` for
everything it doesn't claim**. We mirror it:

```
register_backend_for_device(                          # codegen/common.py:400
    "cuda", KuiperCombinedScheduling, <default cuda PythonWrapperCodegen>)

class KuiperCombinedScheduling(BaseScheduling):
    def __init__(self, scheduler):
        self._triton = TritonScheduling(scheduler)     # fallback (may pick cuBLAS/cuDNN extern)
        self._kuiper = KuiperScheduling(scheduler)     # our codegen

    # --- template nodes: matmul / attention (+ epilogue/prologue fusion) ---
    def codegen_template(self, tmpl, epilogue_nodes, prologue_nodes):
        if self._kuiper.is_kuiper_template(tmpl):       # a Kuiper GEMM/attn shape
            return self._kuiper.codegen_template(tmpl, epilogue_nodes, prologue_nodes)
        return self._triton.codegen_template(tmpl, epilogue_nodes, prologue_nodes)

    # --- fused pointwise / reduction groups ---
    def codegen_node(self, node):
        if self._kuiper.is_kuiper_group(node):          # expressible as map/reduce
            return self._kuiper.codegen_node(node)
        return self._triton.codegen_node(node)

    # --- allow pointwise epilogues to fuse INTO Kuiper templates & groups ---
    def can_fuse_vertical(self, a, b):   ...            # accept where Kuiper supports it
    def can_fuse_horizontal(self, a, b): ...
    # group_fn / codegen_sync / flush / benchmark_* : delegate to self._triton
```

Two complementary ways a matmul becomes Kuiper:

1. **Template node** — Kuiper GEMM registered as an Inductor matmul template (so
   `codegen_template` can fuse pointwise epilogues into it; §6.2).
2. **Autotune choice** — register a Kuiper GEMM `ChoiceCaller` in the `mm`/`addmm`
   lowering's choice list (`torch._inductor.select_algorithm`) so Inductor
   benchmarks Kuiper vs Triton-template vs cuBLAS-extern and picks the fastest —
   or is *forced* to Kuiper for the "run on Kuiper" product build (§11).

**Fallback chain:** Kuiper → Triton/Inductor-default → cuBLAS/cuDNN extern. Because
unclaimed nodes fall through to real Triton, the compiled module is always correct
and CUDA-graphed regardless of Kuiper coverage.

### 4.1 This is all out-of-tree (no torch fork)

Inductor's backend system is **designed for out-of-tree backends**:
`register_backend_for_device` is documented in-source as "the registration API to
equip a new backend at runtime," and the comment right above it cites Intel's
`intel-extension-for-pytorch` as the reference *out-of-tree* backend
(`torch/_inductor/codegen/common.py:379-408`). Everything in this plan is
`import torch._inductor…` + register/subclass from our own `kuipy/inductor/`
package — nothing patches torch. There are two tiers, differing only in API-stability
risk:

- **Tier A — supported hooks (robust).** `register_backend_for_device("cuda", …)`
  (it even accepts a per-device `device_custom_pass`); `config.post_grad_custom_post_pass`
  / `pre_grad_custom_pass` (FX passes via `torch._inductor.custom_graph_pass.CustomGraphPass`);
  `torch.library.custom_op` (+ meta) for opaque Kuiper externs Inductor schedules &
  cudagraphs; `register_lowering` / `register_decomposition`. The **custom-pass +
  custom-op combo alone can express the *entire* plan** — GEMM replacement, epilogue
  fusion done inside the pass, reduce+map by pattern — fully out-of-tree on stable
  hooks, at the cost that *we* do the grouping (pattern-based) instead of reusing
  Inductor's scheduler.
- **Tier B — internal-API subclassing (out-of-tree but fragile).** Subclass
  `BaseScheduling`/`TritonScheduling` and `DefaultHandler`, read `SchedulerNode`
  `inner_fn`, drive `V.ops` tracing. This is what buys the *generic* "reuse Inductor's
  own fusion grouping and only re-target codegen" story (§6.1). Still no fork — you
  import and subclass — but you are coupled to unstable `torch._inductor` internals,
  so the adapter layer + API-pin guard (§13/§14) are mandatory.

**One genuinely awkward spot:** injecting a Kuiper GEMM as an autotune
*choice/template* — `tuned_mm` (`torch/_inductor/kernel/mm.py`) builds its choice
list in-tree with no public "add a choice" hook (that is why CUTLASS lives in-tree).
Out-of-tree that means monkeypatching `tuned_mm` / overriding the `aten.mm` lowering,
or — cleaner — replacing matmuls in a custom pass → Kuiper custom op instead of routing
through autotune. So the GEMM plug-in (P3) leans on Tier A, not the elegant autotune
route.

**When you'd actually need in-tree:** only to change Inductor's fusion
legality/heuristics themselves, or to add a natively-understood external template.
Our scheduler subclass already gets epilogue fusion via `codegen_template`, so a fork
is not required for anything here.

## 5. Node coverage matrix

| Inductor node kind                         | Kuiper mechanism                                  | Kuiper kernel today            | Fallback           |
| ------------------------------------------ | ------------------------------------------------- | ------------------------------ | ------------------ |
| `mm` / `addmm` / `bmm`                      | GEMM template **or** autotune choice              | yes (`Mm/Addmm/BmmImpl`)       | cuBLAS extern      |
| matmul + pointwise epilogue                 | GEMM + epilogue `map` (§6.2)                       | **needs epilogue-GEMM** (upstream) | Triton template / unfused after cuBLAS |
| `scaled_dot_product_attention`              | Kuiper attention template/extern                  | `SdpaImpl` (efficient-attn)    | cuDNN/flash extern |
| fused **pointwise** group (same-shape, contig) | `map_gpu(body)` (§6.1)                          | yes                            | Triton             |
| fused **reduction** group (last-axis, sum/prod/max/min) | `reduce(combiner, pre_map)` / row-reduce | yes (`reduce`, generalize `mean`) | Triton    |
| **broadcast** pointwise (norm/RoPE epilogues) | broadcast-aware `map` (§10)                     | **needs broadcast map** (upstream) | Triton         |
| welford/var, multi-output, gather-in-body, unsupported `ops.*` | —                                | —                              | Triton / extern    |

The matrix *is* the roadmap: each row is an independently-gated capability that
falls back cleanly, so we can land them in any order and measure each in isolation.

## 6. Fusion capabilities

### 6.1 Generic reduce+map (the pointwise/reduction tail)

Inductor exposes each fused body as an `inner_fn` traced against a virtualized
`ops.*` handler (`torch/_inductor/ops_handler.py`); `KernelFormatterHandler` there
already "runs `inner_fn` and formats each op as a **string**". We write the
analogous `KuiperOpsHandler(DefaultHandler)` that returns F\* fragments:

```
class KuiperOpsHandler(DefaultHandler):
    def load(self, buf, index):      return self._input_var(buf, index)   # contiguous same-shape ⇒ "i0","i1",… else Unsupported
    def constant(self, v, dtype):    return cast_constarg(v, dtype)
    def mul(self, a, b):   return f"(mul {a} {b})"
    def add(self, a, b):   return f"(add {a} {b})"
    def rsqrt(self, x):    return f"(rsqrt {x})"
    def where(self, c,a,b):return f"(bwhere {c} {a} {b})"
    def to_dtype(self,x,dt,*_): return f"(fcast #{fstar(dt)} {x})"
    def <no verified primitive>(self,*a): raise KuiperUnsupported(...)
```

- Coverage = the op→primitive table. Growing it = adding one
  `inline_for_extraction` verified def to `Kuiops.Elementwise.fsti` + one handler
  method (`sqrt`, `exp`, `max`, `min`, …).
- For a `Reduction`, trace `inner_fn` → `pre_map`, and map `reduction_type`
  (`sum→add`, `prod→mul`, `max→max`, `min→min`; welford ⇒ Unsupported).
  `add/mul/max/min` have real `is_associative` witnesses; the reduce template has
  an `admit_associativity` hatch for anything that must be admitted.
- `KuiperUnsupported` anywhere while tracing a group ⇒ `is_kuiper_group=False` ⇒
  Triton. The supported subset is self-describing and safe.

### 6.2 Epilogue/prologue fusion into Kuiper templates (the "new opportunity")

Inductor calls `codegen_template(template_node, epilogue_nodes, prologue_nodes)`;
`epilogue_nodes` are the pointwise ops that follow the matmul. A Kuiper GEMM whose
**epilogue is a higher-order `map`** consumes those nodes' `inner_fn` (via the same
`KuiperOpsHandler`) and applies it to each accumulator tile before store — so
`relu(addmm(b, x, w))` becomes **one** Kuiper kernel: `gemm … |> map (fun acc ->
relu (add acc bias))`.

Honest framing of "expose a fusion PyTorch misses":

- Inductor **already** epilogue-fuses into its *Triton* matmul templates, so we
  will not generally beat it there.
- The genuine opening is (i) when Inductor picks the **opaque cuBLAS extern** (no
  epilogue fusion — bias/activation run as a separate kernel) yet a Kuiper GEMM
  template *can* fuse them, and (ii) fusions that require a custom kernel body
  Inductor's fixed template set doesn't cover but Kuiper's higher-order kernels
  express generically.
- **Dependency:** this needs a Kuiper GEMM variant parameterized by an epilogue
  `map` (and, for bias, the same broadcast story as §10 along the N dimension). If
  absent it is an upstream `$KUIPER_HOME` deliverable — proofs belong there, not in
  a template here.

## 7. Kernel build + launch (reuse existing JIT infra)

The F\*→`.cu`→`.so` pipeline is reused unchanged: `kuipy/compile.py::build_kernel`
(in-proc memo `_loaded`, on-disk `cpp_extension` cache, negative cache `_failed`,
cross-process `FileLock`, and the `batch_capture` combined build) + `toolchain.py`.
The backend adds only new `.Inst.fst.j2` renderings (no proofs), keyed by a content
hash of the F\* body + dtypes + shape class so identical instantiations share one
cached `.so`.

Making Inductor's generated wrapper call the kernel:

- **(preferred) codegen-time JIT + runtime shim** — during codegen, render the body,
  `build_kernel(...)` (cached), and emit a call to a tiny
  `kuipy.inductor.launch(key, ins, out)` shim that looks the module up by `key` and
  launches it. Composes with `_loaded`/batch-capture; keeps F\*/nvcc out of the hot
  path.
- **(alt) `torch.library.custom_op`** — register each instantiation as an opaque
  `kuiper::…` op with a meta/fake kernel; emit as `ir.FallbackKernel`/extern.
  Cleaner scheduling integration, more per-op boilerplate.

## 8. CUDA-graph safety (must-fix, P0)

Under `reduce-overhead`, Inductor CUDA-graphs the region *including* Kuiper
kernels. Current wrappers are **not** graph-safe:

- Every wrapper calls `kuiops::sync_current_stream()` (e.g.
  `elementwise/wrapper_elementwise.cu.j2`, `mean/wrapper_mean.cu.j2`) — a stream
  **sync inside a captured graph is illegal**. Under Inductor, run on the current
  stream, no sync (Inductor owns ordering).
- Prefer **writing into the Inductor-provided output buffer** over `torch::empty`
  mid-graph (Inductor already plans the group's output; this also matches the repo
  convention that Kuiper kernels take the output as an argument). Any mid-graph
  allocation must use the caching allocator's graph pool.

Add a `graph_safe` wrapper variant (or flag) to the shared templates.

## 9. Worked examples

**RMSNorm** `x·rsqrt(mean(x²)+eps)·w` → Inductor emits a `Reduction` (Σx² in f32)
+ a `Pointwise` epilogue:
- Reduction → Kuiper: `pre_map = fun i -> square (fcast #f32 i)`, `combiner = add`,
  last-axis row-reduce ⇒ `[m,1]` f32.
- Epilogue loads the reduced scalar **broadcast** across the row and `w` `[d]`
  broadcast across rows ⇒ needs the §10 broadcast map; until then → Triton.

**Linear + bias + SiLU** `silu(addmm(b, x, w))` → one Kuiper GEMM template with
epilogue `map (fun acc -> silu (add acc bias))` (§6.2), instead of cuBLAS + two
pointwise kernels. Exercises the template + epilogue-fusion track end-to-end.

## 10. Broadcasting prerequisite (critical, upstream Kuiper)

The high-value epilogues (RMSNorm weight, the reduced-mean scalar, RoPE `cos`/`sin`,
GEMM bias) all need **broadcast**. `map_gpu2/3` assume one shared `l1_forward`
layout, and the repo lists broadcasting as unsolved (`run()` must not fake it in
Python). A **broadcast/stride-aware `map`** is therefore a `$KUIPER_HOME`
deliverable (indexing/layout proofs). It is the single highest-leverage upstream
task: it unlocks norm/RoPE epilogues *and* GEMM-bias epilogues, i.e. the difference
between "Kuiper owns the reductions + plain pointwise" and "Kuiper owns essentially
the whole non-GEMM/attention surface."

## 11. Measurement configurations (not scope limits)

These are runtime configs of the *same* backend, selected via env
(`KUIPY_INDUCTOR_MODE`), used to answer different questions honestly:

- **`FUSION_ONLY`** — Kuiper claims only pointwise/reduction groups; GEMM/attention
  deliberately stay on cuBLAS/cuDNN (byte-identical to the Triton steelman) so the
  comparison isolates *who fuses the tail, Triton vs Kuiper*. This is the
  `infer_golden_compiled.py`-equivalent reference.
- **`ALL_KUIPER`** — force Kuiper for every supported node (GEMM, attention, fused
  tail); the "run the model on verified kernels" product build. Remainder falls back.
- **`AUTOTUNE`** — Kuiper competes as an autotune choice; Inductor picks the fastest
  per shape. Keeps perf claims honest where Kuiper GEMM may trail cuBLAS.

## 12. Phased plan (independent capability tracks)

- **P0 — scaffold + parity + graph-safety.** `KuiperCombinedScheduling` that claims
  nothing (delegates all to Triton); assert identical numerics + nsys profile vs
  340 ms. Land the graph-safe wrapper variant. De-risks the Inductor API surface and
  the launch/CUDA-graph integration alone.
- **P1 — pointwise `map`.** `KuiperOpsHandler` + claim same-shape contiguous
  single-output groups. Grow `Kuiops.Elementwise.fsti` as needed.
- **P2 — reductions.** `Kuiops.RowReduce.Inst` (generic combiner + pre_map); claim
  last-axis `sum/prod/max/min`.
- **P3 — GEMM plug-in.** Register Kuiper GEMM as matmul template + autotune choice
  (`mm/addmm/bmm`). High value (≈75% of runtime); force or autotune (§11).
- **P4 — epilogue fusion.** GEMM + epilogue `map` (bias/activation); needs the
  epilogue-carrying Kuiper GEMM (upstream if absent). The `relu(mm+b)` opportunity.
- **P5 — broadcast map (upstream).** Consume the stride/broadcast-aware `map`;
  unlock RMSNorm/RoPE and GEMM-bias epilogues end-to-end.
- **P6 — attention.** Kuiper attention template/extern for SDPA.
- **P7 — coverage + perf.** Multi-output groups, tuning, autotune, iterate on nsys.

Tracks are largely independent and each falls back; order by value/risk (P3 is high
value but perf-sensitive; P1/P2 are the safest generic-fusion showcase; P5 is the
key upstream unblock).

## 13. Verification & testing

- **Numerics** — compiled module vs eager reference within `--verify-tol` (fused
  kernels bypass the dispatcher, so the eager `KuiperMode(verify=True)` harness is
  reused for *per-op* Impls but a compile-path output check is added).
- **"Actually Kuiper" guards** — per track, assert the generated wrapper contains
  the Kuiper shim / that `codegen_node`/`codegen_template` claimed the expected
  nodes (matmul template chosen, groups fused), so a silent regression to
  all-Triton/cuBLAS **fails loudly**.
- **Graph capture** — assert `reduce-overhead` records once, no cudagraph skips, no
  recompiles, with Kuiper kernels present.
- **Perf gates** — batch 256 / 64 tok / bf16 decode: `FUSION_ONLY` vs 340 ms;
  `ALL_KUIPER` vs eager `infer.py`; add `profile-golden-*` Makefile targets.
- **API-pin guard** — smoke test that fails clearly if the private `torch._inductor`
  symbols we depend on move; isolate touch-points in `kuipy/inductor/adapter.py`.

## 14. Risks & open questions

- **Private Inductor API churn** — `register_backend_for_device`, `SIMDScheduling`
  /`TritonScheduling`, `OpsHandler`/`DefaultHandler`, `codegen_template`,
  `select_algorithm`, `ir.Pointwise/Reduction` are internal. Pin `torch 2.12.1`;
  wrap behind an adapter; guard with the smoke test.
- **Kuiper GEMM/attention perf vs cuBLAS/cuDNN** — likely trails on some shapes.
  `AUTOTUNE` keeps perf honest; `ALL_KUIPER` forces coverage for the product story.
- **Broadcasting** (§10) and **epilogue-carrying GEMM** (§6.2) are upstream Kuiper
  deliverables gating the highest-value fusions.
- **CUDA-graph safety** of wrappers (§8) — P0.
- **Reduction layout** — matching Inductor's `ranges`/`reduction_ranges` split
  (persistent vs multi-pass) to Kuiper row-reduce; big rows / non-last-axis → fall
  back meanwhile.
- **Type promotion** — honor Inductor `to_dtype` upcasts via `fcast` to stay in tol.
- **Prologue fusion** (pointwise on GEMM inputs) is a stretch vs epilogue.
- **MVP escape hatch** — if the Scheduling API is too unstable, a
  `post_grad_custom_pass` that pattern-replaces reduce+map subgraphs (and matmul+
  epilogue) with `kuiper::…` custom ops (opaque externs Inductor schedules + graphs)
  reuses §7/§8 wholesale, at the cost of pattern-based rather than generic grouping.

## 15. Success criteria

1. An **`ALL_KUIPER` compiled build** where every node Kuiper supports — GEMMs,
   attention, and fused pointwise/reduction groups (broadcasts per P5) — runs on
   verified Kuiper kernels, the remainder falls back to Triton/cuBLAS/cuDNN, the
   whole region is **CUDA-graphed**, and numerics are within tol.
2. A **`FUSION_ONLY` build** that matches **≈340 ms** — proving the generic
   reduce+map tail is competitive with Triton fusion while fully interceptable.
3. Kuiper coverage grows only by adding verified primitives/kernels, never by
   hardcoding fused operators; anything unsupported degrades gracefully.

## 16. File references

- Backend seam: `torch/_inductor/codegen/common.py:400` (`register_backend_for_device`);
  `codegen/cuda_combined_scheduling.py` (`CUDACombinedScheduling` — the precedent,
  incl. `codegen_template(..., epilogue_nodes, prologue_nodes)`);
  `codegen/simd.py:1308,1841` (`SIMDScheduling`, `codegen_node`);
  `codegen/triton.py:6453` (`TritonScheduling`); `torch._inductor.select_algorithm`
  (autotune `ChoiceCaller`s for matmul).
- IR / body tracing: `ir.py:1126` (`Pointwise`), `:1276` (`Reduction`);
  `ops_handler.py:60` (`OpsHandler`), `:991` (`KernelFormatterHandler`),
  `DefaultHandler`.
- Kuiper JIT infra (reused): `kuipy/graph.py` (current `boxed_nop` backend to
  replace), `kuipy/compile.py` (`build_kernel`), `kuipy/toolchain.py`,
  `kuipy/kuiops.py` (`ElementwiseImpl`, `MeanImpl`, `Mm/Addmm/BmmImpl`, `SdpaImpl`,
  `cast_constarg`, `torch_dtype_to_*`), `kuipy/registry.py`.
- Kernel templates: `kuiops/elementwise/*` (`map_gpu` body), `kuiops/reduce/*`
  (`reduce` = combiner + pre_map), `kuiops/mean/*` (row-reduce), `kuiops/mm|addmm|bmm/*`,
  `kuiops/sdpa/*`, `kuiops/Kuiops.Elementwise.fsti` (verified primitives).
- Baselines: `etc/infer_golden_compiled.py` (Triton steelman, 340 ms),
  `etc/infer_golden.py` (manual graph, 375 ms), `KERNELS.md`, `notes.md`.
