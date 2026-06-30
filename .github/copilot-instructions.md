# kuiLLM — Copilot instructions

This repo runs an LLM inference pipeline (Qwen2.5-0.5B) in PyTorch where eligible
`aten` operators are JIT-replaced by **verified Kuiper GPU kernels**. Kuiper is an
F*/Pulse extension that writes separation-logic-verified GPU code and extracts it
to CUDA. The polymorphic kernels live in a separate repo (`$KUIPER_HOME`,
`~/work/kuiper`); this repo instantiates, extracts, compiles, and dispatches them.

## Build / run / test

Always use the project virtualenv interpreter — there is no system install:
`/home/julien/work/kuiLLM/.venv/bin/python` (Makefile targets just call `python3`,
so run them with the venv active or use the venv python directly).

- `make install-kuiper` — copy the built F* toolchain + Kuiper sources from
  `$KUIPER_HOME` into `./inst`. **Run this before anything else** (and after any
  change in the Kuiper repo). Requires `KUIPER_HOME` set (defaults to `~/work/kuiper`).
- `make infer` / `make infer-no-kuiper` — run `infer.py` with / without Kuiper.
- `python infer.py "prompt" --max-new-tokens 8 --temperature 0` — single prompt.
  Other flags: `--prompts FILE`, `--batch N`, `--no-kuiper`, `--timing`,
  `--verify` (run every Kuiper-dispatched op alongside stock PyTorch and report
  relative-Frobenius divergence), `--verify-tol`.
- `make test` — full suite. Single test:
  `.venv/bin/python -m pytest tests/test_jit_ops.py::test_bmm -s`.
  JIT tests require CUDA (they `pytest.skip` otherwise) and the **first run of each
  new kernel instantiation compiles via F*+nvcc (tens of seconds)**; reruns hit the
  on-disk cache.
- `make verify-kuiops` — F*-verify the `kuiops/*.fst{i}` support modules.

## Architecture (the JIT dispatch path)

1. `infer.py` wraps generation in `kuipy.KuiperMode`, a `TorchDispatchMode`
   (`kuipy/__init__.py`). It sees every `aten` call; on a hit it returns a Kuiper
   result, otherwise it falls through to stock PyTorch.
2. `kuipy/registry.py` maps `aten.*` overloads → a singleton `*Impl` family. The
   hot path is deliberately tiny: dict lookup → `impl.supported(...)` → `impl.run(...)`.
3. `kuipy/kuiops.py` holds one `*Impl` class per operator family (subclasses of
   `_Family`). `supported()` returns `None` if no kernel parameterization fits,
   else a spec; `run()` instantiates templates, compiles via `_mod`, and invokes.
4. `kuipy/compile.py` renders a one-line `.fst` (jinja) + a C++ pybind wrapper,
   then `toolchain.py` runs F* → karamel → `.cu/.h`, and `torch.utils.cpp_extension.load`
   builds/caches the `.so`. Loaded modules are memoised in-process; failed builds
   are negatively cached (`_failed`) so we don't re-run F*/nvcc every call.
5. `kuiops/<op>/` holds the jinja templates: `Kuiops.<Op>.Inst.fst.j2` (F*
   instantiation) and `wrapper_<op>.cu.j2` (Torch-tensor glue). Support F* code
   lives in `kuiops/Kuiops.<Op>.fst{i}`.

Caches live in `.kuipy_cache/` (`src/`, `checked/`, `pre/`, `cu/`, `build/`).

## Conventions

- **`supported()` should be as broad as the *kernel's* refinements allow** — do
  not over-restrict dtypes. The source of truth is the typeclass/refinement on the
  Kuiper kernel: e.g. `map` has no `et` refinement, `gemm` needs `scalar et` (int/
  float ok), and `TensorCore2D` needs valid fragment/accumulator types (so f32
  inputs can't use TC2D — fall back to BlockTiling2D). See `MmImpl` for the pattern.
- **No element-type / tile-size branching that selects between fixed kernels** in
  Python or templates. KEEP TEMPLATES MINIMAL (no proofs). Anything with nontrivial
  proof obligations belongs in `$KUIPER_HOME`, not here. Prefer instantiating a
  generic `Klas.<Kernel>.Inst` (e.g. `Klas.GEMM.TensorCore2D.Inst` fixes row-major
  layout) over `Kuiper.Kernel.<...>` when it saves proof steps — but only if the
  Klas isn't hardwired to fixed dtypes/tiles.
- **Naming**: F* modules/templates are in namespace `Kuiops`, titlecased after the
  aten op (`Kuiops.Mm`, `Kuiops.Addmm`). The instantiation template is always
  `Kuiops.<Op>.Inst.fst.j2`; supporting defs go in `Kuiops.<Op>.fst{i}`.
- **Wrapper output dtype**: allocate the output tensor with the *kernel's output*
  dtype, not `A.options()`. TC2D bf16 MM writes f32 output even for bf16 inputs;
  allocating with the input dtype undersizes the buffer → CUDA illegal memory
  access. Pass an `out_scalar` (`torch::kFloat32` etc.) into the wrapper context.
- **Strictness** via `KUIPY_JIT_STRICTNESS` (default 1): `0` = silent fallback to
  PyTorch on any JIT failure, `1` = raise on compile failure, `2` = also raise when
  an op isn't offloaded. Other env flags in `kuipy/config.py`:
  `KUIPY_JIT_VERBOSITY`, `KUIPY_JIT_VERIFY` (full F* verify vs admit-SMT),
  `KUIPY_JIT_FLUSH_CACHE`, `KUIPY_PRINT_PROFILING`.
- **Operator calls**: Do not call aten operators in the Python or C++ integration code, such as `.to()`. There is one exception right now in the form of `mm` which uses a cast at the output because it doesn't handle bf16 x bf16 -> bf16 matmul in Kuiper. Do not use these operators to implement PyTorch broadcasting semantics either; this should be handled by Kuiper kernels (although we do not have a reusable solution for this yet, so it is a known limitation that we do not support broadcasting). `supported()` constraints should reflect the kernel's ability to handle broadcasting, and `run()` should not attempt to implement it in Python.
- **Allocations**: By convention, the Kuiper kernels do not allocate output tensors and instead take the output tensor as argument. In terms of the native_functions.yaml of ATen, If it is OK for an input tensor to be modified and it is in the same alias set as the output, then the Kuiper implementation shall modify that input in place instead of there being a separate argument (this is the case for a unary elementwise operation for instance). The allocation of the output tensor should happen on the CUDA/C++ template side, not in Python. The allocation could also be a copy of some input tensor, for example in the `addmm` operator which copies the C matrix to the output matrix and the Kuiper kernel modifies this in-place. 
- Keep code concise and match existing style; IMPORTANT: AVOID EXCESSIVE COMMENTS. Do new work
  on a fresh git worktree. Commits include:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Reference docs

- `KERNELS.md` — checklist of every observed aten op signature and whether it's
  hooked up (the integration backlog).
- `notes.md` — the original design prompts describing the JIT approach.