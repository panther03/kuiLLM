# kuiLLM — Copilot instructions

This repo contains a system for offloading PyTorch tensor operations to verified GPU kernels
written in Kuiper, a language for memory safe and verified GPU development built on top of 
F* and Pulse. It primarily targets operations used in LLM inference pipelines, currently 
using Qwen2.5-0.5B inference as a main integration target. 

Kuiper supports writing highly polymorphic, higher-order GPU kernels that are generic
in element types, tensor layouts (column major, row major, strided subtiles, etc.),
and even operations. For example, Kuiper implements matrix multiplication kernels 
that can take any layout of 2D matrix, on any floating point element datatype, and with
any generic function as the combinator between the result of A @ B and the C matrix.

Once these higher order aspects that cannot be represented in CUDA are specialized away,
we can _extract_ the specialized instance of the kernel to standard CUDA.

kuiLLM specializes, extracts, compiles, and executes Kuiper kernels just-in-time by 
exposing Python functions that implement `aten` operators. This system is in a package 
called `kuipy`. When a kuipy implementation of an operator is called, the parameters 
such as the datatype and shape of the input tensors inform the JIT extraction process 
and build a specialized instance of that kernel. It caches the built kernel as well. 
`kuipy` also exposes an Inductor pass that allows replacing calls in an operator graph 
with Kuiper kernels. This is used to run end-to-end LLM inference with the Kuiper kernels.

## Architecture

Organization:
- `kuipy/`: The Python code that supports JIT extraction, compilation, etc. and all the 
  glue needed to hook up the CUDA code to ATen/PyTorch-compatible Python operators. Some highlights:
  - `kuipy/kuiops.py` holds one `*Impl` class per operator family (subclasses of
    `_Family`). `supported()` returns `None` if no kernel parameterization fits,
    else a spec; `run()` instantiates templates, compiles via `_mod`, and invokes.
- `kuiops/`: Where the supporting F* and CUDA files for each operator lives. Each
   operator lives in `kuiops/<op>` and gets a `Kuiops.<Op>.Inst.fst.j2` (F*
   instantiation) and `wrapper_<op>.cu.j2` (Torch-tensor glue). Support F* code
   lives in `kuiops/Kuiops.<Op>.fst{i}`. These Jinja templates are what produce 
   the specialized kernel from the information available in the Python context.
   The fstar template gets extracted to CUDA and then linked against the cuda/C++ wrapper 
   that exposes a Torch::Tensor compatible interface.
- `etc/`: Experiments and miscellaneous support files.
- `tests/`: Unit tests for kuipy.
- `infer.py`: Where the main Qwen2.5 integration test lives.

The polymorphic kernels come from Kuiper binary packages or a separate
source repo (`$KUIPER_HOME`, usually `~/work/kuiper`); this repo instantiates,
extracts, compiles, and dispatches them.


## Build / run / test

For development, there is a `kuillm` micromamba environment you can use to 
provide python3 dependencies.
`environment.yml` (used by CI, see `.github/workflows/ci.yml`) declares the same
dependency set for a fresh micromamba/conda setup:
`micromamba create -f environment.yml && micromamba activate kuillm`. It does not
provide CUDA — nvcc 12.x must come from the host.

- `make infer` / `make infer-no-kuiper` — run Qwen2.5 inference with / without using kuiops kernels.
- `make verify`: An integration test that runs inference with every Kuiper-dispatched op alongside
   stock PyTorch and reports relative-Frobenius divergence.
- `make test` — full suite. Single test:
  `python -m pytest tests/test_jit_ops.py::test_bmm -s`.
  JIT tests require CUDA (they `pytest.skip` otherwise) and the **first run of each
  new kernel instantiation compiles via F*+nvcc (tens of seconds)**; reruns hit the
  on-disk cache.
- `make verify-kuiops` — F*-verify the `kuiops/*.fst{i}` support modules.
- The makefile provides an option to reinstall Kuiper, but please ask before running this,
  because usually concurrent work is happening at $KUIPER_HOME (so installing would trigger a
  long rebuild) and nightly/release might be incompatible with the current state of the 
  kuiops repo. Generally speaking, expect `inst/` to be present before starting work 
  and ask if it is not there.

Caches live in `.kuipy_cache/` (`src/`, `checked/`, `pre/`, `cu/`, `build/`).

## Conventions

- **`supported()` should be as broad as the *kernel's* refinements allow** — do
  not over-restrict dtypes. The source of truth is the typeclass/refinement on the
  Kuiper kernel: e.g. `map` has no `et` refinement, the `BlockTiling2D` backend of `mm`
  needs `scalar et` (int/float ok), and `TensorCore2D` needs valid fragment/accumulator types 
  (so f32 inputs can't use TC2D — fall back to BlockTiling2D). See `MmImpl` for the pattern.
- **No element-type / tile-size branching that selects between fixed kernels** in
  Python or templates: we are trying to make an interface that matches the fully general
  polymorphic Kuiper kernel. Some of these kernels will be implemented here in Kuiops,
  but they are usually upstreamed eventually to Kuiper. Once they are in Kuiper, the kernel
  generally comes with a `Klas` instantiation that fixes some parameters like element type or tensor layout etc.
  DON'T use these. Always use the `Kuiper.Kernel.` definition. The Klas version will usually be
  incompatible anyway as Kuiper ops should use an asynchronous kernel launch that accepts an input
  CUDA stream; the default mode that Klas kernels are written in is to spawn a fresh stream and call
  cudaStreamSynchronize(). This is illegal in CUDA graph mode, which is what we use to run inference with kuiops kernels.
- **KEEP TEMPLATES MINIMAL** (no proofs in there, as that verification cost will be paid on
  every JIT compilation). Anything with nontrivial proof obligations belongs in a support module.
  If it is reusable between multiple operators, you can consider adding a module at the root of `kuiops/`.
  Or you can suggest that it be upstreamed to kuiper itself.
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
- **Native autotuning**: `KUIPY_AUTOTUNE=1` exhaustively benchmarks legal Kuiper
  parameterizations and writes shape/device-specific winners to the committed
  `tune_params.json`; `KUIPY_TUNE_PARAMS` overrides its path. Normal runs only
  consume the file. Bump `TUNING_SCHEMA_VERSION` after tuning-relevant changes.
- **Operator calls**: Do not call aten operators in the Python or C++ integration code, such as `.to()`. Do not use these operators to implement PyTorch broadcasting semantics either; this should be handled by Kuiper kernels (although we do not have a reusable solution for this yet, so it is a known limitation that we do not support broadcasting). In general, `supported()` constraints should reflect EXACTLY what the kernel is capable of; any code implemented in the CUDA wrappers and Python is untrusted, and we would like to keep these as minimal wrappers that only do the bare minimum to integrate with the verified implementations. The burden of supporting more edge cases of the Python operators should fall on the Kuiper code, NOT on the Python or CUDA wrappers.
- **Allocations**: By convention, the Kuiper kernels do not allocate output tensors and instead take the output tensor as argument. In terms of the native_functions.yaml of ATen, If it is OK for an input tensor to be modified and it is in the same alias set as the output, then the Kuiper implementation shall modify that input in place instead of there being a separate argument (this is the case for a unary elementwise operation for instance). The allocation of the output tensor should happen on the CUDA/C++ template side, not in Python. The allocation could also be a copy of some input tensor, for example in the `addmm` operator which copies the C matrix to the output matrix and the Kuiper kernel modifies this in-place. 
- Remember that in F* and Pulse, you do not need to write `return` to return a value from a function; the last expression is returned automatically.
- Keep code concise and match existing style; IMPORTANT: AVOID EXCESSIVE COMMENTS. Ask first if you should use a separate worktree or stay on the main one. Commits include:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Reference docs

- `KERNELS.md` — checklist of every observed aten op signature and whether it's
  hooked up (the integration backlog).