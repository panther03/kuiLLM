# kuiLLM

A set of verified, memory-safe Kuiper kernels for LLM inference, and a PyTorch backend that automatically maps operators to those kernels. The PyTorch part is structured as an Inductor pass that replaces suitable operators in the graph with Kuiper implementations. These nodes are then dispatched to JIT-verified, extracted, and compiled Kuiper kernels at runtime (during CUDA graph capture to be specific).

![](etc/diagram.png)

Using the kuiLLM backend is simply a matter of: 

```python
import torch
import kuipy
kuipy.enable() # hooks in Inductor backend
```

## Organization

- `kuipy/`: The Python code that supports JIT extraction, compilation, etc. and all the 
  glue needed to hook up the CUDA code to ATen/PyTorch-compatible Python operators. Some highlights:
  - `kuipy/kuiops.py` holds one `*Impl` class per operator family (subclasses of
    `_Family`). `supported()` returns `None` if no kernel parameterization fits,
    else a spec; `run()` instantiates templates, compiles via `_mod`, and invokes.
  - `kuipy/registry.py` maps each aten op to its family; `kuipy.run(op, **kwargs)`
    returns a drop-in replacement for `op` that always goes through the Kuiper
    kernel (raising instead of falling back), which is what tests and benchmarks use.
  - `kuipy/unverified/` holds unverified CUDA kernels used as benchmarking
    references (see its docstring for conventions).
  - `kuipy/benchmarking.py` is the generic benchmark driver; the benchmarks
    themselves live in `bench_ops.ipynb` at the repo root.
- `kuiops/`: Where the supporting F* and CUDA files for each operator lives. Each
   operator lives in `kuiops/<op>` and gets a `Kuiops.<Op>.Inst.fst.j2` (F*
   instantiation) and `wrapper_<op>.cu.j2` (Torch-tensor glue). Support F* code
   lives in `kuiops/Kuiops.<Op>.fst{i}`. These Jinja templates are what produce 
   the specialized kernel from the information available in the Python context.
   The fstar template gets extracted to CUDA and then linked against the cuda/C++ wrapper 
   that exposes a Torch::Tensor compatible interface.
    - `kuiops/common` has generic kernel implementations and other reusable functions between different operators.
- `etc/`: Experiments and miscellaneous support files.
- `tests/`: Unit tests for kuipy.
- `infer.py`: Where the main Qwen2.5 integration test lives.
- `bench_ops.ipynb`: Kernel benchmarks (verified + unverified) against stock PyTorch.

## Details

The rest of this file is background on *why* the system is shaped this way. For
the engineering specifics — build targets, environment flags, file-naming
conventions, coding rules — see `.github/copilot-instructions.md`, which is kept
current.

### The problem

GPU kernels are where the performance is, and also where the bugs are. A GEMM is
a few hundred lines of index arithmetic, asynchronous copies and barriers, and
essentially all of it is unchecked: a race between a tile load and the barrier
that guards it, or an index that walks off the end of a shared-memory buffer,
produces silently wrong numbers or a crash far from the cause. The usual answer
is to test, but tests only cover the shapes you thought to try, and the
interesting failures are the ones that depend on a particular block count or a
particular interleaving.

Kuiper's answer is to prove the kernel correct instead. It is a DSL embedded in
[F\*](https://fstar-lang.org) and Pulse (F\*'s concurrent separation logic), so a
kernel is checked before it runs for memory safety, data-race freedom, and
functional correctness against a mathematical specification. It then extracts to
ordinary CUDA. The proof is discharged once, at build time; the kernel that runs
is normal compiled code with no residual checking.

The obvious objection is that nobody wants to write a proof per matrix shape.
That is what this repository is about.

### Polymorphism, and why it forces a JIT

The reason a proof per shape is avoidable is that Kuiper kernels are far more
generic than CUDA can express. One matmul is written once and is generic in the
element type, in the *layout* of each operand (row major, column major, a
strided subtile of a larger tensor, a transposed view), in the accumulator type,
and even in the function combining the product with the output — so `mm`,
`addmm`, and a fused activation are the same kernel at different instantiations.
The proof is done once, generically, and holds for every instantiation.

CUDA has no equivalent of a higher-order type-generic kernel, so the genericity
has to be erased before code generation. Erasing it requires knowing the actual
types, layouts and tile sizes, and those are only known when the operator is
actually called with real tensors. Hence the pipeline runs at *runtime*:

    aten op + real tensor metadata
      -> supported()      does a legal parameterization exist for these inputs?
      -> Jinja templates  emit a monomorphic F* instantiation of the generic kernel
      -> F*               check it, then extract to CUDA
      -> nvcc             compile, link against a small Torch-tensor wrapper
      -> cached .so       reused forever after, keyed on the parameterization

The first call for a new parameterization costs tens of seconds; every later
call is a dict lookup. A model has only a handful of distinct GEMM shapes, so
this amortizes almost immediately. `kuiops/<op>/Kuiops.<Op>.Inst.fst.j2` is the
template that performs the specialization; anything with real proof content
lives in a support module beside it, so that the JIT does not pay to re-verify
it on every instantiation.

### Getting the kernels into a real model

Two entry points, for two different purposes.

The direct one is `kuipy.run(aten_op)`, which returns a drop-in replacement for
that operator that always goes through Kuiper and raises rather than silently
falling back. Tests and benchmarks use it, because they need to know they
measured the thing they meant to measure.

The one that matters for inference is an **Inductor post-grad pass**. After
PyTorch has traced, decomposed and functionalized the model into an FX graph of
ATen operators, the pass walks that graph and replaces every node a Kuiper
kernel can serve with a custom op bound to it. This is the right layer to
intervene at: the graph is already canonicalized, so there is a small set of
operator forms to match rather than the whole surface of the Python API, and
adjacent elementwise nodes can be fused into a single kernel's epilogue before
the replacement happens. Running Qwen2.5 end to end is then the integration
test — `infer.py`, with `make verify` running each dispatched op alongside its
stock reference and reporting relative Frobenius divergence.

One consequence is worth calling out because it shapes a lot of the code:
inference runs under **CUDA graph capture**. A captured graph records device
work, so a kernel may not synchronize, may not allocate, and may not make
host-side decisions during capture. This is why the JIT has a batch mode that
extracts every kernel it sees during a first pass, defers execution to PyTorch,
and then compiles them all into a single translation unit — and why kernels take
a stream and never spawn one, and never call `cudaStreamSynchronize`.

### What is actually trusted

The proof covers the Kuiper kernel: no out-of-bounds access, no data race, and
agreement with the functional specification. It does not cover the glue. The
Python that decides a parameterization is legal, and the C++ wrapper that
unpacks `torch::Tensor` into pointers and extents, are untrusted, so the design
rule is that they stay as thin as possible. Specifically:

- `supported()` must accept **exactly** what the kernel's own refinements allow —
  no broader (unsound) and no narrower (leaving performance unclaimed). The
  source of truth is the typeclass constraint in the F\*, not a guess.
- Missing PyTorch semantics — broadcasting, unusual layouts, edge-case dtypes —
  get fixed by making the *kernel* handle them, never by adding a fixup in the
  wrapper. A wrapper that quietly transposes or copies is an unverified kernel
  wearing a verified one's name.

Note also that the default JIT mode admits SMT queries rather than re-proving
each instantiation, since cold-compile latency matters; `KUIPY_JIT_VERIFY=1`
runs the real thing. The generic proof — which is where the guarantee actually
comes from — is checked separately with `make verify-kuiops`.

### Performance

Verified does not have to mean slow, but it does mean some optimizations are
harder to reach, because each one has to be expressed in a way the proof can
follow. The GEMMs here are software-pipelined tensor-core kernels competitive
with hand-written unverified CUDA and within a factor of the vendor library.
`PERFORMANCE.md` documents where the remaining gaps come from and which are
artifacts of the verification framework rather than of the algorithm —
currently, for example, Kuiper cannot alias one shared-memory allocation over
another's lifetime, which costs occupancy in the largest GEMM.

Kernel parameterizations (tile sizes, warp tiling, split-K) are chosen by an
autotuner that benchmarks the legal candidates per shape and records the winners
in `tune_params.json`; ordinary runs only read that file. Unverified reference
implementations of the same kernels live in `kuipy/unverified/` so the cost of
verification can be measured directly rather than guessed at.
