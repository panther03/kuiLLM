# SDPA integration and Karamel extraction notes

## SDPA policy

The ATen integration for
`_scaled_dot_product_efficient_attention` uses the Kuiper naive SDPA kernel with
these constraints:

- A bias tensor is required.
- Dropout, causal attention, and `compute_log_sumexp=True` are unsupported.
- Q, K, V, and bias may use any ordinary PyTorch dtype. Python casts all four
  inputs to f32 before dispatch.
- The result is cast to the original bias dtype.
- The kernel uses the bias as scratch space, so the C++ wrapper clones it.
- The kernel does not compute LSE. The wrapper returns an empty f32 LSE with
  shape `(N, H, 0)`.

## Why Karamel produced `any`

F* can verify generic dependent tuple transformations such as:

```text
[N,H,S,E] -> transpose -> [N,H,E,S] -> fold -> [N*H,E,S]
```

The corresponding concrete index type is `conc shape`, where the nested tuple
type depends on the shape. During extraction, however, shape arguments and
proofs are erased before every generic record call has been specialized.
Expressions involving `fold_outer`, `tail`, or `conc` can therefore lose enough
type information that Karamel represents them as Top (`any`).

The most common failure looked like:

```text
tuple <: any
```

at a `ctlayout.cimap` call, followed by Karamel Warning 26. This is not an F*
proof failure: F* still knows that the tuples and layouts correspond. It is an
extraction failure caused by the erased type at the computational call
boundary.

Adding `cshape` witnesses or more equality proofs was insufficient. Those
witnesses prove the transformation inside F*, but Karamel had already inserted
the Top cast before inlining the generic `ctlayout` record. Similarly, wrapping
a call in `hide` did not reliably remove surrounding computational record
construction from the extracted program.

## Rejected workaround

An early workaround added rank-3/rank-4 fold implementations and fixed
row-major layouts directly to `Kuiper.Kernel.SDPA.Naive`. It extracted, but it
made the polymorphic Kuiper kernel depend on SDPA's current ranks and layouts.
Do not reintroduce that approach.

The Kuiper kernel must remain generic over its input layouts. Concrete rank and
layout choices belong at the JIT instantiation boundary.

## Current extraction pattern

`Kuiper.Kernel.SDPA.Naive.sdpa_naive` receives concrete witnesses for every
derived layout it uses:

- folded Q;
- folded, transposed K;
- once- and twice-folded bias;
- folded V;
- folded output.

The kernel uses those witnesses but does not select row-major/column-major
layouts or fixed ranks itself.

The generic Kuiper layout library exposes small ghost lemmas needed to prove
the witnesses:

- `tlayout_bij_imap` exposes the index map of a bijection-derived layout;
- `fold_bij_gg` exposes that the inverse fold mapping is `unfold_index`;
- row-major and column-major `imap` lemmas expose their concrete arithmetic.

`Kuiops.Sdpa.Inst.fst.j2` then fixes the actual row-major inputs and constructs
the derived `ctlayout` record literals. The fixed-rank proof/index helpers live
in `Kuiops.Sdpa.fst`, not in the generic Kuiper kernel.

For row-major `[N,H,S,Ev]`, folding the first two dimensions is a rank-3
row-major layout `[N*H,S,Ev]`. For row-major K `[N,H,S,E]`, transposing its last
two dimensions and folding produces `[N*H,E,S]` with a rank-3 batched
column-major index:

```text
batch * (S * E) + s * E + e
```

The scalar `fold4_rm_cimap` and `fold_transpose4_rm_cimap` helpers prove that
these concrete formulas equal the abstract fold/bijection layouts.

## Karamel rules learned here

- Do not pass a dependent `conc` tuple through an extra extracted helper
  boundary. Destructure the fixed-rank tuple in the instantiation and pass
  scalar indices to the helper.
- Do not return a `ctlayout` from a support helper when its `cimap` must be
  specialized. Construct the record literal in the `.Inst.fst.j2` function.
- Keep proof-only layout reasoning ghost. Computational `ctlayout` records used
  only to establish an equality can survive extraction and recreate the `any`
  call.
- Use explicit nested product types for fixed-rank transpose/fold helpers when
  `conc` would leave the tuple structure opaque.
- `inline_for_extraction noextract` must be adjacent to its declaration.
  Place `#push-options` before the attribute, not between the attribute and the
  declaration.
- Keep template logic minimal: templates choose concrete layouts and assemble
  witnesses; arithmetic correspondence and proofs belong in verified support
  modules.

## Diagnosing a recurrence

Run a fresh extraction cache, because an existing kernel module can hide an
extraction regression. If Karamel fails, rerun its generated `.krml` input with
`-dmonomorphization` and search the trace for:

```text
<: any
Warning 26
Prims_op_Subtraction
```

Inspect the first active cast, not every later `any` annotation. Usually it
identifies either an opaque `ctlayout.cimap` record boundary or a dependent
tuple crossing a helper boundary.

Use targeted verification rather than full Kuiper verification:

```text
make obj/Kuiper.Kernel.SDPA.Naive.fst.checked
make verify-kuiops
```

Full Kuiper verification includes unrelated intensive TensorCore modules and
is not required for this integration.
