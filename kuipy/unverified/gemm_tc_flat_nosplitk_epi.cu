// =============================================================================
// A flattened, single-file Ampere tensor-core GEMM.
//
//     D = alpha * (A @ B^T) + beta * C,  one block per output tile.
//
// This is a porting reference for a Kuiper implementation, not a kernel
// kuipy dispatches to. It is the same algorithm as the templated
// gemm_tc.cuh, with every C++ mechanism that exists only to serve the JIT
// harness stripped out: no `Params` struct, no `Tile<>` traits class, no
// `Desc` table of function pointers, no C++ template parameters at all.
// Everything that was a template parameter is a macro, and everything
// derived from those macros is a `constexpr int` computed once at the top.
//
// -----------------------------------------------------------------------------
// THIS FILE IS ONE OF FOUR
// -----------------------------------------------------------------------------
// The GEMM is specialized along two independent axes, giving four kernels that
// share their entire k-loop, staging, swizzle and mma sequence -- those parts
// are byte-identical between the files -- and differ only in their k-range and
// their epilogue:
//
//                      | no epilogue (D = A@B^T)      | epilogue (axpby vs C)
//   -------------------+------------------------------+------------------------
//   one block owns k   | gemm_tc_flat_nosplitk_noepi  | gemm_tc_flat_nosplitk_epi
//   k split SPLITS ways| gemm_tc_flat_splitk_noepi    | gemm_tc_flat_splitk_epi
//
// The split-K axis. The no-split-K kernels launch (M/BM)*(N/BN) blocks, one per
// output tile, each walking the whole k dimension. That is the right shape of
// launch whenever it fills the GPU. When it does not -- a decode-shaped GEMM
// like M=256, N=896, K=4864 with a 128x128 tile is 2*7 = 14 blocks on 84 SMs --
// the work is there but it is all stacked along k, which one block traverses
// sequentially. The split-K kernels cut k into SPLITS contiguous slices and
// give each its own block, multiplying the block count by SPLITS, at the price
// of an fp32 workspace and a second pass to reduce it.
//
// The epilogue axis. `noepi` computes D = A@B^T and nothing else; it is what
// aten.mm needs. `epi` computes D = alpha*(A@B^T) + beta*C for a C supplied as
// an arbitrary view (see the C VIEW section in those files); it is what
// aten.addmm needs. Separating them removes the C pointer, the alpha/beta
// scalars, the mode switch and the entire C read from the mm path, which is the
// hotter of the two: in the target inference pipeline only the fused qkv
// projection has a bias, while o_proj, gate, up, down and lm_head do not.
//
// Which one runs, in that pipeline (24 layers, per decode step):
//
//   nosplitk_noepi   o_proj, gate, up, lm_head          73 calls
//   nosplitk_epi     fused qkv (Linear bias)            24 calls
//   splitk_noepi     down_proj (N=896, K=4864, split 4) 24 calls
//   splitk_epi       -- none, but addmm shapes exist elsewhere
//
// So the no-split-K kernels handle ~80% of calls and the no-epilogue kernels
// ~80% of those.
//
// -----------------------------------------------------------------------------
// !!! READ THIS BEFORE PORTING !!!
// -----------------------------------------------------------------------------
// The macros in the CONFIGURATION block below are *tuning parameters*, not
// constants of the algorithm. They are macros here only because C has no other
// way to write a compile-time-polymorphic function, and because in Kuiper the
// equivalent parameters are `nat`-indexed arguments that get specialized away at
// extraction time -- which is exactly what a macro does in C.
//
// A Kuiper port MUST keep BM, BN, BK, WM, WN, SKEW and GROUP as
// parameters of the kernel definition. Do NOT hardcode the values that happen
// to be set below. The whole point of this kernel is that it is tuned per
// (shape, dtype, GPU): the values here are one arbitrary point in that space.
// There are forty distinct tilings in gemm_tc.cu's table and the autotuner
// picks a different one for nearly every shape in the inference pipeline. A
// kernel with these numbers baked in would be a strictly worse kernel.
//
// The same goes for the element type (`elem_t`): it is a typedef selected by a
// macro because this file compiles to one specialization at a time, but the
// algorithm is genuinely polymorphic in it. In Kuiper it is an `et` parameter
// constrained by whatever refinement the tensor-core typeclass imposes.
//
// The things that ARE fixed by the algorithm, and are therefore legitimately
// constant, are only: the 16x16x16 WMMA fragment shape (that is the hardware's
// shape on sm_80/sm_86), the 32-thread warp, the 16-byte cp.async granule, and
// the fp32 accumulator type.
//
// -----------------------------------------------------------------------------
// SPECIALIZATIONS relative to gemm_tc.cuh
// -----------------------------------------------------------------------------
//   * STAGES = 2. The general kernel takes a pipeline depth and waits with
//     `__pipeline_wait_prior(STAGES - 2)`, i.e. it waits for all but the last
//     (STAGES - 2) in-flight cp.async batches, leaving some copies still in
//     flight. Kuiper cannot currently express "wait on a prefix of the in-flight
//     batches" -- it can only drain the pipeline completely. STAGES = 2 is
//     precisely the case where `wait_prior(STAGES - 2)` degenerates to
//     `wait_prior(0)` = drain-everything, so it is the deepest pipeline Kuiper
//     can currently express. Measured cost of this restriction end-to-end on
//     Qwen2.5-0.5B decode: ~0.5%, because at these tile sizes the kernel is
//     shared-memory-throughput bound rather than global-latency bound.
//
//     Specializing it also removes the `for (s = 0; s < STAGES - 1; s++)`
//     prologue loop -- with STAGES = 2 it has exactly one iteration -- and turns
//     `kt % STAGES` into `kt & 1`.
//
//   * B is column-major, always. Every GEMM in the target inference pipeline
//     multiplies activations by a frozen `nn.Linear` weight, which torch hands
//     over as a (K, N) tensor with stride (1, K) -- that is, an (N, K)
//     row-major buffer. The general kernel carries a `BT` flag and supports
//     both; here the row-major-B path is simply deleted. This is not a loss of
//     generality that matters, and it makes the kernel *simpler*, not more
//     complex: with B stored (N, K) the B operand is contiguous along k just
//     like A, so the two staging paths become structurally identical and share
//     one shared-tile leading dimension (LDT).
//
// =============================================================================
// SHAPES AND LAYOUTS
// =============================================================================
// All buffers are dense row-major. Writing `X[i, j]` for the mathematical
// element and `X(i*ld + j)` for the address:
//
//   A   (M, K) row-major, ld = K.   A[m, k] at A(m*K + k).
//   B   (N, K) row-major, ld = K.   B[n, k] at B(n*K + k).
//         Equivalently: the (K, N) matrix with column-major storage, which is
//         how torch labels a transposed / frozen nn.Linear weight.
//   D   (M, N) row-major, ld = N.   Never aliases any input.
//   C   an arbitrary (M, N) VIEW, read-only. It has no layout of its own;
//         see "THE C VIEW" below. The math performed is
//                 D[m, n] = alpha * sum_k A[m,k]*B[n,k] + beta*C[m, n]
//
// DIVISIBILITY PRECONDITIONS. This kernel does no bounds checking whatsoever;
// every predicate that a general GEMM would evaluate per tile is discharged by
// a precondition instead. The caller must guarantee:
//
//     M % BM == 0        N % BN == 0        K % BK == 0
//     A, B, C and D 16-byte aligned; K % VEC == 0 and N % VEC == 0
//
// These are checked by `assert` in the host launcher. In Kuiper they would be
// refinements on the kernel's argument types, which is strictly better: the
// caller would be unable to construct an out-of-range call in the first place.
//
// =============================================================================
// THE C VIEW -- NO LAYOUT IS ASSUMED
// =============================================================================
// C is NOT a matrix with a known layout. It is an arbitrary *view*: a shape
// (M, N) together with a function from an (m, n) index pair to an offset into
// some underlying buffer. That function is `c_index` below, and it is the ONLY
// place in this file that knows anything about how C is stored. Every read of C
// goes through it.
//
// This mirrors Kuiper's `Kuiper.TensorRO`, where a read-only tensor is a
// `rotensor et l` for a `vtlayout l` carrying
//
//     imap : abs d -> GTot (natlt ulen)
//
// -- an arbitrary function from the abstract index space into the base array,
// explicitly NOT required to be an injection. `c_index` is the concrete
// counterpart of that map (Kuiper's `cimap`). The GEMM instance that the
// inference pipeline needs is spelled out in Kuiper as
//
//     bcastC rows cols = extended_layout (vtlayout_of_tlayout (l1_forward cols)) rows
//
// (see Klas.GEMM.TensorCore2D.To.Bcast.Inst): a contiguous length-`cols` vector
// whose layout is *extended* over `rows`, i.e. an imap that ignores the row
// index entirely. That non-injectivity is exactly what broadcasting is.
//
// Because every C access is `C[c_index(m, n)]`, the kernel is correct for ANY
// c_index whatsoever. In particular, replacing its body changes nothing else in
// the file. The strided form below is provided because it covers every case the
// pipeline needs with two integers:
//
//     CS_M = 0, CS_N = 1   length-N bias vector broadcast down the rows.
//                          *** This is what every addmm in the model uses. ***
//                          It is Kuiper's `bcastC`, and it is the default here.
//     CS_M = N, CS_N = 1   dense (M, N) row-major matrix.
//     CS_M = 1, CS_N = 0   length-M vector broadcast across the columns.
//     CS_M = 0, CS_N = 0   a scalar broadcast to everything.
//     CS_M = 1, CS_N = M   dense (M, N) column-major matrix.
//
// The strides are compile-time for the same reason everything else here is: the
// view is resolved at specialization time, so `CS_M = 0` compiles the row index
// out of the address computation instead of multiplying by a runtime zero.
//
// (The two dense-matrix rows above are written with M and N to say what the map
// IS; they are only directly instantiable here when the extent in question is
// known at specialization time, which in the JIT setting it is, since a kernel
// is compiled per shape. Nothing in the kernel depends on the strides being
// compile-time -- making them runtime `int`s costs one multiply per granule and
// forces the scalar path -- so a port is free to take them as arguments. In
// Kuiper the question does not arise: the layout comes in with the tensor.)
//
// >>> THE ONE PLACE LAYOUT LEAKS, AND HOW IT IS FENCED OFF.
// >>> A GEMM epilogue would like to read C with 128-bit loads, and that is only
// >>> meaningful if C's elements are contiguous and aligned along n. So there
// >>> are two read paths below, selected by `if constexpr` on
// >>> C_CONTIGUOUS_ALIGNED_N -- a compile-time predicate, so exactly one of
// >>> them is emitted and there is no runtime branch:
// >>>
// >>>   * the vectorized path, taken when the view happens to be contiguous and
// >>>     aligned along n (CS_N == 1 and CS_M a multiple of VEC, which covers
// >>>     the bias and row-major-matrix cases);
// >>>   * the scalar path, which calls c_index once per element and is
// >>>     therefore correct for EVERY view, including ones with no contiguity
// >>>     at all.
// >>>
// >>> The scalar path is the definition; the vectorized path is an
// >>> optimization that must be *proved* equal to it under its guard. A port
// >>> that only implements the scalar path is correct and simpler, and only
// >>> gives up bandwidth on the C read -- which for the broadcast case is a few
// >>> hundred bytes that stay in L1 anyway.
//
// =============================================================================
// ALGORITHM
// =============================================================================
// Blocking. A thread block owns one BM x BN tile of D and the whole k dimension, which it
// walks in steps of BK. The block tile is partitioned into WM x WN warp tiles,
// one per warp; each warp holds its entire (WM/16) x (WN/16) grid of fp32
// accumulator fragments in registers for the duration of the k-loop, so the
// accumulators are never spilled or re-read. This is the reason for the
// register-pressure ceiling noted at WM/WN below.
//
// Data movement. Each k-step stages a BM x BK slab of A and a BN x BK slab of B
// into shared memory with cp.async (16 bytes per thread per instruction, issued
// from a hoisted per-thread address), double-buffered so the copy for step kt+1
// is in flight while the mmas for step kt run. Within a step, each warp reads
// its operands out of shared memory with `ldmatrix` (via `load_matrix_sync`) and
// issues (WM/16)*(WN/16)*(BK/16) `mma.sync`s.
//
// Reuse. The arithmetic intensity of the whole thing is what the block tile
// buys: a BM x BN x BK step does BM*BN*BK MACs against (BM + BN)*BK elements of
// traffic, so the ratio is BM*BN/(BM + BN). That is the single reason to want
// large square-ish block tiles, and the reason the tuner's preference order
// (see gemm_tc_tuned.py) is "fill the machine first, then maximize BM*BN".
//
// =============================================================================
// WHY IT IS CORRECT
// =============================================================================
// Race freedom in the k-loop rests on three facts, spelled out at the point of
// use below:
//   (1) `__pipeline_wait_prior(0)` at the top of iteration kt retires every
//       cp.async this thread has committed, so the buffer about to be read is
//       fully written *by this thread*;
//   (2) the `__syncthreads()` immediately after extends (1) to the whole block
//       -- shared memory is written by all THREADS threads and read by all of
//       them, so a block-wide barrier is required, a warp-level one is not
//       enough;
//   (3) that same `__syncthreads()` is also what makes the *other* buffer safe
//       to overwrite: its last reader was iteration kt-1, which is ordered
//       before the barrier.
// The epilogue's reuse of the per-warp scratch across the MFRAG bands is
// guarded by `__syncwarp()` -- warp-local, because the scratch band is
// warp-private.
//
// Numerically, accumulation is fp32 throughout -- the mma accumulator and
// the shared scratch -- and the single rounding to elem_t happens once, on
// the final result. The k reduction is performed in one sequential pass in
// fragment order, so the result is deterministic and bitwise reproducible
// across launches.
//
// =============================================================================
// KUIPER-EXPRESSIBILITY NOTES
// =============================================================================
// Deliberately absent, because Kuiper cannot express them:
//   * inline PTX of any kind;
//   * warp shuffles (`__shfl_*`);
//   * atomics;
//   * indexing into the individual registers of a WMMA fragment (fragment `.x[]`
//     under an inferred lane->element mapping). The accumulators are only ever
//     touched through `fill_fragment`, `mma_sync` and `store_matrix_sync`.
//
// Used, and expressible: `nvcuda::wmma` (load_matrix_sync / mma_sync /
// store_matrix_sync / fill_fragment), cp.async via the `__pipeline_*`
// intrinsics, 128-bit vectorized loads and stores, `__syncthreads`, `__syncwarp`.
//
// One caveat that a port cannot fix and should not be surprised by: CUDA lowers
// the *bf16* `load_matrix_sync` through a different builtin than the fp16 one,
// and that builtin does not emit `ldmatrix` -- it emits four generic loads plus
// a register transpose per fragment. bf16 therefore runs ~15% behind fp16 here
// for reasons entirely outside this kernel.
//
// =============================================================================
// BUILDING
// =============================================================================
// Standalone, one specialization per translation unit. Not in kuipy's
// _SOURCES.
//
//   nvcc -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr \
//        -gencode=arch=compute_86,code=sm_86 \
//        -DGEMM_BF16=1 -DBM=128 -DBN=128 -DBK=64 -DWM=64 -DWN=64 \
//        -DCS_M=0 -DCS_N=1 \
//        -c gemm_tc_flat_nosplitk_epi.cu
// =============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <assert.h>

using namespace nvcuda;

// =============================================================================
// CONFIGURATION -- ALL OF THESE ARE TUNING PARAMETERS, NOT CONSTANTS.
// See the warning at the top of the file. In Kuiper these are kernel
// parameters; the values here are one tuned point out of many, not "the"
// values.
// =============================================================================

// Element type of A, B, C and D. Accumulation is always fp32 regardless.
// Set to 1 for __nv_bfloat16, 0 for half.
// Instead of a flag would just be an element type with a typeclass in kuiper.
#ifndef GEMM_BF16
#define GEMM_BF16 0
#endif

// --- Block tile: the BM x BN piece of D one thread block computes, and the BK
// --- depth of one k-step. Bigger BM*BN means more arithmetic per byte staged
// --- (ratio BM*BN/(BM+BN)); bigger BK means fewer barriers per unit of work,
// --- but all three multiply into the shared-memory footprint.
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif

// --- Warp tile: the WM x WN piece of the block tile one warp owns. This fixes
// --- the number of warps at (BM/WM)*(BN/WN) and the accumulator register count
// --- at (WM/16)*(WN/16)*8 per thread. 64x64 is the practical ceiling on
// --- sm_86: 128 accumulator registers/thread, ~210-235 total, no spills.
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 64
#endif

// --- Shared-tile row padding, in elements. NOT cosmetic and NOT free to set to
// --- zero: see the LDT comment below for why the whole kernel's shared-memory
// --- throughput depends on it.
#ifndef SKEW
#define SKEW 8
#endif

// --- L2 block-swizzle group height, in block tiles. See the swizzle comment in
// --- the kernel. Purely a scheduling hint; any value >= 1 is correct.
#ifndef GROUP
#define GROUP 8
#endif

// --- The C view's index map, as strides. See "THE C VIEW" above: these two are
// --- one instance of an arbitrary imap, not an assumption about C's layout.
// --- The default is Kuiper's `bcastC` -- a length-N vector broadcast down the
// --- rows, i.e. a Linear bias, which is what every addmm in the pipeline is.
#ifndef CS_M
#define CS_M 0
#endif
#ifndef CS_N
#define CS_N 1
#endif

#if GEMM_BF16
typedef __nv_bfloat16 elem_t;
#else
typedef half elem_t;
#endif

// -----------------------------------------------------------------------------
// Derived quantities. Everything below is a consequence of the macros above; a
// Kuiper port computes these the same way, from the kernel's index parameters.
// -----------------------------------------------------------------------------

// The hardware's WMMA shape on sm_80/sm_86. This one really is a constant.
constexpr int FRAG = 16;
constexpr int WARP_SIZE = 32;

// Warps per block, and therefore threads per block.
constexpr int WARPS_M = BM / WM;
constexpr int WARPS_N = BN / WN;
constexpr int WARPS = WARPS_M * WARPS_N;
constexpr int THREADS = WARPS * WARP_SIZE;

// Accumulator fragments per warp, and mma steps per k-tile.
constexpr int MFRAG = WM / FRAG;
constexpr int NFRAG = WN / FRAG;
constexpr int KSTEP = BK / FRAG;

// Leading dimension of BOTH shared tiles. Because B is stored (N, K) and A is
// (M, K), both shared tiles are "some number of rows, each BK contiguous
// k-values", so they share a leading dimension -- this is the simplification
// that specializing to column-major B buys.
//
// The +SKEW is load-bearing. A shared row stride of exactly BK makes the row
// stride a multiple of the 32 four-byte banks whenever BK is (BK is always a
// multiple of 16), so the 16 rows a fragment load touches all land in the same
// banks and the load serializes. Padding by SKEW makes the stride odd in units
// of banks and the accesses spread out. Measured in isolation, dropping the
// skew takes the inner loop from 306 to 103 TF/s. SKEW must stay a multiple of
// VEC so that every staged row start remains 16-byte aligned for cp.async.
constexpr int LDT = BK + SKEW;

constexpr int A_ELEMS = BM * LDT;  // one A buffer, in elements
constexpr int B_ELEMS = BN * LDT;  // one B buffer, in elements

// STAGES == 2, specialized: two buffers each for A and B.
constexpr int STAGES = 2;
constexpr int PIPE_BYTES = STAGES * (A_ELEMS + B_ELEMS) * (int)sizeof(elem_t);

// Epilogue scratch. One FRAG-row band of every warp's tile, in fp32, padded for
// the same bank-conflict reason as LDT. See the epilogue for why it exists at
// all, and the block below for why it is not laid on top of the pipeline
// buffers.
constexpr int ESKEW = 4;
constexpr int LDE = WN + ESKEW;
constexpr int EPI_BYTES = WARPS * FRAG * LDE * (int)sizeof(float);

// >>> THE TWO USES DO NOT ALIAS, ON PURPOSE. This is `+`, not `max`.
// >>>
// >>> The pipeline buffers are dead by the time the epilogue runs (one
// >>> __syncthreads() after the k-loop, see the epilogue), so the epilogue
// >>> scratch *could* be laid on top of them at offset 0, reinterpreting one
// >>> allocation from elem_t to float partway through the kernel. gemm_tc.cuh
// >>> does exactly that. This file does not, because Kuiper is not expected to
// >>> be able to express a lifetime-based reuse of a single shared allocation
// >>> at two different element types, and the point of this file is to be
// >>> faithful to what a port can actually do.
// >>>
// >>> Do not "optimize" this back to `max` unless the port really can express
// >>> the aliasing -- but do know what it costs, because it is not nothing.
// >>> Shared memory is the occupancy currency on sm_86 (100 KB per SM): for a
// >>> 128x128x32 tiling `max` is 40960 bytes and `+` is 58368, which is the
// >>> difference between 2 resident blocks per SM and 1. Measured cost of NOT
// >>> aliasing (bf16, same tiling, aliased -> un-aliased):
// >>>     4096^3, 128x128x32 : 94.6 -> 73.1 TF/s   (-29%)
// >>>     4096^3,  32x128x64 : 41.9 -> 39.2 TF/s   (-7%)
// >>>     256x4864x896       : no change
// >>>     256x896x4864, sp4  : no change
// >>>   Qwen2.5-0.5B decode, whole pipeline : 143.4 -> 141.6 tok/s/seq (-1.3%)
// >>> Large shapes pay heavily; the skinny decode shapes do not notice, because
// >>> they never launch enough blocks to fill the machine in the first place.
// >>>
// >>> The split-K kernels do not pay this at all: their output is fp32, which
// >>> is the accumulator type, so they store fragments straight to the
// >>> workspace and have no epilogue scratch to place.
constexpr int SMEM_BYTES = PIPE_BYTES + EPI_BYTES;

// 128-bit access granule, in elements: 8 for both fp16 and bf16. This is the
// widest load/store the ISA has and the required granule for cp.async's
// 16-byte mode, so it is a hardware constant, not a tuning knob.
constexpr int VEC = 16 / (int)sizeof(elem_t);

// Staging schedule. Each tile is copied by the whole block in sweeps: one sweep
// moves THREADS*VEC elements, which is ROW_STEP whole rows of BK. A and B share
// ROW_STEP (same row length BK) and differ only in how many sweeps they need.
constexpr int ROW_STEP = THREADS * VEC / BK;
constexpr int A_ITERS = BM / ROW_STEP;
constexpr int B_ITERS = BN / ROW_STEP;

static_assert(BM % WM == 0 && BN % WN == 0, "warp tile must divide block tile");
static_assert(WM % FRAG == 0 && WN % FRAG == 0 && BK % FRAG == 0,
              "tiles must be whole numbers of 16x16x16 fragments");
static_assert(SKEW % VEC == 0,
              "SKEW must keep shared rows 16-byte aligned for cp.async");
static_assert(THREADS * VEC % BK == 0,
              "a block sweep must cover a whole number of tile rows");
static_assert(BM % ROW_STEP == 0 && BN % ROW_STEP == 0,
              "sweeps must tile the staged slabs exactly");
static_assert(THREADS <= 1024, "block tile needs too many warps");

// Epilogue: a warp writes its WM x WN tile FRAG rows at a time, and each thread
// stores VEC elements per step.
constexpr int BAND_VECS = FRAG * WN / VEC;
static_assert(BAND_VECS % WARP_SIZE == 0,
              "the epilogue band must divide evenly over a warp");

// -----------------------------------------------------------------------------
// Scalar conversion. Round-to-nearest-even; this is the only place the fp32
// accumulation meets the narrow element type, and it happens exactly once per
// output element.
//
// Note there is no fragment-level alternative for bf16: CUDA defines no
// `wmma::fragment<accumulator, 16, 16, 16, __nv_bfloat16>` at all (only fp16
// has an accumulator fragment of the narrow type), so the trick Kuiper's
// KPR_STORE_COMB uses -- declare a second accumulator fragment typed by the
// destination, convert register-wise, store it -- is simply unavailable when
// the output is bf16. That is why the narrowing has to happen on ordinary
// floats, and hence why the accumulators must be spilled to shared scratch
// first.
// -----------------------------------------------------------------------------
__device__ __forceinline__ half to_elem(float v, half) {
    return __float2half_rn(v);
}
__device__ __forceinline__ __nv_bfloat16 to_elem(float v, __nv_bfloat16) {
    return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// -----------------------------------------------------------------------------
// c_index: the C view's index map. THE ONLY PLACE THAT KNOWS HOW C IS STORED.
//
// This is the concrete counterpart of a Kuiper `vtlayout`'s `imap` (its
// `cimap`). Every read of C in this file is `C[c_index(m, n)]`, so the kernel
// is correct for any function whatsoever here -- including non-injective ones,
// which is what broadcasting is. See "THE C VIEW" at the top.
//
// In Kuiper this is not a function the kernel defines; it is a parameter,
// supplied by the `rotensor`'s layout and inlined at specialization time.
// -----------------------------------------------------------------------------
__device__ __forceinline__ size_t c_index(int m, int n) {
    return (size_t)m * (size_t)(CS_M) + (size_t)n * (size_t)(CS_N);
}

// Is this particular view contiguous and 16-byte-alignable along n? If so the
// epilogue may read C with 128-bit loads; if not it falls back to calling
// c_index per element, which is correct for every view. Compile-time, so only
// one of the two paths is ever emitted. See the ">>>" block in "THE C VIEW".
constexpr bool C_CONTIGUOUS_ALIGNED_N = ((CS_N) == 1) && ((CS_M) % VEC == 0);

// =============================================================================
// stage_tiles: issue one k-step's worth of global -> shared cp.async.
// =============================================================================
// This was a C++ lambda in gemm_tc.cuh purely so it could capture the eight or
// so hoisted pointers; it is a free function here because there is nothing a
// closure buys once the captures are written out as parameters.
//
// >>> In Kuiper this should be an `inline_for_extraction noextract` function.
// >>> That is the construct that gives the same result: the definition is
// >>> proved once and separately, but it is beta-reduced into the caller during
// >>> extraction, so the generated CUDA has this body textually inlined at both
// >>> call sites -- which is what we need, because the whole point of the
// >>> hoisted addressing below is that the copy compiles to a straight run of
// >>> cp.async instructions with no call, no spill and no re-derived indices.
// >>> An ordinary (extracted) Kuiper function would produce a real __device__
// >>> call here and the register allocator would lose the hoisted pointers.
//
// The addressing is the reason this kernel is fast. A thread copies the *same
// cells* of every k-tile, so its source addresses differ between k-steps only
// by a fixed stride and its shared destinations not at all. The caller
// therefore keeps `a_src` / `b_src` as running pointers and bumps them by BK
// per k-step, and this function only walks the sweeps. The naive alternative,
// `for (i = tid; i < TILE_VECS; i += THREADS)`, re-derives a 64-bit
// divide/modulo address on every k-tile; replacing it with this form was worth
// ~7% end-to-end (83 -> 89 TF/s at 4096^3). It is *not* a micro-optimization
// that can be skipped: without it the k-loop is dominated by integer
// arithmetic rather than by the copies or the mmas.
//
// Parameters (all pointers already include the thread's own offset within the
// tile, and a_dst/b_dst already select the buffer):
//   a_dst  shared A buffer + this thread's (row*LDT + col)
//   a_src  global A + this thread's cell of the current k-tile
//   b_dst  shared B buffer + this thread's (row*LDT + col)
//   b_src  global B + this thread's cell of the current k-tile
//   K      global row stride, shared by A (M,K) and B (N,K)
//
// Postcondition: exactly A_ITERS + B_ITERS cp.async's are in flight for this
// thread and one batch has been committed. Nothing is readable until a
// matching `__pipeline_wait_prior` and a block-wide barrier.
__device__ __forceinline__ void stage_tiles(elem_t* a_dst, const elem_t* a_src,
                                            elem_t* b_dst, const elem_t* b_src,
                                            int K) {
    // 16 = the cp.async 16-byte mode. Both the source and the destination are
    // 16-byte aligned: the source because K % VEC == 0 and the thread's column
    // offset is a multiple of VEC, the destination because LDT % VEC == 0.
    #pragma unroll
    for (int n = 0; n < A_ITERS; n++)
        __pipeline_memcpy_async(a_dst + n * ROW_STEP * LDT,
                                a_src + (size_t)n * ROW_STEP * K, 16);
    #pragma unroll
    for (int n = 0; n < B_ITERS; n++)
        __pipeline_memcpy_async(b_dst + n * ROW_STEP * LDT,
                                b_src + (size_t)n * ROW_STEP * K, 16);
    __pipeline_commit();
}

// =============================================================================
// The GEMM kernel.
// =============================================================================
// Grid:   x = (M/BM)*(N/BN) block tiles, flattened and then swizzled below
//         1-D: with the whole k-range owned by one block there is nothing
//         for a z dimension to index (contrast the split-K files, whose
//         grid is (tiles, 1, SPLITS)).
// Block:  THREADS threads, 1-D
// Shared: SMEM_BYTES, dynamic
//
// Arguments are passed flat rather than in a `Params` struct; there is no
// __grid_constant__ struct to unpack, so every scalar arrives in constant bank
// 0 exactly as it would in a Kuiper-generated signature.
extern "C" __global__ __launch_bounds__(THREADS) void gemm_tc_epi_kernel(
    const elem_t* __restrict__ A,  // (M, K) row-major
    const elem_t* __restrict__ B,  // (N, K) row-major  == (K, N) col-major
    const elem_t* __restrict__ C,  // an (M, N) view, via c_index
    elem_t* __restrict__ D,        // (M, N) row-major
    int M, int N, int K, float alpha, float beta) {
    extern __shared__ __align__(16) char smem_raw[];
    // Layout of the dynamic allocation:
    //   [ A buf0 | A buf1 | B buf0 | B buf1 | fp32 epilogue scratch ]
    //     \------------- PIPE_BYTES -------------/\-- EPI_BYTES --/
    // The epilogue scratch gets its own storage rather than being laid over
    // the (by then dead) pipeline buffers; see SMEM_BYTES for why.
    elem_t* sA = reinterpret_cast<elem_t*>(smem_raw);
    elem_t* sB = sA + STAGES * A_ELEMS;

    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    // Position of this warp's WM x WN tile inside the block's BM x BN tile.
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    // ---------------------------------------------------------------- swizzle
    // Map the linear block index to a (block_row, block_col) so that
    // consecutively scheduled blocks form a GROUP-tall, N/BN-wide column-major
    // strip rather than a full row. Blocks in a strip share B columns and
    // reuse A rows, so a scheduling wave's working set stays resident in L2;
    // plain row-major ordering streams an entire row of B per block row.
    //
    // This is a pure permutation of block indices -- any bijection is correct,
    // GROUP only affects locality. `rows` handles the final, possibly short,
    // strip when GROUP does not divide M/BM.
    const int nblk_n = N / BN;
    const int nblk_m = M / BM;
    int block_row, block_col;
    {
        const int per_group = GROUP * nblk_n;
        const int gid = blockIdx.x / per_group;
        const int rem = blockIdx.x - gid * per_group;
        const int rows = min(GROUP, nblk_m - gid * GROUP);
        block_row = gid * GROUP + rem % rows;
        block_col = rem / rows;
    }

    // ------------------------------------------------------------- k-range
    // The whole of it. This is the definition of "no split-K", and it is why
    // this kernel can own the reduction and write D directly.
    const int ktiles = K / BK;
    // -------------------------------------------------------- accumulators
    // (WM/16) x (WN/16) fp32 fragments, live in registers across the entire
    // k-loop. Only ever touched via fill/mma/store -- never by indexing .x[].
    wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> acc[MFRAG][NFRAG];
    #pragma unroll
    for (int i = 0; i < MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // ------------------------------------------------- staging addresses
    // This thread's cell within a staged slab: with rows of BK elements and
    // VEC elements per thread, thread `tid` starts at flat element tid*VEC.
    // The same (row, col) is used for both A and B because both slabs have
    // rows of BK -- again, a consequence of B being column-major.
    const int t_row = (tid * VEC) / BK;
    const int t_col = (tid * VEC) % BK;

    // Running global pointers, bumped by BK per k-step. Note both strides are
    // K: A row m starts at m*K, B row n starts at n*K.
    const elem_t* a_src = A + (size_t)(block_row * BM + t_row) * K +
                          t_col;
    const elem_t* b_src = B + (size_t)(block_col * BN + t_row) * K +
                          t_col;
    // Fixed offset of this thread's cell inside a shared buffer.
    const int t_dst = t_row * LDT + t_col;

    // -------------------------------------------------------------- prologue
    // With STAGES == 2 the general `for (s = 0; s < STAGES-1; s++)` prologue is
    // exactly one iteration: fill buffer 0 and start the pipeline. `ktiles >= 1`
    // follows from K % BK == 0 and K > 0, so this is unconditional.
    stage_tiles(sA + t_dst, a_src, sB + t_dst, b_src, K);
    a_src += BK;
    b_src += BK;

    // ---------------------------------------------------------------- k-loop
    for (int kt = 0; kt < ktiles; kt++) {
        // (1) Retire every cp.async this thread has committed. With STAGES == 2
        //     the general form `__pipeline_wait_prior(STAGES - 2)` is
        //     `wait_prior(0)`, i.e. a full drain -- which is the only form
        //     Kuiper can currently express, and the reason this file fixes
        //     STAGES at 2. After this, THIS thread's writes to buffer `buf` are
        //     complete.
        __pipeline_wait_prior(0);
        // (2) ... but buffer `buf` was written by all THREADS threads and is
        //     about to be read by all of them, so a block-wide barrier is
        //     required to make (1) hold for the block. A __syncwarp() would be
        //     unsound here.
        // (3) This same barrier is what makes the OTHER buffer safe to
        //     overwrite below: its last readers were the fragment loads of
        //     iteration kt-1, which precede this barrier in program order.
        __syncthreads();

        const int buf = kt & 1;  // kt % STAGES, specialized
        const elem_t* bufA = sA + buf * A_ELEMS;
        const elem_t* bufB = sB + buf * B_ELEMS;

        // Refill the other buffer, one k-step ahead. Issued BEFORE the math so
        // the copies are in flight while this tile's mmas run -- that overlap
        // is the entire purpose of double buffering, and issuing after the math
        // would serialize them. Safety is (3) above.
        if (kt + 1 < ktiles) {
            const int nbuf = buf ^ 1;
            stage_tiles(sA + nbuf * A_ELEMS + t_dst, a_src,
                        sB + nbuf * B_ELEMS + t_dst, b_src, K);
            a_src += BK;
            b_src += BK;
        }
        // No balancing `__pipeline_commit()` on the else branch: unlike
        // wait_prior(n>0), wait_prior(0) drains whatever is outstanding and
        // does not care how many batches were committed.

        // ------------------------------------------------------------ math
        // The A fragment is row_major: A's shared tile holds A[m, k] at
        // m*LDT + k, and a row_major matrix_a with ldm = LDT reads element
        // (m, k) at ptr[m*ldm + k]. Matches.
        //
        // The B fragment is col_major: B's shared tile holds B[k, n] (in the
        // (K, N) reading) at n*LDT + k, and a col_major matrix_b with
        // ldm = LDT reads element (k, n) at ptr[n*ldm + k]. Matches. This is
        // the whole trick of the column-major-B specialization -- the operand
        // is fed to the tensor core transposed *for free*, by choosing the
        // fragment's layout tag rather than by transposing any data.
        wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, elem_t, wmma::row_major>
            af[MFRAG];
        wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, elem_t, wmma::col_major>
            bf[NFRAG];
        #pragma unroll
        for (int ks = 0; ks < KSTEP; ks++) {
            // MFRAG + NFRAG fragment loads feed MFRAG*NFRAG mmas: the operand
            // reuse inside the warp tile, which is why WM/WN want to be as
            // large as the register file allows.
            #pragma unroll
            for (int i = 0; i < MFRAG; i++)
                wmma::load_matrix_sync(
                    af[i],
                    bufA + (warp_m * WM + i * FRAG) * LDT + ks * FRAG, LDT);
            #pragma unroll
            for (int j = 0; j < NFRAG; j++)
                wmma::load_matrix_sync(
                    bf[j],
                    bufB + (warp_n * WN + j * FRAG) * LDT + ks * FRAG, LDT);
            #pragma unroll
            for (int i = 0; i < MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < NFRAG; j++)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
    }

    // ================================================================ epilogue
    // The accumulators have to become elem_t in global memory, combined with C. Doing
    // that by indexing the accumulator registers would require knowing which
    // lane holds which (m, n) -- an inferred fragment layout, which is exactly
    // what is off-limits (and is unspecified by the WMMA API anyway). And the
    // one sanctioned escape, converting register-wise into a second accumulator
    // fragment typed by the destination (Kuiper's KPR_STORE_COMB), does not
    // exist for bf16: CUDA defines no bf16 accumulator fragment. See the note
    // at to_elem.
    //
    // So: `store_matrix_sync` one FRAG-row band of the warp tile into shared
    // scratch, which is the API's own sanctioned way out of the fragment
    // abstraction, then read it back as ordinary floats. Everything after that
    // -- the narrowing, alpha/beta and the C view -- is plain arithmetic on plain floats, and the
    // global stores come out 128-bit and coalesced because consecutive lanes
    // take consecutive VEC-element chunks of a row.
    //
    // Going through memory costs nothing in time: this happens once per block,
    // not once per k-step. It costs EPI_BYTES of shared memory, which is not
    // nothing -- see SMEM_BYTES.
    //
    // The scratch lives past the pipeline buffers, so this barrier is not
    // needed for storage safety the way it would be if the two aliased (see
    // SMEM_BYTES). It is still needed: the block is about to leave the k-loop,
    // and every warp must have issued its last fragment load before any thread
    // proceeds, so that no warp races ahead into the epilogue while another is
    // still reading a buffer that a straggler could otherwise refill.
    __syncthreads();
    float* scratch =
        reinterpret_cast<float*>(smem_raw + PIPE_BYTES) + warp * FRAG * LDE;

    // Top-left corner of this warp's tile in D.
    const int row_base = block_row * BM + warp_m * WM;
    const int col_base = block_col * BN + warp_n * WN;

    #pragma unroll
    for (int i = 0; i < MFRAG; i++) {
        // Band i occupies rows [row_base + i*FRAG, +FRAG). The scratch is
        // warp-private, so warp-level barriers suffice: this one waits for the
        // previous band's reads before the stores below clobber the scratch.
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            wmma::store_matrix_sync(scratch + j * FRAG, acc[i][j], LDE,
                                    wmma::mem_row_major);
        // ... and this one makes those stores visible to the whole warp, since
        // lane L below reads elements written by other lanes.
        __syncwarp();

        #pragma unroll
        for (int v = lane; v < BAND_VECS; v += WARP_SIZE) {
            // v enumerates VEC-element chunks of the FRAG x WN band.
            const int r = (v * VEC) / WN;   // row within the band
            const int c = (v * VEC) % WN;   // column within the warp tile
            const int gr = row_base + i * FRAG + r;  // row in D
            const int gc = col_base + c;             // column in D

            float val[VEC];
            #pragma unroll
            for (int e = 0; e < VEC; e++) val[e] = scratch[r * LDE + c + e];

            elem_t out[VEC];
            // The C read. Two paths, chosen at compile time by the view's own
            // index map -- see "THE C VIEW" at the top. Neither path assumes
            // anything the map does not say; the scalar one assumes nothing at
            // all.
            if constexpr (C_CONTIGUOUS_ALIGNED_N) {
                // The map is n + m*(multiple of VEC), so VEC consecutive
                // columns are VEC consecutive elements and the base offset is
                // VEC-aligned (gc is a multiple of VEC). One 128-bit load.
                elem_t cv[VEC];
                *reinterpret_cast<int4*>(cv) =
                    *reinterpret_cast<const int4*>(C + c_index(gr, gc));
                #pragma unroll
                for (int e = 0; e < VEC; e++)
                    out[e] = to_elem(alpha * val[e] + beta * to_float(cv[e]),
                                     elem_t());
            } else {
                // Correct for ANY index map, contiguous or not, injective or
                // not. This is the definition the vectorized path above must
                // agree with.
                #pragma unroll
                for (int e = 0; e < VEC; e++)
                    out[e] = to_elem(
                        alpha * val[e] +
                            beta * to_float(C[c_index(gr, gc + e)]),
                        elem_t());
            }
            elem_t* dst = D + (size_t)gr * N + gc;
            *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(out);
        }
    }
}

// =============================================================================
// Host launcher.
// =============================================================================
// Flat on purpose: the general version dispatches through a table of function
// pointers so that one compiled extension can pick any of forty tilings at run
// time without a JIT step. That table is JIT-harness machinery, not part of the
// algorithm, so here there is exactly one kernel and a direct launch.
//
// The kernel neither allocates nor frees anything (this matches the
// convention that Kuiper kernels never allocate); D is the caller's, and
// unlike the split-K variant there is no workspace to provide.
extern "C" cudaError_t gemm_tc_flat_nosplitk_epi_launch(
    const elem_t* A, const elem_t* B, const elem_t* C, elem_t* D, int M,
    int N, int K, float alpha, float beta, cudaStream_t stream) {
    // The divisibility preconditions the kernel relies on instead of bounds
    // checks. In Kuiper these are refinements on the argument types.
    assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
    assert(N % VEC == 0 && K % VEC == 0);
    assert(C != nullptr);

    // Past 48 KB the larger shared-memory carveout is opt-in per kernel.
    if (SMEM_BYTES > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_tc_epi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_BYTES);
        if (e != cudaSuccess) return e;
    }

    dim3 grid((M / BM) * (N / BN));
    gemm_tc_epi_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(
        A, B, C, D, M, N, K, alpha, beta);
    return cudaGetLastError();
}
