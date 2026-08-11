// =============================================================================
// A flattened, single-file Ampere tensor-core GEMM.
//
//     D = A @ B^T,  with k split SPLITS ways.
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
// A Kuiper port MUST keep BM, BN, BK, WM, WN, SKEW and GROUP *and SPLITS* as
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
// SPLITS IS A COMPILE-TIME PARAMETER
// -----------------------------------------------------------------------------
// In gemm_tc.cuh `splits` is a runtime `int`. Here it is a macro, i.e. the
// generated CUDA is specialized to a fixed split count, which is the right
// choice for a Kuiper port for three reasons:
//
//   1. It is chosen by the same mechanism as everything else. The tuner picks
//      SPLITS from the shape at the same moment it picks BM/BN/BK/WM/WN, and
//      the JIT already specializes a kernel per shape. There is no point at
//      which SPLITS is known but BM is not, so making it the one runtime
//      parameter buys nothing.
//   2. It turns the reduction loop into a fully unrolled, fixed-trip-count
//      loop -- worth having, since the reduce kernel is pure memory traffic
//      and its loop body is the whole kernel.
//   3. It makes `SPLITS <= K/BK` (no split may get an empty k-range, see the
//      k-range comment) a precondition relating a compile-time constant to a
//      runtime argument, which is the easy kind. As a runtime parameter it
//      would be a relation between two runtime values that the kernel would
//      have to either check or carry as a refinement through every call.
//
// The cost is that each distinct split count is a distinct extracted kernel.
// Given that each distinct tiling already is, this changes nothing structural.
//
// Note SPLITS == 1 is deliberately rejected by a static_assert below: that is
// not this kernel's job, it is the matching no-split-K file's.
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
//         The math performed is
//                 D[m, n] = sum_k A[m, k] * B[n, k]
//         i.e. A @ B^T in the (N, K) reading of B. There is no C, no alpha
//         and no beta: this is the plain-matmul kernel, and the epilogue
//         operands are not merely predicated off, they are absent from the
//         signature and from the instruction stream.
//   ws  (SPLITS, M, N) fp32 row-major -- the partial-sum workspace. Caller
//         owned, at least SPLITS*M*N floats. Written by pass 1 (each z
//         writes exactly ws[z], so the writes are disjoint), read by pass 2.
//         Needs no initialization: every cell is written before it is read,
//         because the k-range partition below leaves no split empty.
//
// DIVISIBILITY PRECONDITIONS. This kernel does no bounds checking whatsoever;
// every predicate that a general GEMM would evaluate per tile is discharged by
// a precondition instead. The caller must guarantee:
//
//     M % BM == 0        N % BN == 0        K % BK == 0
//     K / BK >= SPLITS           (so no split gets an empty k-range)
//     A, B, ws and D 16-byte aligned; K % VEC == 0 and N % VEC == 0
//
// These are checked by `assert` in the host launcher. In Kuiper they would be
// refinements on the kernel's argument types, which is strictly better: the
// caller would be unable to construct an out-of-range call in the first place.
//
// =============================================================================
// ALGORITHM
// =============================================================================
// Blocking. A thread block owns one BM x BN tile of D *and one of SPLITS
// contiguous slices of k*, which it
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
//
// The epilogue needs no barrier at all -- see the argument at the epilogue
// itself. This is a direct consequence of it touching no shared memory.
//
// Correctness of the split itself: the SPLITS k-ranges are contiguous,
// disjoint and cover [0, K/BK), so summing the SPLITS partials reproduces
// the full k reduction. Each (z, m, n) workspace cell is written by exactly
// one thread and read by exactly one thread of the reduce kernel, and the
// two kernels are ordered by being on the same stream -- so there is no
// inter-block synchronization anywhere in this design.
//
// Numerically, accumulation is fp32 throughout -- the mma accumulator, the
// workspace, and the reduce kernel's running sum -- and the single rounding
// to elem_t happens once, on the final result. Split-K changes the
// summation order of the k reduction (it becomes a two-level tree) and so is
// NOT bitwise identical to the no-split-K kernel, but it is no less accurate
// -- if anything slightly more, since a tree reduction has shallower error
// growth than a single sequential pass. It is still deterministic and
// reproducible across launches of the same configuration.
//
// =============================================================================
// KUIPER-EXPRESSIBILITY NOTES
// =============================================================================
// Deliberately absent, because Kuiper cannot express them:
//   * inline PTX of any kind;
//   * warp shuffles (`__shfl_*`);
//   * atomics -- which is the whole reason split-K needs a separate
//     reduction kernel and an fp32 workspace instead of an atomic
//     accumulate into D;
//   * indexing into the individual registers of a WMMA fragment (fragment `.x[]`
//     under an inferred lane->element mapping). The accumulators are only ever
//     touched through `fill_fragment`, `mma_sync` and `store_matrix_sync`.
//
// Used, and expressible: `nvcuda::wmma` (load_matrix_sync / mma_sync /
// store_matrix_sync / fill_fragment), cp.async via the `__pipeline_*`
// intrinsics, 128-bit vectorized loads and stores, `__syncthreads`.
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
//        -DGEMM_BF16=1 -DSPLITS=4 -DBM=64 -DBN=64 -DBK=64 -DWM=32 -DWN=32 \
//        -c gemm_tc_flat_splitk_noepi.cu
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

// Element type of A, B and D. Accumulation is always fp32 regardless.
// Set to 1 for __nv_bfloat16, 0 for half.
// Instead of a flag would just be an element type with a typeclass in kuiper.
#ifndef GEMM_BF16
#define GEMM_BF16 0
#endif

// --- Number of k-slices, i.e. blocks per output tile. See "SPLITS IS A
// --- COMPILE-TIME PARAMETER" above. Must be >= 2 (use the matching no-split-K
// --- file for 1) and <= K/BK (checked at launch). Larger SPLITS fills more SMs
// --- but linearly increases the workspace traffic the reduce kernel must move,
// --- so it is a real optimum, not a monotone knob.
#ifndef SPLITS
#define SPLITS 4
#endif

// --- Block tile: the BM x BN piece of D one thread block computes, and the BK
// --- depth of one k-step. Bigger BM*BN means more arithmetic per byte staged
// --- (ratio BM*BN/(BM+BN)); bigger BK means fewer barriers per unit of work,
// --- but all three multiply into the shared-memory footprint.
#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 64
#endif

// --- Warp tile: the WM x WN piece of the block tile one warp owns. This fixes
// --- the number of warps at (BM/WM)*(BN/WN) and the accumulator register count
// --- at (WM/16)*(WN/16)*8 per thread. 64x64 is the practical ceiling on
// --- sm_86: 128 accumulator registers/thread, ~210-235 total, no spills.
#ifndef WM
#define WM 32
#endif
#ifndef WN
#define WN 32
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

// --- Reduce kernel block size. Pure memory traffic, so this is not sensitive.
#ifndef REDUCE_THREADS
#define REDUCE_THREADS 256
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

// >>> The pipeline buffers are the ONLY use of shared memory in this kernel.
// >>> There is no epilogue scratch to add or to alias, because the output
// >>> element type here is fp32 -- the accumulator's own type -- so the
// >>> accumulator fragments go straight to global memory via
// >>> `store_matrix_sync`, with no conversion and hence nowhere to stage.
// >>>
// >>> Contrast the no-split-K files, which must narrow fp32 -> elem_t, need a
// >>> shared fp32 scratch to do it in, and (because a Kuiper port is not
// >>> expected to be able to reuse one shared allocation at two element types)
// >>> pay for that scratch on top of their pipeline buffers rather than
// >>> aliasing it over them. That costs them up to 29% on large shapes. This
// >>> kernel does not pay it.
constexpr int SMEM_BYTES = PIPE_BYTES;

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

static_assert(SPLITS >= 2,
              "SPLITS == 1 is the matching no-split-K file's job; using this "
              "kernel for it would add a pointless workspace round trip");

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
// Pass 1: the partial-sum kernel.
// =============================================================================
// Grid:   x = (M/BM)*(N/BN) block tiles, flattened and then swizzled below
//         z = SPLITS, the k-slice index
// Block:  THREADS threads, 1-D
// Shared: SMEM_BYTES, dynamic
//
// Writes ws[blockIdx.z] and nothing else. Takes no C, no alpha, no beta and
// no D: the epilogue belongs to pass 2 and nothing here needs to know it
// exists. This kernel is in fact IDENTICAL in the epi and noepi files --
// splitting k moves the entire epilogue into pass 2, so pass 1 cannot tell
// the two apart. Only the reduce kernel below differs.
//
// Arguments are passed flat rather than in a `Params` struct; there is no
// __grid_constant__ struct to unpack, so every scalar arrives in constant bank
// 0 exactly as it would in a Kuiper-generated signature.
extern "C" __global__ __launch_bounds__(THREADS) void gemm_tc_splitk_noepi_kernel(
    const elem_t* __restrict__ A,  // (M, K) row-major
    const elem_t* __restrict__ B,  // (N, K) row-major  == (K, N) col-major
    float* __restrict__ ws,        // (SPLITS, M, N) fp32
    int M, int N, int K) {
    extern __shared__ __align__(16) char smem_raw[];
    // Layout of the dynamic allocation, in full:
    //   [ A buf0 | A buf1 | B buf0 | B buf1 ]
    //     \------------ PIPE_BYTES ---------/
    elem_t* sA = reinterpret_cast<elem_t*>(smem_raw);
    elem_t* sB = sA + STAGES * A_ELEMS;

    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
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
    // blockIdx.z takes a contiguous, BK-aligned slice of the k tiles. The
    // slice bounds are computed by proportional division rather than
    // `ceil(ktiles/SPLITS)` per split, because the latter can leave a trailing
    // split with an EMPTY range -- and an empty split writes nothing into its
    // workspace slice, which the reduction then reads as uninitialized memory.
    // That was a real, nondeterministic bug in an earlier version. This form
    // gives every split either floor or ceil of ktiles/SPLITS tiles, so no
    // split is empty as long as SPLITS <= ktiles, which the launcher enforces.
    //
    // The ranges are contiguous, disjoint and cover [0, ktiles): split z gets
    // [ktiles*z/SPLITS, ktiles*(z+1)/SPLITS). Summing the SPLITS partials
    // therefore reproduces the full k reduction exactly.
    const int ktiles_all = K / BK;
    const int kt0 = (int)((long long)ktiles_all * blockIdx.z / SPLITS);
    const int kt1 = (int)((long long)ktiles_all * (blockIdx.z + 1) / SPLITS);
    const int ktiles = kt1 - kt0;
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
                          (size_t)kt0 * BK + t_col;
    const elem_t* b_src = B + (size_t)(block_col * BN + t_row) * K +
                          (size_t)kt0 * BK + t_col;
    // Fixed offset of this thread's cell inside a shared buffer.
    const int t_dst = t_row * LDT + t_col;

    // -------------------------------------------------------------- prologue
    // With STAGES == 2 the general `for (s = 0; s < STAGES-1; s++)` prologue is
    // exactly one iteration: fill buffer 0 and start the pipeline. `ktiles >= 1`
    // is a precondition (SPLITS <= K/BK), so this is unconditional.
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
    // There isn't one. The destination is fp32 and so is the accumulator, so
    // the fragments are simply stored where they belong. `store_matrix_sync`
    // accepts a global pointer, and mem_row_major with ldm = N is exactly the
    // workspace's layout, so this is a direct spill of registers to memory with
    // no conversion, no staging and no index arithmetic beyond the base
    // address. (Kuiper's KPR_STORE_COMB does the same thing -- store a fragment
    // to a global pointer with ldm -- so this is expressible.)
    //
    // Measured against routing these values through the shared scratch the way
    // the no-split-K kernels must: bitwise identical, and 0-3% faster
    // (256x896x4864 sp4 66.8 -> 66.1us, 128x896x4864 sp8 38.8 -> 37.7us,
    // 256x1152x896 sp2 18.3 -> 17.9us, 256x4864x896 sp2 81.1 -> 78.8us).
    //
    // NO BARRIER IS NEEDED HERE, unlike in the no-split-K kernels, and the
    // reason is worth stating because it is easy to add one "just in case":
    //   * no shared memory is touched below, so there is nothing to order
    //     against the k-loop's reads and writes of sA/sB;
    //   * no cp.async is outstanding. The prologue committed one batch and each
    //     iteration except the last committed one more, and every iteration
    //     begins with `wait_prior(0)`, which drains all of them. The last
    //     iteration commits nothing (its `kt + 1 < ktiles` is false), so on
    //     exit the count is zero;
    //   * each thread writes only registers it owns, to addresses no other
    //     thread writes.
    //
    // Each block writes the BM x BN sub-block of ws[z] that it owns, and
    // distinct (block, z) own disjoint sub-blocks, so the SPLITS slices are
    // filled independently with no synchronization between them anywhere.
    float* wsz = ws + (size_t)blockIdx.z * M * N +
                 (size_t)(block_row * BM + warp_m * WM) * N +
                 (block_col * BN + warp_n * WN);
    #pragma unroll
    for (int i = 0; i < MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            wmma::store_matrix_sync(wsz + (size_t)(i * FRAG) * N + j * FRAG,
                                    acc[i][j], N, wmma::mem_row_major);
}

// =============================================================================
// Pass 2: the reduction.
// =============================================================================
// Sums the SPLITS fp32 partials in `ws`, narrows to elem_t, writes D.
// There is no C here, so "epilogue" means only the narrowing back to
// elem_t. The reduction still has to be a separate pass, because no block
// of pass 1 has seen the whole k range.
//
// Grid-stride over 128-bit output granules, one thread per granule, so the grid
// size is decoupled from M*N and the kernel is correct for any launch
// configuration. Cost is a (SPLITS + 1) * M * N * 4-byte pass over memory --
// entirely bandwidth-bound, no math worth speaking of -- which is the price of
// split-K and the reason the tuner is conservative about using it.
//
// The `for (s...)` loop has a compile-time trip count because SPLITS is a
// macro, so it fully unrolls into SPLITS independent load streams; that is one
// of the reasons to make SPLITS compile-time (see the header).
extern "C" __global__ void gemm_tc_reduce_noepi_kernel(
    const float* __restrict__ ws,  // (SPLITS, M, N) fp32
    elem_t* __restrict__ D,        // (M, N)
    int M, int N) {
    const size_t total = (size_t)M * N / VEC;  // number of 128-bit granules
    const size_t stride = (size_t)M * N;       // distance between splits
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t off = i * VEC;  // flat element index into D

        float val[VEC];
        #pragma unroll
        for (int e = 0; e < VEC; e++) val[e] = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; s++) {
            const float* src = ws + s * stride + off;
            #pragma unroll
            for (int e = 0; e < VEC; e += 4) {
                // float4, not VEC floats: the workspace is fp32 while D is
                // elem_t, so one 128-bit granule of D is VEC elements but
                // VEC/4 float4s of workspace.
                const float4 v = *reinterpret_cast<const float4*>(src + e);
                val[e] += v.x;
                val[e + 1] += v.y;
                val[e + 2] += v.z;
                val[e + 3] += v.w;
            }
        }

        elem_t out[VEC];
        #pragma unroll
        for (int e = 0; e < VEC; e++) out[e] = to_elem(val[e], elem_t());
        *reinterpret_cast<int4*>(D + off) =
            *reinterpret_cast<const int4*>(out);
    }
}

// =============================================================================
// Host launcher.
// =============================================================================
// Flat on purpose: the general version dispatches through a table of function
// pointers so that one compiled extension can pick any of forty tilings at run
// time without a JIT step. That table is JIT-harness machinery, not part of the
// algorithm, so here there are exactly two kernels and two direct launches.
//
// `ws` must be at least SPLITS*M*N floats and is required, not optional. The
// caller owns it; neither kernel allocates or frees (this matches the
// convention that Kuiper kernels never allocate). Its contents need not be
// initialized: pass 1 writes every cell before pass 2 reads any.
//
// Both launches go on the same stream, which is what orders pass 2 after
// pass 1 -- there is no other synchronization, and in particular nothing
// device-wide inside either kernel.
extern "C" cudaError_t gemm_tc_flat_splitk_noepi_launch(
    const elem_t* A, const elem_t* B, elem_t* D, float* ws, int M, int N,
    int K, cudaStream_t stream) {
    // The divisibility preconditions the kernels rely on instead of bounds
    // checks. In Kuiper these are refinements on the argument types.
    assert(M % BM == 0 && N % BN == 0 && K % BK == 0);
    assert(N % VEC == 0 && K % VEC == 0);
    assert(ws != nullptr);
    // No split may be empty; see the k-range comment. Note this is the one
    // precondition that could not be a static_assert, since K is runtime --
    // but because SPLITS is compile-time it is still just a bound on K.
    assert(K / BK >= SPLITS);

    // Past 48 KB the larger shared-memory carveout is opt-in per kernel.
    if (SMEM_BYTES > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_tc_splitk_noepi_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_BYTES);
        if (e != cudaSuccess) return e;
    }

    dim3 grid((M / BM) * (N / BN), 1, SPLITS);
    gemm_tc_splitk_noepi_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(A, B, ws, M, N, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    const size_t granules = (size_t)M * N / VEC;
    int blocks = (int)((granules + REDUCE_THREADS - 1) / REDUCE_THREADS);
    if (blocks > 65535) blocks = 65535;  // grid-stride handles the rest
    gemm_tc_reduce_noepi_kernel<<<blocks, REDUCE_THREADS, 0, stream>>>(
        ws, D, M, N);
    return cudaGetLastError();
}
