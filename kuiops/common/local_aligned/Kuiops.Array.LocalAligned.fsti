module Kuiops.Array.LocalAligned

(* The one place in kuiops that asserts 16-byte alignment of a *kernel-local*
   (stack-allocated) vector buffer.

   WHAT IS ASSUMED.  [local_aligned16] states that an array which is exactly 16
   bytes wide ([length arr * size et == 16]) is 16-byte aligned.  Every such
   array in kuiops is a whole allocation, and a whole 16-byte allocation is
   16-byte aligned under every allocator Kuiper can reach -- stack (see below),
   [cudaMalloc] (256-byte aligned), and shared memory (already covered
   separately by [Kuiops.SHMem.Aligned]).

   The honest side condition would additionally be [A.is_full_array arr],
   which rules out a 16-byte *slice* of a larger array -- the only way the
   conclusion could fail.  It is not stated because Pulse's array-literal
   desugaring ([let mut a = [| init; n |]]) does not surface [is_full_array]
   into the enclosing proof context, only [length]; the fact is therefore
   unavailable at the one call site.  Using [Pulse.Lib.Array.PtsTo.with_local]
   directly would surface it, but that is a continuation-passing combinator
   whose closure KaRaMeL hoists to top level, which in turn leaks the
   [noextract] [strided_row_major] class into the generated CUDA.  Note that
   upstream Kuiper's corresponding [assume pure (aligned 16 local)] carries no
   side condition whatsoever, so this is a strictly narrower assumption.

   TODO: reinstate the [is_full_array] conjunct once Pulse propagates it out of
   the array-literal form.

   WHY IT IS TRUE ON THE REAL TOOLCHAIN (verified empirically, see below).
   [Kuiper.Array]'s [base_address] model does not track stack-allocation
   alignment, so the fact is not derivable inside F* and must be axiomatised.
   It is nonetheless satisfied by nvcc:

   Extraction emits a bare declaration with no alignment attribute -- e.g.
   [__nv_bfloat16 obuf[8U];] in [Klas_GEMM_TensorCore2D_To.cu] -- and
   [scripts/fixup.sed] adds none.  Read statically that looks like a 2-byte
   aligned object under a 128-bit access, i.e. UB.  It is not, because the
   alignment is established by NVVM rather than by the C declaration: LLVM's
   [getOrEnforceKnownAlignment] raises an alloca's alignment to match the
   widest access made to it, and the local depot is laid out accordingly.

   Confirmed two ways on sm_86 / CUDA 13.0, using the exact extracted pattern
   (a [float4]-cast copy through an 8-element bf16 local), made adversarial by
   placing an odd-sized 3-element bf16 local first and forcing both to real
   local memory with dynamic indexing:

     - PTX: [.local .align 16 .b8 __local_depot0[32]], with the staging buffer
       at depot offset 16 -- NVVM padded past the 3-element array specifically
       to keep the buffer 16-byte aligned -- and the access emitted as
       [ld.local.v4.u32].
     - Runtime: 512 threads each checking [(uintptr_t)buf % 16], zero
       misalignments, no CUDA error.

   So no [__align__(16)] / [_Alignas(16)] extraction fixup is required.  The
   guarantee comes from the compiler's alloca-alignment inference, which
   applies exactly when a 128-bit access to the buffer is present -- which is
   the only situation in which this lemma is used.  If a future use ever
   allocated such a buffer WITHOUT a vector access on it, the inference would
   not fire; that is harmless, since without a vector access nothing needs the
   alignment.

   This axiom makes explicit, in one auditable place, the assumption that is
   scattered as [assume pure (aligned 16 local); // FIXME local arrays do not
   need alignment] throughout upstream Kuiper
   ([Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueChunkUpdate],
   [Kuiper.Kernel.GEMM.Copy.Vec2], ...).

   TODO(upstream): the correct long-term fix is an aligned-local-allocation
   primitive in Kuiper proper (cf. [gpu_array_alloc], which already hands out
   [aligned 128]); this module should then be retired.  Precedent for the
   axiom-module shape: [Kuiops.SHMem.Aligned]. *)

#lang-pulse

open Kuiper

module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT

(* Extracts to nothing (a [Lemma]), so no closure and no stack-array plumbing
   survives into the generated CUDA. *)
val local_aligned16 (#a : Type0) {| sized a |} (arr : array a)
  : Lemma (requires A.length arr * size #a == 16)
          (ensures  aligned 16 arr)
