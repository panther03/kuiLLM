module Kuiops.Array.LocalAligned

(* The one place in kuiops that asserts 16-byte alignment of a *kernel-local*
   (stack-allocated) vector buffer.

   WHAT IS ASSUMED.  [with_aligned_local16] is [Pulse.Lib.Array.with_local]
   specialised to a buffer of exactly [chunk et] elements (i.e. exactly 16
   bytes: [chunk et * size et == 16]), whose scoped [body] additionally gets
   [pure (aligned 16 arr)].  Nothing else is assumed: the fact is available
   only for the buffer this combinator itself allocates, never for a
   pre-existing array, so the axiom cannot be instantiated at an arbitrary
   [array et].

   WHY IT IS TRUE ON THE REAL TOOLCHAIN.  A 128-bit ([int4]) load/store, and
   cp.async's 16-byte granule, require the operand to be 16-byte aligned.  The
   whole point of such a buffer is to be the staging register/local for one
   [int4] transfer, so the emitted C declaration must carry 16-byte alignment
   (e.g. a KaRaMeL [KRML_ALIGNED_TYPE] / a C11 [_Alignas(16)] / an
   [__align__(16)] attribute on the local array).  [Kuiper.Array]'s
   [base_address] model does not track stack-allocation alignment, so the fact
   is not derivable and must be axiomatised here.

   >>> TODO(extraction): CONFIRM the generated CUDA declaration of this buffer
   >>> is in fact 16-byte aligned.  A plain 8-element uint16 buffer (bf16) is
   >>> only 2-byte aligned, and a 128-bit cast-store through it is UB -- if
   >>> KaRaMeL does not emit an alignment attribute, this axiom is FALSE and the
   >>> buffer declaration must be post-processed to add an [__align__(16)]
   >>> attribute (see the repo's extraction fixup step).  This is a real
   >>> correctness obligation, not cosmetic.

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

inline_for_extraction noextract
fn with_aligned_local16
  (#a : Type0) {| sized a |}
  (init : a)
  (len : SZ.t { SZ.v len * size #a == 16 })
  (#pre : slprop)
  (ret_t : Type0)
  (#post : ret_t -> slprop)
  (body : (arr : array a) -> stt ret_t
            (pre **
             A.pts_to arr (Seq.create (SZ.v len) init) **
             pure (A.is_full_array arr /\ A.length arr == SZ.v len /\ aligned 16 arr))
            (fun r -> post r ** (exists* v. A.pts_to arr v)))
  requires pre
  returns  r : ret_t
  ensures  post r
