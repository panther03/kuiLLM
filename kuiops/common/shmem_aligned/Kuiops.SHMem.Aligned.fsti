module Kuiops.SHMem.Aligned

(* The one place in kuiops that asserts 16-byte alignment of a shared array.

   [cp.async] with a 16-byte granule requires both operands to be 16-byte
   aligned.  Kuiper places every shared array in the block's dynamic shared
   segment (`extern __shared__`), whose base CUDA guarantees to be 16-byte
   aligned, but [Kuiper.Array]'s [base_address] model does not track where in
   that segment a given array lands, so the fact is not derivable.

   The obligation this axiom shifts onto the caller: every shared array
   declared *before* [a] in the block's [shmems_desc] must have a byte size
   that is a multiple of 16.  A kernel that only allocates tiles of
   [rows * (cols + pad)] elements discharges this by choosing tile parameters
   whose byte size is a multiple of 16, which is anyway forced by the
   vectorized staging.

   Precedent for the same gap: [Kuiper.Array.Core.fst] assumes
   [aligned 128] for kernel-local arrays, and
   [Kuiper.Example.PipelineCopy.fst] assumes exactly this fact inline.

   TODO(upstream): [Kuiper.SHMem] should hand out an alignment fact with the
   array, computed from the preceding [shmem_desc] entries. *)

#lang-pulse

open Kuiper
open Kuiper.SHMem { is_block_array }

val shmem_aligned16
  (#et : Type0)
  (a : array et { is_block_array a })
  : Lemma (aligned 16 a)
