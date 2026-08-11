module Kuiops.SHMem.Aligned

#lang-pulse

open Kuiper
open Kuiper.SHMem { is_block_array }

(* Axiom; see the .fsti header for what is assumed and why.  Stated as an
   explicit [admit()] rather than under [--admit_smt_queries]: a scoped flag
   silently discharges every obligation in its extent, so a single missing
   [#pop-options] would void the module, and it does not show up in an [admit]
   audit. *)
let shmem_aligned16 (#et : Type0) (a : array et { is_block_array a }) = admit()
