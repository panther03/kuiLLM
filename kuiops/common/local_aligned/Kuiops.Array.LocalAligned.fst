module Kuiops.Array.LocalAligned

#lang-pulse

open Kuiper

module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT

(* Axiom; see the .fsti header for what is assumed and why it holds on the
   real toolchain.  Stated as an explicit [admit()] rather than under
   [--admit_smt_queries]: a scoped flag silently discharges every obligation in
   its extent, so a single missing [#pop-options] would void the module, and it
   does not show up in an [admit] audit. *)
let local_aligned16 (#a : Type0) {| sized a |} (arr : array a) = admit()
