module Kuiops.Array.LocalAligned

#lang-pulse

open Kuiper

module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT

(* Axiom; see the .fsti header for what is assumed and why it holds on the
   real toolchain. *)
#push-options "--admit_smt_queries true"
let local_aligned16 (#a : Type0) {| sized a |} (arr : array a) = ()
#pop-options
