module Kuiops.Array.LocalAligned

#lang-pulse

open Kuiper

module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT

(* The alignment fact for the freshly allocated stack buffer is discharged by
   admitted SMT (see the .fsti header); everything else is real Pulse framing
   over [Pulse.Lib.Array.with_local]. *)
#push-options "--admit_smt_queries true"
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
{
  let mut arr = [| init; len |];
  body arr
}
#pop-options
