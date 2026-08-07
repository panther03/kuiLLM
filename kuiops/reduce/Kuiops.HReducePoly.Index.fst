module Kuiops.HReducePoly.Index

(* Flat index builders for the reduction entry points.

   [Kuiops.HReducePoly.Spec.conc_snoc] does this generically, but it recurses
   over an erased shape, so Karamel cannot extract it (it casts the result to
   the Top type and bails). Each supported rank therefore gets a monomorphic,
   fully inlinable builder here, verified once -- as opposed to being emitted,
   together with its lemma, into every JIT instantiation. *)

open Kuiper
open Kuiper.Shape
open Kuiops.HReducePoly.Spec

module SZ = Kuiper.SizeT

inline_for_extraction noextract
let index1 (d0 : szp)
  (i : conc (INil <: shape 0)) (j : szlt (SZ.v d0))
  : conc ((SZ.v d0) @| INil) =
  (j, ())

let index1_up (d0 : szp) (i : conc (INil <: shape 0)) (j : szlt (SZ.v d0))
  : Lemma (up (index1 d0 i j) ==
           abs_snoc #0 #(INil <: shape 0) #(SZ.v d0) (up i) (SZ.v j))
  = ()

inline_for_extraction noextract
let index2 (d0 d1 : szp)
  (i : conc ((SZ.v d0) @| INil)) (j : szlt (SZ.v d1))
  : conc ((SZ.v d0) @| (SZ.v d1) @| INil) =
  let (i0, ()) = i in
  (i0, (j, ()))

let index2_up (d0 d1 : szp)
  (i : conc ((SZ.v d0) @| INil)) (j : szlt (SZ.v d1))
  : Lemma (up (index2 d0 d1 i j) ==
           abs_snoc #1 #((SZ.v d0) @| INil) #(SZ.v d1) (up i) (SZ.v j))
  = ()

inline_for_extraction noextract
let index3 (d0 d1 d2 : szp)
  (i : conc ((SZ.v d0) @| (SZ.v d1) @| INil)) (j : szlt (SZ.v d2))
  : conc ((SZ.v d0) @| (SZ.v d1) @| (SZ.v d2) @| INil) =
  let (i0, (i1, ())) = i in
  (i0, (i1, (j, ())))

let index3_up (d0 d1 d2 : szp)
  (i : conc ((SZ.v d0) @| (SZ.v d1) @| INil)) (j : szlt (SZ.v d2))
  : Lemma (up (index3 d0 d1 d2 i j) ==
           abs_snoc #2 #((SZ.v d0) @| (SZ.v d1) @| INil) #(SZ.v d2)
             (up i) (SZ.v j))
  = ()

inline_for_extraction noextract
let index4 (d0 d1 d2 d3 : szp)
  (i : conc ((SZ.v d0) @| (SZ.v d1) @| (SZ.v d2) @| INil))
  (j : szlt (SZ.v d3))
  : conc ((SZ.v d0) @| (SZ.v d1) @| (SZ.v d2) @| (SZ.v d3) @| INil) =
  let (i0, (i1, (i2, ()))) = i in
  (i0, (i1, (i2, (j, ()))))

let index4_up (d0 d1 d2 d3 : szp)
  (i : conc ((SZ.v d0) @| (SZ.v d1) @| (SZ.v d2) @| INil))
  (j : szlt (SZ.v d3))
  : Lemma (up (index4 d0 d1 d2 d3 i j) ==
           abs_snoc #3 #((SZ.v d0) @| (SZ.v d1) @| (SZ.v d2) @| INil)
             #(SZ.v d3) (up i) (SZ.v j))
  = ()
