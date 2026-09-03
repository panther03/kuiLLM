module Kuiops.GEMM.T.FlipFlopBarrier2

(* ====================================================================== *)
(* Fork of [Kuiper.Kernel.GEMM.FlipFlopBarrier2] for the transposed-B      *)
(* ("TN") GEMM, where the shared-memory B tile is stored COLUMN-major.     *)
(*                                                                         *)
(* Upstream's barrier contract names the ROW-major chunk partition of the  *)
(* B tile at dims (bk, bn):                                                *)
(*                                                                         *)
(*     own_strided_chunks m2 (ematrix_subtile eB bk bn (it/2) mcol) ...       *)
(*                                                                         *)
(* [Copy.Vec2.in_chunk] keys off the DIMENSIONS, not the layout, so no     *)
(* choice of [l2] can make that partition match a column-major tile's      *)
(* physical address order [nn*bk + kk].  Whatever partition the barrier    *)
(* names, the copy must produce -- and [cp_array2_vec] ties source and     *)
(* destination to the same (rows, cols) -- so keeping the (bk, bn)         *)
(* partition would force the GLOBAL column-major B read onto a stride-k    *)
(* access pattern.  Hence this fork.                                       *)
(*                                                                         *)
(* The ONLY change is on the B side: [own_strided_chunks m2] becomes         *)
(* [own_strided_chunks_cm m2], and likewise for [live_strided_chunks].      *)
(* Since [own_strided_chunks_cm m em] unfolds DEFINITIONALLY to            *)
(* [own_strided_chunks (atranspose m) (ctranspose em)], this is an         *)
(* instantiation of upstream's proof at a transposed view, not a second    *)
(* proof.  The A side is untouched.                                        *)
(*                                                                         *)
(* TODO(upstream): the version worth contributing upstream is a barrier    *)
(* parameterized by the two operands' ownership predicates, which would    *)
(* let NN and TN share one module instead of forking.  That is an          *)
(* unproven refactor (six-ish slprop parameters plus the lemmas relating   *)
(* them) and was deliberately NOT attempted here.                          *)
(* ====================================================================== *)


#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.EMatrix
open Kuiper.Math { even, odd }
open Kuiper.Tensor.Tiling

open Kuiper.Tensor
module B = Kuiper.Barrier
module SZ = Kuiper.SizeT
module CV = Kuiper.Kernel.GEMM.Copy.Vec2

open Kuiops.Tensor.Transpose2 { atranspose, ctranspose,
                                 lemma_ctranspose_involutive, atranspose_back }

let own_strided_chunks
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (em : chest2 et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (chunk et #_ #hvc) rows cols nthr tid ij}).
    tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)

let live_strided_chunks
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
=
  exists* em.
    own_strided_chunks m em nthr tid

(* --- Column-major B-tile partition, by delegation to the transposed view. --- *)

let own_strided_chunks_cm
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (em : chest2 et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
= own_strided_chunks (atranspose m) (ctranspose em) nthr tid

let live_strided_chunks_cm
  (#et : Type0) {| sized et, hvc: has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
= live_strided_chunks (atranspose m) nthr tid

let bp_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (em : chest2 et rows cols)
  (nthr : pos)
  : slprop
  = m |-> Frac (1.0R /. nthr) em

(* ---- Array2-level strided-chunk (de)composition, exported for reuse ----
   These operate on an arbitrary [array2 et l] (any layout, not just full
   ones), so the skewed pipeline tiles of SuperGEMM can be partitioned with
   the same proofs. *)

ghost
fn split_array2_into_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#em : chest2 et rows cols)
  (nthr : pos)
  requires
    m |-> em
  ensures
    pure (SZ.fits (l.ulen))
  ensures
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid

ghost
fn join_array2_from_strided_chunks
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#em : chest2 et rows cols)
  (nthr : pos)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+ (tid : natlt nthr).
      own_strided_chunks m em nthr tid
  ensures
    m |-> em

ghost
fn join_array2_from_strided_chunks_underspec
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (nthr : pos)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+ (tid : natlt nthr).
      live_strided_chunks m nthr tid
  ensures
    live m

(* The odd branch indexes tile [it / 2].  Make the arithmetic fact explicit;
   recent F* versions verify each leaf goal independently. *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let half_lt_quot (it shared bk : nat)
  : Lemma (requires shared > 0 /\ bk > 0 /\ shared % bk = 0 /\ it < 2 * shared / bk)
          (ensures it / 2 < shared / bk)
          [SMTPat (it / 2); SMTPat (shared / bk)]
  = let q = shared / bk in
    FStar.Math.Lemmas.lemma_div_exact shared bk;
    assert (2 * shared == (2 * q) * bk);
    FStar.Math.Lemmas.multiple_division_lemma (2 * q) bk;
    FStar.Math.Lemmas.euclidean_division_definition it 2
#pop-options

let barrier_p
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    if it >= 2 * shared / bk then
      emp
    else if even it then
      (exists* em1. bp_sharing m1 em1 nthr) **
      (exists* em2. bp_sharing m2 em2 nthr)
    else
      let mrow = bid / (cols/bn) in
      let mcol = bid % (cols/bn) in
      half_lt_quot it shared bk;
      own_strided_chunks m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr tid **
      own_strided_chunks_cm m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr tid

#push-options "--z3rlimit 40"
let barrier_q
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.barrier_side nthr =
  fun it tid ->
    if it >= 2 * shared / bk then
      emp
    else if even it then
      live_strided_chunks m1 nthr tid **
      live_strided_chunks_cm m2 nthr tid
    else
      let mrow = bid / (cols/bn) in
      let mcol = bid % (cols/bn) in
      half_lt_quot it shared bk;
      bp_sharing m1 (ematrix_subtile eA bm bk mrow (it / 2)) nthr **
      bp_sharing m2 (ematrix_subtile eB bk bn (it / 2) mcol) nthr
#pop-options

let contract
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk))
  (sar2 : larray etB (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : B.contract nthr =
{
  B.rin  = barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
  B.rout = barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid;
}

let barrier_tok
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk))
  (sar2 : larray etB (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  : slprop
  = B.barrier_tok (contract eA eB l1 l2 sar1 sar2 nthr bid)

(* The proof of correctness. *)
ghost
fn barrier_p_to_q_transform
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (l1 : full_layout2 bm bk)
  (l2 : full_layout2 bk bn)
  (sar1 : larray etA (bm * bk))
  (sar2 : larray etB (bk * bn))
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (#_ : squash (chunk etB /?+ bk)) // TN: transposed view has cols = bk
  (#_ : squash (chunk etA /?+ bk))
  (#_ : squash (chunk etA * nthr /?+ (bm * bk)))
  (#_ : squash (chunk etB * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (l1.ulen)))
  (#_ : squash (SZ.fits (l2.ulen)))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      barrier_p eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid
  ensures
    forall+ (tid : natlt nthr).
      barrier_q eA eB (from_array l1 sar1) (from_array l2 sar2) nthr bid it tid

(* Per-thread helpers for odd iterations. *)
ghost
fn fold_barrier_p_odd
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (mrow : nat{mrow == bid / (cols/bn)})
  (mcol : nat{mcol == bid % (cols/bn)})
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    own_strided_chunks m1 (ematrix_subtile eA bm bk mrow bkIdx) nthr tid **
    own_strided_chunks_cm m2 (ematrix_subtile eB bk bn bkIdx mcol) nthr tid
  ensures
    barrier_p eA eB m1 m2 nthr bid (2 * bkIdx + 1) tid

ghost
fn unfold_barrier_q_odd
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (mrow : nat{mrow == bid / (cols/bn)})
  (mcol : nat{mcol == bid % (cols/bn)})
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx + 1) tid
  ensures
    bp_sharing m1 (ematrix_subtile eA bm bk mrow bkIdx) nthr **
    bp_sharing m2 (ematrix_subtile eB bk bn bkIdx mcol) nthr

(* Per-thread helpers for even iterations. *)
ghost
fn fold_barrier_p_even
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    (exists* em1. bp_sharing m1 em1 nthr) **
    (exists* em2. bp_sharing m2 em2 nthr)
  ensures
    barrier_p eA eB m1 m2 nthr bid (2 * bkIdx) tid

ghost
fn unfold_barrier_q_even
  (#etA : Type0) (#etB : Type0)
  {| sized etA, has_vec_cpy etA, sized etB, has_vec_cpy etB |}
  (#rows #shared #cols : pos)
  (eA : chest2 etA rows shared)
  (eB : chest2 etB shared cols)
  (#bm : pos{bm /?+ rows})
  (#bk : pos{bk /?+ shared})
  (#bn : pos{bn /?+ cols})
  (#l1 : layout2 bm bk)
  (#l2 : layout2 bk bn)
  (m1 : array2 etA l1)
  (m2 : array2 etB l2)
  (nthr : pos)
  (bid : natlt (rows/bm * (cols/bn)))
  (bkIdx : natlt (shared / bk))
  (tid : natlt nthr)
  requires
    barrier_q eA eB m1 m2 nthr bid (2 * bkIdx) tid
  ensures
    live_strided_chunks m1 nthr tid **
    live_strided_chunks_cm m2 nthr tid
