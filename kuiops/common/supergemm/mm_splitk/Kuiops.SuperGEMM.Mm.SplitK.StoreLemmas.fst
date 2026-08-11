module Kuiops.SuperGEMM.Mm.SplitK.StoreLemmas

(* Pure support for the direct fragment store: relate the content of a warp's
   [wm x wn] workspace tile to the real accumulator [rAcc], one [frag x frag]
   subtile at a time. *)

open Kuiper
open Kuiper.Chest { chest2, acc2 }
open Kuiper.EMatrix { lemma_approximates_intro }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiper.TensorCore.Base { FragAcc, value_for }
open Kuiops.SuperGEMM.Mm.Params { frag }

module ML = FStar.Math.Lemmas

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"

let frag_div (a : nat) : Lemma (frag /? (a * frag) /\ (a * frag) / frag == a)
= ML.cancel_mul_mod a frag; ML.cancel_mul_div a frag

(* The [frag x frag] subtiles of [e] strictly before position [(ito, jto)] in
   row-major order all approximate the matching subtile of [r]. *)
let subtiles_approx
  (#et : Type0) {| scalar et |} {| real_like et |}
  (a b : nat)
  (e : chest2 et (a * frag) (b * frag))
  (r : chest2 real (a * frag) (b * frag))
  (ito jto : nat)
  : prop
= frag_div a; frag_div b;
  forall (i : natlt a) (j : natlt b).
    (i < ito \/ (i == ito /\ j < jto)) ==>
      ematrix_subtile e frag frag i j %~ ematrix_subtile r frag frag i j

(* Per-fragment approximation fact for the accumulator content [em0], packaged
   at top level so its nonlinear flat-index bound is discharged once. *)
let frags_approx
  (#et : Type0) {| scalar et |} {| real_like et |}
  (a b : nat)
  (em0 : seq (value_for et FragAcc frag frag frag))
  (r : chest2 real (a * frag) (b * frag))
  : prop
= frag_div a; frag_div b;
  Seq.length em0 == a * b /\
  (forall (i : natlt a) (j : natlt b).
     Seq.index em0 (i * b + j) %~ ematrix_subtile r frag frag i j)

(* Once every subtile approximates, so does the whole tile. *)
let tile_approx_from_subtiles
  (#et : Type0) {| scalar et |} {| real_like et |}
  (a b : nat)
  (e : chest2 et (a * frag) (b * frag))
  (r : chest2 real (a * frag) (b * frag))
  (_ : squash (subtiles_approx a b e r a 0))
  : Lemma (e %~ r)
= frag_div a; frag_div b;
  introduce forall (x : natlt (a * frag)) (y : natlt (b * frag)). acc2 e x y %~ acc2 r x y
  with begin
    let i = x / frag in
    let j = y / frag in
    let x' = x % frag in
    let y' = y % frag in
    ML.lemma_div_mod x frag;
    ML.lemma_div_mod y frag;
    introduce i >= a ==> False
    with _. ML.lemma_mult_le_right frag a i;
    introduce j >= b ==> False
    with _. ML.lemma_mult_le_right frag b j;
    assert (acc2 (ematrix_subtile e frag frag i j) x' y' == acc2 e (i * frag + x') (j * frag + y'));
    assert (acc2 (ematrix_subtile r frag frag i j) x' y' == acc2 r (i * frag + x') (j * frag + y'))
  end;
  lemma_approximates_intro e r

(* A finished row extends the prefix by one. *)
let subtiles_approx_row_done
  (#et : Type0) {| scalar et |} {| real_like et |}
  (a b : nat)
  (e : chest2 et (a * frag) (b * frag))
  (r : chest2 real (a * frag) (b * frag))
  (ito : nat)
  : Lemma (requires subtiles_approx a b e r ito b)
          (ensures subtiles_approx a b e r (ito + 1) 0)
= frag_div a; frag_div b

#pop-options
