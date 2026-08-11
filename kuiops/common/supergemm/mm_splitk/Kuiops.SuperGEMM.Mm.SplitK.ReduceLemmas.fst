module Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas

(* The matrix pass 2 computes: cell [(i,j)] of the output is the sum, over the
   [splits] row slabs of the workspace, of workspace cell [(z*rows+i, j)]. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas { sum_upto, rsum_upto, sum_upto_approx }

module ML = FStar.Math.Lemmas

#set-options "--fuel 1 --ifuel 1 --z3rlimit 10"

let ws_cell
  (#et : Type0) {| scalar et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (i j z : nat)
  : GTot et
= if i < rows && j < cols && z < splits && z * rows + i < mws
  then acc2 eW (z * rows + i) j
  else zero

let rws_cell
  (#mws #cols : nat) (rows splits : nat)
  (rW : chest2 real mws cols) (i j z : nat)
  : GTot real
= if i < rows && j < cols && z < splits && z * rows + i < mws
  then acc2 rW (z * rows + i) j
  else 0.0R

let reduce_ws
  (#mws #cols : nat) (rows splits : nat)
  (rW : chest2 real mws cols)
  : GTot (chest2 real rows cols)
= mk2 (fun (i : natlt rows) (j : natlt cols) ->
         rsum_upto (rws_cell rows splits rW i j) splits)

let ws_cell_approx1
  (#et : Type0) {| scalar et, real_like et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (rW : chest2 real mws cols)
  (i j z : nat)
  : Lemma (requires eW %~ rW)
          (ensures ws_cell rows splits eW i j z %~ rws_cell rows splits rW i j z)
= (a0 #et)

(* The accumulator-typed fold of one output cell approximates that cell of
   [reduce_ws]. *)
let reduce_ws_cell_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (rW : chest2 real mws cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (requires eW %~ rW)
          (ensures sum_upto (ws_cell rows splits eW i j) splits
                   %~ acc2 (reduce_ws rows splits rW) i j)
= introduce forall (z : nat).
      z < splits ==> ws_cell rows splits eW i j z %~ rws_cell rows splits rW i j z
  with introduce _ ==> _
  with _. ws_cell_approx1 rows splits eW rW i j z;
  sum_upto_approx (ws_cell rows splits eW i j) (rws_cell rows splits rW i j) splits

(* Row [z*rows+i] of the workspace is what split [z] contributed to output row
   [i]; stated as the rewriting the reduce kernel's loop performs. *)
let ws_cell_step
  (#et : Type0) {| scalar et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (i : natlt rows) (j : natlt cols) (z : nat)
  : Lemma (requires z < splits /\ z * rows + i < mws)
          (ensures sum_upto (ws_cell rows splits eW i j) (z + 1)
                   == add (sum_upto (ws_cell rows splits eW i j) z)
                          (acc2 eW (z * rows + i) j))
= ()

let ws_row_bound (rows splits mws : nat) (z i : nat)
  : Lemma (requires mws == splits * rows /\ z < splits /\ i < rows)
          (ensures z * rows + i < mws)
= ML.lemma_mult_le_right rows (z + 1) splits;
  ML.distributivity_add_left z 1 rows
