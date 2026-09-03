module Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas

(* The matrix pass 2 computes: cell [(i,j)] of the output is the sum, over the
   [splits] row slabs of the workspace, of workspace cell [(z*rows+i, j)]. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling { ematrix_subtile }
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

let ws_run
  (#et : Type0) {| scalar et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (i base e z : nat)
  : GTot et
= ws_cell rows splits eW i (base + e) z

let rec sum_upto_ext
  (#et : Type0) {| scalar et |}
  (f g : nat -> GTot et) (t : nat)
  : Lemma (requires forall (z : nat). z < t ==> f z == g z)
          (ensures sum_upto f t == sum_upto g t)
          (decreases t)
= if t = 0 then () else sum_upto_ext f g (t - 1)

let ws_run_sum
  (#et : Type0) {| scalar et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (i base y t : nat)
  : Lemma (sum_upto (ws_run rows splits eW i base y) t
           == sum_upto (ws_cell rows splits eW i (base + y)) t)
= sum_upto_ext (ws_run rows splits eW i base y)
               (ws_cell rows splits eW i (base + y)) t

let ws_run_sum_all
  (#et : Type0) {| scalar et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (i base c t : nat)
  : Lemma (forall (y : natlt c).
             sum_upto (ws_run rows splits eW i base y) t
             == sum_upto (ws_cell rows splits eW i (base + y)) t)
= introduce forall (y : natlt c).
    sum_upto (ws_run rows splits eW i base y) t
    == sum_upto (ws_cell rows splits eW i (base + y)) t
  with ws_run_sum rows splits eW i base y t

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
  with ws_cell_approx1 rows splits eW rW i j z;
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

(* What pass 2 must land in D: the reduced workspace, post-mapped. *)
let gran_target
  (#mws #cols : nat) (rows splits : nat)
  (rW : chest2 real mws cols) (post_map_r : real -> real)
  : GTot (chest2 real rows cols)
= chest_map post_map_r (reduce_ws rows splits rW)

(* One [1 x c] granule of D approximates the matching tile of [gran_target] as
   soon as each of its cells is the post-mapped fold of that output column. *)
let gran_approx
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc |} {| scalar et_d, real_like et_d |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et_acc mws cols) (rW : chest2 real mws cols)
  (post_map : et_acc -> et_d) (post_map_r : real -> real { post_map %~ post_map_r })
  (c : pos { c /? cols })
  (v : chest2 et_d 1 c)
  (di : natlt rows) (dj : natlt (cols / c))
  (_ : squash (1 /? rows))
  : Lemma
      (requires eW %~ rW /\
                (forall (y : natlt c).
                   dj * c + y < cols /\
                   acc2 v 0 y
                   == post_map (sum_upto (ws_cell rows splits eW di (dj * c + y)) splits)))
      (ensures v %~ ematrix_subtile (gran_target rows splits rW post_map_r) 1 c di dj)
= introduce forall (a : natlt 1) (b : natlt c).
    acc2 v a b %~ acc2 (ematrix_subtile (gran_target rows splits rW post_map_r) 1 c di dj) a b
  with reduce_ws_cell_approx rows splits eW rW di (dj * c + b);
  lemma_approximates_intro v
    (ematrix_subtile (gran_target rows splits rW post_map_r) 1 c di dj)
