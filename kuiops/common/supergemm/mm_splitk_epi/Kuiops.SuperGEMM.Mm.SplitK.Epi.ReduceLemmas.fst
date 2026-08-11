module Kuiops.SuperGEMM.Mm.SplitK.Epi.ReduceLemmas

(* The matrix the epilogue-aware split-K pass 2 must produce.

   This is [Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas] with the unary
   [post_map] replaced by a binary [comb] whose first argument is read from the
   C view.  The reduction itself -- [reduce_ws], [ws_cell] and their
   approximation lemmas -- is imported unchanged from that module; only the
   post-reduction cell map differs, so only that is restated here.

   WHY THE EPILOGUE IS HERE AND CANNOT BE IN PASS 1.  [comb] is affine in the
   accumulated value (the instantiation the pipeline uses is
   [alpha * acc + beta * c]).  Pass 1 produces [splits] PARTIAL sums, one per
   block, and no block has seen the whole k range; applying [comb] there would
   add the [beta * c] term once per split rather than once in total.  Pass 2 is
   the first point at which a complete k reduction exists for an output
   element, so it is the only place the epilogue can correctly go.  That is a
   correctness constraint, not a stylistic choice, and it is what makes the
   no-epi / epi distinction structural rather than a runtime branch. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas { sum_upto }
open Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas { ws_cell, reduce_ws, reduce_ws_cell_approx }

#set-options "--fuel 1 --ifuel 1 --z3rlimit 10"

(* What pass 2 must land in D: the reduced workspace, combined cellwise with
   the C view.  [reduce_ws] is exactly the no-epi reduction; the epilogue is
   the outer [chest_comb]. *)
let gran_target
  (#mws #cols : nat) (rows splits : nat)
  (rC : chest2 real rows cols)
  (rW : chest2 real mws cols)
  (comb_r : binop real)
  : GTot (chest2 real rows cols)
= chest_comb comb_r rC (reduce_ws rows splits rW)

(* One output column of the granule, as a function of the split index: the
   term [Kuiops.SuperGEMM.Mm.SplitK.Reduce] folds over.  Naming it lets the
   innermost accumulation loop be stated -- and verified -- without any tensor
   in scope; see [Kuiops.SuperGEMM.Mm.SplitK.Epi.Reduce.acc_chunk]. *)
let ws_run
  (#et : Type0) {| scalar et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (di base e z : nat)
  : GTot et
= ws_cell rows splits eW di (base + e) z

(* [sum_upto] only sees its argument as a function value, so the SMT solver
   cannot relate [sum_upto (ws_run .. base y)] to
   [sum_upto (ws_cell .. (base + y))] -- the two partial applications are
   distinct opaque terms even though they agree at every point.  Pointwise
   agreement does transfer through the fold, by induction. *)
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
  (eW : chest2 et mws cols) (di base y t : nat)
  : Lemma (sum_upto (ws_run rows splits eW di base y) t
           == sum_upto (ws_cell rows splits eW di (base + y)) t)
= sum_upto_ext (ws_run rows splits eW di base y)
               (ws_cell rows splits eW di (base + y)) t

(* One [1 x c] granule of D approximates the matching tile of [gran_target] as
   soon as each of its cells is [comb] of the C cell and the fold of that
   output column over the splits. *)
let gran_approx
  (#et_c #et_acc #et_d : Type0)
  {| scalar et_c, real_like et_c |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_d, real_like et_d |}
  (#mws #cols : nat) (rows splits : nat)
  (eC : chest2 et_c rows cols) (rC : chest2 real rows cols)
  (eW : chest2 et_acc mws cols) (rW : chest2 real mws cols)
  (comb : et_c -> et_acc -> et_d)
  (comb_r : binop real { approx2 comb comb_r })
  (c : pos { c /? cols })
  (v : chest2 et_d 1 c)
  (di : natlt rows) (dj : natlt (cols / c))
  (_ : squash (1 /? rows))
  : Lemma
      (requires eW %~ rW /\ eC %~ rC /\
                (forall (y : natlt c).
                   dj * c + y < cols /\
                   acc2 v 0 y
                   == comb (acc2 eC di (dj * c + y))
                           (sum_upto (ws_run rows splits eW di (dj * c) y) splits)))
      (ensures v %~ ematrix_subtile (gran_target rows splits rC rW comb_r) 1 c di dj)
= introduce forall (a : natlt 1) (b : natlt c).
    acc2 v a b
    %~ acc2 (ematrix_subtile (gran_target rows splits rC rW comb_r) 1 c di dj) a b
  with begin
    ws_run_sum rows splits eW di (dj * c) b splits;
    reduce_ws_cell_approx rows splits eW rW di (dj * c + b)
  end;
  lemma_approximates_intro v
    (ematrix_subtile (gran_target rows splits rC rW comb_r) 1 c di dj)

(* [ws_run] is a partial application, so the SMT solver treats
   [sum_upto (ws_run .. base y) t] and [sum_upto (ws_cell .. (base+y)) t] as
   unrelated opaque terms even though they are pointwise equal.  This is the
   quantified form of [ws_run_sum], to be dropped into the Pulse context once
   so that both spellings are interchangeable. *)
let ws_run_sum_all
  (#et : Type0) {| scalar et |}
  (#mws #cols : nat) (rows splits : nat)
  (eW : chest2 et mws cols) (di base c t : nat)
  : Lemma (forall (y : natlt c).
             sum_upto (ws_run rows splits eW di base y) t
             == sum_upto (ws_cell rows splits eW di (base + y)) t)
= introduce forall (y : natlt c).
    sum_upto (ws_run rows splits eW di base y) t
    == sum_upto (ws_cell rows splits eW di (base + y)) t
  with ws_run_sum rows splits eW di base y t
