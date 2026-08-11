module Kuiops.SuperGEMM.Mm.SplitK.Compose

(* Composing the two passes: reducing the workspace pass 1 wrote reproduces the
   full k reduction, so the split is invisible in the top-level spec. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas
open Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas
open Kuiops.SuperGEMM.Mm.SplitK.WsLemmas { ws_target, ws_target_slab }

module MS = Kuiper.Spec.GEMM
module ML = FStar.Math.Lemmas

#set-options "--fuel 1 --ifuel 1 --z3rlimit 15"

let rec rsum_upto_ext (f g : nat -> GTot real) (t : nat)
  : Lemma (requires forall (z : nat). z < t ==> f z == g z)
          (ensures rsum_upto f t == rsum_upto g t)
          (decreases t)
= if t = 0 then () else rsum_upto_ext f g (t - 1)

#push-options "--z3rlimit 20 --split_queries always"
let reduce_ws_target
  (#rows #cols #shared : pos) (mws splits ks : pos)
  (rA : chest2 real rows shared) (rB : chest2 real cols shared)
  (sq : squash (shared == splits * ks /\ mws == splits * rows))
  : Lemma (reduce_ws rows splits (ws_target mws splits ks rA rB sq)
           == MS.matmul rA (mtranspose rB))
= ML.cancel_mul_div splits ks;
  ML.cancel_mul_div splits rows;
  let rW = ws_target mws splits ks rA rB sq in
  let rBt = mtranspose rB in
  introduce forall (i : natlt rows) (j : natlt cols).
      acc2 (reduce_ws rows splits rW) i j == acc2 (MS.matmul rA rBt) i j
  with begin
    introduce forall (z : nat). z < splits ==>
        rws_cell rows splits rW i j z
        == (if z < shared / ks then acc2 (split_partial ks rA rBt z) i j else 0.0R)
    with introduce _ ==> _
    with _. begin
      ws_row_bound rows splits mws z i;
      ws_target_slab mws splits ks rA rB sq z
    end;
    rsum_upto_ext (rws_cell rows splits rW i j)
      (fun z -> if z < shared / ks then acc2 (split_partial ks rA rBt z) i j else 0.0R)
      splits;
    split_sum_cell ks rA rBt splits i j;
    split_sum_full ks rA rBt
  end;
  Kuiper.Chest.ext (reduce_ws rows splits rW) (MS.matmul rA rBt)
#pop-options

let gran_target_matmul
  (#rows #cols #shared : pos) (mws splits ks : pos)
  (rA : chest2 real rows shared) (rB : chest2 real cols shared)
  (post_map_r : real -> real)
  (sq : squash (shared == splits * ks /\ mws == splits * rows))
  : Lemma (gran_target rows splits (ws_target mws splits ks rA rB sq) post_map_r
           == chest_map post_map_r (MS.matmul rA (mtranspose rB)))
= reduce_ws_target mws splits ks rA rB sq
