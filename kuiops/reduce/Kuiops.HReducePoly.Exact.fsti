module Kuiops.HReducePoly.Exact

(* Exact reduction over the innermost dimension of a tensor. Each outer
   (batch) index is reduced by one CUDA block. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Functions { is_associative }
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiops.HReducePoly.Spec
module SZ = Kuiper.SizeT

(* [index]/[index_up] build a flat index of the input shape from a batch index
   and a position along the reduced axis. [Kuiops.HReducePoly.Spec.conc_snoc]
   does exactly this, but recurses over an erased shape and so is not
   extractable; callers pass a monomorphic builder from
   [Kuiops.HReducePoly.Index] instead. *)

inline_for_extraction noextract
// `sized` needed because we allocate a shared memory array dynamically with element type `et`
type reduce_ty (et_i et et_o : Type0) {| sized et |} =
  fn (#r : erased nat)
     (#d : shape r)
     (cd : cshape d { batches_ok d })
     (f : (et -> et -> et) { is_associative f })
     (pre_map : et_i -> et)
     (post_map : et -> et_o)
     (cols : szp)
     (nth : szp {
       nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
     (index : conc d -> szlt cols -> conc (snoc_shape d cols))
     (index_up : (i:conc d -> j:szlt cols ->
       Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
     (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
     (#lout : tlayout d) {| ctlayout lout |}
     (a : tensor et_i lin { is_global a })
     (out : tensor et_o lout { is_global out })
     (s : stream_t)
     (#va : chest (snoc_shape d cols) et_i)
     (#vout : chest d et_o)
     (#e : epoch_t)
  preserves
    cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (a |-> va) ** on gpu_loc (out |-> vout)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (
        (a |-> va) **
        (out |-> mk d (fun i -> reduced f pre_map post_map va i))))

inline_for_extraction noextract
val reduce (#et_i #et #et_o : Type0) {| sized et |} : reduce_ty et_i et et_o
