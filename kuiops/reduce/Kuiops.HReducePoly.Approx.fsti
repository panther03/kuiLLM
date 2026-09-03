module Kuiops.HReducePoly.Approx

(* Reduction over the innermost dimension of a tensor, specified over the
   reals. Unlike [Kuiops.HReducePoly.Exact], the concrete combiner is only
   required to *approximate* an associative real operation, so the (real)
   reduction order the kernel picks is unobservable in the specification.
   Each outer (batch) index is reduced by one CUDA block. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Math
open Kuiper.Functions
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiops.HReducePoly.Spec
open Kuiops.Maps { approx1 }

module SZ = Kuiper.SizeT

(* The output chest approximates the real reduction of [vr]. *)
let out_approx
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#rank : nat) (#d : shape rank) (#cols : pos)
  (f_r : real -> real -> real)
  (pre_map_r post_map_r : real -> real)
  (vr : chest (snoc_shape d cols) real)
  (vout' : chest d et_o)
  : prop
  = forall (i : abs d).
      acc vout' i %~ reduced f_r pre_map_r post_map_r vr i

(* [index]/[index_up] build a flat index of the input shape from a batch index
   and a position along the reduced axis. [Kuiops.HReducePoly.Spec.conc_snoc]
   does exactly this, but recurses over an erased shape and so is not
   extractable; callers pass a monomorphic builder from
   [Kuiops.HReducePoly.Index] instead. *)
inline_for_extraction noextract
fn reduce
  (#et_i : Type0) {| scalar et_i, real_like et_i |}
  (#et : Type0) {| scalar et, real_like et |}
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#r : erased nat)
  (#d : shape r)
  (cd : cshape d { batches_ok d })
  (f : et -> et -> et)
  (f_r : (real -> real -> real) { is_associative f_r /\ approx2 f f_r })
  (pre_map : et_i -> et)
  (pre_map_r : (real -> real) { approx1 pre_map pre_map_r })
  (post_map : et -> et_o)
  (post_map_r : (real -> real) { approx1 post_map post_map_r })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (index : conc d -> szlt cols -> conc (snoc_shape d cols))
  (index_up : (i:conc d -> j:szlt cols ->
    Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (input : tensor et_i lin { is_global input })
  (output : tensor et_o lout { is_global output })
  (s : stream_t)
  (#vin : chest (snoc_shape d cols) et_i)
  (#vr : chest (snoc_shape d cols) real)
  (#vout : chest d et_o)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires on gpu_loc (input |-> vin) ** on gpu_loc (output |-> vout)
  requires pure (vin %~ vr)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc (
        (input |-> vin) **
        (exists* (vout' : chest d et_o).
          (output |-> vout') **
          pure (out_approx f_r pre_map_r post_map_r vr vout'))))
