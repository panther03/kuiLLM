module Kuiops.Mean

(* Implementation of the verified last-axis mean kernel; see Kuiops.Mean.fsti
   for the spec. [mean_gpu] delegates to the parallel per-row tree reduction
   [Kuiper.Kernel.RowReduce.row_reduce] (one block per row), with
   the mean's division by the row length folded into the reduction's pre-map:
   summing (cell / divisor) over a row equals (row sum / divisor) in the reals,
   and the [div_approx] rule lifts the pre-map's approximation. *)

#lang-pulse
open Pulse.Lib.Pledge
open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common
open Kuiper.Real
open Kuiper.Tensor

module SZ = Kuiper.SizeT
module KR = Kuiper.Kernel.RowReduce
module R = FStar.Real
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }

inline_for_extraction noextract
fn mean_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (divisor : et)
  (divisor_r : real { divisor_r =!= 0.0R /\ v_approximates divisor divisor_r })
  (a : array2 et (l2_row_major (SZ.v m) (SZ.v n)) { is_global a })
  (output : array1 et (l1_forward (SZ.v m)) { is_global output })
  (#sa : chest2 et (SZ.v m) (SZ.v n))
  (ra : chest2 real (SZ.v m) (SZ.v n))
  (#sout : erased (chest1 et (SZ.v m)))
  (s : stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (a |-> sa) ** on gpu_loc (output |-> sout) **
    pure (sa %~ ra)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (exists* (sout' : chest1 et (SZ.v m)).
          (a |-> sa) ** (output |-> sout') **
          pure (forall (i : nat). i < SZ.v m ==>
                (acc1 sout' i) %~ (mean_row_real ra divisor_r i))))
{
  launch (KR.row_reduce_kd #et
            (fun (x : et) -> div x divisor)
            (fun (z : real) -> z /. divisor_r)
            m n max_threads a output sa ra sout) s;
}

(* Instantiation helper: bake the reduced length `len` (> 0) into the element
   divisor `of_int len` and the real divisor `R.of_int (Int64.v len)`. The
   element/real approximation comes from `of_int_approx`, and `R.of_int` of a
   positive int is nonzero (a Z3-real primitive fact), discharging the `/.`
   side-condition folded into the reduction's pre-map. *)
inline_for_extraction noextract
fn mean_inst
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (len : Int64.t { Int64.v len > 0 })
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (a : array2 et (l2_row_major (SZ.v m) (SZ.v n)) { is_global a })
  (output : array1 et (l1_forward (SZ.v m)) { is_global output })
  (#sa : chest2 et (SZ.v m) (SZ.v n))
  (ra : chest2 real (SZ.v m) (SZ.v n))
  (#sout : erased (chest1 et (SZ.v m)))
  (s : stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (a |-> sa) ** on gpu_loc (output |-> sout) **
    pure (sa %~ ra)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (exists* (sout' : chest1 et (SZ.v m)).
          (a |-> sa) ** (output |-> sout') **
          pure (forall (i : nat). i < SZ.v m ==>
                (acc1 sout' i) %~ (mean_row_real ra (R.of_int (Int64.v len)) i))))
{
  let divisor : et = Kuiper.Floating.Base.of_int len;
  of_int_approx #et len;
  mean_gpu #et m n divisor (R.of_int (Int64.v len)) a output ra s;
}
