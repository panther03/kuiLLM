module Kuiops.Mean

(* torch.mean(input, dim=-1, keepdim=True) over a contiguous tensor, reducing
   the *last* axis. A rank-N contiguous tensor whose last axis is reduced is,
   by a zero-cost reshape, an [m, n] row-major matrix (m = product of the
   leading dims, n = the reduced length); its keepdim output is the [m, 1]
   column, i.e. an [m] vector. So the kernel is a thin wrapper over the
   parallel per-row tree reduction [Kuiper.Kernel.HReduce.Block]: one GPU block
   per output row cooperatively sums that row, with the mean's division by the
   row length folded into the reduction's pre-map. Reducing an arbitrary middle
   axis (which would need a strided view) is intentionally out of scope; the
   Python `supported()` gates `dim` to the last axis. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Seq.Common
open Kuiper.Real

module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
module Array2 = Kuiper.Array2
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }

(* Real-valued spec of one output cell: the mean of row [i] is the sum of its
   [n] cells, each divided by [divisor_r] (= the real value of [n]). Dividing
   per cell before summing equals (sum / n) in the reals, and the whole
   pipeline is only ever related up to the approximation relation [%~]. *)
let mean_row_real (#m #n : nat)
  (ra : ematrix real m n) (divisor_r : real { divisor_r =!= 0.0R }) (i : nat { i < m })
  : real
  = rsum (lseq_map (fun (z : real) -> z /. divisor_r) (ematrix_row ra i))

(* One block per row, tree-reducing that row's cells into `output[i]` and
   dividing by `divisor`. `divisor`/`divisor_r` are the reduced length as an
   element / real; the instantiation supplies them from the compile-time
   length. *)
inline_for_extraction noextract
fn mean_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (divisor : et)
  (divisor_r : real { divisor_r =!= 0.0R /\ v_approximates divisor divisor_r })
  (a : Array2.t et (l2_row_major (SZ.v m) (SZ.v n)) { Array2.is_global a })
  (output : Array1.t et (l1_forward (SZ.v m)) { Array1.is_global output })
  (#sa : ematrix et (SZ.v m) (SZ.v n))
  (ra : ematrix real (SZ.v m) (SZ.v n))
  (#sout : erased (lseq et (SZ.v m)))
  preserves
    cpu **
    on gpu_loc (a |-> sa)
  requires
    on gpu_loc (output |-> sout) **
    pure (sa %~ ra)
  ensures
    exists* (sout' : lseq et (SZ.v m)).
      on gpu_loc (output |-> sout') **
      pure (forall (i : nat). i < SZ.v m ==>
            (sout' @! i) %~ (mean_row_real ra divisor_r i))

(* Instantiation entry point: bakes the reduced length `len` (> 0) into the
   element and real divisors. The JIT instantiates `mean_inst #et lenL`. *)
inline_for_extraction noextract
fn mean_inst
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (len : Int64.t { Int64.v len > 0 })
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (a : Array2.t et (l2_row_major (SZ.v m) (SZ.v n)) { Array2.is_global a })
  (output : Array1.t et (l1_forward (SZ.v m)) { Array1.is_global output })
  (#sa : ematrix et (SZ.v m) (SZ.v n))
  (ra : ematrix real (SZ.v m) (SZ.v n))
  (#sout : erased (lseq et (SZ.v m)))
  preserves
    cpu **
    on gpu_loc (a |-> sa)
  requires
    on gpu_loc (output |-> sout) **
    pure (sa %~ ra)
  ensures
    exists* (sout' : lseq et (SZ.v m)).
      on gpu_loc (output |-> sout') **
      pure (forall (i : nat). i < SZ.v m ==>
            (sout' @! i) %~ (mean_row_real ra (FStar.Real.of_int (Int64.v len)) i))
