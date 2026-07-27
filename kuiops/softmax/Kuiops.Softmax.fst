module Kuiops.Softmax
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }

module K = Kuiper.Kernel.RowSoftmax
module SZ = Kuiper.SizeT

(* RowSoftmax is a composite kernel: it launches several GPU kernels and syncs in
   between to observe their results, so it cannot take a stream / be graph-captured. *)
inline_for_extraction noextract
fn softmax_rm
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (nth : szp { nth <= max_threads })
  (a : array2 et (l2_row_major m n) { is_global a })
  (#sa : chest2 et m n)
  (ra : chest2 real m n)
  preserves cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  ensures
    exists* (sa' : chest2 et m n).
      on gpu_loc (a |-> sa') **
      pure (sa' %~ K.row_softmax_real #(SZ.v m) #(SZ.v n) ra)
{
  K.row_softmax_gpu #et m n nth a #sa ra;
}
