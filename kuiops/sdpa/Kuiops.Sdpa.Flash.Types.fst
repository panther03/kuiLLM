module Kuiops.Sdpa.Flash.Types

#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Kernel.FlashAttention.KernelDesc

module SZ = Kuiper.SizeT

inline_for_extraction noextract
let flash_scale_cimap
  (nw bm : szp) (lane : szlt bm)
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (x : conc ((SZ.v nw / 1) @| (SZ.v bm / SZ.v bm) @| INil))
  : r:sz {
      SZ.v r ==
        (stride_subtile_layout
          (l2_row_major (SZ.v nw) (SZ.v bm))
          1 (SZ.v bm) 0 (SZ.v lane)).imap.f (up x) } =
  match x with
  | (i, (j, ())) ->
    assert (SZ.v j == 0);
    i *^ bm +^ j *^ bm +^ lane

noeq inline_for_extraction noextract
type flash_views
  (et_ab et_acc : Type0) (nw d : nat) =
{
  shQv : array2 et_ab (l2_row_major 16 d);
  shKv : array2 et_ab (l2_row_major (nw * 16) d);
  shVv : array2 et_ab (l2_row_major (nw * 16) d);
  shSv : array2 et_acc (l2_row_major (nw * 16) 16);
  shPv : array2 et_ab (l2_row_major (nw * 16) 16);
  shPVv : array2 et_acc (l2_row_major (nw * 16) 16);
  shcwv : array2 et_acc (l2_row_major nw 16);
  shMv : array2 et_acc (l2_row_major nw 16);
  shLv : array2 et_acc (l2_row_major nw 16);
  shscalev : array2 et_acc (l2_row_major nw 16);
  shOv : array2 et_acc (l2_row_major (nw * 16) d);
  shgmv : array1 et_acc (l1_forward 16);
  shglv : array1 et_acc (l1_forward 16);
}
