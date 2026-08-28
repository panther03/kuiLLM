module Kuiops.Sdpa.Flash.KfBlock

#lang-pulse

open Kuiper
open Kuiper.Floating
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiops.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float

inline_for_extraction noextract
fn sdpa_flash_q_load
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |}
  {| floating et_acc |}
  (bm d nthr : szp)
  (b hq sq : szp)
  (#lgQ : layout4 b hq sq d) {| ctlayout lgQ |}
  (#lm #ll : layout1 bm) (#lO : layout2 (SZ.v bm) (SZ.v d))
  {| ctlayout lm |} {| ctlayout ll |} {| ctlayout lO |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (shQ : array2 et_ab (l2_row_major bm d))
  (shm : array1 et_acc lm) (shl : array1 et_acc ll)
  (shO : array2 et_acc lO)
  (tid : szlt nthr) (lane : szlt warp_size)
  (bi : szlt b) (r0 : sz) (rows : szp) (group : szp) (kvh : sz)
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fQ : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  requires
    (gQ |-> Frac fQ eQ) **
    strided_cells2 shQ (SZ.v nthr) (SZ.v tid) **
    strided_cells2 shO BW.warp_size (SZ.v lane) **
    if_ (lane_active bm lane) (ml_cells bm shm shl lane)
  ensures
    (gQ |-> Frac fQ eQ) **
    strided_cells2_v shQ (SZ.v nthr) (SZ.v tid)
      (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group)
         eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) **
    strided_cells2_v shO BW.warp_size (SZ.v lane) (ozero (SZ.v bm) (SZ.v d)) **
    if_ (lane_active bm lane)
      (ml_cells_v bm shm shl lane (neg infinity) zero)

inline_for_extraction noextract
fn sdpa_flash_causal_mask
  (bm bn sk sq rows : szp)
  (r0 : sz) (causal : bool)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v sk + SZ.v bm + 1)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  returns nkt : sz
  ensures pure (SZ.v nkt <= SZ.v sk / SZ.v bn + 1 /\
                SZ.v nkt == SF.key_tiles (SZ.v bn) (SZ.v bm) (SZ.v sq) (SZ.v sk)
                              (SZ.v rows) (SZ.v r0) causal)

inline_for_extraction noextract
fn sdpa_flash_combine_partials
  (#et : Type0) {| floating et |}
  (nw bm : szp)
  (#lgm #lgl : layout1 bm) {| ctlayout lgm |} {| ctlayout lgl |}
  (shM shL shscale : array2 et (l2_row_major nw bm))
  (shgm : array1 et lgm) (shgl : array1 et lgl)
  (w : szlt nw) (lane : szlt warp_size)
  (#fM #fL : perm)
  (#eM #eL : chest2 et (SZ.v nw) (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  preserves
    (shM |-> Frac fM eM) ** (shL |-> Frac fL eL)
  requires
    if_ (combine_active bm w lane)
      (combine_cells nw bm shscale shgm shgl lane)
  ensures
    if_ (combine_active bm w lane)
      (combine_cells_v nw bm shscale shgm shgl lane
        (SF.gscale_col eM (SZ.v (clamp_lt bm lane)))
        (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
        (SF.gsum eM eL (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
           (SZ.v (clamp_lt bm lane)) (SZ.v nw)))

inline_for_extraction noextract
fn sdpa_flash_o_store
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nw bm d rows b hq sq : szp)
  (#lscale : layout2 nw bm) (#lgl : layout1 bm)
  (#lout : layout4 b hq sq d)
  {| ctlayout lscale |} {| ctlayout lgl |} {| ctlayout lout |}
  (shscale : array2 et_acc lscale)
  (shO : array2 et_acc (l2_row_major (SZ.v nw * SZ.v bm) d))
  (shgl : array1 et_acc lgl)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (bi : szlt b) (kvh : sz) (group : szp)
  (r0 : sz) (w : szlt nw) (lane : szlt warp_size)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fscale #fO #fgl : perm)
  (#escale : chest2 et_acc (SZ.v nw) (SZ.v bm))
  (#eO : chest2 et_acc (SZ.v nw * SZ.v bm) (SZ.v d))
  (#egl : chest1 et_acc (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d + BW.warp_size)))
  preserves
    (shscale |-> Frac fscale escale) **
    (shO |-> Frac fO eO) **
    (shgl |-> Frac fgl egl)
  requires if_ (w = 0sz)
    (out_store_cells b hq sq bm d rows gout
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane))
  ensures if_ (w = 0sz)
    (out_store_cells_v nw b hq sq bm d rows gout escale eO egl
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane))
