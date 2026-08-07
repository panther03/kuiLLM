module Kuiops.Sdpa.Flash.Thread

(* Adapts [sdpa_flash_kf] to the per-thread pre/post conditions demanded by
   the kernel descriptor. *)

#lang-pulse

open Kuiper
open Kuiper.Floating
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiper.Bijection
open Kuiper.EMatrix
open Kuiper.ForEvery
open Kuiper.Shape
open Pulse.Lib.Pledge
open Kuiops.Sdpa.Flash.Types
open Kuiops.Sdpa.Flash.Barrier
open Kuiops.Sdpa.Flash.Kf

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module FD = Kuiops.Sdpa.Flash.KernelDesc
module Vals = Kuiops.Sdpa.Flash.Vals
module Trade = Pulse.Lib.Trade

inline_for_extraction noextract
fn sdpa_flash_thread
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |} {| floating_real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (nblk : szp)
  (nw nthr : szp {
    SZ.v nthr == block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| TRO.cvtlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (has_mask : bool) (scale : et_acc)
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (SZ.fits (16 * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (16 * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (#_ : squash (SZ.fits (SZ.v sk + 32)))
  (#_ : squash (SZ.fits (SZ.v sk + 32 + SZ.v nw)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
  requires
    gpu **
    flash_thread_pre nw nthr
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout (flash_views_of nw d sh)
      #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
      #(fK /. (SZ.v nblk) /. (SZ.v nthr))
      #(fV /. (SZ.v nblk) /. (SZ.v nthr))
      #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
      #eQ #eK #eV #emask
      (SZ.v bid <: natlt
        (SZ.v b * SZ.v hkv * SZ.v tiles))
      (SZ.v tid / BW.warp_size)
      (SZ.v tid % BW.warp_size) **
    thread_id (SZ.v nthr) tid **
    block_id (SZ.v nblk) bid **
    B.barrier_tok
      (barrier_contract nw d
        (flash_views_of nw d sh).shQv
        (flash_eQsh d b hq hkv group sq rows tiles eQ
          (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_views_of nw d sh).shMv
        (flash_views_of nw d sh).shLv
        (flash_eM nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
          (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_eL nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
          (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_views_of nw d sh).shscalev
        (flash_views_of nw d sh).shOv
        (flash_views_of nw d sh).shglv
        (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))) **
    B.barrier_state 0
  ensures
    gpu **
    flash_thread_post nw nthr
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout (flash_views_of nw d sh)
      #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
      #(fK /. (SZ.v nblk) /. (SZ.v nthr))
      #(fV /. (SZ.v nblk) /. (SZ.v nthr))
      #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
      #eQ #eK #eV #emask
      (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
      (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
      (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
      (SZ.v bid <: natlt
        (SZ.v b * SZ.v hkv * SZ.v tiles))
      (SZ.v tid / BW.warp_size)
      (SZ.v tid % BW.warp_size) **
    thread_id (SZ.v nthr) tid **
    block_id (SZ.v nblk) bid **
    B.barrier_tok
      (barrier_contract nw d
        (flash_views_of nw d sh).shQv
        (flash_eQsh d b hq hkv group sq rows tiles eQ
          (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_views_of nw d sh).shMv
        (flash_views_of nw d sh).shLv
        (flash_eM nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
          (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_eL nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
          (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_views_of nw d sh).shscalev
        (flash_views_of nw d sh).shOv
        (flash_views_of nw d sh).shglv
        (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))) **
    B.barrier_state 3
{
  let w = sdpa_flash_w nw nthr tid;
  let lane = sdpa_flash_lane nw nthr tid;
  let rt : szlt tiles = bid %^ tiles;
  let bh = bid /^ tiles;
  let kvh : szlt hkv = bh %^ hkv;
  let bi : szlt b = bh /^ hkv;
  let r0 = rt *^ 16sz;
  assert pure (
    SZ.v bi ==
      flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
        (SZ.v bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  assert pure (
    SZ.v kvh ==
      flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
        (SZ.v bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  assert pure (
    SZ.v rt ==
      flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles)
        (SZ.v bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  assert pure (SZ.fits (SZ.v r0 + 16));
  assert pure (
    SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows));

  unfold flash_thread_pre nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout (flash_views_of nw d sh)
    #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
    #(fK /. (SZ.v nblk) /. (SZ.v nthr))
    #(fV /. (SZ.v nblk) /. (SZ.v nthr))
    #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
    #eQ #eK #eV #emask
    (SZ.v bid <: natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles))
    (SZ.v tid / BW.warp_size)
    (SZ.v tid % BW.warp_size);
  assert pure (SZ.v w == SZ.v tid / BW.warp_size);
  assert pure (SZ.v lane == SZ.v tid % BW.warp_size);
  rewrite each (SZ.v tid / BW.warp_size) as (SZ.v w);
  rewrite each (SZ.v tid % BW.warp_size) as (SZ.v lane);
  FD.flash_b0_to_descriptor nw d (flash_views_of nw d sh)
    (SZ.v w) (SZ.v lane)
    (SZ.v tid <: natlt (block_threads nw));
  flash_ml_to_pre nw (flash_views_of nw d sh).shMv (flash_views_of nw d sh).shLv
    (SZ.v w) (SZ.v lane) w lane;
  flash_combine_to_pre nw
    (flash_views_of nw d sh).shscalev (flash_views_of nw d sh).shgmv (flash_views_of nw d sh).shglv
    (SZ.v w) (SZ.v lane) w lane;
  rewrite
    (when_ (SZ.v w = 0)
      (out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi
          (SZ.v b) (SZ.v hkv) (SZ.v tiles)
          (SZ.v bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_bid_kvh
          (SZ.v b) (SZ.v hkv) (SZ.v tiles)
          (SZ.v bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles)
          (SZ.v bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
        (SZ.v lane)))
    as
    (when_ (SZ.v w = 0)
      (out_store_cells b hq sq 16sz d rows gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group)
        (SZ.v r0) (SZ.v lane)));
  flash_output_to_pre b hq sq rows d gout
    (SZ.v bi)
    (SZ.v kvh) (SZ.v group) (SZ.v r0)
    (SZ.v w) (SZ.v lane) w lane;

  tensor_extract_slice_ro gK 0 (SZ.v bi);
  let gKb = sliceof gK 0 (SZ.v bi);
  rewrite each (sliceof gK 0 (SZ.v bi)) as gKb;
  tensor_extract_slice_ro gKb 0 (SZ.v kvh);
  let gKkv = sliceof gKb 0 (SZ.v kvh);
  rewrite each (sliceof gKb 0 (SZ.v kvh)) as gKkv;
  tensor_extract_slice_ro gV 0 (SZ.v bi);
  let gVb = sliceof gV 0 (SZ.v bi);
  rewrite each (sliceof gV 0 (SZ.v bi)) as gVb;
  tensor_extract_slice_ro gVb 0 (SZ.v kvh);
  let gVkv = sliceof gVb 0 (SZ.v kvh);
  rewrite each (sliceof gVb 0 (SZ.v kvh)) as gVkv;
  lem_is_global_iff_sliceof gK 0 (SZ.v bi);
  lem_is_global_iff_sliceof gKb 0 (SZ.v kvh);
  lem_is_global_iff_sliceof gV 0 (SZ.v bi);
  lem_is_global_iff_sliceof gVb 0 (SZ.v kvh);
  assert pure (Kuiper.Tensor.is_global gKkv);
  assert pure (Kuiper.Tensor.is_global gVkv);

  let eKkv : chest2 et_ab (SZ.v sk) (SZ.v d) =
    chest_slice 0 (SZ.v kvh)
      (chest_slice 0 (SZ.v bi) eK);
  let eVkv : chest2 et_ab (SZ.v sk) (SZ.v d) =
    chest_slice 0 (SZ.v kvh)
      (chest_slice 0 (SZ.v bi) eV);
  unfold flash_jt_local d
    (flash_warp_k (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_v (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_s (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_p (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_pv (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_cw (flash_views_of nw d sh) (SZ.v w))
    (SZ.v lane);
  fold sdpa_flash_jt_frame d sk b hq sq
    (flash_warp_k (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_v (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_s (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_p (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_pv (flash_views_of nw d sh) (SZ.v w))
    (flash_warp_cw (flash_views_of nw d sh) (SZ.v w))
    gKkv gVkv gmask
    #(fK /. (SZ.v nblk) /. (SZ.v nthr))
    #(fV /. (SZ.v nblk) /. (SZ.v nthr))
    #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
    #eKkv #eVkv #emask (SZ.v lane);
  rewrite each w as (sdpa_flash_w nw nthr tid);
  rewrite each lane as (sdpa_flash_lane nw nthr tid);
  let _ = ctlayout_slice
    (l2_row_major (SZ.v nw) 16)
    #(c_l2_row_major (SZ.v nw) 16sz)
    0 (SZ.v (sdpa_flash_w nw nthr tid));
  fold sdpa_flash_pre #et_ab #et_acc
    nw nthr d sk b hq sq rows
    #lgQ #_ #_ #lgmask #lout #_
    gQ gKkv gVkv gmask gout
    (flash_views_of nw d sh).shQv
    (flash_warp_k (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_v (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_s (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_p (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_pv (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_cw (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_views_of nw d sh).shMv (flash_views_of nw d sh).shLv
    (flash_eM nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
      (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_eL nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
      (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_views_of nw d sh).shscalev (flash_views_of nw d sh).shOv (flash_views_of nw d sh).shgmv (flash_views_of nw d sh).shglv
    (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    tid bi r0 group kvh
    #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
    #(fK /. (SZ.v nblk) /. (SZ.v nthr))
    #(fV /. (SZ.v nblk) /. (SZ.v nthr))
    #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
    #eQ #eKkv #eVkv #emask;
  sdpa_flash_kf nw nthr d sk b hq sq rows
    #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_
    #_ #_ #_ #_ #_
    #(ctlayout_slice
      (l2_row_major (SZ.v nw) 16)
      #(c_l2_row_major (SZ.v nw) 16sz)
      0 (SZ.v (sdpa_flash_w nw nthr tid)))
    #(c_subtile_layout
      (l2_row_major (SZ.v nw * 16) (SZ.v d))
      #(c_l2_row_major (SZ.v nw * 16) d)
      16 (SZ.v d) (SZ.v (sdpa_flash_w nw nthr tid)) 0)
    #(c_subtile_layout
      (l2_row_major (SZ.v nw * 16) (SZ.v d))
      #(c_l2_row_major (SZ.v nw * 16) d)
      16 (SZ.v d) (SZ.v (sdpa_flash_w nw nthr tid)) 0)
    #(c_subtile_layout
      (l2_row_major (SZ.v nw * 16) 16)
      #(c_l2_row_major (SZ.v nw * 16) 16sz)
      16 16 (SZ.v (sdpa_flash_w nw nthr tid)) 0)
    #(c_subtile_layout
      (l2_row_major (SZ.v nw * 16) 16)
      #(c_l2_row_major (SZ.v nw * 16) 16sz)
      16 16 (SZ.v (sdpa_flash_w nw nthr tid)) 0)
    #(c_subtile_layout
      (l2_row_major (SZ.v nw * 16) 16)
      #(c_l2_row_major (SZ.v nw * 16) 16sz)
      16 16 (SZ.v (sdpa_flash_w nw nthr tid)) 0)
    gQ gKkv gVkv gmask gout
    (flash_views_of nw d sh).shQv
    (flash_warp_k (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_v (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_s (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_p (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_pv (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_cw (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_views_of nw d sh).shMv (flash_views_of nw d sh).shLv (flash_views_of nw d sh).shscalev (flash_views_of nw d sh).shOv (flash_views_of nw d sh).shgmv (flash_views_of nw d sh).shglv
    tid bi r0 group kvh causal has_mask scale;

  unfold sdpa_flash_post #et_ab #et_acc
    nw nthr d sk b hq sq rows
    #lgQ #_ #_ #lgmask #lout #_
    gQ gKkv gVkv gmask gout
    (flash_views_of nw d sh).shQv
    (flash_warp_k (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_v (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_s (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_p (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_pv (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_cw (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_views_of nw d sh).shMv (flash_views_of nw d sh).shLv
    (flash_eM nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
      (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_eL nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
      (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_views_of nw d sh).shscalev (flash_views_of nw d sh).shOv (flash_views_of nw d sh).shglv
    (Vals.flash_escale_at nw d b hq sq rows sk eQ eKkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
    (Vals.flash_eO_at nw d b hq sq rows sk eQ eKkv eVkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
    (Vals.flash_egl_at nw d b hq sq rows sk eQ eKkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
    tid bi r0 group kvh
    #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
    #(fK /. (SZ.v nblk) /. (SZ.v nthr))
    #(fV /. (SZ.v nblk) /. (SZ.v nthr))
    #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
    #eQ #eKkv #eVkv #emask;
  unfold sdpa_flash_jt_frame d sk b hq sq
    (flash_warp_k (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_v (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_s (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_p (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_pv (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_cw (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    gKkv gVkv gmask
    #(fK /. (SZ.v nblk) /. (SZ.v nthr))
    #(fV /. (SZ.v nblk) /. (SZ.v nthr))
    #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
    #eKkv #eVkv #emask
    (SZ.v (sdpa_flash_lane nw nthr tid));
  fold flash_jt_local d
    (flash_warp_k (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_v (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_s (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_p (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_pv (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (flash_warp_cw (flash_views_of nw d sh) (SZ.v (sdpa_flash_w nw nthr tid)))
    (SZ.v (sdpa_flash_lane nw nthr tid));
  flash_if_when_reindex
    (fun (l : natlt BW.warp_size) ->
      out_store_cells_v nw b hq sq 16sz d rows gout
        (Vals.flash_escale_at nw d b hq sq rows sk eQ eKkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
        (Vals.flash_eO_at nw d b hq sq rows sk eQ eKkv eVkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
        (Vals.flash_egl_at nw d b hq sq rows sk eQ eKkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) l)
    (SZ.v (sdpa_flash_w nw nthr tid))
    (SZ.v (sdpa_flash_lane nw nthr tid))
    (sdpa_flash_w nw nthr tid)
    (sdpa_flash_lane nw nthr tid);
  flash_gm_from_post (flash_views_of nw d sh).shgmv
    (SZ.v (sdpa_flash_w nw nthr tid))
    (SZ.v (sdpa_flash_lane nw nthr tid))
    (sdpa_flash_w nw nthr tid)
    (sdpa_flash_lane nw nthr tid);

  rewrite
    (gKkv |-> Frac
      (fK /. (SZ.v nblk) /. (SZ.v nthr)) eKkv)
    as
    (gKkv |-> Frac
      (fK /. (SZ.v nblk) /. (SZ.v nthr))
      (chest_slice 0 (SZ.v kvh)
        (chest_slice 0 (SZ.v bi) eK)));
  rewrite
    (gVkv |-> Frac
      (fV /. (SZ.v nblk) /. (SZ.v nthr)) eVkv)
    as
    (gVkv |-> Frac
      (fV /. (SZ.v nblk) /. (SZ.v nthr))
      (chest_slice 0 (SZ.v kvh)
        (chest_slice 0 (SZ.v bi) eV)));
  rewrite each gKkv as (sliceof gKb 0 (SZ.v kvh));
  tensor_restore_slice gKb 0 (SZ.v kvh);
  rewrite each gKb as (sliceof gK 0 (SZ.v bi));
  tensor_restore_slice gK 0 (SZ.v bi);
  rewrite each gVkv as (sliceof gVb 0 (SZ.v kvh));
  tensor_restore_slice gVb 0 (SZ.v kvh);
  rewrite each gVb as (sliceof gV 0 (SZ.v bi));
  tensor_restore_slice gV 0 (SZ.v bi);

  rewrite
    (when_ (SZ.v (sdpa_flash_w nw nthr tid) = 0)
      (out_store_cells_v nw b hq sq 16sz d rows gout
        (Vals.flash_escale_at nw d b hq sq rows sk eQ eKkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
        (Vals.flash_eO_at nw d b hq sq rows sk eQ eKkv eVkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
        (Vals.flash_egl_at nw d b hq sq rows sk eQ eKkv emask has_mask causal scale (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0))
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0)
        (SZ.v (sdpa_flash_lane nw nthr tid))))
    as
    (when_ (SZ.v (sdpa_flash_w nw nthr tid) = 0)
      (out_store_cells_v nw b hq sq 16sz d rows gout
        (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_bid_bi
          (SZ.v b) (SZ.v hkv) (SZ.v tiles)
          (SZ.v bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_bid_kvh
          (SZ.v b) (SZ.v hkv) (SZ.v tiles)
          (SZ.v bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles)
          (SZ.v bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
        (SZ.v (sdpa_flash_lane nw nthr tid))));
  rewrite each (SZ.v (sdpa_flash_w nw nthr tid))
    as (SZ.v tid / BW.warp_size);
  rewrite each (SZ.v (sdpa_flash_lane nw nthr tid))
    as (SZ.v tid % BW.warp_size);
  fold flash_thread_post nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout (flash_views_of nw d sh)
    #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
    #(fK /. (SZ.v nblk) /. (SZ.v nthr))
    #(fV /. (SZ.v nblk) /. (SZ.v nthr))
    #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
    #eQ #eK #eV #emask
    (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (SZ.v bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (SZ.v bid <: natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles))
    (SZ.v tid / BW.warp_size)
    (SZ.v tid % BW.warp_size);
}
