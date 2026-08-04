module Kuiops.Sdpa.Flash

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
open Kuiops.Sdpa.Flash.KfSub
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module FD = Kuiops.Sdpa.Flash.KernelDesc
module Trade = Pulse.Lib.Trade

unfold
let flash_scale_tile
  (#et : Type0) (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (lane : natlt 16) : slprop =
  exists* (e : chest2 et (SZ.v nw) 1).
    tensor_pts_to
      (array2_stride_subtile shscale 1 16 0 lane) e

ghost
fn flash_combine_to_b2_keep_gm
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgm shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  requires
    if_ (combine_active 16sz w lane)
      (combine_cells nw 16sz shscale shgm shgl lane)
  ensures
    if_ (combine_active 16sz w lane)
      (flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgl (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgm (SZ.v (clamp_lt 16sz lane)))
{
  let active = combine_active 16sz w lane;
  if active {
    if_elim_true (combine_cells nw 16sz shscale shgm shgl lane);
    unfold (combine_cells nw 16sz shscale shgm shgl lane);
    unfold (cell_full_n shgm (SZ.v (clamp_lt 16sz lane)));
    unfold (cell_full_n shgl (SZ.v (clamp_lt 16sz lane)));
    with vgm. assert (
      tensor_pts_to_cell shgm
        (idx1 (SZ.v (clamp_lt 16sz lane))) vgm);
    with vgl. assert (
      tensor_pts_to_cell shgl
        (idx1 (SZ.v (clamp_lt 16sz lane))) vgl);
    fold (cell_full shgm (SZ.v (clamp_lt 16sz lane)));
    fold (cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    fold (flash_scale_tile nw shscale
      (SZ.v (clamp_lt 16sz lane)));
    assert pure (combine_active 16sz w lane);
    if_intro_true' (combine_active 16sz w lane) (
      (flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgl (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgm (SZ.v (clamp_lt 16sz lane))));
  } else {
    if_elim_false (combine_cells nw 16sz shscale shgm shgl lane);
    assert pure (combine_active 16sz w lane == false);
    if_intro_false' (combine_active 16sz w lane) (
      (flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgl (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgm (SZ.v (clamp_lt 16sz lane))));
  }
}

ghost
fn flash_b2_scale_to_descriptor
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  (tid : szlt (FD.block_threads nw))
  requires
    pure (SZ.v w == FD.thread_w nw (SZ.v tid) /\
          SZ.v lane == FD.thread_lane nw (SZ.v tid)) **
    if_ (combine_active 16sz w lane)
      (flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgl (SZ.v (clamp_lt 16sz lane)))
  ensures
    FD.b2_scale_pre nw shscale shgl (SZ.v tid)
{
  let active = combine_active 16sz w lane;
  if active {
    if_elim_true' active (
      flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz lane)) **
      cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    unfold flash_scale_tile nw shscale
      (SZ.v (clamp_lt 16sz lane));
    let old_lane : natlt 16 = SZ.v (clamp_lt 16sz lane);
    let new_lane : natlt 16 =
      FD.clamp_nat_lt 16 (FD.thread_lane nw (SZ.v tid));
    assert pure (old_lane == new_lane);
    rewrite each (SZ.v (clamp_lt 16sz lane)) as old_lane;
    rewrite
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 old_lane) e)
      as
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 new_lane) e);
    rewrite
      (cell_full shgl old_lane)
      as
      (cell_full shgl new_lane);
    rewrite each new_lane
      as (FD.clamp_nat_lt 16
        (FD.thread_lane nw (SZ.v tid)));
    assert pure (FD.b2_active nw (SZ.v tid) == true);
    if_intro_true' (FD.b2_active nw (SZ.v tid)) (
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0
            (FD.clamp_nat_lt 16
              (FD.thread_lane nw (SZ.v tid)))) e)
      ** cell_full shgl
        (FD.clamp_nat_lt 16
          (FD.thread_lane nw (SZ.v tid))));
  } else {
    if_elim_false' active (
      flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz lane)) **
      cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    assert pure (FD.b2_active nw (SZ.v tid) == false);
    if_intro_false' (FD.b2_active nw (SZ.v tid)) (
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0
            (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane))));
    if_rewrite_bool false (FD.b2_active nw (SZ.v tid)) (
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0
            (FD.clamp_nat_lt 16 (FD.thread_lane nw (SZ.v tid))))
          e)
      ** cell_full shgl
        (FD.clamp_nat_lt 16 (FD.thread_lane nw (SZ.v tid))));
  }
}

inline_for_extraction noextract
fn sdpa_flash_kf
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (nw nthr d sk : szp { SZ.v nthr == FD.block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| ctlayout lgmask |} {| ctlayout lout |} {| ctlayout lcw |}
  {| ctlayout lK |} {| ctlayout lV |} {| ctlayout lS |}
  {| ctlayout lP |} {| ctlayout lPVc |}
  {| strided_row_major lK |} {| strided_row_major lV |}
  {| strided_row_major lS |} {| strided_row_major lP |}
  {| strided_row_major lPVc |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array2 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array2 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgm shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (bi : szlt b) (r0 : sz) (group : szp) (kvh : sz)
  (causal : bool) (scale : et_acc)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash ((Kuiper.Barrier.Warp.warp_size / 16) /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (16 /?+ (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (tlayout_ulen lcw)))
  (#_ : squash (SZ.fits (16 * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (16 * SZ.v d + Kuiper.Barrier.Warp.warp_size)))
  (#_ : squash (SZ.fits (SZ.v r0 + 16)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (#_ : squash (SZ.fits (SZ.v sk + 32)))
  (#_ : squash (SZ.fits (SZ.v sk + 32 + SZ.v nw)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  requires
    sdpa_flash_pre #et_ab #et_acc nw nthr d sk b hq sq rows
      #lgQ #lgK #lgV #lgmask #lout #lcw
      gQ gK gV gmask gout shQ shK shV shS shP shPVc shcw
      shM shL shscale shO shgm shgl tid bi r0 group kvh
      #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask
  ensures
    sdpa_flash_post #et_ab #et_acc nw nthr d sk b hq sq rows
      #lgQ #lgK #lgV #lgmask #lout #lcw
      gQ gK gV gmask gout shQ shK shV shS shP shPVc shcw
      shM shL shscale shO shgl tid bi r0 group kvh
      #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask **
    if_ (combine_active 16sz
      (sdpa_flash_w nw nthr tid) (sdpa_flash_lane nw nthr tid))
      (cell_full shgm
        (SZ.v (clamp_lt 16sz (sdpa_flash_lane nw nthr tid))))
{
  unfold sdpa_flash_pre #et_ab #et_acc nw nthr d sk b hq sq rows
    #lgQ #lgK #lgV #lgmask #lout #lcw
    gQ gK gV gmask gout shQ shK shV shS shP shPVc shcw
    shM shL shscale shO shgm shgl tid bi r0 group kvh
    #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask;
  rewrite each (sdpa_flash_w nw nthr tid) as (tid /^ 32sz);
  rewrite each (sdpa_flash_lane nw nthr tid) as (tid %^ 32sz);

  // TODO: remove me. just makes the file take forever to verify.
  admit ();

  sdpa_flash_block_prologue nw nthr d b hq sq rows
    gQ shQ shM shL shscale shO shgl tid bi r0 group kvh;

  with eQsh. assert (
    shQ |-> Frac (1.0R /. (SZ.v nthr)) eQsh);

  assert (jt_rest #et_ab #et_acc d sk b hq sq
    shK shV shS shP
    (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v (tid /^ 32sz)) 0)
    shQ shPVc shcw
    (FD.row shM (SZ.v (tid /^ 32sz))) (FD.row shL (SZ.v (tid /^ 32sz)))
    gK gV gmask
    #(1.0R /. (SZ.v nthr)) #fKg #fVg #fmask #eQsh #eKg #eVg #emask
    (SZ.v (tid %^ 32sz)));

  let w = sdpa_flash_w nw nthr tid;
  let lane = sdpa_flash_lane nw nthr tid;
  let irow : szlt 16 = clamp_lt 16sz lane;
  let r = r0 +^ irow;
  let rr : szlt rows = clamp_lt rows r;
  let qh0 = kvh *^ group +^ (rr /^ sq);
  let qh : szlt hq = clamp_lt hq qh0;
  let qpos : szlt sq = rr %^ sq;
  let row_active = r <^ rows;
  let cbound = qpos +^ (sk -^ sq);
  let nkt = sdpa_flash_causal_mask 16sz 16sz sk sq rows r0 causal;
  let mut jt : sz = w;
  let mut iter : sz = 0sz;
  while (!jt <^ nkt)
    invariant
      exists* (vjt : sz).
        jt |-> vjt **
        live iter **
        pure (SZ.v !iter <= SZ.v vjt) **
        gpu **
        thread_id (FD.block_threads nw) tid **
        B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
        B.barrier_state 1 **
        (gQ |-> Frac fQ eQ) **
        jt_rest #et_ab #et_acc d sk b hq sq
          shK shV shS shP
          (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v (tid /^ 32sz)) 0)
          shQ shPVc shcw
          (FD.row shM (SZ.v (tid /^ 32sz))) (FD.row shL (SZ.v (tid /^ 32sz)))
          gK gV gmask
          #(1.0R /. (SZ.v nthr)) #fKg #fVg #fmask #eQsh #eKg #eVg #emask
          (SZ.v (tid %^ 32sz)) **
        if_ (combine_active 16sz (tid /^ 32sz) (tid %^ 32sz))
          (combine_cells nw 16sz shscale shgm shgl (tid %^ 32sz)) **
        if_ (tid /^ 32sz = 0sz)
          (out_store_cells b hq sq 16sz d rows gout
            (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0)
            (SZ.v (tid %^ 32sz)))
    decreases (SZ.v nkt - SZ.v !iter)
  {
    let vjt = !jt;
    assert pure (SZ.v vjt < SZ.v nkt);
    assert pure (SZ.v vjt <= SZ.v sk / 16);
    let k0 = vjt *^ 16sz;
    assert pure (SZ.fits (SZ.v k0 + 16));
    sdpa_flash_jt_body d sk b hq sq
      #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_
      #_ #_ #_ #_ #_ #_ #_ #_
      #_ #_ #_ #_
      #(ctlayout_slice
        (l2_row_major (SZ.v nw) 16)
        #(c_l2_row_major (SZ.v nw) 16sz)
        0 (SZ.v (tid /^ 32sz)))
      #(ctlayout_slice
        (l2_row_major (SZ.v nw) 16)
        #(c_l2_row_major (SZ.v nw) 16sz)
        0 (SZ.v (tid /^ 32sz)))
      #_ #_ #_ #_ #_
      #(c_subtile_layout
        (l2_row_major (SZ.v nw * 16) d)
        #(c_l2_row_major (SZ.v nw * 16) d)
        16 (SZ.v d) (SZ.v (tid /^ 32sz)) 0)
      #_ #_ #_ #_ #_
      (tid %^ 32sz) nthr tid shK shV shS shP
      (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v (tid /^ 32sz)) 0)
      shQ shPVc shcw
      (FD.row shM (SZ.v (tid /^ 32sz))) (FD.row shL (SZ.v (tid /^ 32sz)))
      gK gV gmask bi qh qpos k0 cbound row_active causal scale;
    assert pure (SZ.fits (SZ.v !jt + SZ.v nw));
    let next = !jt +^ nw;
    assert pure (SZ.v next == SZ.v !jt + SZ.v nw);
    assert pure (SZ.v !jt < SZ.v next);
    assert pure (SZ.fits (SZ.v !iter + 1));
    (* [jt := next] must not be the final statement: karamel would turn the
       loop into a [for] whose increment reads [next], which is scoped to the
       body, and the emitted CUDA would not compile. *)
    jt := next;
    iter := !iter +^ 1sz;
  };

  unfold jt_rest #et_ab #et_acc d sk b hq sq
    shK shV shS shP
    (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v (tid /^ 32sz)) 0)
    shQ shPVc shcw
    (FD.row shM (SZ.v (tid /^ 32sz))) (FD.row shL (SZ.v (tid /^ 32sz)))
    gK gV gmask
    #(1.0R /. (SZ.v nthr)) #fKg #fVg #fmask #eQsh #eKg #eVg #emask
    (SZ.v (tid %^ 32sz));
  assert pure (FD.thread_w nw (SZ.v tid) == SZ.v (tid /^ 32sz));
  assert pure (FD.thread_lane nw (SZ.v tid) == SZ.v (tid %^ 32sz));
  block_row_cell_reindex shM
    (SZ.v (tid /^ 32sz) <: natlt (SZ.v nw))
    (FD.thread_w nw (SZ.v tid))
    (SZ.v (tid %^ 32sz) <: natlt BW.warp_size)
    (FD.thread_lane nw (SZ.v tid));
  block_row_cell_reindex shL
    (SZ.v (tid /^ 32sz) <: natlt (SZ.v nw))
    (FD.thread_w nw (SZ.v tid))
    (SZ.v (tid %^ 32sz) <: natlt BW.warp_size)
    (FD.thread_lane nw (SZ.v tid));
  fold FD.b1_pre nw shM shL (SZ.v tid);
  sdpa_flash_block_barrier1 nw nthr d
    shQ shM shL shscale shO shgl tid;

  sdpa_flash_combine_partials nw 16sz
    shM shL shscale shgm shgl (tid /^ 32sz) (tid %^ 32sz);

  flash_combine_to_b2_keep_gm nw shscale shgm shgl
    (tid /^ 32sz) (tid %^ 32sz);
  if combine_active 16sz (tid /^ 32sz) (tid %^ 32sz) {
    if_elim_true (
      flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
      cell_full shgl
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
      cell_full shgm
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    assert pure (
      combine_active 16sz (tid /^ 32sz) (tid %^ 32sz));
    if_intro_true (
      flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
      cell_full shgl
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    if_intro_true (
      cell_full shgm
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    if_rewrite_bool true
      (combine_active 16sz (tid /^ 32sz) (tid %^ 32sz))
      (flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
       cell_full shgl
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    if_rewrite_bool true
      (combine_active 16sz (tid /^ 32sz) (tid %^ 32sz))
      (cell_full shgm
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
  } else {
    if_elim_false (
      flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
      cell_full shgl
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
      cell_full shgm
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    assert pure (
      combine_active 16sz (tid /^ 32sz) (tid %^ 32sz) == false);
    if_intro_false (
      flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
      cell_full shgl
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    if_intro_false (
      cell_full shgm
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    if_rewrite_bool false
      (combine_active 16sz (tid /^ 32sz) (tid %^ 32sz))
      (flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))) **
       cell_full shgl
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
    if_rewrite_bool false
      (combine_active 16sz (tid /^ 32sz) (tid %^ 32sz))
      (cell_full shgm
        (SZ.v (clamp_lt 16sz (tid %^ 32sz))));
  };
  assert pure (FD.thread_w nw (SZ.v tid) == SZ.v (tid /^ 32sz));
  assert pure (FD.thread_lane nw (SZ.v tid) == SZ.v (tid %^ 32sz));
  flash_b2_scale_to_descriptor nw shscale shgl
    (tid /^ 32sz) (tid %^ 32sz)
    (tid <: szlt (FD.block_threads nw));
  block_o_tile_reindex nw d shO
    (SZ.v (tid /^ 32sz) <: natlt (SZ.v nw))
    (FD.thread_w nw (SZ.v tid))
    (SZ.v (tid %^ 32sz) <: natlt BW.warp_size)
    (FD.thread_lane nw (SZ.v tid));
  fold FD.b2_pre nw d shscale shO shgl (SZ.v tid);
  sdpa_flash_block_barrier2 nw nthr d
    shQ shM shL shscale shO shgl tid;

  assert pure (
    SZ.v (tid %^ 32sz) == SZ.v (sdpa_flash_lane nw nthr tid));
  stride_reindex shK
    (SZ.v (tid %^ 32sz)) (SZ.v (sdpa_flash_lane nw nthr tid));
  stride_reindex shV
    (SZ.v (tid %^ 32sz)) (SZ.v (sdpa_flash_lane nw nthr tid));
  row_reindex shP
    (SZ.v (tid %^ 32sz)) (SZ.v (sdpa_flash_lane nw nthr tid));
  cell_reindex shcw
    (SZ.v (tid %^ 32sz)) (SZ.v (sdpa_flash_lane nw nthr tid));
  rewrite
    (if_ (tid /^ 32sz = 0sz)
      (out_store_cells b hq sq 16sz d rows gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0)
        (SZ.v (tid %^ 32sz))))
    as
    (if_ (sdpa_flash_w nw nthr tid = 0sz)
      (out_store_cells b hq sq 16sz d rows gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0)
        (SZ.v (sdpa_flash_lane nw nthr tid))));
  sdpa_flash_o_store nw 16sz d rows b hq sq
    #_ #_ #_
    #(c_l2_row_major (SZ.v nw) 16sz)
    #(c_l1_forward 16)
    #_
    shscale shO shgl gout bi kvh group r0
      (sdpa_flash_w nw nthr tid) (sdpa_flash_lane nw nthr tid);
  fold sdpa_flash_post #et_ab #et_acc nw nthr d sk b hq sq rows
    #lgQ #lgK #lgV #lgmask #lout #lcw
    gQ gK gV gmask gout shQ shK shV shS shP shPVc shcw
    shM shL shscale shO shgl tid bi r0 group kvh
    #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask;
}

unfold let flash_nwarps : nat = 4

inline_for_extraction noextract
let flash_shmems
  (et_ab et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (nw d : szp)
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  : list shmem_desc =
  [ SHArray et_ab (16sz *^ d)
  ; SHArray et_ab (nw *^ 16sz *^ d)
  ; SHArray et_ab (nw *^ 16sz *^ d)
  ; SHArray et_acc (nw *^ 16sz *^ 16sz)
  ; SHArray et_ab (nw *^ 16sz *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz *^ d)
  ; SHArray et_acc 16sz
  ; SHArray et_acc 16sz
  ]

inline_for_extraction noextract
let flash_views_of
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (nw d : szp)
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  : flash_views et_ab et_acc (SZ.v nw) (SZ.v d) =
  let
    (q, (k, (v, (s, (p, (pv, (cw,
    (m, (l, (scale, (o, (gm, (gl, _))))))))))))) = sh in
  {
    shQv = from_array (l2_row_major 16 (SZ.v d)) q;
    shKv = from_array
      (l2_row_major (SZ.v nw * 16) (SZ.v d)) k;
    shVv = from_array
      (l2_row_major (SZ.v nw * 16) (SZ.v d)) v;
    shSv = from_array
      (l2_row_major (SZ.v nw * 16) 16) s;
    shPv = from_array
      (l2_row_major (SZ.v nw * 16) 16) p;
    shPVv = from_array
      (l2_row_major (SZ.v nw * 16) 16) pv;
    shcwv = from_array
      (l2_row_major (SZ.v nw) 16) cw;
    shMv = from_array (l2_row_major (SZ.v nw) 16) m;
    shLv = from_array (l2_row_major (SZ.v nw) 16) l;
    shscalev = from_array (l2_row_major (SZ.v nw) 16) scale;
    shOv = from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) o;
    shgmv = from_array (l1_forward 16) gm;
    shglv = from_array (l1_forward 16) gl;
  }

inline_for_extraction noextract
let flash_warp_k
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_ab
      (subtile_layout (l2_row_major (nw * 16) d) 16 (d <: pos) w 0) =
  array2_subtile v.shKv 16 (d <: pos) w 0

inline_for_extraction noextract
let flash_warp_v
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_ab
      (subtile_layout (l2_row_major (nw * 16) d) 16 (d <: pos) w 0) =
  array2_subtile v.shVv 16 (d <: pos) w 0

inline_for_extraction noextract
let flash_warp_s
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_acc
      (subtile_layout (l2_row_major (nw * 16) 16) 16 16 w 0) =
  array2_subtile v.shSv 16 16 w 0

inline_for_extraction noextract
let flash_warp_p
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_ab
      (subtile_layout (l2_row_major (nw * 16) 16) 16 16 w 0) =
  array2_subtile v.shPv 16 16 w 0

inline_for_extraction noextract
let flash_warp_pv
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_acc
      (subtile_layout (l2_row_major (nw * 16) 16) 16 16 w 0) =
  array2_subtile v.shPVv 16 16 w 0

inline_for_extraction noextract
let flash_warp_cw
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array1 et_acc
      (tlayout_slice (l2_row_major nw 16) 0 w) =
  FD.row v.shcwv w

ghost
fn flash_forevery4_intro
  (p : natlt flash_nwarps -> slprop)
  requires p 0 ** p 1 ** p 2 ** p 3
  ensures forall+ (w : natlt flash_nwarps). p w
{
  forevery_intro_false p;
  forevery_insert p 0;
  forevery_insert p 1;
  forevery_insert p 2;
  forevery_insert p 3;
  forevery_unrefine p;
}

ghost
fn flash_forevery4_elim
  (p : natlt flash_nwarps -> slprop)
  requires forall+ (w : natlt flash_nwarps). p w
  ensures p 0 ** p 1 ** p 2 ** p 3
{
  forevery_natlt_pop flash_nwarps p;
  forevery_natlt_pop 3
    (fun (w : natlt 3) -> p (natlt_coerce w));
  forevery_natlt_pop 2
    (fun (w : natlt 2) -> p (natlt_coerce w));
  forevery_natlt_pop 1
    (fun (w : natlt 1) -> p (natlt_coerce w));
  forevery_elim_empty
    (fun (w : natlt 0) -> p (natlt_coerce w));
}

let flash_stride_partition_bij
  (rows cols : nat) (nthr : pos)
  : ((tid : natlt nthr & FD.stride_index2 rows cols nthr tid)
      =~ (natlt rows & natlt cols)) =
{
  ff = (fun (x :
      (tid : natlt nthr & FD.stride_index2 rows cols nthr tid)) ->
    x._2)
    <: ((tid : natlt nthr & FD.stride_index2 rows cols nthr tid) ->
        GTot (natlt rows & natlt cols));
  gg = (fun (ij : natlt rows & natlt cols) ->
    (| ((ij._1 * cols + ij._2) % nthr), ij |))
    <: ((natlt rows & natlt cols) ->
        GTot (tid : natlt nthr &
          FD.stride_index2 rows cols nthr tid));
}

unfold
let flash_abs1_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
{
  ff = (fun (i, ()) -> i);
  gg = (fun i -> idx1 i);
}

ghost
fn flash_cells2_gather
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt rows) (j : natlt cols).
      exists* (v : et). Cell a (idx2 i j) |-> Frac 1.0R v)
  ensures live a
{
  let vf = forevery_exists_2
    (fun (i : natlt rows) (j : natlt cols) (v : et) ->
      Cell a (idx2 i j) |-> Frac 1.0R v);
  let e : chest2 et rows cols = mk2 vf;
  forevery_map_2
    (fun (i : natlt rows) (j : natlt cols) ->
      Cell a (idx2 i j) |-> Frac 1.0R (vf i j))
    (fun (i : natlt rows) (j : natlt cols) ->
      Cell a (idx2 i j) |-> Frac 1.0R (acc e (idx2 i j)))
    fn i j {
      rewrite
        (Cell a (idx2 i j) |-> Frac 1.0R (vf i j))
        as
        (Cell a (idx2 i j) |-> Frac 1.0R (acc e (idx2 i j)));
    };
  tensor_iraise2 a #1.0R #e;
  fold live a;
}

ghost
fn flash_cells1_gather
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt len).
      exists* (v : et). Cell a (idx1 i) |-> Frac 1.0R v)
  ensures live a
{
  let vf = forevery_exists
    (fun (i : natlt len) (v : et) ->
      Cell a (idx1 i) |-> Frac 1.0R v);
  let e : chest1 et len = mk1 vf;
  forevery_map #(natlt len)
    (fun i -> Cell a (idx1 i) |-> Frac 1.0R (vf i))
    (fun i -> Cell a (idx1 i) |-> Frac 1.0R (acc e (idx1 i)))
    fn i {
      rewrite
        (Cell a (idx1 i) |-> Frac 1.0R (vf i))
        as
        (Cell a (idx1 i) |-> Frac 1.0R (acc e (idx1 i)));
    };
  forevery_iso_back flash_abs1_bij
    (fun i -> Cell a i |-> Frac 1.0R (acc e i));
  tensor_implode a #1.0R #e;
  fold live a;
}

ghost
fn flash_strided_cells2_gather
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr). FD.strided_cells2 a nthr tid)
  ensures live a
{
  forevery_map #(natlt nthr)
    (fun tid -> FD.strided_cells2 a nthr tid)
    (fun tid ->
      forall+ (ij : FD.stride_index2 rows cols nthr tid).
        exists* (v : et).
          Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v)
    fn tid {
      unfold FD.strided_cells2 a nthr tid;
    };
  forevery_flatten_dep
    (fun (tid : natlt nthr)
      (ij : FD.stride_index2 rows cols nthr tid) ->
      exists* (v : et).
        Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v);
  forevery_iso (flash_stride_partition_bij rows cols nthr)
    (fun x ->
      exists* (v : et).
        Cell a (idx2 x._2._1 x._2._2) |-> Frac 1.0R v);
  forevery_map #(natlt rows & natlt cols)
    (fun ij ->
      exists* (v : et).
        Cell a
          (idx2
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._1
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._2)
          |-> Frac 1.0R v)
    (fun ij ->
      exists* (v : et).
        Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v)
    fn ij {
      with v. assert (
        tensor_pts_to_cell a #1.0R
          (idx2
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._1
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._2)
          v);
      rewrite
        (tensor_pts_to_cell a #1.0R
          (idx2
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._1
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._2)
          v)
        as
        (tensor_pts_to_cell a #1.0R (idx2 ij._1 ij._2) v);
    };
  forevery_unflatten'
    (fun (ij : natlt rows & natlt cols) ->
      exists* (v : et).
        Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v);
  flash_cells2_gather a;
}

ghost
fn flash_split_strided_cells2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires live a
  ensures
    forall+ (tid : natlt nthr).
      FD.strided_cells2 a nthr tid
{
  unfold live a;
  with e. assert (a |-> e);
  tensor_explode2 a;
  forevery_map #(natlt rows & natlt cols)
    (fun ij ->
      Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
        (acc e (idx2 ij._1 ij._2)))
    (fun ij ->
      Cell a
        (idx2
          ((flash_stride_partition_bij rows cols nthr).gg ij)._2._1
          ((flash_stride_partition_bij rows cols nthr).gg ij)._2._2)
        |-> Frac 1.0R
          (acc e
            (idx2
              ((flash_stride_partition_bij rows cols nthr).gg ij)._2._1
              ((flash_stride_partition_bij rows cols nthr).gg ij)._2._2)))
    fn ij {
      rewrite
        (Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
          (acc e (idx2 ij._1 ij._2)))
        as
        (Cell a
          (idx2
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._1
            ((flash_stride_partition_bij rows cols nthr).gg ij)._2._2)
          |-> Frac 1.0R
            (acc e
              (idx2
                ((flash_stride_partition_bij rows cols nthr).gg ij)._2._1
                ((flash_stride_partition_bij rows cols nthr).gg ij)._2._2)));
    };
  forevery_iso_back (flash_stride_partition_bij rows cols nthr)
    (fun (x :
      (tid : natlt nthr & FD.stride_index2 rows cols nthr tid)) ->
      Cell a (idx2 x._2._1 x._2._2) |-> Frac 1.0R
        (acc e (idx2 x._2._1 x._2._2)));
  forevery_unflatten_dep
    (fun (tid : natlt nthr)
      (ij : FD.stride_index2 rows cols nthr tid) ->
      Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
        (acc e (idx2 ij._1 ij._2)));
  forevery_map #(natlt nthr)
    (fun tid ->
      forall+ (ij : FD.stride_index2 rows cols nthr tid).
        Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
          (acc e (idx2 ij._1 ij._2)))
    (fun tid -> FD.strided_cells2 a nthr tid)
    fn tid {
      forevery_map
        (fun (ij : FD.stride_index2 rows cols nthr tid) ->
          Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
            (acc e (idx2 ij._1 ij._2)))
        (fun (ij : FD.stride_index2 rows cols nthr tid) ->
          exists* (v : et).
            Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v)
        fn ij { () };
      fold FD.strided_cells2 a nthr tid;
    };
}

ghost
fn flash_gather_strided_cells2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr).
      FD.strided_cells2 a nthr tid)
  ensures live a
{
  flash_strided_cells2_gather a nthr;
}

ghost
fn flash_warp_split_stride
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#_ : squash (SZ.v warp_size / 16 /? rows))
  (#_ : squash (16 /? cols))
  requires live a
  ensures
    forall+ (lane : natlt BW.warp_size).
      exists* (r : chest2 et (rows / (SZ.v warp_size / 16)) (cols / 16)).
        array2_stride_subtile a (SZ.v warp_size / 16) 16
          (lane / 16) (lane % 16) |-> Frac 1.0R r
{
  unfold live a;
  with e. assert (a |-> e);
  array2_stride_tile a (SZ.v warp_size / 16) 16;
  forevery_unfactor' BW.warp_size (SZ.v warp_size / 16) 16
    (fun (tr : natlt (SZ.v warp_size / 16)) (tc : natlt 16) ->
      array2_stride_subtile a (SZ.v warp_size / 16) 16 tr tc
        |-> Frac 1.0R
          (ematrix_stride_subtile e (SZ.v warp_size / 16) 16 tr tc));
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      array2_stride_subtile a (SZ.v warp_size / 16) 16
        (lane / 16) (lane % 16) |-> Frac 1.0R
          (ematrix_stride_subtile e (SZ.v warp_size / 16) 16
            (lane / 16) (lane % 16)))
    (fun lane ->
      exists* (r : chest2 et
        (rows / (SZ.v warp_size / 16)) (cols / 16)).
        array2_stride_subtile a (SZ.v warp_size / 16) 16
          (lane / 16) (lane % 16) |-> Frac 1.0R r)
    fn lane { () };
}

ghost
fn flash_warp_gather_stride
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#_ : squash (SZ.v warp_size / 16 /? rows))
  (#_ : squash (16 /? cols))
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (lane : natlt BW.warp_size).
      exists* (r : chest2 et (rows / (SZ.v warp_size / 16)) (cols / 16)).
        array2_stride_subtile a (SZ.v warp_size / 16) 16
          (lane / 16) (lane % 16) |-> Frac 1.0R r)
  ensures live a
{
  let rf = forevery_exists
    (fun (lane : natlt BW.warp_size)
      (r : chest2 et (rows / (SZ.v warp_size / 16)) (cols / 16)) ->
      array2_stride_subtile a (SZ.v warp_size / 16) 16
        (lane / 16) (lane % 16) |-> Frac 1.0R r);
  forevery_ext #(natlt BW.warp_size)
    (fun lane ->
      array2_stride_subtile a (SZ.v warp_size / 16) 16
        (lane / 16) (lane % 16) |-> Frac 1.0R (rf lane))
    (fun lane ->
      array2_stride_subtile a (SZ.v warp_size / 16) 16
        (lane / 16) (lane % 16) |-> Frac 1.0R
          (rf ((lane / 16) * 16 + (lane % 16))));
  forevery_factor' BW.warp_size (SZ.v warp_size / 16) 16
    (fun (tr : natlt (SZ.v warp_size / 16)) (tc : natlt 16) ->
      array2_stride_subtile a (SZ.v warp_size / 16) 16 tr tc
        |-> Frac 1.0R (rf (tr * 16 + tc)));
  array2_stride_untile' a (SZ.v warp_size / 16) 16
    (fun tr tc -> rf (tr * 16 + tc));
  fold live a;
}

ghost
fn flash_split_rows16
  (#et : Type0) (#l : layout2 16 16)
  (#_ : squash (SZ.fits (tlayout_ulen l)))
  (a : array2 et l)
  requires live a
  ensures
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> row_subtile a lane)
{
  unfold live a;
  with e. assert (a |-> e);
  array2_tile a 1 16;
  forevery_rw_size2 16 16 1 1
    #(fun (tr : natlt 16) (tc : natlt 1) ->
      array2_subtile a 1 16 tr tc |-> Frac 1.0R
        (ematrix_subtile e 1 16 tr tc));
  forevery_map #(natlt 16)
    (fun tr ->
      forall+ (tc : natlt 1).
        array2_subtile a 1 16 tr tc |-> Frac 1.0R
          (ematrix_subtile e 1 16 tr tc))
    (fun tr -> row_subtile a tr)
    fn tr {
      forevery_singleton_elim
        (fun (tc : natlt 1) ->
          array2_subtile a 1 16 tr tc |-> Frac 1.0R
            (ematrix_subtile e 1 16 tr tc));
      fold row_subtile a tr;
    };
  forevery_natlt_extend BW.warp_size
    (fun (lane : natlt 16) -> row_subtile a lane);
  forevery_unrefine_pred'
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    (fun lane _ -> row_subtile a (lane <: natlt 16));
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      when__ (lane < 16)
        (fun _ -> row_subtile a (lane <: natlt 16)))
    (fun lane ->
      when__ (lane < 16) (fun _ -> row_subtile a lane))
    fn lane { () };
}

ghost
fn flash_gather_rows16
  (#et : Type0) (#l : layout2 16 16)
  (#_ : squash (SZ.fits (tlayout_ulen l)))
  (a : array2 et l)
  requires
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> row_subtile a lane)
  ensures live a
{
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      when__ (lane < 16) (fun _ -> row_subtile a lane))
    (fun lane ->
      when_ (lane < 16)
        (row_subtile a (FD.clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> row_subtile a lane))
        as
        (when_ (lane < 16)
          (row_subtile a (FD.clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      row_subtile a (FD.clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> row_subtile a (FD.clamp_nat_lt 16 lane))
    (fun lane -> row_subtile a (lane <: natlt 16))
    fn lane {
      rewrite
        (row_subtile a (FD.clamp_nat_lt 16 lane))
        as
        (row_subtile a (lane <: natlt 16));
    };
  forevery_natlt_restrict BW.warp_size
    (fun (lane : natlt 16) -> row_subtile a lane);
  let rf = forevery_exists
    (fun (tr : natlt 16) (r : chest2 et 1 16) ->
      array2_subtile a 1 16 tr 0 |-> Frac 1.0R r);
  forevery_map #(natlt 16)
    (fun tr ->
      array2_subtile a 1 16 tr 0 |-> Frac 1.0R (rf tr))
    (fun tr ->
      forall+ (tc : natlt 1).
        array2_subtile a 1 16 tr tc |-> Frac 1.0R (rf tr))
    fn tr {
      forevery_singleton_intro
        (fun (tc : natlt 1) ->
          array2_subtile a 1 16 tr tc |-> Frac 1.0R (rf tr));
    };
  forevery_rw_size2 16 16 1 1
    #(fun (tr : natlt 16) (tc : natlt 1) ->
      array2_subtile a 1 16 tr tc |-> Frac 1.0R (rf tr));
  array2_untile' a 1 16 (fun tr _ -> rf tr);
  fold live a;
}

ghost
fn flash_split_cells16
  (#et : Type0)
  (a : array1 et (l1_forward 16))
  requires live a
  ensures
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> cell_full a lane)
{
  unfold live a;
  with e. assert (a |-> e);
  tensor_explode a;
  forevery_iso flash_abs1_bij
    (fun i -> Cell a i |-> Frac 1.0R (acc e i));
  forevery_map #(natlt 16)
    (fun i ->
      Cell a (idx1 i) |-> Frac 1.0R (acc e (idx1 i)))
    (fun i -> cell_full a i)
    fn i {
      fold cell_full a i;
    };
  forevery_natlt_extend BW.warp_size
    (fun (lane : natlt 16) -> cell_full a lane);
  forevery_unrefine_pred'
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    (fun lane _ -> cell_full a (lane <: natlt 16));
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      when__ (lane < 16)
        (fun _ -> cell_full a (lane <: natlt 16)))
    (fun lane ->
      when__ (lane < 16) (fun _ -> cell_full a lane))
    fn lane { () };
}

ghost
fn flash_gather_cells16
  (#et : Type0)
  (a : array1 et (l1_forward 16))
  requires
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> cell_full a lane)
  ensures live a
{
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      when__ (lane < 16) (fun _ -> cell_full a lane))
    (fun lane ->
      when_ (lane < 16)
        (cell_full a (FD.clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> cell_full a lane))
        as
        (when_ (lane < 16)
          (cell_full a (FD.clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      cell_full a (FD.clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> cell_full a (FD.clamp_nat_lt 16 lane))
    (fun lane -> cell_full a (lane <: natlt 16))
    fn lane {
      rewrite
        (cell_full a (FD.clamp_nat_lt 16 lane))
        as
        (cell_full a (lane <: natlt 16));
    };
  forevery_natlt_restrict BW.warp_size
    (fun (lane : natlt 16) -> cell_full a lane);
  forevery_map #(natlt 16)
    (fun i -> cell_full a i)
    (fun i ->
      exists* (v : et). Cell a (idx1 i) |-> Frac 1.0R v)
    fn i {
      unfold cell_full a i;
    };
  flash_cells1_gather a;
}

ghost
fn flash_share_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  requires live a
  ensures
    forall+ (_ : natlt k).
      exists* (e : chest d et). a |-> Frac (1.0R /. k) e
{
  unfold live a;
  with e. assert (a |-> e);
  tensor_share_n a k;
  forevery_map #(natlt k)
    (fun _ -> a |-> Frac (1.0R /. k) e)
    (fun _ ->
      exists* (x : chest d et). a |-> Frac (1.0R /. k) x)
    fn _ { () };
}

ghost
fn flash_gather_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  requires
    forall+ (_ : natlt k).
      exists* (e : chest d et). a |-> Frac (1.0R /. k) e
  ensures live a
{
  let ef = forevery_exists
    (fun (_ : natlt k) (e : chest d et) ->
      a |-> Frac (1.0R /. k) e);
  forevery_map #(natlt k)
    (fun i -> a |-> Frac (1.0R /. k) (ef i))
    (fun _ ->
      exists* (s : chest d et).
        tensor_pts_to a #(1.0R /. k) s)
    fn _ { () };
  tensor_gather_n_underspec a k;
  fold live a;
}

ghost
fn flash_split_ml
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (a : array2 et (l2_row_major (SZ.v nw) 16))
  requires live a
  ensures
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ ->
        cell_full (FD.row a w) lane)
{
  unfold live a;
  with e. assert (a |-> e);
  tensor_ilower2 a;
  forevery_map_2
    (fun (w : natlt (SZ.v nw)) (lane : natlt 16) ->
      Cell a (idx2 w lane) |-> Frac 1.0R (acc e (idx2 w lane)))
    (fun (w : natlt (SZ.v nw)) (lane : natlt 16) ->
      cell_full (FD.row a w) lane)
    fn w lane {
      rewrite each (FD.row a w) as (sliceof a 0 w);
      tensor_slice_cell_eq a 0 w (idx1 lane) 1.0R
        (acc e (idx2 w lane));
      rewrite
        (Cell a (idx2 w lane) |-> Frac 1.0R
          (acc e (idx2 w lane)))
        as
        (Cell (sliceof a 0 w) (idx1 lane) |-> Frac 1.0R
          (acc e (idx2 w lane)));
      fold cell_full (sliceof a 0 w) lane;
    };
  forevery_map #(natlt (SZ.v nw))
    (fun w ->
      forall+ (lane : natlt 16).
        cell_full (FD.row a w) lane)
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          cell_full (FD.row a w) lane))
    fn w {
      forevery_natlt_extend BW.warp_size
        (fun (lane : natlt 16) ->
          cell_full (FD.row a w) lane);
      forevery_unrefine_pred'
        (fun (lane : natlt BW.warp_size) -> lane < 16)
        (fun lane _ ->
          cell_full (FD.row a w) (lane <: natlt 16));
      forevery_map #(natlt BW.warp_size)
        (fun lane ->
          when__ (lane < 16) (fun _ ->
            cell_full (FD.row a w) (lane <: natlt 16)))
        (fun lane ->
          when__ (lane < 16) (fun _ ->
            cell_full (FD.row a w) lane))
        fn lane { () };
    };
}

ghost
fn flash_gather_ml
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (a : array2 et (l2_row_major (SZ.v nw) 16))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ ->
        cell_full (FD.row a w) lane)
  ensures live a
{
  forevery_map #(natlt (SZ.v nw))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          cell_full (FD.row a w) lane))
    (fun w ->
      forall+ (lane : natlt 16).
        exists* (v : et).
          Cell a (idx2 w lane) |-> Frac 1.0R v)
    fn w {
      forevery_map #(natlt BW.warp_size)
        (fun lane ->
          when__ (lane < 16) (fun _ ->
            cell_full (FD.row a w) lane))
        (fun lane ->
          when_ (lane < 16)
            (cell_full (FD.row a w) (FD.clamp_nat_lt 16 lane)))
        fn lane {
          rewrite
            (when__ (lane < 16) (fun _ ->
              cell_full (FD.row a w) lane))
            as
            (when_ (lane < 16)
              (cell_full (FD.row a w)
                (FD.clamp_nat_lt 16 lane)));
        };
      forevery_refine_pred
        (fun (lane : natlt BW.warp_size) ->
          cell_full (FD.row a w) (FD.clamp_nat_lt 16 lane))
        (fun lane -> lane < 16);
      forevery_map #(lane : natlt BW.warp_size { lane < 16 })
        (fun lane ->
          cell_full (FD.row a w) (FD.clamp_nat_lt 16 lane))
        (fun lane ->
          cell_full (FD.row a w) (lane <: natlt 16))
        fn lane {
          rewrite
            (cell_full (FD.row a w) (FD.clamp_nat_lt 16 lane))
            as
            (cell_full (FD.row a w) (lane <: natlt 16));
        };
      forevery_natlt_restrict BW.warp_size
        (fun (lane : natlt 16) ->
          cell_full (FD.row a w) lane);
      forevery_map #(natlt 16)
        (fun lane ->
          cell_full (FD.row a w) lane)
        (fun lane ->
          exists* (v : et).
            Cell a (idx2 w lane) |-> Frac 1.0R v)
        fn lane {
          unfold cell_full (FD.row a w) lane;
          with v. assert (
            tensor_pts_to_cell (FD.row a w) #1.0R
              (idx1 lane) v);
          rewrite each (FD.row a w) as (sliceof a 0 w);
          tensor_slice_cell_eq a 0 w (idx1 lane) 1.0R v;
          rewrite
            (tensor_pts_to_cell (sliceof a 0 w) #1.0R
              (idx1 lane) v)
            as
            (tensor_pts_to_cell a #1.0R (idx2 w lane) v);
        };
    };
  flash_cells2_gather a;
}

unfold
let flash_combine_local
  (#et : Type0) (nw : szp)
  (vscale : array2 et (l2_row_major (SZ.v nw) 16))
  (vgm vgl : array1 et (l1_forward 16))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size)
  : slprop =
  when_ (w = 0 /\ lane < 16)
    ((exists* (e : chest2 et (SZ.v nw) 1).
       tensor_pts_to
         (array2_stride_subtile vscale 1 16 0
           (FD.clamp_nat_lt 16 lane)) e)
     ** cell_full vgm (FD.clamp_nat_lt 16 lane)
     ** cell_full vgl (FD.clamp_nat_lt 16 lane))

let flash_combine_idx (nw : szp) : Type0 =
  wl : (natlt (SZ.v nw) & natlt BW.warp_size) {
    wl._1 = 0 /\ wl._2 < 16}

let flash_combine_bij (nw : szp)
  : (flash_combine_idx nw =~ natlt 16) =
{
  ff = (fun (wl : flash_combine_idx nw) -> wl._2 <: natlt 16)
    <: (flash_combine_idx nw -> GTot (natlt 16));
  gg = (fun (lane : natlt 16) -> (0, lane))
    <: (natlt 16 -> GTot (flash_combine_idx nw));
}

ghost
fn flash_split_combine
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (vscale : array2 et (l2_row_major (SZ.v nw) 16))
  (vgm vgl : array1 et (l1_forward 16))
  requires live vscale ** live vgm ** live vgl
  ensures
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      flash_combine_local nw vscale vgm vgl w lane
{
  unfold live vscale;
  with escale. assert (vscale |-> escale);
  array2_stride_tile vscale 1 16;
  forevery_singleton_elim
    (fun (tr : natlt 1) ->
      forall+ (lane : natlt 16).
        array2_stride_subtile vscale 1 16 tr lane
          |-> Frac 1.0R
            (ematrix_stride_subtile escale 1 16 tr lane));
  forevery_map #(natlt 16)
    (fun lane ->
      array2_stride_subtile vscale 1 16 0 lane |-> Frac 1.0R
        (ematrix_stride_subtile escale 1 16 0 lane))
    (fun lane ->
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0 lane) e)
    fn lane { () };
  flash_split_cells16 vgm;
  flash_split_cells16 vgl;
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      when__ (lane < 16) (fun _ -> cell_full vgm lane))
    (fun lane ->
      when_ (lane < 16)
        (cell_full vgm (FD.clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> cell_full vgm lane))
        as
        (when_ (lane < 16)
          (cell_full vgm (FD.clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      cell_full vgm (FD.clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> cell_full vgm (FD.clamp_nat_lt 16 lane))
    (fun lane -> cell_full vgm (lane <: natlt 16))
    fn lane {
      rewrite
        (cell_full vgm (FD.clamp_nat_lt 16 lane))
        as
        (cell_full vgm (lane <: natlt 16));
    };
  forevery_natlt_restrict BW.warp_size
    (fun (lane : natlt 16) -> cell_full vgm lane);
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      when__ (lane < 16) (fun _ -> cell_full vgl lane))
    (fun lane ->
      when_ (lane < 16)
        (cell_full vgl (FD.clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> cell_full vgl lane))
        as
        (when_ (lane < 16)
          (cell_full vgl (FD.clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      cell_full vgl (FD.clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> cell_full vgl (FD.clamp_nat_lt 16 lane))
    (fun lane -> cell_full vgl (lane <: natlt 16))
    fn lane {
      rewrite
        (cell_full vgl (FD.clamp_nat_lt 16 lane))
        as
        (cell_full vgl (lane <: natlt 16));
    };
  forevery_natlt_restrict BW.warp_size
    (fun (lane : natlt 16) -> cell_full vgl lane);
  forevery_zip3
    (fun (lane : natlt 16) ->
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0 lane) e)
    (fun (lane : natlt 16) -> cell_full vgm lane)
    (fun (lane : natlt 16) -> cell_full vgl lane);
  forevery_map #(natlt 16)
    (fun lane ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0 lane) e)
      ** cell_full vgm lane
      ** cell_full vgl lane)
    (fun lane ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            ((flash_combine_bij nw).gg lane)._2) e)
      ** cell_full vgm ((flash_combine_bij nw).gg lane)._2
      ** cell_full vgl ((flash_combine_bij nw).gg lane)._2)
    fn lane {
      assert pure (((flash_combine_bij nw).gg lane)._2 == lane);
      rewrite each lane as ((flash_combine_bij nw).gg lane)._2;
    };
  forevery_iso_back (flash_combine_bij nw)
    (fun (wl : flash_combine_idx nw) ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (wl._2 <: natlt 16)) e)
      ** cell_full vgm (wl._2 <: natlt 16)
      ** cell_full vgl (wl._2 <: natlt 16));
  forevery_map #(flash_combine_idx nw)
    (fun wl ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (wl._2 <: natlt 16)) e)
      ** cell_full vgm (wl._2 <: natlt 16)
      ** cell_full vgl (wl._2 <: natlt 16))
    (fun wl ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (FD.clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (FD.clamp_nat_lt 16 wl._2)
      ** cell_full vgl (FD.clamp_nat_lt 16 wl._2))
    fn wl {
      rewrite each (wl._2 <: natlt 16)
        as (FD.clamp_nat_lt 16 wl._2);
    };
  forevery_unrefine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (FD.clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (FD.clamp_nat_lt 16 wl._2)
      ** cell_full vgl (FD.clamp_nat_lt 16 wl._2))
    (fun wl -> wl._1 = 0 /\ wl._2 < 16);
  forevery_map #(natlt (SZ.v nw) & natlt BW.warp_size)
    (fun wl ->
      when_ (wl._1 = 0 /\ wl._2 < 16)
        ((exists* (e : chest2 et (SZ.v nw) 1).
           tensor_pts_to
             (array2_stride_subtile vscale 1 16 0
               (FD.clamp_nat_lt 16 wl._2)) e)
         ** cell_full vgm (FD.clamp_nat_lt 16 wl._2)
         ** cell_full vgl (FD.clamp_nat_lt 16 wl._2)))
    (fun wl ->
      flash_combine_local nw vscale vgm vgl wl._1 wl._2)
    fn wl {
      rewrite
        (when_ (wl._1 = 0 /\ wl._2 < 16)
          ((exists* (e : chest2 et (SZ.v nw) 1).
             tensor_pts_to
               (array2_stride_subtile vscale 1 16 0
                  (FD.clamp_nat_lt 16 wl._2)) e)
           ** cell_full vgm (FD.clamp_nat_lt 16 wl._2)
           ** cell_full vgl (FD.clamp_nat_lt 16 wl._2)))
        as
        (flash_combine_local nw vscale vgm vgl wl._1 wl._2);
    };
  forevery_unflatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      flash_combine_local nw vscale vgm vgl w lane);
}

ghost
fn flash_gather_combine
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (vscale : array2 et (l2_row_major (SZ.v nw) 16))
  (vgm vgl : array1 et (l1_forward 16))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      flash_combine_local nw vscale vgm vgl w lane
  ensures live vscale ** live vgm ** live vgl
{
  forevery_flatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      flash_combine_local nw vscale vgm vgl w lane);
  forevery_map #(natlt (SZ.v nw) & natlt BW.warp_size)
    (fun wl ->
      flash_combine_local nw vscale vgm vgl wl._1 wl._2)
    (fun wl ->
      when_ (wl._1 = 0 /\ wl._2 < 16)
        ((exists* (e : chest2 et (SZ.v nw) 1).
           tensor_pts_to
             (array2_stride_subtile vscale 1 16 0
               (FD.clamp_nat_lt 16 wl._2)) e)
         ** cell_full vgm (FD.clamp_nat_lt 16 wl._2)
         ** cell_full vgl (FD.clamp_nat_lt 16 wl._2)))
    fn wl {
      unfold flash_combine_local nw vscale vgm vgl wl._1 wl._2;
    };
  forevery_refine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (FD.clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (FD.clamp_nat_lt 16 wl._2)
      ** cell_full vgl (FD.clamp_nat_lt 16 wl._2))
    (fun wl -> wl._1 = 0 /\ wl._2 < 16);
  forevery_map #(flash_combine_idx nw)
    (fun wl ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (FD.clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (FD.clamp_nat_lt 16 wl._2)
      ** cell_full vgl (FD.clamp_nat_lt 16 wl._2))
    (fun wl ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (wl._2 <: natlt 16)) e)
      ** cell_full vgm (wl._2 <: natlt 16)
      ** cell_full vgl (wl._2 <: natlt 16))
    fn wl {
      rewrite each (FD.clamp_nat_lt 16 wl._2)
        as (wl._2 <: natlt 16);
    };
  forevery_iso (flash_combine_bij nw)
    (fun (wl : flash_combine_idx nw) ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (wl._2 <: natlt 16)) e)
      ** cell_full vgm (wl._2 <: natlt 16)
      ** cell_full vgl (wl._2 <: natlt 16));
  forevery_map #(natlt 16)
    (fun lane ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            ((flash_combine_bij nw).gg lane)._2) e)
      ** cell_full vgm ((flash_combine_bij nw).gg lane)._2
      ** cell_full vgl ((flash_combine_bij nw).gg lane)._2)
    (fun lane ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0 lane) e)
      ** cell_full vgm lane
      ** cell_full vgl lane)
    fn lane {
      rewrite each ((flash_combine_bij nw).gg lane)._2 as lane;
    };
  forevery_unzip3
    (fun (lane : natlt 16) ->
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0 lane) e)
    (fun (lane : natlt 16) -> cell_full vgm lane)
    (fun (lane : natlt 16) -> cell_full vgl lane);

  let sf = forevery_exists
    (fun (lane : natlt 16) (e : chest2 et (SZ.v nw) 1) ->
      tensor_pts_to
        (array2_stride_subtile vscale 1 16 0 lane) e);
  forevery_map #(natlt 16)
    (fun lane ->
      tensor_pts_to
        (array2_stride_subtile vscale 1 16 0 lane) (sf lane))
    (fun lane ->
      forall+ (tr : natlt 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 tr lane) (sf lane))
    fn lane {
      forevery_singleton_intro
        (fun (tr : natlt 1) ->
          tensor_pts_to
            (array2_stride_subtile vscale 1 16 tr lane) (sf lane));
    };
  forevery_commute
    (fun (lane : natlt 16) (tr : natlt 1) ->
      tensor_pts_to
        (array2_stride_subtile vscale 1 16 tr lane) (sf lane));
  array2_stride_untile' vscale 1 16
    (fun _tr lane -> sf lane);
  fold live vscale;

  forevery_map #(natlt 16)
    (fun lane -> cell_full vgm lane)
    (fun lane ->
      exists* (v : et).
        Cell vgm (idx1 lane) |-> Frac 1.0R v)
    fn lane { unfold cell_full vgm lane; };
  flash_cells1_gather vgm;
  forevery_map #(natlt 16)
    (fun lane -> cell_full vgl lane)
    (fun lane ->
      exists* (v : et).
        Cell vgl (idx1 lane) |-> Frac 1.0R v)
    fn lane { unfold cell_full vgl lane; };
  flash_cells1_gather vgl;
}

let flash_query_bij
  (hkv group sq hq rows : pos {
    hq == hkv * group /\ rows == group * sq })
  : (natlt hq & natlt sq =~ natlt hkv & natlt rows) =
{
  ff = (fun (qh, qpos) ->
    let kg = (bij_nat_prod #hkv #group).gg
      (qh <: natlt (hkv * group)) in
    (kg._1,
      ((bij_nat_prod #group #sq).ff
        (kg._2, qpos) <: natlt rows)));
  gg = (fun (kvh, r) ->
    let gp = (bij_nat_prod #group #sq).gg
      (r <: natlt (group * sq)) in
    (((bij_nat_prod #hkv #group).ff
        (kvh, gp._1) <: natlt hq),
      gp._2));
  ff_gg = (fun (kvh, r) ->
    bij_inv_bk (bij_nat_prod #group #sq)
      (r <: natlt (group * sq));
    bij_inv_fwd (bij_nat_prod #hkv #group)
      (kvh,
        ((bij_nat_prod #group #sq).gg
          (r <: natlt (group * sq)))._1));
  gg_ff = (fun (qh, qpos) ->
    bij_inv_bk (bij_nat_prod #hkv #group)
      (qh <: natlt (hkv * group));
    bij_inv_fwd (bij_nat_prod #group #sq)
      (((bij_nat_prod #hkv #group).gg
        (qh <: natlt (hkv * group)))._2, qpos));
}

let flash_pair4_bij (#a #b #c #d : Type0)
  : (a & b & c & d =~ (a & b) & (c & d)) =
{
  ff = (fun (x, y, z, w) -> ((x, y), (z, w)));
  gg = (fun ((x, y), (z, w)) -> (x, y, z, w));
}

let flash_output_logical_bij
  (b hkv group sq hq rows d : pos {
    hq == hkv * group /\ rows == group * sq })
  : (abs (b @| hq @| sq @| d @| INil)
      =~ (natlt b & natlt hkv & natlt rows & natlt d)) =
{
  ff = (fun (bi, (qh, (qpos, (dd, ())))) ->
    let kr = (flash_query_bij hkv group sq hq rows).ff
      (qh, qpos) in
    (bi, kr._1, kr._2, dd));
  gg = (fun (bi, kvh, r, dd) ->
    let qq = (flash_query_bij hkv group sq hq rows).gg
      (kvh, r) in
    (bi, (qq._1, (qq._2, (dd, ())))));
  ff_gg = (fun (bi, kvh, r, dd) ->
    bij_inv_bk (flash_query_bij hkv group sq hq rows)
      (kvh, r));
  gg_ff = (fun (bi, (qh, (qpos, (dd, ())))) ->
    bij_inv_fwd (flash_query_bij hkv group sq hq rows)
      (qh, qpos));
}

unfold
let flash_padded_idx
  (b hkv tiles d : pos) : Type0 =
  natlt b & natlt hkv & natlt tiles & natlt 16 & natlt d

unfold
let flash_padded_active
  (rows : pos) (#b #hkv #tiles #d : pos)
  (x : flash_padded_idx b hkv tiles d) : prop =
  let (_, _, rt, i, _) = x in
  rt * 16 + i < rows

unfold
let flash_active_padded_idx
  (b hkv rows tiles d : pos) : Type0 =
  x : flash_padded_idx b hkv tiles d {
    flash_padded_active rows x }

let flash_padded_logical
  (rows : pos) (#b #hkv #tiles #d : pos)
  (x : flash_padded_idx b hkv tiles d)
  : natlt b & natlt hkv & natlt rows & natlt d =
  let (bi, kvh, rt, i, dd) = x in
  (bi, kvh, FD.clamp_nat_lt rows (rt * 16 + i), dd)

ghost
fn flash_when_to_when__ (p : prop) (q : slprop)
  requires when_ p q
  ensures when__ p (fun _ -> q)
{
  rewrite (when_ p q) as (when__ p (fun _ -> q));
}

ghost
fn flash_when__to_when (p : prop) (q : slprop)
  requires when__ p (fun _ -> q)
  ensures when_ p q
{
  rewrite (when__ p (fun _ -> q)) as (when_ p q);
}

ghost
fn flash_when__elim_true
  (b : bool { b == true })
  (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures q ()
{
  rewrite (when__ b q) as q ();
}

ghost
fn flash_when__elim_false
  (b : bool { b == false })
  (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures emp
{
  rewrite (when__ b q) as emp;
}

ghost
fn flash_when__intro_true
  (b : bool { b == true })
  (q : squash (b2t b) -> slprop)
  requires q ()
  ensures when__ b q
{
  rewrite q () as (when__ b q);
}

ghost
fn flash_when__intro_false
  (b : bool { b == false })
  (q : squash (b2t b) -> slprop)
  ensures when__ b q
{
  rewrite emp as (when__ b q);
}

ghost
fn flash_when_elim_true (b : bool { b == true }) (q : slprop)
  requires when_ b q
  ensures q
{
  rewrite (when_ b q) as q;
}

ghost
fn flash_when_elim_false (b : bool { b == false }) (q : slprop)
  requires when_ b q
  ensures emp
{
  rewrite (when_ b q) as emp;
}

ghost
fn flash_when_intro_true (b : bool { b == true }) (q : slprop)
  requires q
  ensures when_ b q
{
  rewrite q as (when_ b q);
}

ghost
fn flash_when_intro_false (b : bool { b == false }) (q : slprop)
  ensures when_ b q
{
  rewrite emp as (when_ b q);
}

ghost
fn flash_when_reindex (p q : prop) (r : slprop)
  requires pure (p <==> q) ** when_ p r
  ensures when_ q r
{
  rewrite (when_ p r) as (when_ q r);
}

ghost
fn flash_when_prop_elim_true
  (p : prop) (b : bool { b == t2b p /\ b == true }) (q : slprop)
  requires when_ p q
  ensures q
{
  rewrite (when_ p q) as q;
}

ghost
fn flash_when_prop_elim_false
  (p : prop) (b : bool { b == t2b p /\ b == false }) (q : slprop)
  requires when_ p q
  ensures emp
{
  rewrite (when_ p q) as emp;
}

ghost
fn flash_when_prop_intro_true
  (p : prop) (b : bool { b == t2b p /\ b == true }) (q : slprop)
  requires q
  ensures when_ p q
{
  rewrite q as (when_ p q);
}

ghost
fn flash_when_prop_intro_false
  (p : prop) (b : bool { b == t2b p /\ b == false }) (q : slprop)
  ensures when_ p q
{
  rewrite emp as (when_ p q);
}

unfold
let flash_active_padded_bij
  (b hkv rows tiles d : pos { rows <= tiles * 16 })
  : (flash_active_padded_idx b hkv rows tiles d
      =~ (natlt b & natlt hkv & natlt rows & natlt d)) =
{
  ff = (fun (x : flash_active_padded_idx b hkv rows tiles d) ->
    flash_padded_logical rows x);
  gg = (fun (bi, kvh, r, dd) ->
    (bi, kvh,
      (r / 16 <: natlt tiles),
      (r % 16 <: natlt 16),
      dd));
  ff_gg = (fun (bi, kvh, r, dd) ->
    FStar.Math.Lemmas.lemma_div_mod (r <: nat) 16);
  gg_ff = (fun (x : flash_active_padded_idx b hkv rows tiles d) ->
    let (bi, kvh, rt, i, dd) = x in
    assert (flash_padded_active rows x);
    assert (rt * 16 + i < rows);
    assert (FD.clamp_nat_lt rows (rt * 16 + i) == rt * 16 + i);
    FStar.Math.Lemmas.lemma_div_plus
      (i <: nat) (rt <: nat) 16;
    FStar.Math.Lemmas.small_div (i <: nat) 16;
    FStar.Math.Lemmas.small_mod (i <: nat) 16);
}

unfold
let flash_block_bij
  (b hkv tiles : pos)
  : (natlt b & natlt hkv & natlt tiles
      =~ natlt (b * hkv * tiles)) =
{
  ff = (fun ((bi, kvh, rt) :
      natlt b & natlt hkv & natlt tiles) ->
    ((bi * hkv + kvh) * tiles + rt
      <: natlt (b * hkv * tiles)));
  gg = (fun bid ->
    let bh = bid / tiles in
    ((bh / hkv <: natlt b),
      (bh % hkv <: natlt hkv),
      (bid % tiles <: natlt tiles)));
  ff_gg = (fun bid ->
    FStar.Math.Lemmas.lemma_div_mod (bid <: nat) tiles;
    FStar.Math.Lemmas.lemma_div_mod
      ((bid / tiles) <: nat) hkv);
  gg_ff = (fun (bi, kvh, rt) ->
    FStar.Math.Lemmas.lemma_div_plus
      (rt <: nat) ((bi * hkv + kvh) <: nat) tiles;
    FStar.Math.Lemmas.small_div (rt <: nat) tiles;
    FStar.Math.Lemmas.small_mod (rt <: nat) tiles;
    FStar.Math.Lemmas.lemma_div_plus
      (kvh <: nat) (bi <: nat) hkv;
    FStar.Math.Lemmas.small_div (kvh <: nat) hkv;
    FStar.Math.Lemmas.small_mod (kvh <: nat) hkv);
}

let flash_owner_idx
  (b hkv tiles d : pos) : Type0 =
  bid : natlt (b * hkv * tiles) &
  (lane : natlt BW.warp_size &
    out_stride_index2 16 d BW.warp_size lane)

unfold
let flash_padded_owner_bij
  (b hkv tiles d : pos)
  : (flash_padded_idx b hkv tiles d
      =~ flash_owner_idx b hkv tiles d) =
{
  ff = (fun (bi, kvh, rt, i, dd) ->
    let bid = (flash_block_bij b hkv tiles).ff
      (bi, kvh, rt) in
    let lane_ij =
      (flash_stride_partition_bij 16 d BW.warp_size).gg
        (i, dd) in
    (| bid, lane_ij |));
  gg = (fun owner ->
    let (bi, kvh, rt) =
      (flash_block_bij b hkv tiles).gg owner._1 in
    let ij = owner._2._2 in
    (bi, kvh, rt, ij._1, ij._2));
  ff_gg = (fun owner ->
    bij_inv_bk (flash_block_bij b hkv tiles) owner._1;
    bij_inv_fwd
      (flash_stride_partition_bij 16 d BW.warp_size)
      owner._2);
  gg_ff = (fun (bi, kvh, rt, i, dd) ->
    bij_inv_fwd (flash_block_bij b hkv tiles)
      (bi, kvh, rt);
    bij_inv_bk
      (flash_stride_partition_bij 16 d BW.warp_size)
      (i, dd));
}

unfold
let flash_bid_rt
  (b hkv tiles : pos)
  (bid : natlt (b * hkv * tiles)) : natlt tiles =
  bid % tiles

unfold
let flash_bid_kvh
  (b hkv tiles : pos)
  (bid : natlt (b * hkv * tiles)) : natlt hkv =
  (bid / tiles) % hkv

unfold
let flash_bid_bi
  (b hkv tiles : pos)
  (bid : natlt (b * hkv * tiles)) : natlt b =
  (bid / tiles) / hkv

unfold
let flash_block_output
  (#et : Type0)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  : slprop =
  forall+ (lane : natlt BW.warp_size).
    out_store_cells b hq sq 16sz d rows gout
      (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (SZ.v group)
      (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
      lane

ghost
fn flash_split_output
  (#et : Type0)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  requires live gout
  ensures
    forall+ (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)).
      flash_block_output b hq hkv group sq rows tiles d gout bid
{
  unfold live gout;
  with eout. assert (gout |-> eout);
  tensor_explode gout;
  forevery_map
    (fun (idx : abs (b @| hq @| sq @| d @| INil)) ->
      Cell gout idx |-> acc eout idx)
    (fun idx ->
      exists* (v : et). Cell gout idx |-> Frac 1.0R v)
    fn _ { () };
  forevery_iso
    (flash_output_logical_bij
      (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
      (SZ.v hq) (SZ.v rows) (SZ.v d))
    (fun idx ->
      exists* (v : et). Cell gout idx |-> Frac 1.0R v);
  forevery_map
    (fun (x : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      exists* (v : et).
        Cell gout
          ((flash_output_logical_bij
            (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
            (SZ.v hq) (SZ.v rows) (SZ.v d)).gg x)
          |-> Frac 1.0R v)
    (fun (x : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      out_cell b hq sq d gout
        x._1 x._2 (SZ.v group) x._3 x._4)
    fn x {
      with v. assert (
        tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
            (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
            (SZ.v hq) (SZ.v rows) (SZ.v d)).gg x)
          v);
      rewrite
        (tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
            (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
            (SZ.v hq) (SZ.v rows) (SZ.v d)).gg x)
          v)
        as
        (tensor_pts_to_cell gout #1.0R
          (idx4 x._1
            (out_qh (SZ.v hq) (SZ.v sq) x._2
              (SZ.v group) x._3)
            (out_qpos (SZ.v sq) x._3)
            x._4)
          v);
      fold out_cell b hq sq d gout
        x._1 x._2 (SZ.v group) x._3 x._4;
    };
  forevery_iso
    (bij_sym (flash_active_padded_bij
      (SZ.v b) (SZ.v hkv) (SZ.v rows) (SZ.v tiles) (SZ.v d)))
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      out_cell b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4);
  forevery_unrefine_pred
    (fun (x : flash_padded_idx
      (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)) ->
      let y = flash_padded_logical (SZ.v rows) x in
      out_cell b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4)
    (flash_padded_active (SZ.v rows));
  forevery_iso
    (flash_padded_owner_bij
      (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d))
    (fun (x : flash_padded_idx
      (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)) ->
      when_ (flash_padded_active (SZ.v rows) x)
        (let y = flash_padded_logical (SZ.v rows) x in
        out_cell b hq sq d gout
          y._1 y._2 (SZ.v group) y._3 y._4));
  forevery_map
    (fun (owner : flash_owner_idx
      (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)) ->
      let x = (flash_padded_owner_bij
        (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner in
      when_ (flash_padded_active (SZ.v rows) x)
        (let y = flash_padded_logical (SZ.v rows) x in
         out_cell b hq sq d gout
           y._1 y._2 (SZ.v group) y._3 y._4))
    (fun owner ->
      let bid = owner._1 in
      let ij = owner._2._2 in
      when_ (
        flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
          + ij._1 < SZ.v rows)
        (out_cell b hq sq d gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
              + ij._1)
            ij._2))
    fn owner {
      let x = (flash_padded_owner_bij
        (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner;
      let (bi, kvh, rt, i, dd) = x;
      let owner_active =
        flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
          + owner._2._2._1 < SZ.v rows;
      let active = t2b owner_active;
      assert pure (
        owner_active <==>
        flash_padded_active (SZ.v rows)
          ((flash_padded_owner_bij
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner));
      assert pure (owner_active <==> rt * 16 + i < SZ.v rows);
      rewrite each
        (flash_padded_owner_bij
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner
        as x;
      rewrite each
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
          + owner._2._2._1 < SZ.v rows)
        as owner_active;
      if active {
        flash_when_prop_elim_true
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          let y = flash_padded_logical (SZ.v rows) x in
          out_cell b hq sq d gout
            y._1 y._2 (SZ.v group) y._3 y._4);
        assert pure (FD.clamp_nat_lt (SZ.v rows) (rt * 16 + i)
          == rt * 16 + i);
        rewrite
          (let y = flash_padded_logical (SZ.v rows) x in
           out_cell b hq sq d gout
             y._1 y._2 (SZ.v group) y._3 y._4)
          as
          (out_cell b hq sq d gout bi kvh (SZ.v group)
            (rt * 16 + i) dd);
        rewrite each
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
          as bi;
        rewrite each
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
          as kvh;
        rewrite each
          (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
          as rt;
        rewrite each owner._2._2._1 as i;
        rewrite each owner._2._2._2 as dd;
        flash_when_prop_intro_true
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          out_cell b hq sq d gout bi kvh (SZ.v group)
            (rt * 16 + i) dd);
        rewrite each bi as
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1);
        rewrite each kvh as
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1);
        rewrite each rt as
          (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1);
        rewrite each i as owner._2._2._1;
        rewrite each dd as owner._2._2._2;
      } else {
        flash_when_prop_elim_false
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          let y = flash_padded_logical (SZ.v rows) x in
          out_cell b hq sq d gout
            y._1 y._2 (SZ.v group) y._3 y._4);
        flash_when_prop_intro_false
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          out_cell b hq sq d gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
            (SZ.v group)
            (flash_bid_rt
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
              + owner._2._2._1)
            owner._2._2._2);
      };
    };
  forevery_unflatten_dep
    (fun
      (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
      (li : (lane : natlt BW.warp_size &
        out_stride_index2 16 (SZ.v d) BW.warp_size lane)) ->
      let ij = li._2 in
      when_ (
        flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
          + ij._1 < SZ.v rows)
        (out_cell b hq sq d gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
            + ij._1)
          ij._2));
  forevery_map
    (fun (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      forall+ (li : (lane : natlt BW.warp_size &
        out_stride_index2 16 (SZ.v d) BW.warp_size lane)).
        let ij = li._2 in
        when_ (
          flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
            + ij._1 < SZ.v rows)
          (out_cell b hq sq d gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
              + ij._1)
            ij._2))
    (fun bid -> flash_block_output
      b hq hkv group sq rows tiles d gout bid)
    fn bid {
      forevery_unflatten_dep
        (fun (lane : natlt BW.warp_size)
          (ij : out_stride_index2
            16 (SZ.v d) BW.warp_size lane) ->
          when_ (
            flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
              + ij._1 < SZ.v rows)
            (out_cell b hq sq d gout
              (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (SZ.v group)
              (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid
                * 16 + ij._1)
              ij._2));
      forevery_map
        (fun (lane : natlt BW.warp_size) ->
          forall+ (ij : out_stride_index2
            16 (SZ.v d) BW.warp_size lane).
            when_ (
              flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid
                * 16 + ij._1 < SZ.v rows)
              (out_cell b hq sq d gout
                (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                (SZ.v group)
                (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid
                  * 16 + ij._1)
                ij._2))
        (fun lane ->
          out_store_cells b hq sq 16sz d rows gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
            lane)
        fn lane {
          fold out_store_cells b hq sq 16sz d rows gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
            lane;
        };
      fold flash_block_output
        b hq hkv group sq rows tiles d gout bid;
    };
}

ghost
fn flash_gather_output
  (#et : Type0)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  requires
    pure (SZ.fits (tlayout_ulen lout)) **
    (forall+ (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)).
      flash_block_output b hq hkv group sq rows tiles d gout bid)
  ensures live gout
{
  forevery_map
    (fun (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      flash_block_output b hq hkv group sq rows tiles d gout bid)
    (fun bid ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane)
    fn bid {
      unfold flash_block_output
        b hq hkv group sq rows tiles d gout bid;
    };
  forevery_map
    (fun (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane)
    (fun bid ->
      forall+ (lane : natlt BW.warp_size)
        (ij : out_stride_index2
          16 (SZ.v d) BW.warp_size lane).
        when_ (
          flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
            + ij._1 < SZ.v rows)
          (out_cell b hq sq d gout
              (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (SZ.v group)
              (flash_bid_rt
                (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                + ij._1)
              ij._2))
    fn bid {
      forevery_map
        (fun (lane : natlt BW.warp_size) ->
          out_store_cells b hq sq 16sz d rows gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
            lane)
        (fun lane ->
          forall+ (ij : out_stride_index2
            16 (SZ.v d) BW.warp_size lane).
            when_ (
              flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                + ij._1 < SZ.v rows)
              (out_cell b hq sq d gout
                  (flash_bid_bi
                    (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (flash_bid_kvh
                    (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (SZ.v group)
                  (flash_bid_rt
                    (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                    + ij._1)
                  ij._2))
        fn lane {
          unfold out_store_cells b hq sq 16sz d rows gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
            lane;
        };
    };
  forevery_map
    (fun (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      forall+ (lane : natlt BW.warp_size)
        (ij : out_stride_index2
          16 (SZ.v d) BW.warp_size lane).
        when_ (
          flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
            + ij._1 < SZ.v rows)
          (out_cell b hq sq d gout
              (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (SZ.v group)
              (flash_bid_rt
                (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                + ij._1)
              ij._2))
    (fun bid ->
      forall+ (li : (lane : natlt BW.warp_size &
        out_stride_index2 16 (SZ.v d) BW.warp_size lane)).
        let ij = li._2 in
        when_ (
          flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
            + ij._1 < SZ.v rows)
          (out_cell b hq sq d gout
              (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (SZ.v group)
              (flash_bid_rt
                (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                + ij._1)
              ij._2))
    fn bid {
      forevery_flatten_dep
        (fun (lane : natlt BW.warp_size)
          (ij : out_stride_index2
            16 (SZ.v d) BW.warp_size lane) ->
          when_ (
            flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
              + ij._1 < SZ.v rows)
            (out_cell b hq sq d gout
                (flash_bid_bi
                  (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                (flash_bid_kvh
                  (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                (SZ.v group)
                (flash_bid_rt
                  (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                  + ij._1)
                ij._2));
    };
  forevery_flatten_dep
    (fun
      (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
      (li : (lane : natlt BW.warp_size &
        out_stride_index2 16 (SZ.v d) BW.warp_size lane)) ->
      let ij = li._2 in
      when_ (
        flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
          + ij._1 < SZ.v rows)
        (out_cell b hq sq d gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
              + ij._1)
            ij._2));
  forevery_map
    (fun (owner : flash_owner_idx
      (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)) ->
      let bid = owner._1 in
      let ij = owner._2._2 in
      when_ (
        flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
          + ij._1 < SZ.v rows)
        (out_cell b hq sq d gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
              + ij._1)
            ij._2))
    (fun owner ->
      let x = (flash_padded_owner_bij
        (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner in
      when_ (flash_padded_active (SZ.v rows) x)
        (let y = flash_padded_logical (SZ.v rows) x in
        out_cell b hq sq d gout
          y._1 y._2 (SZ.v group) y._3 y._4))
    fn owner {
      let x = (flash_padded_owner_bij
        (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner;
      let (bi, kvh, rt, i, dd) = x;
      let owner_active =
        flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
          + owner._2._2._1 < SZ.v rows;
      let padded_active = flash_padded_active (SZ.v rows) x;
      let active = t2b owner_active;
      assert pure (owner_active <==> padded_active);
      assert pure (active == t2b padded_active);
      if active {
        flash_when_prop_elim_true
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          out_cell b hq sq d gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
            (SZ.v group)
            (flash_bid_rt
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
              + owner._2._2._1)
            owner._2._2._2);
        rewrite each
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
          as bi;
        rewrite each
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
          as kvh;
        rewrite each
          (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
          as rt;
        rewrite each owner._2._2._1 as i;
        rewrite each owner._2._2._2 as dd;
        assert pure (FD.clamp_nat_lt (SZ.v rows) (rt * 16 + i)
          == rt * 16 + i);
        rewrite
          (out_cell b hq sq d gout bi kvh (SZ.v group)
            (rt * 16 + i) dd)
          as
          (let y = flash_padded_logical (SZ.v rows) x in
           out_cell b hq sq d gout
             y._1 y._2 (SZ.v group) y._3 y._4);
        rewrite each x as
          ((flash_padded_owner_bij
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner);
        flash_when_prop_intro_true
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          let y = flash_padded_logical (SZ.v rows)
            ((flash_padded_owner_bij
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner) in
          out_cell b hq sq d gout
            y._1 y._2 (SZ.v group) y._3 y._4);
      } else {
        flash_when_prop_elim_false
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          out_cell b hq sq d gout
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1)
            (SZ.v group)
            (flash_bid_rt
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
              + owner._2._2._1)
            owner._2._2._2);
        flash_when_prop_intro_false
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          let y = flash_padded_logical (SZ.v rows)
            ((flash_padded_owner_bij
              (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)).gg owner) in
          out_cell b hq sq d gout
            y._1 y._2 (SZ.v group) y._3 y._4);
      }
    };
  forevery_iso_back
    (flash_padded_owner_bij
      (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d))
    (fun (x : flash_padded_idx
      (SZ.v b) (SZ.v hkv) (SZ.v tiles) (SZ.v d)) ->
      when_ (flash_padded_active (SZ.v rows) x)
        (let y = flash_padded_logical (SZ.v rows) x in
        out_cell b hq sq d gout
          y._1 y._2 (SZ.v group) y._3 y._4));
  forevery_refine_pred
    (fun x ->
      let y = flash_padded_logical (SZ.v rows) x in
      out_cell b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4)
    (flash_padded_active (SZ.v rows));
  forevery_iso
    (flash_active_padded_bij
      (SZ.v b) (SZ.v hkv) (SZ.v rows) (SZ.v tiles) (SZ.v d))
    (fun (x : flash_active_padded_idx
      (SZ.v b) (SZ.v hkv) (SZ.v rows) (SZ.v tiles) (SZ.v d)) ->
      let y = flash_padded_logical (SZ.v rows) x in
      out_cell b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4);
  forevery_map
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      out_cell b hq sq d gout
        ((flash_active_padded_bij
          (SZ.v b) (SZ.v hkv) (SZ.v rows)
          (SZ.v tiles) (SZ.v d)).ff
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).gg y))._1
        ((flash_active_padded_bij
          (SZ.v b) (SZ.v hkv) (SZ.v rows)
          (SZ.v tiles) (SZ.v d)).ff
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).gg y))._2
        (SZ.v group)
        ((flash_active_padded_bij
          (SZ.v b) (SZ.v hkv) (SZ.v rows)
          (SZ.v tiles) (SZ.v d)).ff
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).gg y))._3
        ((flash_active_padded_bij
          (SZ.v b) (SZ.v hkv) (SZ.v rows)
          (SZ.v tiles) (SZ.v d)).ff
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).gg y))._4)
    (fun y ->
      out_cell b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4)
    fn y {
      bij_inv_bk
        (flash_active_padded_bij
          (SZ.v b) (SZ.v hkv) (SZ.v rows)
          (SZ.v tiles) (SZ.v d))
        y;
      rewrite
        (out_cell b hq sq d gout
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).ff
            ((flash_active_padded_bij
              (SZ.v b) (SZ.v hkv) (SZ.v rows)
              (SZ.v tiles) (SZ.v d)).gg y))._1
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).ff
            ((flash_active_padded_bij
              (SZ.v b) (SZ.v hkv) (SZ.v rows)
              (SZ.v tiles) (SZ.v d)).gg y))._2
          (SZ.v group)
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).ff
            ((flash_active_padded_bij
              (SZ.v b) (SZ.v hkv) (SZ.v rows)
              (SZ.v tiles) (SZ.v d)).gg y))._3
          ((flash_active_padded_bij
            (SZ.v b) (SZ.v hkv) (SZ.v rows)
            (SZ.v tiles) (SZ.v d)).ff
            ((flash_active_padded_bij
              (SZ.v b) (SZ.v hkv) (SZ.v rows)
              (SZ.v tiles) (SZ.v d)).gg y))._4)
        as
        (out_cell b hq sq d gout
          y._1 y._2 (SZ.v group) y._3 y._4);
    };
  forevery_map
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      out_cell b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4)
    (fun y ->
      exists* (v : et).
        tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
            (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
            (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
          v)
    fn y {
      unfold out_cell b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4;
      with v. assert (
        tensor_pts_to_cell gout #1.0R
          (idx4 y._1
            (out_qh (SZ.v hq) (SZ.v sq) y._2
              (SZ.v group) y._3)
            (out_qpos (SZ.v sq) y._3)
            y._4)
          v);
      rewrite
        (tensor_pts_to_cell gout #1.0R
          (idx4 y._1
            (out_qh (SZ.v hq) (SZ.v sq) y._2
              (SZ.v group) y._3)
            (out_qpos (SZ.v sq) y._3)
            y._4)
          v)
        as
        (tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
            (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
            (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
          v);
    };
  forevery_iso
    (flash_pair4_bij
      #(natlt (SZ.v b)) #(natlt (SZ.v hkv))
      #(natlt (SZ.v rows)) #(natlt (SZ.v d)))
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      exists* (v : et).
        tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
            (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
            (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
          v);
  let vf = forevery_exists
    (fun (yy : (natlt (SZ.v b) & natlt (SZ.v hkv)) &
      (natlt (SZ.v rows) & natlt (SZ.v d))) (v : et) ->
      let y = (flash_pair4_bij
        #(natlt (SZ.v b)) #(natlt (SZ.v hkv))
        #(natlt (SZ.v rows)) #(natlt (SZ.v d))).gg yy in
      tensor_pts_to_cell gout #1.0R
        ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
        v);
  let all_bij =
    bij_comp
      (flash_output_logical_bij
        (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
        (SZ.v hq) (SZ.v rows) (SZ.v d))
      (flash_pair4_bij
        #(natlt (SZ.v b)) #(natlt (SZ.v hkv))
        #(natlt (SZ.v rows)) #(natlt (SZ.v d)));
  let eout : chest (b @| hq @| sq @| d @| INil) et =
    mk (b @| hq @| sq @| d @| INil)
      (fun idx -> vf (all_bij.ff idx));
  forevery_map
    (fun (yy : (natlt (SZ.v b) & natlt (SZ.v hkv)) &
      (natlt (SZ.v rows) & natlt (SZ.v d))) ->
      let y = (flash_pair4_bij
        #(natlt (SZ.v b)) #(natlt (SZ.v hkv))
        #(natlt (SZ.v rows)) #(natlt (SZ.v d))).gg yy in
      tensor_pts_to_cell gout #1.0R
        ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
        (vf yy))
    (fun yy ->
      tensor_pts_to_cell gout #1.0R (all_bij.gg yy) (vf yy))
    fn yy {
      assert pure (
        all_bij.gg yy ==
        (flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg
          ((flash_pair4_bij
            #(natlt (SZ.v b)) #(natlt (SZ.v hkv))
            #(natlt (SZ.v rows)) #(natlt (SZ.v d))).gg yy));
      rewrite
        (tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
            (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
            (SZ.v hq) (SZ.v rows) (SZ.v d)).gg
            ((flash_pair4_bij
              #(natlt (SZ.v b)) #(natlt (SZ.v hkv))
              #(natlt (SZ.v rows)) #(natlt (SZ.v d))).gg yy))
          (vf yy))
        as
        (tensor_pts_to_cell gout #1.0R (all_bij.gg yy) (vf yy));
    };
  forevery_map
    (fun (yy : (natlt (SZ.v b) & natlt (SZ.v hkv)) &
      (natlt (SZ.v rows) & natlt (SZ.v d))) ->
      tensor_pts_to_cell gout #1.0R (all_bij.gg yy) (vf yy))
    (fun yy ->
      tensor_pts_to_cell gout #1.0R
        (all_bij.gg yy) (acc eout (all_bij.gg yy)))
    fn yy {
      bij_inv_bk all_bij yy;
      assert pure (
        acc eout (all_bij.gg yy) == vf yy);
      rewrite
        (tensor_pts_to_cell gout #1.0R (all_bij.gg yy) (vf yy))
        as
        (tensor_pts_to_cell gout #1.0R
          (all_bij.gg yy) (acc eout (all_bij.gg yy)));
    };
  forevery_iso_back all_bij
    (fun idx ->
      tensor_pts_to_cell gout #1.0R idx (acc eout idx));
  tensor_implode gout;
  fold live gout;
}

unfold
let flash_views_live
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d) : slprop =
  live v.shQv ** live v.shKv ** live v.shVv **
  live v.shSv ** live v.shPv ** live v.shPVv **
  live v.shcwv ** live v.shMv ** live v.shLv **
  live v.shscalev ** live v.shOv **
  live v.shgmv ** live v.shglv

ghost
fn flash_open_shmems
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (nw d : szp)
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  requires live_c_shmems sh
  ensures flash_views_live (flash_views_of nw d sh)
{
  unfold_c_shmems sh (`%flash_shmems);
  let
    (q, (k, (v, (s, (p, (pv, (cw,
    (m, (l, (scale, (o, (gm, (gl, _))))))))))))) = sh;
  gpu_pts_to_ref q;
  gpu_pts_to_ref k;
  gpu_pts_to_ref v;
  gpu_pts_to_ref s;
  gpu_pts_to_ref p;
  gpu_pts_to_ref pv;
  gpu_pts_to_ref cw;
  gpu_pts_to_ref m;
  gpu_pts_to_ref l;
  gpu_pts_to_ref scale;
  gpu_pts_to_ref o;
  gpu_pts_to_ref gm;
  gpu_pts_to_ref gl;
  tensor_abs' (l2_row_major 16 (SZ.v d)) q;
  tensor_abs' (l2_row_major (SZ.v nw * 16) (SZ.v d)) k;
  tensor_abs' (l2_row_major (SZ.v nw * 16) (SZ.v d)) v;
  tensor_abs' (l2_row_major (SZ.v nw * 16) 16) s;
  tensor_abs' (l2_row_major (SZ.v nw * 16) 16) p;
  tensor_abs' (l2_row_major (SZ.v nw * 16) 16) pv;
  tensor_abs' (l2_row_major (SZ.v nw) 16) cw;
  tensor_abs' (l2_row_major (SZ.v nw) 16) m;
  tensor_abs' (l2_row_major (SZ.v nw) 16) l;
  tensor_abs' (l2_row_major (SZ.v nw) 16) scale;
  tensor_abs' (l2_row_major (SZ.v nw * 16) (SZ.v d)) o;
  tensor_abs' (l1_forward 16) gm;
  tensor_abs' (l1_forward 16) gl;
  let views = flash_views_of nw d sh;
  with eQ. assert (
    from_array (l2_row_major 16 (SZ.v d)) q |-> eQ);
  rewrite (from_array (l2_row_major 16 (SZ.v d)) q |-> eQ)
    as (views.shQv |-> eQ);
  fold live views.shQv;
  with eK. assert (
    from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) k |-> eK);
  rewrite
    (from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) k |-> eK)
    as (views.shKv |-> eK);
  fold live views.shKv;
  with eV. assert (
    from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) v |-> eV);
  rewrite
    (from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) v |-> eV)
    as (views.shVv |-> eV);
  fold live views.shVv;
  with eS. assert (
    from_array (l2_row_major (SZ.v nw * 16) 16) s |-> eS);
  rewrite (from_array (l2_row_major (SZ.v nw * 16) 16) s |-> eS)
    as (views.shSv |-> eS);
  fold live views.shSv;
  with eP. assert (
    from_array (l2_row_major (SZ.v nw * 16) 16) p |-> eP);
  rewrite (from_array (l2_row_major (SZ.v nw * 16) 16) p |-> eP)
    as (views.shPv |-> eP);
  fold live views.shPv;
  with ePV. assert (
    from_array (l2_row_major (SZ.v nw * 16) 16) pv |-> ePV);
  rewrite (from_array (l2_row_major (SZ.v nw * 16) 16) pv |-> ePV)
    as (views.shPVv |-> ePV);
  fold live views.shPVv;
  with ecw. assert (
    from_array (l2_row_major (SZ.v nw) 16) cw |-> ecw);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) cw |-> ecw)
    as (views.shcwv |-> ecw);
  fold live views.shcwv;
  with eM. assert (
    from_array (l2_row_major (SZ.v nw) 16) m |-> eM);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) m |-> eM)
    as (views.shMv |-> eM);
  fold live views.shMv;
  with eL. assert (
    from_array (l2_row_major (SZ.v nw) 16) l |-> eL);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) l |-> eL)
    as (views.shLv |-> eL);
  fold live views.shLv;
  with escale. assert (
    from_array (l2_row_major (SZ.v nw) 16) scale |-> escale);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) scale |-> escale)
    as (views.shscalev |-> escale);
  fold live views.shscalev;
  with eO. assert (
    from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) o |-> eO);
  rewrite
    (from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) o |-> eO)
    as (views.shOv |-> eO);
  fold live views.shOv;
  with egm. assert (from_array (l1_forward 16) gm |-> egm);
  rewrite (from_array (l1_forward 16) gm |-> egm)
    as (views.shgmv |-> egm);
  fold live views.shgmv;
  with egl. assert (from_array (l1_forward 16) gl |-> egl);
  rewrite (from_array (l1_forward 16) gl |-> egl)
    as (views.shglv |-> egl);
  fold live views.shglv;
  rewrite each views as (flash_views_of nw d sh);
  fold flash_views_live (flash_views_of nw d sh);
}

ghost
fn flash_close_shmems
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (nw d : szp)
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  requires flash_views_live (flash_views_of nw d sh)
  ensures live_c_shmems sh
{
  unfold flash_views_live (flash_views_of nw d sh);
  unfold live (flash_views_of nw d sh).shQv;
  unfold live (flash_views_of nw d sh).shKv;
  unfold live (flash_views_of nw d sh).shVv;
  unfold live (flash_views_of nw d sh).shSv;
  unfold live (flash_views_of nw d sh).shPv;
  unfold live (flash_views_of nw d sh).shPVv;
  unfold live (flash_views_of nw d sh).shcwv;
  unfold live (flash_views_of nw d sh).shMv;
  unfold live (flash_views_of nw d sh).shLv;
  unfold live (flash_views_of nw d sh).shscalev;
  unfold live (flash_views_of nw d sh).shOv;
  unfold live (flash_views_of nw d sh).shgmv;
  unfold live (flash_views_of nw d sh).shglv;
  tensor_concr (flash_views_of nw d sh).shQv;
  tensor_concr (flash_views_of nw d sh).shKv;
  tensor_concr (flash_views_of nw d sh).shVv;
  tensor_concr (flash_views_of nw d sh).shSv;
  tensor_concr (flash_views_of nw d sh).shPv;
  tensor_concr (flash_views_of nw d sh).shPVv;
  tensor_concr (flash_views_of nw d sh).shcwv;
  tensor_concr (flash_views_of nw d sh).shMv;
  tensor_concr (flash_views_of nw d sh).shLv;
  tensor_concr (flash_views_of nw d sh).shscalev;
  tensor_concr (flash_views_of nw d sh).shOv;
  tensor_concr (flash_views_of nw d sh).shgmv;
  tensor_concr (flash_views_of nw d sh).shglv;
  let q = fst sh;
  let k = fst (snd sh);
  let v = fst (snd (snd sh));
  let s = fst (snd (snd (snd sh)));
  let p = fst (snd (snd (snd (snd sh))));
  let pv = fst (snd (snd (snd (snd (snd sh)))));
  let cw = fst (snd (snd (snd (snd (snd (snd sh))))));
  let m = fst (snd (snd (snd (snd (snd (snd (snd sh)))))));
  let l = fst (snd (snd (snd (snd (snd (snd (snd (snd sh))))))));
  let scale = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))));
  let o = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))));
  let gm = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))))));
  let gl = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))))));
  lem_from_array_core #et_ab #2
    #(16 @| SZ.v d @| INil) #(l2_row_major 16 (SZ.v d)) q;
  lem_from_array_core #et_ab #2
    #(SZ.v nw * 16 @| SZ.v d @| INil)
    #(l2_row_major (SZ.v nw * 16) (SZ.v d)) k;
  lem_from_array_core #et_ab #2
    #(SZ.v nw * 16 @| SZ.v d @| INil)
    #(l2_row_major (SZ.v nw * 16) (SZ.v d)) v;
  lem_from_array_core #et_acc #2
    #(SZ.v nw * 16 @| 16 @| INil)
    #(l2_row_major (SZ.v nw * 16) 16) s;
  lem_from_array_core #et_ab #2
    #(SZ.v nw * 16 @| 16 @| INil)
    #(l2_row_major (SZ.v nw * 16) 16) p;
  lem_from_array_core #et_acc #2
    #(SZ.v nw * 16 @| 16 @| INil)
    #(l2_row_major (SZ.v nw * 16) 16) pv;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) cw;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) m;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) l;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) scale;
  lem_from_array_core #et_acc #2
    #(SZ.v nw * 16 @| SZ.v d @| INil)
    #(l2_row_major (SZ.v nw * 16) (SZ.v d)) o;
  lem_from_array_core #et_acc #1
    #(16 @| INil) #(l1_forward 16) gm;
  lem_from_array_core #et_acc #1
    #(16 @| INil) #(l1_forward 16) gl;
  with sq. assert (core (flash_views_of nw d sh).shQv |-> sq);
  rewrite (core (flash_views_of nw d sh).shQv |-> sq) as (q |-> sq);
  with sk. assert (core (flash_views_of nw d sh).shKv |-> sk);
  rewrite (core (flash_views_of nw d sh).shKv |-> sk) as (k |-> sk);
  with sv. assert (core (flash_views_of nw d sh).shVv |-> sv);
  rewrite (core (flash_views_of nw d sh).shVv |-> sv) as (v |-> sv);
  with ss. assert (core (flash_views_of nw d sh).shSv |-> ss);
  rewrite (core (flash_views_of nw d sh).shSv |-> ss) as (s |-> ss);
  with sp. assert (core (flash_views_of nw d sh).shPv |-> sp);
  rewrite (core (flash_views_of nw d sh).shPv |-> sp) as (p |-> sp);
  with spv. assert (core (flash_views_of nw d sh).shPVv |-> spv);
  rewrite (core (flash_views_of nw d sh).shPVv |-> spv)
    as (pv |-> spv);
  with scw. assert (core (flash_views_of nw d sh).shcwv |-> scw);
  rewrite (core (flash_views_of nw d sh).shcwv |-> scw)
    as (cw |-> scw);
  with sm. assert (core (flash_views_of nw d sh).shMv |-> sm);
  rewrite (core (flash_views_of nw d sh).shMv |-> sm) as (m |-> sm);
  with sl. assert (core (flash_views_of nw d sh).shLv |-> sl);
  rewrite (core (flash_views_of nw d sh).shLv |-> sl) as (l |-> sl);
  with sscale. assert (
    core (flash_views_of nw d sh).shscalev |-> sscale);
  rewrite (core (flash_views_of nw d sh).shscalev |-> sscale)
    as (scale |-> sscale);
  with so. assert (core (flash_views_of nw d sh).shOv |-> so);
  rewrite (core (flash_views_of nw d sh).shOv |-> so) as (o |-> so);
  with sgm. assert (core (flash_views_of nw d sh).shgmv |-> sgm);
  rewrite (core (flash_views_of nw d sh).shgmv |-> sgm)
    as (gm |-> sgm);
  with sgl. assert (core (flash_views_of nw d sh).shglv |-> sgl);
  rewrite (core (flash_views_of nw d sh).shglv |-> sgl)
    as (gl |-> sgl);
  rewrite each q as (fst sh);
  rewrite each k as (fst (snd sh));
  rewrite each v as (fst (snd (snd sh)));
  rewrite each s as (fst (snd (snd (snd sh))));
  rewrite each p as (fst (snd (snd (snd (snd sh)))));
  rewrite each pv as (fst (snd (snd (snd (snd (snd sh))))));
  rewrite each cw as
    (fst (snd (snd (snd (snd (snd (snd sh)))))));
  rewrite each m as
    (fst (snd (snd (snd (snd (snd (snd (snd sh))))))));
  rewrite each l as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))));
  rewrite each scale as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))));
  rewrite each o as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))))));
  rewrite each gm as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))))));
  rewrite each gl as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))))))));
  fold_c_shmems sh (`%flash_shmems);
}

ghost
fn flash_split_warp_tiles
  (#et : Type0) (nw cols : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v cols)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v cols)))
  requires live a
  ensures
    forall+ (w : natlt (SZ.v nw)).
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0)
{
  unfold live a;
  with e. assert (a |-> e);
  array2_tile a 16 (SZ.v cols);
  forevery_rw_size2
    (SZ.v nw * 16 / 16) (SZ.v nw)
    (SZ.v cols / SZ.v cols) 1
    #(fun (w : natlt (SZ.v nw)) (tc : natlt 1) ->
      array2_subtile a 16 (SZ.v cols) w tc
        |-> Frac 1.0R
          (ematrix_subtile e 16 (SZ.v cols) w tc));
  forevery_map
    (fun (w : natlt (SZ.v nw)) ->
      forall+ (tc : natlt 1).
        array2_subtile a 16 (SZ.v cols) w tc
          |-> Frac 1.0R
            (ematrix_subtile e 16 (SZ.v cols) w tc))
    (fun w ->
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0))
    fn w {
      forevery_singleton_elim
        (fun (tc : natlt 1) ->
          array2_subtile a 16 (SZ.v cols) w tc
            |-> Frac 1.0R
              (ematrix_subtile e 16 (SZ.v cols) w tc));
      fold live (array2_subtile a 16 (SZ.v cols <: pos) w 0);
    };
}

ghost
fn flash_gather_warp_tiles
  (#et : Type0) (nw cols : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v cols)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v cols)))
  requires
    forall+ (w : natlt (SZ.v nw)).
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0)
  ensures live a
{
  forevery_map
    (fun (w : natlt (SZ.v nw)) ->
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0))
    (fun w ->
      forall+ (tc : natlt 1).
        exists* (e : chest2 et 16 (SZ.v cols)).
          array2_subtile a 16 (SZ.v cols) w tc |-> Frac 1.0R e)
    fn w {
      unfold live (array2_subtile a 16 (SZ.v cols <: pos) w 0);
      with e. assert (
        array2_subtile a 16 (SZ.v cols) w 0 |-> Frac 1.0R e);
      forevery_singleton_intro
        (fun (tc : natlt 1) ->
          exists* (x : chest2 et 16 (SZ.v cols)).
            array2_subtile a 16 (SZ.v cols) w tc |-> Frac 1.0R x);
    };
  let ef = forevery_exists_2
    (fun (w : natlt (SZ.v nw)) (tc : natlt 1)
      (e : chest2 et 16 (SZ.v cols)) ->
      array2_subtile a 16 (SZ.v cols) w tc |-> Frac 1.0R e);
  forevery_rw_size2
    (SZ.v nw) (SZ.v nw * 16 / 16)
    1 (SZ.v cols / SZ.v cols)
    #(fun (w : natlt (SZ.v nw)) (tc : natlt 1) ->
      array2_subtile a 16 (SZ.v cols) w tc
        |-> Frac 1.0R (ef w tc));
  array2_untile' a 16 (SZ.v cols <: pos) ef;
  fold live a;
}

unfold
let flash_jt_local
  (#et_ab #et_acc : Type0)
  (d : szp { 16 /?+ SZ.v d })
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPV : layout2 16 16)
  (#lcw : layout1 16)
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shPV : array2 et_acc lPV)
  (shcw : array1 et_acc lcw)
  (lane : natlt BW.warp_size) : slprop =
  (exists* (e : chest2 et_ab
      (16 / warp_row_span) (SZ.v d / 16)).
    array2_stride_subtile shK warp_row_span 16
      (lane / 16) (lane % 16) |-> Frac 1.0R e)
  ** (exists* (e : chest2 et_ab
      (16 / warp_row_span) (SZ.v d / 16)).
    array2_stride_subtile shV warp_row_span 16
      (lane / 16) (lane % 16) |-> Frac 1.0R e)
  ** (exists* (e : chest2 et_acc 16 16).
    shS |-> Frac (1.0R /. BW.warp_size) e)
  ** when__ (lane < 16) (fun _ -> row_subtile shP lane)
  ** when__ (lane < 16) (fun _ -> cell_full shcw lane)
  ** (exists* (e : chest2 et_acc 16 16).
    shPV |-> Frac (1.0R /. BW.warp_size) e)

ghost
fn flash_split_jt
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  requires
    live v.shKv ** live v.shVv ** live v.shSv **
    live v.shPv ** live v.shPVv ** live v.shcwv
  ensures
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane
{
  flash_split_warp_tiles nw d v.shKv;
  forevery_map
    (fun w -> live (flash_warp_k v w))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_k v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    fn w { flash_warp_split_stride (flash_warp_k v w); };

  flash_split_warp_tiles nw d v.shVv;
  forevery_map
    (fun w -> live (flash_warp_v v w))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_v v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    fn w { flash_warp_split_stride (flash_warp_v v w); };

  flash_split_warp_tiles nw 16sz v.shSv;
  forevery_map
    (fun w -> live (flash_warp_s v w))
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_s v w |-> Frac
            (1.0R /. BW.warp_size) e)
    fn w {
      flash_share_tensor (flash_warp_s v w) BW.warp_size;
    };

  flash_split_warp_tiles nw 16sz v.shPv;
  forevery_map
    (fun w -> live (flash_warp_p v w))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          row_subtile (flash_warp_p v w) lane))
    fn w { flash_split_rows16 (flash_warp_p v w); };

  flash_split_warp_tiles nw 16sz v.shPVv;
  forevery_map
    (fun w -> live (flash_warp_pv v w))
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_pv v w |-> Frac
            (1.0R /. BW.warp_size) e)
    fn w {
      flash_share_tensor (flash_warp_pv v w) BW.warp_size;
    };

  flash_split_ml nw v.shcwv;

  forevery_zip4_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane));
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane))
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e);
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      ((exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane))
      **
      (when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e)))
    (fun w lane ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    fn w lane {
      fold flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane;
    };
}

ghost
fn flash_gather_jt
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane
  ensures
    live v.shKv ** live v.shVv ** live v.shSv **
    live v.shPv ** live v.shPVv ** live v.shcwv
{
  forevery_map_2
    (fun w lane ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e))
    fn w lane {
      unfold flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane;
    };
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane))
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e);

  forevery_map
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_k v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    (fun w -> live (flash_warp_k v w))
    fn w { flash_warp_gather_stride (flash_warp_k v w); };
  flash_gather_warp_tiles nw d v.shKv;

  forevery_map
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_v v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    (fun w -> live (flash_warp_v v w))
    fn w { flash_warp_gather_stride (flash_warp_v v w); };
  flash_gather_warp_tiles nw d v.shVv;

  forevery_map
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_s v w |-> Frac
            (1.0R /. BW.warp_size) e)
    (fun w -> live (flash_warp_s v w))
    fn w {
      flash_gather_tensor (flash_warp_s v w) BW.warp_size;
    };
  flash_gather_warp_tiles nw 16sz v.shSv;

  forevery_map
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          row_subtile (flash_warp_p v w) lane))
    (fun w -> live (flash_warp_p v w))
    fn w { flash_gather_rows16 (flash_warp_p v w); };
  flash_gather_warp_tiles nw 16sz v.shPv;

  forevery_map
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_pv v w |-> Frac
            (1.0R /. BW.warp_size) e)
    (fun w -> live (flash_warp_pv v w))
    fn w {
      flash_gather_tensor (flash_warp_pv v w) BW.warp_size;
    };
  flash_gather_warp_tiles nw 16sz v.shPVv;

  flash_gather_ml nw v.shcwv;
}

unfold
let flash_b0_local
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) : slprop =
  FD.strided_cells2 v.shQv (FD.block_threads nw)
    (w * BW.warp_size + lane) **
  FD.strided_cells2
    (array2_subtile v.shOv 16 (SZ.v d <: pos) w 0)
    BW.warp_size lane

ghost
fn flash_split_b0
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  requires live v.shQv ** live v.shOv
  ensures
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      flash_b0_local nw d v w lane
{
  flash_split_strided_cells2 v.shQv (FD.block_threads nw);
  forevery_factor (FD.block_threads nw)
    (SZ.v nw) BW.warp_size
    (fun tid ->
      FD.strided_cells2 v.shQv (FD.block_threads nw)
        tid);
  flash_split_warp_tiles nw d v.shOv;
  forevery_map
    (fun (w : natlt (SZ.v nw)) ->
      live (array2_subtile v.shOv
        16 (SZ.v d <: pos) w 0))
    (fun (w : natlt (SZ.v nw)) ->
      forall+ (lane : natlt BW.warp_size).
        FD.strided_cells2
          (array2_subtile v.shOv
            16 (SZ.v d <: pos) w 0)
          BW.warp_size lane)
    fn w {
      flash_split_strided_cells2
        (array2_subtile v.shOv
          16 (SZ.v d <: pos) w 0)
        BW.warp_size;
    };
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      FD.strided_cells2 v.shQv (FD.block_threads nw)
        (w * BW.warp_size + lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      FD.strided_cells2
        (array2_subtile v.shOv
          16 (SZ.v d <: pos) w 0)
        BW.warp_size lane);
  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      FD.strided_cells2 v.shQv (FD.block_threads nw)
        (w * BW.warp_size + lane) **
      FD.strided_cells2
        (array2_subtile v.shOv
          16 (SZ.v d <: pos) w 0)
        BW.warp_size lane)
    (fun w lane -> flash_b0_local nw d v w lane)
    fn w lane {
      fold flash_b0_local nw d v w lane;
    };
}

unfold
let flash_thread_pre
  (#et_ab #et_acc : Type0)
  (nw nthr : szp { SZ.v nthr == FD.block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  (w : natlt (SZ.v nw))
  (lane : natlt BW.warp_size) : slprop =
  (gQ |-> Frac fQ eQ) **
  (gK |-> Frac fK eK) **
  (gV |-> Frac fV eV) **
  (gmask |-> Frac fmask emask) **
  flash_b0_local nw d v w lane **
  when__ (lane < 16) (fun _ ->
    cell_full (FD.row v.shMv w) lane) **
  when__ (lane < 16) (fun _ ->
    cell_full (FD.row v.shLv w) lane) **
  flash_jt_local d
    (flash_warp_k v w) (flash_warp_v v w)
    (flash_warp_s v w) (flash_warp_p v w)
    (flash_warp_pv v w) (flash_warp_cw v w) lane **
  flash_combine_local nw v.shscalev v.shgmv v.shglv w lane **
  when_ (w = 0)
    (out_store_cells b hq sq 16sz d rows gout
      (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (SZ.v group)
      (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
      lane)

unfold
let flash_thread_post
  (#et_ab #et_acc : Type0)
  (nw nthr : szp { SZ.v nthr == FD.block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  (w : natlt (SZ.v nw))
  (lane : natlt BW.warp_size) : slprop =
  (gQ |-> Frac fQ eQ) **
  (gK |-> Frac fK eK) **
  (gV |-> Frac fV eV) **
  (gmask |-> Frac fmask emask) **
  (exists* (e : chest2 et_ab 16 (SZ.v d)).
    v.shQv |-> Frac (1.0R /. (FD.block_threads nw)) e) **
  FD.b1_post nw v.shMv v.shLv
    (w * BW.warp_size + lane
      <: natlt (FD.block_threads nw)) **
  FD.b2_post nw d v.shscalev v.shOv v.shglv
    (w * BW.warp_size + lane
      <: natlt (FD.block_threads nw)) **
  flash_jt_local d
    (flash_warp_k v w) (flash_warp_v v w)
    (flash_warp_s v w) (flash_warp_p v w)
    (flash_warp_pv v w) (flash_warp_cw v w) lane **
  when_ (w = 0)
    (out_store_cells b hq sq 16sz d rows gout
      (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (SZ.v group)
      (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
      lane) **
  when_ (w = 0 /\ lane < 16)
    (cell_full v.shgmv (FD.clamp_nat_lt 16 lane))

unfold
let flash_block_state
  (#et_ab : Type0)
  (nblk : szp)
  (b hq hkv group sq rows tiles sk d : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (bid : natlt (SZ.v nblk)) : slprop =
  (gQ |-> Frac (fQ /. (SZ.v nblk)) eQ) **
  (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
  (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
  (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
  flash_block_output b hq hkv group sq rows tiles d gout
    (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles))

let flash_w0_idx (nw : szp) : Type0 =
  wl : (natlt (SZ.v nw) & natlt BW.warp_size) { wl._1 = 0 }

let flash_w0_bij (nw : szp)
  : (flash_w0_idx nw =~ natlt BW.warp_size) =
{
  ff = (fun (wl : flash_w0_idx nw) -> wl._2)
    <: (flash_w0_idx nw -> GTot (natlt BW.warp_size));
  gg = (fun (lane : natlt BW.warp_size) -> (0, lane))
    <: (natlt BW.warp_size -> GTot (flash_w0_idx nw));
}

ghost
fn flash_output_add_warps
  (#et : Type0)
  (nw : szp)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  requires flash_block_output
    b hq hkv group sq rows tiles d gout bid
  ensures
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane)
{
  unfold flash_block_output
    b hq hkv group sq rows tiles d gout bid;
  forevery_map
    (fun lane ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        lane)
    (fun lane ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        ((flash_w0_bij nw).gg lane)._2)
    fn lane {
      rewrite each ((flash_w0_bij nw).gg lane)._2 as lane;
    };
  forevery_iso_back (flash_w0_bij nw)
    (fun wl ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        wl._2);
  forevery_unrefine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        wl._2)
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      wl._1 = 0);
  forevery_unflatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane));
}

ghost
fn flash_output_remove_warps
  (#et : Type0)
  (nw : szp)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane)
  ensures flash_block_output
    b hq hkv group sq rows tiles d gout bid
{
  forevery_flatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane));
  forevery_refine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        wl._2)
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      wl._1 = 0);
  forevery_iso (flash_w0_bij nw)
    (fun wl ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        wl._2);
  forevery_map
    (fun lane ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        ((flash_w0_bij nw).gg lane)._2)
    (fun lane ->
      out_store_cells b hq sq 16sz d rows gout
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        lane)
    fn lane {
      rewrite each ((flash_w0_bij nw).gg lane)._2 as lane;
    };
  fold flash_block_output
    b hq hkv group sq rows tiles d gout bid;
}

ghost
fn flash_gather_gm
  (#et : Type0)
  (nw : szp)
  (gm : array1 et (l1_forward 16))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when_ (w = 0 /\ lane < 16)
        (cell_full gm (FD.clamp_nat_lt 16 lane))
  ensures live gm
{
  forevery_flatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      when_ (w = 0 /\ lane < 16)
        (cell_full gm (FD.clamp_nat_lt 16 lane)));
  forevery_refine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      cell_full gm (FD.clamp_nat_lt 16 wl._2))
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      wl._1 = 0 /\ wl._2 < 16);
  forevery_rw_type
    (wl : (natlt (SZ.v nw) & natlt BW.warp_size) {
      wl._1 = 0 /\ wl._2 < 16})
    (flash_combine_idx nw)
    (fun wl -> cell_full gm (FD.clamp_nat_lt 16 wl._2));
  forevery_iso (flash_combine_bij nw)
    (fun wl -> cell_full gm (FD.clamp_nat_lt 16 wl._2));
  forevery_map #(natlt 16)
    (fun lane ->
      cell_full gm
        (FD.clamp_nat_lt 16 ((flash_combine_bij nw).gg lane)._2))
    (fun lane -> cell_full gm lane)
    fn lane {
      rewrite each
        (FD.clamp_nat_lt 16 ((flash_combine_bij nw).gg lane)._2)
        as lane;
    };
  forevery_map #(natlt 16)
    (fun lane -> cell_full gm lane)
    (fun lane ->
      exists* (x : et).
        Cell gm (idx1 lane) |-> Frac 1.0R x)
    fn lane { unfold cell_full gm lane; };
  flash_cells1_gather gm;
}

ghost
fn flash_unfactor_threads
  (nw : szp)
  (p : natlt (SZ.v nw) -> natlt BW.warp_size -> slprop)
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      p w lane
  ensures
    forall+ (tid : natlt (SZ.v nw * BW.warp_size)).
      p (tid / BW.warp_size) (tid % BW.warp_size)
{
  forevery_unfactor' (SZ.v nw * BW.warp_size)
    (SZ.v nw) BW.warp_size p;
}

ghost
fn flash_factor_threads
  (nw : szp)
  (p : natlt (SZ.v nw) -> natlt BW.warp_size -> slprop)
  requires
    forall+ (tid : natlt (SZ.v nw * BW.warp_size)).
      p (tid / BW.warp_size) (tid % BW.warp_size)
  ensures
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      p w lane
{
  forevery_factor' (SZ.v nw * BW.warp_size)
    (SZ.v nw) BW.warp_size p;
}

ghost
fn flash_b0_to_descriptor
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size)
  (tid : natlt (FD.block_threads nw))
  requires
    pure (FD.thread_w nw tid == w /\
          FD.thread_lane nw tid == lane) **
    flash_b0_local nw d v w lane
  ensures FD.b0_pre nw d v.shQv v.shOv tid
{
  unfold flash_b0_local nw d v w lane;
  rewrite each w as (FD.thread_w nw tid);
  rewrite each lane as (FD.thread_lane nw tid);
  FStar.Math.Lemmas.lemma_div_mod
    (tid <: nat) BW.warp_size;
  rewrite each
    (FD.thread_w nw tid * BW.warp_size +
      FD.thread_lane nw tid)
    as tid;
  fold FD.b0_pre nw d v.shQv v.shOv tid;
}

ghost
fn flash_cell_full_to_n
  (#et : Type0) (#l : layout1 16)
  (a : array1 et l) (i : natlt 16)
  requires cell_full a i
  ensures cell_full_n a i
{
  unfold cell_full a i;
  with x. assert (Cell a (idx1 i) |-> Frac 1.0R (x <: et));
  tensor_pts_to_cell_eq a (idx1 i) 1.0R x;
  rewrite
    (Cell a (idx1 i) |-> Frac 1.0R x)
    as
    (tensor_pts_to_cell a (idx1 i) x);
  fold cell_full_n a i;
}

ghost
fn flash_cell_full_from_n
  (#et : Type0) (#l : layout1 16)
  (a : array1 et l) (i : natlt 16)
  requires cell_full_n a i
  ensures cell_full a i
{
  unfold cell_full_n a i;
  with x. assert (tensor_pts_to_cell a (idx1 i) (x <: et));
  tensor_pts_to_cell_eq a (idx1 i) 1.0R x;
  rewrite
    (tensor_pts_to_cell a (idx1 i) x)
    as
    (Cell a (idx1 i) |-> Frac 1.0R x);
  fold cell_full a i;
}

ghost
fn flash_ml_to_pre
  (#et : Type0)
  (nw : szp)
  (m l : array2 et (l2_row_major (SZ.v nw) 16))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size)
  (ws : szlt nw) (ls : szlt BW.warp_size)
  requires
    pure (SZ.v ws == w /\ SZ.v ls == lane) **
    when__ (lane < 16) (fun _ ->
      cell_full (FD.row m w) lane) **
    when__ (lane < 16) (fun _ ->
      cell_full (FD.row l w) lane)
  ensures
    if_ (lane_active 16sz ls)
      (ml_cells 16sz (FD.row m (SZ.v ws))
        (FD.row l (SZ.v ws)) ls)
{
  rewrite each w as (SZ.v ws);
  rewrite each lane as (SZ.v ls);
  rewrite
    (when__ (SZ.v ls < 16) (fun _ ->
      cell_full (FD.row m (SZ.v ws)) (SZ.v ls)))
    as
    (when__ (lane_active 16sz ls) (fun _ ->
      cell_full (FD.row m (SZ.v ws)) (SZ.v ls)));
  rewrite
    (when__ (SZ.v ls < 16) (fun _ ->
      cell_full (FD.row l (SZ.v ws)) (SZ.v ls)))
    as
    (when__ (lane_active 16sz ls) (fun _ ->
      cell_full (FD.row l (SZ.v ws)) (SZ.v ls)));
  if lane_active 16sz ls {
    rewrite
      (when__ l_True (fun _ ->
        cell_full (FD.row m (SZ.v ws)) (SZ.v ls)))
      as
      (cell_full (FD.row m (SZ.v ws)) (SZ.v ls));
    rewrite
      (when__ l_True (fun _ ->
        cell_full (FD.row l (SZ.v ws)) (SZ.v ls)))
      as
      (cell_full (FD.row l (SZ.v ws)) (SZ.v ls));
    flash_cell_full_to_n
      (FD.row m (SZ.v ws)) (SZ.v ls);
    flash_cell_full_to_n
      (FD.row l (SZ.v ws)) (SZ.v ls);
    assert pure (
      SZ.v (clamp_lt 16sz ls) == SZ.v ls);
    rewrite each (SZ.v ls) as
      (SZ.v (clamp_lt 16sz ls));
    fold ml_cells 16sz (FD.row m (SZ.v ws))
      (FD.row l (SZ.v ws)) ls;
    if_intro_true (
      ml_cells 16sz (FD.row m (SZ.v ws))
        (FD.row l (SZ.v ws)) ls);
  } else {
    rewrite
      (when__ l_False (fun _ ->
        cell_full (FD.row m (SZ.v ws)) (SZ.v ls)))
      as emp;
    rewrite
      (when__ l_False (fun _ ->
        cell_full (FD.row l (SZ.v ws)) (SZ.v ls)))
      as emp;
    if_intro_false (
      ml_cells 16sz (FD.row m (SZ.v ws))
        (FD.row l (SZ.v ws)) ls);
  }
}

ghost
fn flash_combine_to_pre
  (#et : Type0)
  (nw : szp)
  (scale : array2 et (l2_row_major (SZ.v nw) 16))
  (gm gl : array1 et (l1_forward 16))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size)
  (ws : szlt nw) (ls : szlt BW.warp_size)
  requires
    pure (SZ.v ws == w /\ SZ.v ls == lane) **
    flash_combine_local nw scale gm gl w lane
  ensures
    if_ (combine_active 16sz ws ls)
      (combine_cells nw 16sz scale gm gl ls)
{
  unfold flash_combine_local nw scale gm gl w lane;
  rewrite each w as (SZ.v ws);
  rewrite each lane as (SZ.v ls);
  rewrite
    (when_ (SZ.v ws = 0 /\ SZ.v ls < 16)
      ((exists* (e : chest2 et (SZ.v nw) 1).
         tensor_pts_to
          (array2_stride_subtile scale 1 16 0
            (FD.clamp_nat_lt 16 (SZ.v ls))) e)
       ** cell_full gm (FD.clamp_nat_lt 16 (SZ.v ls))
       ** cell_full gl (FD.clamp_nat_lt 16 (SZ.v ls))))
    as
    (if_ (combine_active 16sz ws ls)
      (combine_cells nw 16sz scale gm gl ls));
}

ghost
fn flash_output_to_pre
  (#et : Type0)
  (b hq sq rows d : szp)
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bi : natlt (SZ.v b)) (kvh : nat) (group : pos)
  (r0 : nat)
  (w : nat) (lane : natlt BW.warp_size)
  (ws : sz) (ls : szlt BW.warp_size)
  requires
    pure (SZ.v ws == w /\ SZ.v ls == lane) **
    when_ (w = 0)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 lane)
  ensures
    if_ (ws = 0sz)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 (SZ.v ls))
{
  rewrite each w as (SZ.v ws);
  rewrite each lane as (SZ.v ls);
  rewrite
    (when_ (SZ.v ws = 0)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 (SZ.v ls)))
    as
    (if_ (ws = 0sz)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 (SZ.v ls)));
}

ghost
fn flash_output_from_post
  (#et : Type0)
  (b hq sq rows d : szp)
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bi : natlt (SZ.v b)) (kvh : nat) (group : pos)
  (r0 : nat)
  (w : nat) (lane : natlt BW.warp_size)
  (ws : sz) (ls : szlt BW.warp_size)
  requires
    pure (SZ.v ws == w /\ SZ.v ls == lane) **
    if_ (ws = 0sz)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 (SZ.v ls))
  ensures
    when_ (w = 0)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 lane)
{
  rewrite
    (if_ (ws = 0sz)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 (SZ.v ls)))
    as
    (when_ (SZ.v ws = 0)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 (SZ.v ls)));
  rewrite
    (when_ (SZ.v ws = 0)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 (SZ.v ls)))
    as
    (when_ (w = 0)
      (out_store_cells b hq sq 16sz d rows gout
        bi kvh group r0 lane));
}

ghost
fn flash_gm_from_post
  (#et : Type0)
  (gm : array1 et (l1_forward 16))
  (w : nat) (lane : natlt BW.warp_size)
  (ws : sz) (ls : szlt BW.warp_size)
  requires
    pure (SZ.v ws == w /\ SZ.v ls == lane) **
    if_ (combine_active 16sz ws ls)
      (cell_full gm (SZ.v (clamp_lt 16sz ls)))
  ensures
    when_ (w = 0 /\ lane < 16)
      (cell_full gm (FD.clamp_nat_lt 16 lane))
{
  if combine_active 16sz ws ls {
    if_elim_true (
      cell_full gm (SZ.v (clamp_lt 16sz ls)));
    assert pure (w = 0 /\ lane < 16);
    assert pure (
      SZ.v (clamp_lt 16sz ls) ==
      FD.clamp_nat_lt 16 lane);
    rewrite
      (cell_full gm (SZ.v (clamp_lt 16sz ls)))
      as
      (cell_full gm (FD.clamp_nat_lt 16 lane));
    flash_when_prop_intro_true
      (w = 0 /\ lane < 16)
      (t2b (w = 0 /\ lane < 16))
      (cell_full gm (FD.clamp_nat_lt 16 lane));
  } else {
    if_elim_false (
      cell_full gm (SZ.v (clamp_lt 16sz ls)));
    assert pure (~(w = 0 /\ lane < 16));
    flash_when_prop_intro_false
      (w = 0 /\ lane < 16)
      (t2b (w = 0 /\ lane < 16))
      (cell_full gm (FD.clamp_nat_lt 16 lane));
  }
}

ghost
fn flash_gather_thread_tensor
  (#et : Type0) (#r : nat) (#ds : shape r)
  (#l : tlayout ds)
  (nw nthr : szp {
    SZ.v nthr == SZ.v nw * BW.warp_size })
  (a : tensor et l)
  (#f : perm) (#e : chest ds et)
  requires
    forall+ (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size).
      a |-> Frac (f /. (SZ.v nthr)) e
  ensures a |-> Frac f e
{
  flash_unfactor_threads nw
    (fun _w _lane ->
      a |-> Frac (f /. (SZ.v nthr)) e);
  forevery_map
    (fun (_ : natlt (SZ.v nw * BW.warp_size)) ->
      a |-> Frac (f /. (SZ.v nthr)) e)
    (fun (_ : natlt (SZ.v nw * BW.warp_size)) ->
      a |-> Frac
        (f /. (SZ.v nw * BW.warp_size)) e)
    fn _ {
      rewrite
        (a |-> Frac (f /. (SZ.v nthr)) e)
        as
        (a |-> Frac
          (f /. (SZ.v nw * BW.warp_size)) e);
    };
  tensor_gather_n a (SZ.v nw * BW.warp_size) #f;
}

ghost
fn flash_setup
  (#et_ab : Type0)
  (nblk : szp)
  (b hq hkv group sq rows tiles sk d : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  ()
  norewrite
  requires
    (gQ |-> Frac fQ eQ) **
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (gmask |-> Frac fmask emask) **
    live gout
  ensures
    (forall+ (bid : natlt (SZ.v nblk)).
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid) **
    emp
{
  tensor_share_n gQ (SZ.v nblk);
  tensor_share_n gK (SZ.v nblk);
  tensor_share_n gV (SZ.v nblk);
  tensor_share_n gmask (SZ.v nblk);
  flash_split_output b hq hkv group sq rows tiles d gout;
  forevery_rw_size
    (SZ.v b * SZ.v hkv * SZ.v tiles) (SZ.v nblk)
    #(fun bid ->
      flash_block_output b hq hkv group sq rows tiles d gout bid);
  forevery_map
    (fun (bid : natlt (SZ.v nblk)) ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane)
    (fun (bid : natlt (SZ.v nblk)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)))
    fn bid {
      fold flash_block_output
        b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles));
    };
  forevery_zip
    (fun _ -> gmask |-> Frac (fmask /. (SZ.v nblk)) emask)
    (fun (bid : natlt (SZ.v nblk)) ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane);
  forevery_zip
    (fun _ -> gV |-> Frac (fV /. (SZ.v nblk)) eV)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_zip
    (fun _ -> gK |-> Frac (fK /. (SZ.v nblk)) eK)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_zip
    (fun _ -> gQ |-> Frac (fQ /. (SZ.v nblk)) eQ)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_map #(natlt (SZ.v nblk))
    (fun bid ->
      (gQ |-> Frac (fQ /. (SZ.v nblk)) eQ) **
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane))
    (fun bid ->
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid)
    fn bid {
      fold flash_block_output
        b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles));
      fold flash_block_state nblk
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid;
    };
}

ghost
fn flash_teardown
  (#et_ab : Type0)
  (nblk : szp)
  (b hq hkv group sq rows tiles sk d : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lout |}
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (SZ.v nblk)).
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid) **
    emp
  ensures
    (gQ |-> Frac fQ eQ) **
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (gmask |-> Frac fmask emask) **
    live gout
{
  assert pure (SZ.fits (tlayout_ulen lout));
  forevery_map #(natlt (SZ.v nblk))
    (fun bid ->
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid)
    (fun bid ->
      (gQ |-> Frac (fQ /. (SZ.v nblk)) eQ) **
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)))
    fn bid {
      unfold flash_block_state nblk
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid;
    };
  forevery_unzip
    (fun _ -> gQ |-> Frac (fQ /. (SZ.v nblk)) eQ)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_unzip
    (fun _ -> gK |-> Frac (fK /. (SZ.v nblk)) eK)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_unzip
    (fun _ -> gV |-> Frac (fV /. (SZ.v nblk)) eV)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_unzip
    (fun _ -> gmask |-> Frac (fmask /. (SZ.v nblk)) emask)
    (fun (bid : natlt (SZ.v nblk)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  tensor_gather_n gQ (SZ.v nblk) #fQ;
  tensor_gather_n gK (SZ.v nblk) #fK;
  tensor_gather_n gV (SZ.v nblk) #fV;
  tensor_gather_n gmask (SZ.v nblk) #fmask;
  forevery_rw_size
    (SZ.v nblk) (SZ.v b * SZ.v hkv * SZ.v tiles)
    #(fun (bid : natlt (SZ.v nblk)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_map
    (fun (bid : natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (fun bid ->
      flash_block_output b hq hkv group sq rows tiles d gout bid)
    fn bid {
      rewrite
        (flash_block_output b hq hkv group sq rows tiles d gout
          (bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)))
        as
        (flash_block_output b hq hkv group sq rows tiles d gout bid);
    };
  flash_gather_output b hq hkv group sq rows tiles d gout;
}

ghost
fn flash_block_setup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| scalar et_acc |}
  (nblk : szp)
  (nw nthr : szp {
    SZ.v nthr == FD.block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  (bid : natlt (SZ.v nblk))
  ()
  norewrite
  requires
    live_c_shmems sh **
    flash_block_state nblk b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask bid
  ensures
    (forall+ (tid : natlt (SZ.v nthr)).
      flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout (flash_views_of nw d sh)
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        (tid / BW.warp_size) (tid % BW.warp_size)) **
    emp
{
  admit();
  unfold flash_block_state nblk
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout #fQ #fK #fV #fmask
    #eQ #eK #eV #emask bid;
  tensor_share_n gQ (SZ.v nthr);
  tensor_share_n gK (SZ.v nthr);
  tensor_share_n gV (SZ.v nthr);
  tensor_share_n gmask (SZ.v nthr);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask);
  forevery_zip4_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask);

  flash_output_add_warps nw
    b hq hkv group sq rows tiles d gout
    (bid <: natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles));
  flash_open_shmems nw d sh;
  let v = flash_views_of nw d sh;
  rewrite each (flash_views_of nw d sh) as v;
  unfold flash_views_live v;
  flash_split_b0 nw d v;
  flash_split_ml nw v.shMv;
  flash_split_ml nw v.shLv;
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (FD.row v.shMv w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (FD.row v.shLv w) lane));
  flash_split_jt nw d v;
  flash_split_combine nw v.shscalev v.shgmv v.shglv;
  forevery_zip4_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_b0_local nw d v w lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (FD.row v.shMv w) lane) **
      when__ (lane < 16) (fun _ ->
        cell_full (FD.row v.shLv w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_combine_local nw
        v.shscalev v.shgmv v.shglv w lane);
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
      (gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
      (gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
      (gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_b0_local nw d v w lane **
      (when__ (lane < 16) (fun _ ->
        cell_full (FD.row v.shMv w) lane) **
       when__ (lane < 16) (fun _ ->
        cell_full (FD.row v.shLv w) lane)) **
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane **
      flash_combine_local nw
        v.shscalev v.shgmv v.shglv w lane);
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      ((gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
       (gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
       (gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
       (gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask)) **
      (flash_b0_local nw d v w lane **
       (when__ (lane < 16) (fun _ ->
          cell_full (FD.row v.shMv w) lane) **
        when__ (lane < 16) (fun _ ->
          cell_full (FD.row v.shLv w) lane)) **
       flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane **
       flash_combine_local nw
        v.shscalev v.shgmv v.shglv w lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (((gQ |-> Frac
          (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
        (gK |-> Frac
          (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
        (gV |-> Frac
          (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
        (gmask |-> Frac
          (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask)) **
       (flash_b0_local nw d v w lane **
        (when__ (lane < 16) (fun _ ->
           cell_full (FD.row v.shMv w) lane) **
         when__ (lane < 16) (fun _ ->
           cell_full (FD.row v.shLv w) lane)) **
        flash_jt_local d
          (flash_warp_k v w) (flash_warp_v v w)
          (flash_warp_s v w) (flash_warp_p v w)
          (flash_warp_pv v w) (flash_warp_cw v w) lane **
        flash_combine_local nw
          v.shscalev v.shgmv v.shglv w lane)) **
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane)
    fn w lane {
      fold flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane;
    };
  rewrite each v as (flash_views_of nw d sh);
  flash_unfactor_threads nw
    (fun w lane ->
      flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout (flash_views_of nw d sh)
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane);
  forevery_rw_size
    (SZ.v nw * BW.warp_size) (SZ.v nthr)
    #(fun (tid : natlt
      (SZ.v nw * BW.warp_size)) ->
      flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout (flash_views_of nw d sh)
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        (tid / BW.warp_size) (tid % BW.warp_size));
}

ghost
fn flash_block_teardown
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| scalar et_acc |}
  (nblk : szp)
  (nw nthr : szp {
    SZ.v nthr == FD.block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  (bid : natlt (SZ.v nblk))
  ()
  norewrite
  requires
    (forall+ (tid : natlt (SZ.v nthr)).
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout (flash_views_of nw d sh)
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        (tid / BW.warp_size) (tid % BW.warp_size)) **
    emp
  ensures
    live_c_shmems sh **
    flash_block_state nblk b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask bid
{
  admit ();
  let v = flash_views_of nw d sh;
  rewrite each (flash_views_of nw d sh) as v;
  forevery_rw_size
    (SZ.v nthr) (SZ.v nw * BW.warp_size)
    #(fun (tid : natlt (SZ.v nthr)) ->
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        (tid / BW.warp_size) (tid % BW.warp_size));
  flash_factor_threads nw
    (fun w lane ->
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane);
  forevery_map_2
    (fun w lane ->
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
      (gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
      (gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
      (gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask) **
      (exists* (e : chest2 et_ab 16 (SZ.v d)).
        v.shQv |-> Frac
          (1.0R /. (FD.block_threads nw)) e) **
      FD.b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw)) **
      FD.b2_post nw d v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw)) **
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane **
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane) **
      when_ (w = 0 /\ lane < 16)
        (cell_full v.shgmv (FD.clamp_nat_lt 16 lane)))
    fn w lane {
      unfold flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane;
    };
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab 16 (SZ.v d)).
        v.shQv |-> Frac
          (1.0R /. (FD.block_threads nw)) e)
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      FD.b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw)))
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      FD.b2_post nw d v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw)))
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane))
    _;

  flash_gather_thread_tensor nw nthr gQ
    #(fQ /. (SZ.v nblk)) #eQ;
  flash_gather_thread_tensor nw nthr gK
    #(fK /. (SZ.v nblk)) #eK;
  flash_gather_thread_tensor nw nthr gV
    #(fV /. (SZ.v nblk)) #eV;
  flash_gather_thread_tensor nw nthr gmask
    #(fmask /. (SZ.v nblk)) #emask;

  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_ab 16 (SZ.v d)).
        v.shQv |-> Frac
          (1.0R /. (FD.block_threads nw)) e);
  flash_gather_tensor v.shQv (FD.block_threads nw);

  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      FD.b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw)))
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shMv |-> Frac
          (1.0R /. (FD.block_threads nw)) e) **
      (exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shLv |-> Frac
          (1.0R /. (FD.block_threads nw)) e))
    fn w lane {
      unfold FD.b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw));
    };
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shMv |-> Frac
          (1.0R /. (FD.block_threads nw)) e)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shLv |-> Frac
          (1.0R /. (FD.block_threads nw)) e);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shMv |-> Frac
          (1.0R /. (FD.block_threads nw)) e);
  flash_gather_tensor v.shMv (FD.block_threads nw);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shLv |-> Frac
          (1.0R /. (FD.block_threads nw)) e);
  flash_gather_tensor v.shLv (FD.block_threads nw);

  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      FD.b2_post nw d v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw)))
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shscalev |-> Frac
          (1.0R /. (FD.block_threads nw)) e) **
      (exists* (e : chest2 et_acc
        (SZ.v nw * 16) (SZ.v d)).
        v.shOv |-> Frac
          (1.0R /. (FD.block_threads nw)) e) **
      (exists* (e : chest1 et_acc 16).
        v.shglv |-> Frac
          (1.0R /. (FD.block_threads nw)) e))
    fn w lane {
      unfold FD.b2_post nw d
        v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (FD.block_threads nw));
    };
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shscalev |-> Frac
          (1.0R /. (FD.block_threads nw)) e)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc
        (SZ.v nw * 16) (SZ.v d)).
        v.shOv |-> Frac
          (1.0R /. (FD.block_threads nw)) e)
    _;
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shscalev |-> Frac
          (1.0R /. (FD.block_threads nw)) e);
  flash_gather_tensor v.shscalev (FD.block_threads nw);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc
        (SZ.v nw * 16) (SZ.v d)).
        v.shOv |-> Frac
          (1.0R /. (FD.block_threads nw)) e);
  flash_gather_tensor v.shOv (FD.block_threads nw);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest1 et_acc 16).
        v.shglv |-> Frac
          (1.0R /. (FD.block_threads nw)) e);
  flash_gather_tensor v.shglv (FD.block_threads nw);

  flash_gather_jt nw d v;
  flash_output_remove_warps nw
    b hq hkv group sq rows tiles d gout
    (bid <: natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles));
  flash_gather_gm nw v.shgmv;

  fold flash_views_live v;
  rewrite each v as (flash_views_of nw d sh);
  flash_close_shmems nw d sh;
  fold flash_block_state nblk
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout #fQ #fK #fV #fmask
    #eQ #eK #eV #emask bid;
}

inline_for_extraction noextract
fn sdpa_flash_thread
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (nblk : szp)
  (nw nthr : szp {
    SZ.v nthr == FD.block_threads nw })
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
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| ctlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : array4 et_ab lgmask)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (scale : et_acc)
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
      (FD.barrier_contract nw d
        (flash_views_of nw d sh).shQv
        (flash_views_of nw d sh).shMv
        (flash_views_of nw d sh).shLv
        (flash_views_of nw d sh).shscalev
        (flash_views_of nw d sh).shOv
        (flash_views_of nw d sh).shglv) **
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
      (SZ.v bid <: natlt
        (SZ.v b * SZ.v hkv * SZ.v tiles))
      (SZ.v tid / BW.warp_size)
      (SZ.v tid % BW.warp_size) **
    thread_id (SZ.v nthr) tid **
    block_id (SZ.v nblk) bid **
    B.barrier_tok
      (FD.barrier_contract nw d
        (flash_views_of nw d sh).shQv
        (flash_views_of nw d sh).shMv
        (flash_views_of nw d sh).shLv
        (flash_views_of nw d sh).shscalev
        (flash_views_of nw d sh).shOv
        (flash_views_of nw d sh).shglv) **
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
  flash_b0_to_descriptor nw d (flash_views_of nw d sh)
    (SZ.v w) (SZ.v lane)
    (SZ.v tid <: natlt (FD.block_threads nw));
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
    (flash_views_of nw d sh).shMv (flash_views_of nw d sh).shLv (flash_views_of nw d sh).shscalev (flash_views_of nw d sh).shOv (flash_views_of nw d sh).shgmv (flash_views_of nw d sh).shglv
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
    tid bi r0 group kvh causal scale;

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
    (flash_views_of nw d sh).shMv (flash_views_of nw d sh).shLv (flash_views_of nw d sh).shscalev (flash_views_of nw d sh).shOv (flash_views_of nw d sh).shglv
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
  flash_output_from_post b hq sq rows d gout
    (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0)
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
      (out_store_cells b hq sq 16sz d rows gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0)
        (SZ.v (sdpa_flash_lane nw nthr tid))))
    as
    (when_ (SZ.v (sdpa_flash_w nw nthr tid) = 0)
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
    (SZ.v bid <: natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles))
    (SZ.v tid / BW.warp_size)
    (SZ.v tid % BW.warp_size);
}

inline_for_extraction noextract
let sdpa_flash_kd
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (nblk : szp { SZ.v nblk <= max_blocks })
  (nw nthr : szp {
    SZ.v nthr == FD.block_threads nw /\
    SZ.v nthr <= max_threads })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.fits (SZ.v rows + 15) /\
    SZ.v tiles == (SZ.v rows + 15) / 16 /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| ctlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : array4 et_ab lgmask { Kuiper.Tensor.is_global gmask })
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (scale : et_acc)
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
  : kernel_desc
      ((gQ |-> Frac fQ eQ) **
       (gK |-> Frac fK eK) **
       (gV |-> Frac fV eV) **
       (gmask |-> Frac fmask emask) **
       live gout)
      ((gQ |-> Frac fQ eQ) **
       (gK |-> Frac fK eK) **
       (gV |-> Frac fV eV) **
       (gmask |-> Frac fmask emask) **
       live gout)
= {
  nblk;
  nthr;
  shmems_desc = flash_shmems et_ab et_acc nw d;
  barrier_contract = (fun _bid sh ->
    FD.barrier_contract nw d
      (flash_views_of nw d sh).shQv
      (flash_views_of nw d sh).shMv
      (flash_views_of nw d sh).shLv
      (flash_views_of nw d sh).shscalev
      (flash_views_of nw d sh).shOv
      (flash_views_of nw d sh).shglv);
  barrier_count = (fun _ -> 3);
  barrier_ok = (fun _bid sh ->
    FD.barrier_ok nw d
      (flash_views_of nw d sh).shQv
      (flash_views_of nw d sh).shMv
      (flash_views_of nw d sh).shLv
      (flash_views_of nw d sh).shscalev
      (flash_views_of nw d sh).shOv
      (flash_views_of nw d sh).shglv);
  frame = emp;
  block_pre = (fun bid ->
    flash_block_state nblk
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask bid);
  block_post = (fun bid ->
    flash_block_state nblk
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask bid);
  setup = flash_setup nblk
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout
    #fQ #fK #fV #fmask #eQ #eK #eV #emask;
  teardown = flash_teardown nblk
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout
    #fQ #fK #fV #fmask #eQ #eK #eV #emask;
  block_frame = (fun _sh _bid -> emp);
  block_setup = flash_block_setup nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout
    #fQ #fK #fV #fmask #eQ #eK #eV #emask;
  block_teardown = flash_block_teardown nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout
    #fQ #fK #fV #fmask #eQ #eK #eV #emask;
  kpre = (fun sh bid tid ->
    flash_thread_pre nw nthr
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout (flash_views_of nw d sh)
      #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
      #(fK /. (SZ.v nblk) /. (SZ.v nthr))
      #(fV /. (SZ.v nblk) /. (SZ.v nthr))
      #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
      #eQ #eK #eV #emask
      (bid <: natlt
        (SZ.v b * SZ.v hkv * SZ.v tiles))
      (tid / BW.warp_size) (tid % BW.warp_size));
  kpost = (fun sh bid tid ->
    flash_thread_post nw nthr
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout (flash_views_of nw d sh)
      #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
      #(fK /. (SZ.v nblk) /. (SZ.v nthr))
      #(fV /. (SZ.v nblk) /. (SZ.v nthr))
      #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
      #eQ #eK #eV #emask
      (bid <: natlt
        (SZ.v b * SZ.v hkv * SZ.v tiles))
      (tid / BW.warp_size) (tid % BW.warp_size));
  f = sdpa_flash_thread nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout causal scale;
  block_pre_sendable = (fun _ -> magic());
  block_post_sendable = (fun _ -> magic());
  kpre_sendable = (fun _ _ _ _ -> magic());
  kpost_sendable = (fun _ _ _ _ -> magic());
}

inline_for_extraction noextract
fn sdpa_flash_async
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  // TODO: why do we have an nblk? why is it not just b * hkv * tiles etc.?
  (nblk : szp { SZ.v nblk <= max_blocks })
  // And an nthr?
  (nw nthr : szp {
    SZ.v nthr == FD.block_threads nw /\
    SZ.v nthr <= max_threads })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.fits (SZ.v rows + 15) /\
    SZ.v tiles == (SZ.v rows + 15) / 16 /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| ctlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : array4 et_ab lgmask { Kuiper.Tensor.is_global gmask })
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (scale : et_acc)
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
  (s : stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (
      (gQ |-> Frac fQ eQ) **
      (gK |-> Frac fK eK) **
      (gV |-> Frac fV eV) **
      (gmask |-> Frac fmask emask) **
      live gout)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (
        (gQ |-> Frac fQ eQ) **
        (gK |-> Frac fK eK) **
        (gV |-> Frac fV eV) **
        (gmask |-> Frac fmask emask) **
        live gout))
{
  launch (sdpa_flash_kd nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout causal scale) s;
}
