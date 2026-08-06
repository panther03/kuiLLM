module Kuiops.Sdpa.Flash.Kf

(* [sdpa_flash_kf]: the per-thread kernel body, composed from the warp-level
   primitives in [KfSub] and the block-level phases in [KfBlock]/[KfBarrier]. *)

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
open Kuiops.Sdpa.Flash.KfBlock
open Kuiops.Sdpa.Flash.KfBarrier

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module Trade = Pulse.Lib.Trade
open Kuiper.TensorRO { vtlayout_of_tlayout }

inline_for_extraction noextract
fn sdpa_flash_kf
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (nw nthr d sk : szp { SZ.v nthr == block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| TRO.cvtlayout lgmask |} {| ctlayout lout |} {| ctlayout lcw |}
  {| ctlayout lK |} {| ctlayout lV |} {| ctlayout lS |}
  {| ctlayout lP |} {| ctlayout lPVc |}
  {| strided_row_major (vtlayout_of_tlayout lK) |} {| strided_row_major (vtlayout_of_tlayout lV) |}
  {| strided_row_major (vtlayout_of_tlayout lS) |} {| strided_row_major (vtlayout_of_tlayout lP) |}
  {| strided_row_major (vtlayout_of_tlayout lPVc) |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array2 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array2 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : TRO.roarray4 et_ab lgmask)
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
  (causal : bool) (has_mask : bool) (scale : et_acc)
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


  sdpa_flash_block_prologue nw nthr d b hq sq rows
    gQ shQ shM shL shscale shO shgl tid bi r0 group kvh;

  with eQsh. assert (
    shQ |-> Frac (1.0R /. (SZ.v nthr)) eQsh);

  assert (jt_rest #et_ab #et_acc d sk b hq sq
    shK shV shS shP
    (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v (tid /^ 32sz)) 0)
    shQ shPVc shcw
    (row shM (SZ.v (tid /^ 32sz))) (row shL (SZ.v (tid /^ 32sz)))
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
        thread_id (block_threads nw) tid **
        B.barrier_tok (barrier_contract nw d shQ shM shL shscale shO shgl) **
        B.barrier_state 1 **
        (gQ |-> Frac fQ eQ) **
        jt_rest #et_ab #et_acc d sk b hq sq
          shK shV shS shP
          (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v (tid /^ 32sz)) 0)
          shQ shPVc shcw
          (row shM (SZ.v (tid /^ 32sz))) (row shL (SZ.v (tid /^ 32sz)))
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
      (row shM (SZ.v (tid /^ 32sz))) (row shL (SZ.v (tid /^ 32sz)))
      gK gV gmask bi qh qpos k0 cbound row_active causal has_mask scale;
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
    (row shM (SZ.v (tid /^ 32sz))) (row shL (SZ.v (tid /^ 32sz)))
    gK gV gmask
    #(1.0R /. (SZ.v nthr)) #fKg #fVg #fmask #eQsh #eKg #eVg #emask
    (SZ.v (tid %^ 32sz));
  assert pure (thread_w nw (SZ.v tid) == SZ.v (tid /^ 32sz));
  assert pure (thread_lane nw (SZ.v tid) == SZ.v (tid %^ 32sz));
  block_row_cell_reindex shM
    (SZ.v (tid /^ 32sz) <: natlt (SZ.v nw))
    (thread_w nw (SZ.v tid))
    (SZ.v (tid %^ 32sz) <: natlt BW.warp_size)
    (thread_lane nw (SZ.v tid));
  block_row_cell_reindex shL
    (SZ.v (tid /^ 32sz) <: natlt (SZ.v nw))
    (thread_w nw (SZ.v tid))
    (SZ.v (tid %^ 32sz) <: natlt BW.warp_size)
    (thread_lane nw (SZ.v tid));
  fold b1_pre nw shM shL (SZ.v tid);
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
  assert pure (thread_w nw (SZ.v tid) == SZ.v (tid /^ 32sz));
  assert pure (thread_lane nw (SZ.v tid) == SZ.v (tid %^ 32sz));
  flash_b2_scale_to_descriptor nw shscale shgl
    (tid /^ 32sz) (tid %^ 32sz)
    (tid <: szlt (block_threads nw));
  block_o_tile_reindex nw d shO
    (SZ.v (tid /^ 32sz) <: natlt (SZ.v nw))
    (thread_w nw (SZ.v tid))
    (SZ.v (tid %^ 32sz) <: natlt BW.warp_size)
    (thread_lane nw (SZ.v tid));
  fold b2_pre nw d shscale shO shgl (SZ.v tid);
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
