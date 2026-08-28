module Kuiops.Sdpa.Flash.KfBarrier

(* Block-level barrier phases of [sdpa_flash_kf]: the prologue, the two
   inter-phase barriers, and the reindexing that relates the block's view of
   the shared tiles to the barrier descriptor's. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiops.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Floating
open Kuiper.Shape
open Kuiper.Bijection
open Kuiper.Tensor.Layout.Bijection
open Pulse.Lib.Trade
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiper.ForEvery
open Kuiper.Ghost.TensorTranspose
open Kuiper.EMatrix
open Kuiops.Sdpa.Flash.KfSub
open Kuiops.Sdpa.Flash.KfBlock
open Kuiops.Sdpa.Flash.Vals
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module Trade = Pulse.Lib.Trade
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float
module SS = Kuiops.Sdpa.Flash.Spec.Step
module FT = Kuiops.Sdpa.Flash.Types

#push-options "--fuel 2 --ifuel 2"

let barrier_stage0_def
  (#et_ab #et_acc : Type0) {| scalar et_acc |}
  (nw d : szp) (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (eQsh : chest2 et_ab 16 (SZ.v d))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (eM eL : chest2 et_acc (SZ.v nw) 16)
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (escale : chest2 et_acc (SZ.v nw) 16)
  (eO : chest2 et_acc (SZ.v nw * 16) (SZ.v d))
  (egl : chest1 et_acc 16)
  (tid : natlt (block_threads nw))
  : Lemma
      ((barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl
          escale eO egl).rin 0 tid == b0_pre nw d shQ eQsh shO tid /\
       (barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl
          escale eO egl).rout 0 tid == b0_post nw d shQ eQsh shO tid)
= ()

#pop-options

(* Ownership of the row-major cells visited by
   [for (idx = tid; idx < rows*cols; idx += nthr)]. *)

(* The prologue initializes M/L through q_load's generic [if_] predicate,
   whereas the per-warp tile body uses [when__].  They describe the same
   active lanes for the fixed tensor-core row extent. *)
ghost
fn raw_cell_to_cell
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l) (i : natlt len) (#v : erased et)
  requires tensor_pts_to_cell a (idx1 i) v
  ensures Cell a (idx1 i) |-> Frac 1.0R v
{
  unfold tensor_pts_to_cell a (idx1 i) v;
  tensor_pts_to_cell_eq a (idx1 i) 1.0R v;
}

ghost
fn ml_if_to_when_v
  (#et : Type0)
  (#lm #ll : layout1 16)
  (shm : array1 et lm) (shl : array1 et ll)
  (lane : szlt warp_size) (vm vl : et)
  requires
    if_ (lane_active 16sz lane) (ml_cells_v 16sz shm shl lane vm vl)
  ensures
    when__ (SZ.v lane < 16) (fun _ -> cell_full_v shm (SZ.v lane) vm) **
    when__ (SZ.v lane < 16) (fun _ -> cell_full_v shl (SZ.v lane) vl)
{
  let active = lane <^ 16sz;
  if active {
    if_elim_true (ml_cells_v 16sz shm shl lane vm vl);
    unfold (ml_cells_v 16sz shm shl lane vm vl);
    unfold (cell_full_n_v shm (SZ.v (clamp_lt 16sz lane)) vm);
    unfold (cell_full_n_v shl (SZ.v (clamp_lt 16sz lane)) vl);
    assert pure (SZ.v (clamp_lt 16sz lane) == SZ.v lane);
    rewrite each (SZ.v (clamp_lt 16sz lane)) as (SZ.v lane);
    raw_cell_to_cell shm (SZ.v lane) #vm;
    raw_cell_to_cell shl (SZ.v lane) #vl;
    fold (cell_full_v shm (SZ.v lane) vm);
    fold (cell_full_v shl (SZ.v lane) vl);
    when__intro_true (SZ.v lane < 16) (fun _ -> cell_full_v shm (SZ.v lane) vm);
    when__intro_true (SZ.v lane < 16) (fun _ -> cell_full_v shl (SZ.v lane) vl);
  } else {
    if_elim_false (ml_cells_v 16sz shm shl lane vm vl);
    assert pure ((SZ.v lane < 16) == false);
    when__intro_false (SZ.v lane < 16) (fun _ -> cell_full_v shm (SZ.v lane) vm);
    when__intro_false (SZ.v lane < 16) (fun _ -> cell_full_v shl (SZ.v lane) vl);
  }
}

inline_for_extraction noextract
fn sdpa_flash_block_prologue
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| floating et_acc |}
  (nw nthr d : szp { SZ.v nthr == block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  {| ctlayout lgQ |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (bi : szlt b) (r0 : sz) (group : szp) (kvh : sz)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (16 /?+ (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (16 * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (16 * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.fits (SZ.v r0 + 16)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fQ : perm)
  (#eM #eL : chest2 et_acc (SZ.v nw) 16)
  (#escale : chest2 et_acc (SZ.v nw) 16)
  (#eO : chest2 et_acc (SZ.v nw * 16) (SZ.v d))
  (#egl : chest1 et_acc 16)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  requires
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shM shL eM eL shscale shO shgl escale eO egl) **
    B.barrier_state 0 **
    (gQ |-> Frac fQ eQ) **
    b0_raw nw d shQ shO (SZ.v tid) **
    if_ (lane_active 16sz (tid %^ 32sz))
      (ml_cells 16sz
        (row shM (SZ.v (tid /^ 32sz)))
        (row shL (SZ.v (tid /^ 32sz)))
        (tid %^ 32sz))
  ensures
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shM shL eM eL shscale shO shgl escale eO egl) **
    B.barrier_state 1 **
    (gQ |-> Frac fQ eQ) **
    b0_post nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shO (SZ.v tid) **
    when__ (SZ.v (tid %^ 32sz) < 16) (fun _ ->
      cell_full_v (row shM (SZ.v (tid /^ 32sz))) (SZ.v (tid %^ 32sz))
        (neg infinity))
    ** when__ (SZ.v (tid %^ 32sz) < 16) (fun _ ->
      cell_full_v (row shL (SZ.v (tid /^ 32sz))) (SZ.v (tid %^ 32sz)) zero)
{
  let w : szlt nw = tid /^ 32sz;
  let lane : szlt warp_size = tid %^ 32sz;
  assert pure (block_threads nw == SZ.v nthr);
  assert pure (SZ.v w == thread_w nw (SZ.v tid));
  assert pure (SZ.v lane == thread_lane nw (SZ.v tid));

  unfold b0_raw nw d shQ shO (SZ.v tid);
  unfold FT.strided_cells2 shQ (block_threads nw) (SZ.v tid);
  unfold FT.strided_cells2
    (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw (SZ.v tid)) 0)
    BW.warp_size (thread_lane nw (SZ.v tid));
  rewrite each (thread_w nw (SZ.v tid)) as (SZ.v w);
  rewrite each (thread_lane nw (SZ.v tid)) as (SZ.v lane);
  forevery_rw_type
    (FT.stride_index2 16 (SZ.v d) (block_threads nw) (SZ.v tid))
    (stride_index2 16 (SZ.v d) (SZ.v nthr) (SZ.v tid))
    (fun ij -> exists* (v : et_ab). tensor_pts_to_cell shQ (idx2 ij._1 ij._2) v);
  fold strided_cells2 shQ (SZ.v nthr) (SZ.v tid);
  forevery_rw_type
    (FT.stride_index2 16 (SZ.v d) BW.warp_size (SZ.v lane))
    (stride_index2 16 (SZ.v d) BW.warp_size (SZ.v lane))
    (fun ij -> exists* (v : et_acc).
      tensor_pts_to_cell
        (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v w) 0)
        (idx2 ij._1 ij._2) v);
  fold strided_cells2
    (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v w) 0)
    BW.warp_size (SZ.v lane);

  sdpa_flash_q_load 16sz d nthr b hq sq
    #_ #_ #_ #_
    #_
    #(ctlayout_slice
      (l2_row_major (SZ.v nw) 16)
      #(c_l2_row_major (SZ.v nw) 16sz)
      0 (SZ.v w))
    #(ctlayout_slice
      (l2_row_major (SZ.v nw) 16)
      #(c_l2_row_major (SZ.v nw) 16sz)
      0 (SZ.v w))
    #(c_subtile_layout
      (l2_row_major (SZ.v nw * 16) d)
      #(c_l2_row_major (SZ.v nw * 16) d)
      16 (SZ.v d) (SZ.v w) 0)
    gQ shQ
    (row shM (SZ.v w))
    (row shL (SZ.v w))
    (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v w) 0)
    tid lane bi r0 rows group kvh;

  unfold strided_cells2_v shQ (SZ.v nthr) (SZ.v tid) (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0));
  unfold strided_cells2_v
    (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v w) 0)
    BW.warp_size (SZ.v lane) (FT.ozero 16 (SZ.v d));
  forevery_rw_type
    (stride_index2 16 (SZ.v d) (SZ.v nthr) (SZ.v tid))
    (FT.stride_index2 16 (SZ.v d) (block_threads nw) (SZ.v tid))
    (fun ij -> tensor_pts_to_cell shQ (idx2 ij._1 ij._2)
                 (acc2 (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) ij._1 ij._2));
  forevery_rw_type
    (stride_index2 16 (SZ.v d) BW.warp_size (SZ.v lane))
    (FT.stride_index2 16 (SZ.v d) BW.warp_size (SZ.v lane))
    (fun ij ->
      tensor_pts_to_cell
        (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v w) 0)
        (idx2 ij._1 ij._2) (acc2 (FT.ozero 16 (SZ.v d)) ij._1 ij._2));
  fold FT.strided_cells2_v shQ (block_threads nw) (SZ.v tid) (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0));
  fold FT.strided_cells2_v
    (array2_subtile shO 16 (SZ.v d <: pos) (SZ.v w) 0)
    BW.warp_size (SZ.v lane) (FT.ozero 16 (SZ.v d));
  rewrite each (SZ.v w) as (thread_w nw (SZ.v tid));
  rewrite each (SZ.v lane) as (thread_lane nw (SZ.v tid));
  fold b0_pre nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shO (SZ.v tid);

  barrier_stage0_def nw d shQ
    (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0))
    shM shL eM eL shscale shO shgl escale eO egl (SZ.v tid);
  rewrite (b0_pre nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shO (SZ.v tid))
       as ((barrier_contract nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shM shL eM eL shscale shO shgl escale eO egl).rin 0 (SZ.v tid));
  B.barrier_wait ();
  rewrite ((barrier_contract nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shM shL eM eL shscale shO shgl escale eO egl).rout 0 (SZ.v tid))
       as (b0_post nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shO (SZ.v tid));

  ml_if_to_when_v
    (row shM (SZ.v (tid /^ 32sz)))
    (row shL (SZ.v (tid /^ 32sz)))
    (tid %^ 32sz) (neg infinity) zero;
}

inline_for_extraction noextract
fn sdpa_flash_block_barrier1
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |}
  (nw nthr d : szp { SZ.v nthr == block_threads nw })
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (#eQsh : chest2 et_ab 16 (SZ.v d))
  (#eM #eL : chest2 et_acc (SZ.v nw) 16)
  (#escale : chest2 et_acc (SZ.v nw) 16)
  (#eO : chest2 et_acc (SZ.v nw * 16) (SZ.v d))
  (#egl : chest1 et_acc 16)
  (#_ : squash (16 /?+ SZ.v d))
  requires
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl) **
    B.barrier_state 1 **
    b1_pre_v nw shM shL eM eL (SZ.v tid)
  ensures
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl) **
    B.barrier_state 2 **
    b1_post_v nw shM shL eM eL (SZ.v tid)
{
  rewrite (b1_pre_v nw shM shL eM eL (SZ.v tid))
       as ((barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl).rin 1 (SZ.v tid));
  B.barrier_wait ();
  rewrite ((barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl).rout 1 (SZ.v tid))
       as (b1_post_v nw shM shL eM eL (SZ.v tid));
}

inline_for_extraction noextract
fn sdpa_flash_block_barrier2
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |}
  (nw nthr d : szp { SZ.v nthr == block_threads nw })
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (#eQsh : chest2 et_ab 16 (SZ.v d))
  (#eM #eL : chest2 et_acc (SZ.v nw) 16)
  (#escale : chest2 et_acc (SZ.v nw) 16)
  (#eO : chest2 et_acc (SZ.v nw * 16) (SZ.v d))
  (#egl : chest1 et_acc 16)
  (#_ : squash (16 /?+ SZ.v d))
  requires
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl) **
    B.barrier_state 2 **
    b2_pre_v nw d shscale shO shgl escale eO egl (SZ.v tid)
  ensures
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl) **
    B.barrier_state 3 **
    b2_post_v nw d shscale shO shgl escale eO egl (SZ.v tid)
{
  rewrite (b2_pre_v nw d shscale shO shgl escale eO egl (SZ.v tid))
       as ((barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl).rin 2 (SZ.v tid));
  B.barrier_wait ();
  rewrite ((barrier_contract nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl).rout 2 (SZ.v tid))
       as (b2_post_v nw d shscale shO shgl escale eO egl (SZ.v tid));
}

unfold
let sdpa_flash_gm_cell
  (#et : Type0) (#lgm : layout1 16)
  (nw : szp) (shgm : array1 et lgm) (w : szlt nw) (lane : szlt warp_size) : slprop
= if_ (combine_active 16sz w lane)
     (cell_full_n shgm (SZ.v (clamp_lt 16sz lane)))

ghost
fn block_row_cell_reindex
  (#et : Type0) (#rows : nat) (#l : layout2 rows 16)
  (a : array2 et l)
  (w1 w2 : natlt rows)
  (lane1 lane2 : natlt BW.warp_size)
  requires
    pure (w1 == w2 /\ lane1 == lane2) **
    when__ (lane1 < 16) (fun _ -> cell_full (row a w1) lane1)
  ensures
    when__ (lane2 < 16) (fun _ -> cell_full (row a w2) lane2)
{
  rewrite
    (when__ (lane1 < 16) (fun _ -> cell_full (row a w1) lane1))
  as
    (when__ (lane2 < 16) (fun _ -> cell_full (row a w2) lane2));
}

unfold
let sdpa_flash_b2_scale_local
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size) : slprop
= if_ (combine_active 16sz w lane)
    (exists* (e : chest2 et (SZ.v nw) 1).
      tensor_pts_to
        (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e)
    ** cell_full shgl (SZ.v (clamp_lt 16sz lane))

ghost
fn combine_to_b2_local
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
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)))
{
  let active = combine_active 16sz w lane;
  if active {
    if_elim_true (combine_cells nw 16sz shscale shgm shgl lane);
    unfold (combine_cells nw 16sz shscale shgm shgl lane);
    unfold (cell_full_n shgm (SZ.v (clamp_lt 16sz lane)));
    unfold (cell_full_n shgl (SZ.v (clamp_lt 16sz lane)));
    with vgm. assert (
      tensor_pts_to_cell shgm (idx1 (SZ.v (clamp_lt 16sz lane))) vgm);
    with vgl. assert (
      tensor_pts_to_cell shgl (idx1 (SZ.v (clamp_lt 16sz lane))) vgl);
    raw_cell_to_cell shgl (SZ.v (clamp_lt 16sz lane)) #vgl;
    fold (cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    assert pure (combine_active 16sz w lane);
    if_intro_true' (combine_active 16sz w lane) (
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    drop_ (tensor_pts_to_cell shgm
      (idx1 (SZ.v (clamp_lt 16sz lane))) vgm);
  } else {
    if_elim_false (combine_cells nw 16sz shscale shgm shgl lane);
    if_intro_false (
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    assert pure (combine_active 16sz w lane == false);
    if_rewrite_bool false (combine_active 16sz w lane) (
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e)
      ** cell_full shgl (SZ.v (clamp_lt 16sz lane)));
  }
}

ghost
fn scale_column_reindex
  (#et : Type0) (nw : szp)
  (a : array2 et (l2_row_major (SZ.v nw) 16))
  (i j : natlt 16)
  requires
    pure (i == j) **
    (exists* (e : chest2 et (SZ.v nw) 1).
      tensor_pts_to (array2_stride_subtile a 1 16 0 i) e)
  ensures
    (exists* (e : chest2 et (SZ.v nw) 1).
      tensor_pts_to (array2_stride_subtile a 1 16 0 j) e)
{
  rewrite
    (exists* (e : chest2 et (SZ.v nw) 1).
      tensor_pts_to (array2_stride_subtile a 1 16 0 i) e)
  as
    (exists* (e : chest2 et (SZ.v nw) 1).
      tensor_pts_to (array2_stride_subtile a 1 16 0 j) e);
}

ghost
fn vector_cell_reindex
  (#et : Type0) (#l : layout1 16)
  (a : array1 et l) (i j : natlt 16)
  requires pure (i == j) ** cell_full a i
  ensures cell_full a j
{
  rewrite (cell_full a i) as (cell_full a j);
}

ghost
fn b2_scale_to_descriptor
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  (tid : szlt (block_threads nw))
  requires
    pure (SZ.v w == thread_w nw (SZ.v tid) /\
          SZ.v lane == thread_lane nw (SZ.v tid)) **
    if_ (combine_active 16sz w lane)
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)))
  ensures
    b2_scale_pre nw shscale shgl (SZ.v tid)
{
  let active = combine_active 16sz w lane;
  if active {
    if_elim_true (
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    let old_lane : natlt 16 = SZ.v (clamp_lt 16sz lane);
    let new_lane : natlt 16 =
      clamp_nat_lt 16 (thread_lane nw (SZ.v tid));
    assert pure (old_lane == new_lane);
    scale_column_reindex nw shscale
      (SZ.v (clamp_lt 16sz lane) <: natlt 16)
      (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)));
    vector_cell_reindex shgl
      (SZ.v (clamp_lt 16sz lane) <: natlt 16)
      (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)));
    assert pure (b2_active nw (SZ.v tid) == true);
    if_intro_true' (b2_active nw (SZ.v tid)) (
      ((exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0
            (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))))
          e)
        ** cell_full shgl (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))));
  } else {
    if_elim_false (
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)));
    assert pure (b2_active nw (SZ.v tid) == false);
    if_intro_false (
      ((exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0
            (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))))
          e)
        ** cell_full shgl (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))));
    if_rewrite_bool false (b2_active nw (SZ.v tid)) (
      ((exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0
            (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))))
          e)
        ** cell_full shgl (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))));
  }
}

ghost
fn flash_combine_to_b2_keep_gm_v
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgm shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  (escale : chest2 et (SZ.v nw) 1) (vgm vgl : et)
  requires
    if_ (combine_active 16sz w lane)
      (combine_cells_v nw 16sz shscale shgm shgl lane escale vgm vgl)
  ensures
    if_ (combine_active 16sz w lane)
      (flash_scale_tile_v nw shscale escale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl **
       cell_full shgm (SZ.v (clamp_lt 16sz lane)))
{
  let active = combine_active 16sz w lane;
  if active {
    if_elim_true (combine_cells_v nw 16sz shscale shgm shgl lane
                    escale vgm vgl);
    unfold (combine_cells_v nw 16sz shscale shgm shgl lane escale vgm vgl);
    unfold (cell_full_n_v shgm (SZ.v (clamp_lt 16sz lane)) vgm);
    unfold (cell_full_n_v shgl (SZ.v (clamp_lt 16sz lane)) vgl);
    fold (cell_full shgm (SZ.v (clamp_lt 16sz lane)));
    fold (cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl);
    fold (flash_scale_tile_v nw shscale escale
           (SZ.v (clamp_lt 16sz lane)));
    assert pure (combine_active 16sz w lane);
    if_intro_true' (combine_active 16sz w lane) (
      (flash_scale_tile_v nw shscale escale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl **
       cell_full shgm (SZ.v (clamp_lt 16sz lane))));
  } else {
    if_elim_false (combine_cells_v nw 16sz shscale shgm shgl lane
                     escale vgm vgl);
    assert pure (combine_active 16sz w lane == false);
    if_intro_false' (combine_active 16sz w lane) (
      (flash_scale_tile_v nw shscale escale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl **
       cell_full shgm (SZ.v (clamp_lt 16sz lane))));
  }
}

ghost
fn flash_combine_split_gm_v
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgm shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  (escale : chest2 et (SZ.v nw) 1) (vgl : et)
  requires
    if_ (combine_active 16sz w lane)
      (flash_scale_tile_v nw shscale escale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl **
       cell_full shgm (SZ.v (clamp_lt 16sz lane)))
  ensures
    if_ (combine_active 16sz w lane)
      (flash_scale_tile_v nw shscale escale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl)
    ** if_ (combine_active 16sz w lane)
      (cell_full shgm (SZ.v (clamp_lt 16sz lane)))
{
  split (combine_active 16sz w lane)
    (flash_scale_tile_v nw shscale escale (SZ.v (clamp_lt 16sz lane)))
    (cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl **
     cell_full shgm (SZ.v (clamp_lt 16sz lane)));
  split (combine_active 16sz w lane)
    (cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl)
    (cell_full shgm (SZ.v (clamp_lt 16sz lane)));
  combine (combine_active 16sz w lane)
    (flash_scale_tile_v nw shscale escale (SZ.v (clamp_lt 16sz lane)))
    (cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl);
}

ghost
fn flash_b2_scale_to_descriptor_v
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  (tid : szlt (block_threads nw))
  (escale : chest2 et (SZ.v nw) 16) (egl : chest1 et 16)
  (escale_t : chest2 et (SZ.v nw) 1) (vgl : et)
  requires
    pure (SZ.v w == thread_w nw (SZ.v tid) /\
          SZ.v lane == thread_lane nw (SZ.v tid) /\
          escale_t == ematrix_stride_subtile escale 1 16 0
                        (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))) /\
          vgl == acc1 egl
                   (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))) **
    if_ (combine_active 16sz w lane)
      (flash_scale_tile_v nw shscale escale_t
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl)
  ensures
    b2_scale_pre_v nw shscale shgl escale egl (SZ.v tid)
{
  let active = combine_active 16sz w lane;
  if active {
    if_elim_true' active (
      flash_scale_tile_v nw shscale escale_t
        (SZ.v (clamp_lt 16sz lane)) **
      cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl);
    unfold flash_scale_tile_v nw shscale escale_t
      (SZ.v (clamp_lt 16sz lane));
    let old_lane : natlt 16 = SZ.v (clamp_lt 16sz lane);
    let new_lane : natlt 16 =
      clamp_nat_lt 16 (thread_lane nw (SZ.v tid));
    assert pure (old_lane == new_lane);
    rewrite each (SZ.v (clamp_lt 16sz lane)) as old_lane;
    rewrite
      (tensor_pts_to
        (array2_stride_subtile shscale 1 16 0 old_lane) escale_t)
      as
      (tensor_pts_to
        (array2_stride_subtile shscale 1 16 0 new_lane)
        (ematrix_stride_subtile escale 1 16 0 new_lane));
    rewrite
      (cell_full_v shgl old_lane vgl)
      as
      (cell_full_v shgl new_lane (acc1 egl new_lane));
    rewrite each new_lane
      as (clamp_nat_lt 16
        (thread_lane nw (SZ.v tid)));
    assert pure (b2_active nw (SZ.v tid) == true);
    if_intro_true' (b2_active nw (SZ.v tid)) (
      tensor_pts_to
        (array2_stride_subtile shscale 1 16 0
          (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))))
        (ematrix_stride_subtile escale 1 16 0
          (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))))
      ** cell_full_v shgl
        (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))
        (acc1 egl (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))));
  } else {
    if_elim_false' active (
      flash_scale_tile_v nw shscale escale_t
        (SZ.v (clamp_lt 16sz lane)) **
      cell_full_v shgl (SZ.v (clamp_lt 16sz lane)) vgl);
    assert pure (b2_active nw (SZ.v tid) == false);
    if_intro_false' (b2_active nw (SZ.v tid)) (
      tensor_pts_to
        (array2_stride_subtile shscale 1 16 0
          (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))))
        (ematrix_stride_subtile escale 1 16 0
          (clamp_nat_lt 16 (thread_lane nw (SZ.v tid))))
      ** cell_full_v shgl
        (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))
        (acc1 egl (clamp_nat_lt 16 (thread_lane nw (SZ.v tid)))));
  }
}

ghost
fn flash_pin_o
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (eQsh : chest2 et_ab 16 (SZ.v d))
  (eKg eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  (nkt iters : nat)
  (w1 : natlt (SZ.v nw)) (lane1 : natlt BW.warp_size)
  (tid : natlt (block_threads nw))
  requires
    (array2_stride_subtile
      (array2_subtile shO 16 (SZ.v d <: pos) w1 0)
      warp_row_span 16 (lane1 / 16) (lane1 % 16)
      |-> Frac 1.0R
            (ematrix_stride_subtile
              (SS.run_O emask has_mask causal bi kvh group (SZ.v rows) r0
                 scale eQsh eKg eVg (SZ.v nw) w1 iters)
              warp_row_span 16 (lane1 / 16) (lane1 % 16)))
    ** pure (
      w1 == thread_w nw tid /\ lane1 == thread_lane nw tid /\
      eQsh == SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0 /\
      nkt == SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal /\
      iters == SF.warp_iters (SZ.v nw) nkt w1)
  ensures
    b2_o_pre_v nw d shO
      (flash_eO_at nw d b hq sq rows sk eQ eKg eVg emask has_mask causal
         scale bi kvh group r0)
      tid
{
  flash_eO_at_def nw d b hq sq rows sk eQ eKg eVg emask has_mask causal
    scale bi kvh group r0 w1;
  rewrite
    (array2_stride_subtile
      (array2_subtile shO 16 (SZ.v d <: pos) w1 0)
      warp_row_span 16 (lane1 / 16) (lane1 % 16)
      |-> Frac 1.0R
            (ematrix_stride_subtile
              (SS.run_O emask has_mask causal bi kvh group (SZ.v rows) r0
                 scale eQsh eKg eVg (SZ.v nw) w1 iters)
              warp_row_span 16 (lane1 / 16) (lane1 % 16)))
    as
    (b2_o_pre_v nw d shO
      (flash_eO_at nw d b hq sq rows sk eQ eKg eVg emask has_mask causal
         scale bi kvh group r0)
      tid);
}
