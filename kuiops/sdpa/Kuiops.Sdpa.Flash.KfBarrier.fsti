module Kuiops.Sdpa.Flash.KfBarrier

(* Block-level barrier phases of [sdpa_flash_kf]: the prologue, the two
   inter-phase barriers, and the reindexing that relates the block's view of
   the shared tiles to the barrier descriptor's. *)

#lang-pulse

open Kuiper
open Kuiper.Floating
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiops.Sdpa.Flash.Types
open Kuiops.Sdpa.Flash.KfSub
open Kuiops.Sdpa.Flash.KfBlock

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float


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
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  requires
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shM shL shscale shO shgl) **
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
    B.barrier_tok (barrier_contract nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shM shL shscale shO shgl) **
    B.barrier_state 1 **
    (gQ |-> Frac fQ eQ) **
    b0_post nw d shQ (SF.q_tile 16 (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) shO (SZ.v tid) **
    when__ (SZ.v (tid %^ 32sz) < 16) (fun _ ->
      cell_full_v (row shM (SZ.v (tid /^ 32sz))) (SZ.v (tid %^ 32sz))
        (neg infinity))
    ** when__ (SZ.v (tid %^ 32sz) < 16) (fun _ ->
      cell_full_v (row shL (SZ.v (tid /^ 32sz))) (SZ.v (tid %^ 32sz)) zero)

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
  (#_ : squash (16 /?+ SZ.v d))
  requires
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL shscale shO shgl) **
    B.barrier_state 1 **
    b1_pre nw shM shL (SZ.v tid)
  ensures
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL shscale shO shgl) **
    B.barrier_state 2 **
    b1_post nw shM shL (SZ.v tid)

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
  (#_ : squash (16 /?+ SZ.v d))
  requires
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL shscale shO shgl) **
    B.barrier_state 2 **
    b2_pre nw d shscale shO shgl (SZ.v tid)
  ensures
    gpu **
    thread_id (block_threads nw) tid **
    B.barrier_tok (barrier_contract nw d shQ eQsh shM shL shscale shO shgl) **
    B.barrier_state 3 **
    b2_post nw d shscale shO shgl (SZ.v tid)

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

ghost
fn block_o_tile_reindex
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (16 /?+ (SZ.v nw * 16)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (w1 w2 : natlt (SZ.v nw))
  (lane1 lane2 : natlt BW.warp_size)
  requires
    pure (w1 == w2 /\ lane1 == lane2) **
    (exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
      tensor_pts_to
        (array2_stride_subtile
          (array2_subtile a 16 (SZ.v d <: pos) w1 0)
          warp_row_span 16 (lane1 / 16) (lane1 % 16))
        e)
  ensures
    (exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
      tensor_pts_to
        (array2_stride_subtile
          (array2_subtile a 16 (SZ.v d <: pos) w2 0)
          warp_row_span 16 (lane2 / 16) (lane2 % 16))
        e)

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

ghost
fn flash_b2_scale_to_descriptor
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
      (flash_scale_tile nw shscale
        (SZ.v (clamp_lt 16sz lane)) **
       cell_full shgl (SZ.v (clamp_lt 16sz lane)))
  ensures
    b2_scale_pre nw shscale shgl (SZ.v tid)
