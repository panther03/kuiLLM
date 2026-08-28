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
open Kuiops.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiops.Sdpa.Flash.Vals
open Kuiops.Sdpa.Flash.Types
open Kuiops.Sdpa.Flash.KfSub
open Kuiops.Sdpa.Flash.KfBlock

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float
module SS = Kuiops.Sdpa.Flash.Spec.Step


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
