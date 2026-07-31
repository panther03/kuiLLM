module Kuiops.Sdpa.Flash.KernelDesc

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.ForEvery
open Kuiper.Kernel.FlashAttention.KernelDesc

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp

unfold let warp_row_span : nat = 2

(* Concrete, content-free ownership states for the three block barriers in
   the flash-attention block.  Values are existential throughout: the
   description proves only permission transfer and bounds safety. *)

let block_threads (nw : szp) : nat = SZ.v nw * BW.warp_size

let thread_w (nw : szp) (tid : natlt (block_threads nw)) : natlt (SZ.v nw) =
  tid / BW.warp_size

let thread_lane (nw : szp) (tid : natlt (block_threads nw)) : natlt BW.warp_size =
  tid % BW.warp_size

let stride_index2 (rows cols : nat) (nthr : pos) (tid : natlt nthr) : Type0 =
  ij:(natlt rows & natlt cols) { (ij._1 * cols + ij._2) % nthr == tid }

unfold
let strided_cells2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos) (tid : natlt nthr) : slprop
= forall+ (ij : stride_index2 rows cols nthr tid).
    exists* (v : et). Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v

unfold
let cell_full
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l) (i : natlt len) : slprop
= exists* (v : et). Cell a (idx1 i) |-> Frac 1.0R v

let row_layout
  (#et : Type0) (#rows #cols : erased nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : erased nat { i < rows }) : layout1 cols
= tlayout_slice l 0 i

inline_for_extraction noextract
let row
  (#et : Type0) (#rows #cols : erased nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : erased nat { i < rows })
  : array1 et (row_layout a i)
= sliceof a 0 i

unfold
let b0_pre
  (#et_q #et_o : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_q (l2_row_major 16 (SZ.v d)))
  (shO : array2 et_o (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  strided_cells2 shQ (block_threads nw) tid **
  strided_cells2 (array2_subtile shO 16 (SZ.v d <: pos) w 0) BW.warp_size lane

unfold
let b0_post
  (#et_q #et_o : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_q (l2_row_major 16 (SZ.v d)))
  (shO : array2 et_o (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  (exists* (e : chest2 et_q 16 (SZ.v d)).
     shQ |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (o : chest2 et_o (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile
          (array2_subtile shO 16 (SZ.v d <: pos) w 0)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R o)

unfold
let b1_pre_one
  (#et : Type0) (nw : szp)
  (#lM : layout2 (SZ.v nw) 16)
  (shM : array2 et lM)
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  when__ (lane < 16) (fun _ ->
    cell_full (row shM w) (lane <: natlt 16))

unfold
let b1_pre
  (#et : Type0) (nw : szp)
  (#lM #lL : layout2 (SZ.v nw) 16)
  (shM : array2 et lM) (shL : array2 et lL)
  (tid : natlt (block_threads nw)) : slprop
= b1_pre_one nw shM tid ** b1_pre_one nw shL tid

unfold
let b1_post
  (#et : Type0) (nw : szp)
  (#lM #lL : layout2 (SZ.v nw) 16)
  (shM : array2 et lM) (shL : array2 et lL)
  (_tid : natlt (block_threads nw)) : slprop
= (exists* (e : chest2 et (SZ.v nw) 16).
     shM |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (e : chest2 et (SZ.v nw) 16).
     shL |-> Frac (1.0R /. (block_threads nw)) e)

unfold
let b2_o_pre
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
    array2_stride_subtile
      (array2_subtile shO 16 (SZ.v d <: pos) w 0)
      warp_row_span 16 (lane / 16) (lane % 16)
      |-> Frac 1.0R e

unfold
let b2_active
  (nw : szp) (tid : natlt (block_threads nw)) : GTot bool
= t2b (thread_w nw tid = 0 /\ thread_lane nw tid < 16)

let clamp_nat_lt (n : pos) (x : nat) : natlt n =
  if x < n then x else 0

unfold
let b2_scale_pre
  (#et : Type0) (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  if_ (b2_active nw tid)
    ((exists* (e : chest2 et (SZ.v nw) 1).
       tensor_pts_to (array2_stride_subtile shscale 1 16 0 (clamp_nat_lt 16 lane)) e)
     ** cell_full shgl (clamp_nat_lt 16 lane))

unfold
let b2_pre
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et (l1_forward 16))
  (tid : natlt (block_threads nw)) : slprop
= b2_o_pre nw d shO tid ** b2_scale_pre nw shscale shgl tid

unfold
let b2_post
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et (l1_forward 16))
  (_tid : natlt (block_threads nw)) : slprop
= (exists* (e : chest2 et (SZ.v nw) 16).
     shscale |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (e : chest2 et (SZ.v nw * 16) (SZ.v d)).
     shO |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (e : chest1 et 16).
     shgl |-> Frac (1.0R /. (block_threads nw)) e)

let barrier_rin
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  : B.barrier_side (block_threads nw)
= fun it tid ->
    if it = 0 then b0_pre nw d shQ shO tid
    else if it = 1 then b1_pre nw shM shL tid
    else if it = 2 then b2_pre nw d shscale shO shgl tid
    else emp

let barrier_rout
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  : B.barrier_side (block_threads nw)
= fun it tid ->
    if it = 0 then b0_post nw d shQ shO tid
    else if it = 1 then b1_post nw shM shL tid
    else if it = 2 then b2_post nw d shscale shO shgl tid
    else emp

let barrier_contract
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  : B.contract (block_threads nw)
= {
  rin = barrier_rin nw d shQ shM shL shscale shO shgl;
  rout = barrier_rout nw d shQ shM shL shscale shO shgl;
}

let barrier_count (_nw : szp) : GTot nat = 3

ghost
fn barrier_ok
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (it : nat)
  requires
    forall+ (tid : natlt (block_threads nw)).
      barrier_rin nw d shQ shM shL shscale shO shgl it tid
  ensures
    forall+ (tid : natlt (block_threads nw)).
      barrier_rout nw d shQ shM shL shscale shO shgl it tid
