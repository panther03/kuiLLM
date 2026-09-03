module Kuiops.Sdpa.Flash.Barrier

(* The block barrier contract: the ghost transforms that carry ownership
   across each of the three barrier phases, and the bridges between the
   per-thread shared-memory views and the barrier pre/post predicates. *)

#lang-pulse


open Kuiper
open Kuiper.Bijection
open Kuiper.EMatrix
open Kuiper.ForEvery
open Kuiper.Shape
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Tiling
open Kuiops.Array2.Strided
open Kuiper.Kernel.FlashAttention.KernelDesc

open Kuiper.TensorCore
open Pulse.Lib.Pledge
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module FC = Kuiper.Float.Casts
module Trade = Pulse.Lib.Trade
module BW = Kuiper.Barrier.Warp

let stride_partition_bij
  (rows cols : nat) (nthr : pos)
  : ((tid : natlt nthr & stride_index2 rows cols nthr tid)
      =~ (natlt rows & natlt cols))
= {
  ff = (fun (x : (tid : natlt nthr & stride_index2 rows cols nthr tid)) ->
    x._2)
    <: ((tid : natlt nthr & stride_index2 rows cols nthr tid) ->
        GTot (natlt rows & natlt cols));
  gg = (fun (ij : natlt rows & natlt cols) ->
    (| ((ij._1 * cols + ij._2) % nthr), ij |))
    <: ((natlt rows & natlt cols) ->
        GTot (tid : natlt nthr & stride_index2 rows cols nthr tid));
}

let abs1_eq (len : nat)
  : Lemma (abs (len @| INil) == (natlt len & unit))
      [SMTPat (abs (len @| INil))]
= ()

unfold
let abs1_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
{
  ff = (fun (i, ()) -> i);
  gg = (fun i -> idx1 i);
}

ghost
fn cells2_gather
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt rows) (j : natlt cols).
      exists* (v : et). Cell a (idx2 i j) |-> Frac 1.0R v)
  ensures
    exists* (e : chest2 et rows cols). a |-> Frac 1.0R e

ghost
fn cells1_gather
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt len).
      exists* (v : et). Cell a (idx1 i) |-> Frac 1.0R v)
  ensures
    exists* (e : chest1 et len). a |-> Frac 1.0R e

ghost
fn strided_cells2_gather
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr). strided_cells2 a nthr tid)
  ensures
    exists* (e : chest2 et rows cols). a |-> Frac 1.0R e

ghost
fn cells1_gather_v
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l) (e : chest1 et len)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt len). Cell a (idx1 i) |-> Frac 1.0R (acc1 e i))
  ensures a |-> Frac 1.0R e

ghost
fn b0_o_transform
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  requires
    forall+ (tid : natlt (block_threads nw)).
      strided_cells2
        (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
        BW.warp_size (thread_lane nw tid)
  ensures
    forall+ (tid : natlt (block_threads nw)).
      exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile
          (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
          warp_row_span 16
          (thread_lane nw tid / 16) (thread_lane nw tid % 16)
          |-> Frac 1.0R e

ghost
fn row_cell_to_matrix
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (r : natlt rows) (c : natlt cols)
  requires cell_full (row a r) c
  ensures
    exists* (v : et). Cell a (idx2 r c) |-> Frac 1.0R v

ghost
fn b1_matrix_transform
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (a : array2 et (l2_row_major (SZ.v nw) 16))
  requires
    forall+ (tid : natlt (block_threads nw)).
      b1_pre_one nw a tid
  ensures
    forall+ (_tid : natlt (block_threads nw)).
      exists* (e : chest2 et (SZ.v nw) 16).
        a |-> Frac (1.0R /. (block_threads nw)) e

ghost
fn b2_o_transform
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  requires
    forall+ (tid : natlt (block_threads nw)).
      b2_o_pre nw d shO tid
  ensures
    forall+ (_tid : natlt (block_threads nw)).
      exists* (e : chest2 et (SZ.v nw * 16) (SZ.v d)).
        shO |-> Frac (1.0R /. (block_threads nw)) e

let b2_active_idx (nw : szp) : Type0 =
  wl : (natlt (SZ.v nw) & natlt BW.warp_size) {
    wl._1 == 0 /\ wl._2 < 16}

let b2_active_lane_bij (nw : szp)
  : (b2_active_idx nw =~ natlt 16)
= {
  ff = (fun (wl : b2_active_idx nw) -> wl._2 <: natlt 16)
    <: (b2_active_idx nw -> GTot (natlt 16));
  gg = (fun (lane : natlt 16) -> (0, lane))
    <: (natlt 16 -> GTot (b2_active_idx nw));
}

unfold
let scale_tile
  (#et : Type0) (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (lane : natlt 16) : slprop
= exists* (e : chest2 et (SZ.v nw) 1).
    tensor_pts_to (array2_stride_subtile shscale 1 16 0 lane) e

unfold
let scale_tile_v
  (#et : Type0) (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (escale : chest2 et (SZ.v nw) 16)
  (lane : natlt 16) : slprop
= tensor_pts_to (array2_stride_subtile shscale 1 16 0 lane)
    (ematrix_stride_subtile escale 1 16 0 lane)

ghost
fn b2_scale_transform
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  requires
    forall+ (tid : natlt (block_threads nw)).
      b2_scale_pre nw shscale shgl tid
  ensures
    (forall+ (_tid : natlt (block_threads nw)).
      exists* (e : chest2 et (SZ.v nw) 16).
        shscale |-> Frac (1.0R /. (block_threads nw)) e) **
    (forall+ (_tid : natlt (block_threads nw)).
      exists* (e : chest1 et 16).
        shgl |-> Frac (1.0R /. (block_threads nw)) e)

ghost
fn b2_o_transform_v
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (eO : chest2 et (SZ.v nw * 16) (SZ.v d))
  requires
    forall+ (tid : natlt (block_threads nw)).
      b2_o_pre_v nw d shO eO tid
  ensures
    forall+ (_tid : natlt (block_threads nw)).
      shO |-> Frac (1.0R /. (block_threads nw)) eO

ghost
fn b2_scale_transform_v
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (escale : chest2 et (SZ.v nw) 16) (egl : chest1 et 16)
  requires
    forall+ (tid : natlt (block_threads nw)).
      b2_scale_pre_v nw shscale shgl escale egl tid
  ensures
    (forall+ (_tid : natlt (block_threads nw)).
      shscale |-> Frac (1.0R /. (block_threads nw)) escale) **
    (forall+ (_tid : natlt (block_threads nw)).
      shgl |-> Frac (1.0R /. (block_threads nw)) egl)

ghost
fn barrier_ok
  (#et_ab #et_acc : Type0) {| scalar et_acc |}
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
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
  (it : nat)
  requires
    forall+ (tid : natlt (block_threads nw)).
      barrier_rin nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl it tid
  ensures
    forall+ (tid : natlt (block_threads nw)).
      barrier_rout nw d shQ eQsh shM shL eM eL shscale shO shgl escale eO egl it tid

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
      cell_full (row m w) lane) **
    when__ (lane < 16) (fun _ ->
      cell_full (row l w) lane)
  ensures
    if_ (lane_active 16sz ls)
      (ml_cells 16sz (row m (SZ.v ws))
        (row l (SZ.v ws)) ls)

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

(* Turn a warp-0 [if_] over machine indices back into the [when_] over
   erased indices the thread postcondition is stated with. *)
ghost
fn flash_if_when_reindex
  (p : natlt BW.warp_size -> slprop)
  (w : nat) (lane : natlt BW.warp_size)
  (ws : sz) (ls : szlt BW.warp_size)
  requires
    pure (SZ.v ws == w /\ SZ.v ls == lane) **
    if_ (ws = 0sz) (p (SZ.v ls))
  ensures
    when_ (w = 0) (p lane)

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
      (cell_full gm (clamp_nat_lt 16 lane))
