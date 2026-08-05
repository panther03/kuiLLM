module Kuiops.Sdpa.Flash.Split

(* Generic [forevery] plumbing: splitting and gathering per-thread and
   per-warp ownership of the global and shared tensors. *)

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

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module Trade = Pulse.Lib.Trade


ghost
fn flash_forevery4_intro
  (p : natlt flash_nwarps -> slprop)
  requires p 0 ** p 1 ** p 2 ** p 3
  ensures forall+ (w : natlt flash_nwarps). p w

ghost
fn flash_forevery4_elim
  (p : natlt flash_nwarps -> slprop)
  requires forall+ (w : natlt flash_nwarps). p w
  ensures p 0 ** p 1 ** p 2 ** p 3

ghost
fn flash_cells2_gather
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt rows) (j : natlt cols).
      exists* (v : et). Cell a (idx2 i j) |-> Frac 1.0R v)
  ensures live a

ghost
fn flash_cells1_gather
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt len).
      exists* (v : et). Cell a (idx1 i) |-> Frac 1.0R v)
  ensures live a

ghost
fn flash_strided_cells2_gather
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr). strided_cells2 a nthr tid)
  ensures live a

ghost
fn flash_split_strided_cells2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires live a
  ensures
    forall+ (tid : natlt nthr).
      strided_cells2 a nthr tid

ghost
fn flash_gather_strided_cells2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr).
      strided_cells2 a nthr tid)
  ensures live a

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

ghost
fn flash_split_rows16
  (#et : Type0) (#l : layout2 16 16)
  (#_ : squash (SZ.fits (tlayout_ulen l)))
  (a : array2 et l)
  requires live a
  ensures
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> row_subtile a lane)

ghost
fn flash_gather_rows16
  (#et : Type0) (#l : layout2 16 16)
  (#_ : squash (SZ.fits (tlayout_ulen l)))
  (a : array2 et l)
  requires
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> row_subtile a lane)
  ensures live a

ghost
fn flash_split_cells16
  (#et : Type0)
  (a : array1 et (l1_forward 16))
  requires live a
  ensures
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> cell_full a lane)

ghost
fn flash_gather_cells16
  (#et : Type0)
  (a : array1 et (l1_forward 16))
  requires
    forall+ (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ -> cell_full a lane)
  ensures live a

ghost
fn flash_share_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  requires live a
  ensures
    forall+ (_ : natlt k).
      exists* (e : chest d et). a |-> Frac (1.0R /. k) e

ghost
fn flash_gather_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  requires
    forall+ (_ : natlt k).
      exists* (e : chest d et). a |-> Frac (1.0R /. k) e
  ensures live a

ghost
fn flash_split_ml
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (a : array2 et (l2_row_major (SZ.v nw) 16))
  requires live a
  ensures
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ ->
        cell_full (row a w) lane)

ghost
fn flash_gather_ml
  (#et : Type0) (nw : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (a : array2 et (l2_row_major (SZ.v nw) 16))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when__ (lane < 16) (fun _ ->
        cell_full (row a w) lane)
  ensures live a

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

ghost
fn flash_when_to_when__ (p : prop) (q : slprop)
  requires when_ p q
  ensures when__ p (fun _ -> q)

ghost
fn flash_when__to_when (p : prop) (q : slprop)
  requires when__ p (fun _ -> q)
  ensures when_ p q

ghost
fn flash_when__elim_true
  (b : bool { b == true })
  (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures q ()

ghost
fn flash_when__elim_false
  (b : bool { b == false })
  (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures emp

ghost
fn flash_when__intro_true
  (b : bool { b == true })
  (q : squash (b2t b) -> slprop)
  requires q ()
  ensures when__ b q

ghost
fn flash_when__intro_false
  (b : bool { b == false })
  (q : squash (b2t b) -> slprop)
  ensures when__ b q

ghost
fn flash_when_elim_true (b : bool { b == true }) (q : slprop)
  requires when_ b q
  ensures q

ghost
fn flash_when_elim_false (b : bool { b == false }) (q : slprop)
  requires when_ b q
  ensures emp

ghost
fn flash_when_intro_true (b : bool { b == true }) (q : slprop)
  requires q
  ensures when_ b q

ghost
fn flash_when_intro_false (b : bool { b == false }) (q : slprop)
  ensures when_ b q

ghost
fn flash_when_reindex (p q : prop) (r : slprop)
  requires pure (p <==> q) ** when_ p r
  ensures when_ q r

ghost
fn flash_when_prop_elim_true
  (p : prop) (b : bool { b == t2b p /\ b == true }) (q : slprop)
  requires when_ p q
  ensures q

ghost
fn flash_when_prop_elim_false
  (p : prop) (b : bool { b == t2b p /\ b == false }) (q : slprop)
  requires when_ p q
  ensures emp

ghost
fn flash_when_prop_intro_true
  (p : prop) (b : bool { b == t2b p /\ b == true }) (q : slprop)
  requires q
  ensures when_ p q

ghost
fn flash_when_prop_intro_false
  (p : prop) (b : bool { b == t2b p /\ b == false }) (q : slprop)
  ensures when_ p q

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

ghost
fn flash_gather_gm
  (#et : Type0)
  (nw : szp)
  (gm : array1 et (l1_forward 16))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when_ (w = 0 /\ lane < 16)
        (cell_full gm (clamp_nat_lt 16 lane))
  ensures live gm

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

ghost
fn flash_cell_full_to_n
  (#et : Type0) (#l : layout1 16)
  (a : array1 et l) (i : natlt 16)
  requires cell_full a i
  ensures cell_full_n a i

ghost
fn flash_cell_full_from_n
  (#et : Type0) (#l : layout1 16)
  (a : array1 et l) (i : natlt 16)
  requires cell_full_n a i
  ensures cell_full a i

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

(* Read-only counterpart of [flash_gather_thread_tensor], for the additive mask:
   it is a [rotensor] so that a broadcast (non-injective) layout is admissible. *)
ghost
fn flash_gather_thread_rotensor
  (#et : Type0) (#r : nat) (#ds : shape r)
  (#l : TRO.vtlayout ds)
  (nw nthr : szp {
    SZ.v nthr == SZ.v nw * BW.warp_size })
  (a : TRO.rotensor et l)
  (#f : perm) (#e : chest ds et)
  requires
    forall+ (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size).
      a |-> Frac (f /. (SZ.v nthr)) e
  ensures a |-> Frac f e
