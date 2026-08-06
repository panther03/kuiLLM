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
open Kuiper.Array2.Strided
open Kuiper.Kernel.FlashAttention.KernelDesc

open Kuiper.TensorCore
open Pulse.Lib.Pledge
open Kuiops.Sdpa.Flash.Split

module SZ = Kuiper.SizeT
module FC = Kuiper.Float.Casts
module Trade = Pulse.Lib.Trade
module BW = Kuiper.Barrier.Warp

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
      assert pure (acc e (idx2 i j) == vf i j);
      rewrite
        (Cell a (idx2 i j) |-> Frac 1.0R (vf i j))
        as
        (Cell a (idx2 i j) |-> Frac 1.0R (acc e (idx2 i j)));
    };
  tensor_iraise2 a #1.0R #e;
}

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
{
  let vf = forevery_exists
    (fun (i : natlt len) (v : et) ->
      Cell a (idx1 i) |-> Frac 1.0R v);
  let e : chest1 et len = mk1 vf;
  forevery_map #(natlt len)
    (fun i -> Cell a (idx1 i) |-> Frac 1.0R (vf i))
    (fun i -> Cell a (idx1 i) |-> Frac 1.0R (acc e (idx1 i)))
    fn i {
      assert pure (acc e (idx1 i) == vf i);
      rewrite
        (Cell a (idx1 i) |-> Frac 1.0R (vf i))
        as
        (Cell a (idx1 i) |-> Frac 1.0R (acc e (idx1 i)));
    };
  forevery_iso_back (abs1_bij #len)
    (fun i -> Cell a i |-> Frac 1.0R (acc e i));
  forevery_map #(abs (len @| INil))
    (fun i -> Cell a i |-> Frac 1.0R (acc e i))
    (fun i -> pts_to_cell (core a) #1.0R (l.imap.f i) (acc e i))
    fn i {
      tensor_pts_to_cell_eq a i 1.0R (acc e i);
      rewrite
        (Cell a i |-> Frac 1.0R (acc e i))
        as
        (pts_to_cell (core a) #1.0R (l.imap.f i) (acc e i));
    };
  tensor_iraise a #1.0R #e;
}

ghost
fn strided_cells2_gather
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr). strided_cells2 a nthr tid)
  ensures
    exists* (e : chest2 et rows cols). a |-> Frac 1.0R e
{
  forevery_map #(natlt nthr)
    (fun tid -> strided_cells2 a nthr tid)
    (fun tid ->
      forall+ (ij : stride_index2 rows cols nthr tid).
        exists* (v : et). Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v)
    fn tid {
      unfold strided_cells2 a nthr tid;
    };
  forevery_flatten_dep
    (fun (tid : natlt nthr) (ij : stride_index2 rows cols nthr tid) ->
      exists* (v : et). Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v);
  forevery_iso (stride_partition_bij rows cols nthr)
    (fun x ->
      exists* (v : et). Cell a (idx2 x._2._1 x._2._2) |-> Frac 1.0R v);
  forevery_map #(natlt rows & natlt cols)
    (fun ij ->
      exists* (v : et).
        Cell a
          (idx2
            ((stride_partition_bij rows cols nthr).gg ij)._2._1
            ((stride_partition_bij rows cols nthr).gg ij)._2._2)
          |-> Frac 1.0R v)
    (fun ij ->
      exists* (v : et). Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v)
    fn ij {
      with v. assert (
        tensor_pts_to_cell a #1.0R
          (idx2
            ((stride_partition_bij rows cols nthr).gg ij)._2._1
            ((stride_partition_bij rows cols nthr).gg ij)._2._2)
          v);
      assert pure (
        ((stride_partition_bij rows cols nthr).gg ij)._2 == ij);
      rewrite
        (tensor_pts_to_cell a #1.0R
          (idx2
            ((stride_partition_bij rows cols nthr).gg ij)._2._1
            ((stride_partition_bij rows cols nthr).gg ij)._2._2)
          v)
        as
        (tensor_pts_to_cell a #1.0R (idx2 ij._1 ij._2) v);
    };
  forevery_unflatten'
    (fun (ij : natlt rows & natlt cols) ->
      exists* (v : et). Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v);
  cells2_gather a;
}

ghost
fn cells2_gather_v
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (e : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : natlt rows) (j : natlt cols).
      Cell a (idx2 i j) |-> Frac 1.0R (acc2 e i j))
  ensures a |-> Frac 1.0R e
{
  forevery_map_2
    (fun (i : natlt rows) (j : natlt cols) ->
      Cell a (idx2 i j) |-> Frac 1.0R (acc2 e i j))
    (fun (i : natlt rows) (j : natlt cols) ->
      Cell a (idx2 i j) |-> Frac 1.0R (acc e (idx2 i j)))
    fn i j {
      assert pure (acc e (idx2 i j) == acc2 e i j);
      rewrite
        (Cell a (idx2 i j) |-> Frac 1.0R (acc2 e i j))
        as
        (Cell a (idx2 i j) |-> Frac 1.0R (acc e (idx2 i j)));
    };
  tensor_iraise2 a #1.0R #e;
}

ghost
fn strided_cells2_gather_v
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos) (e : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr). strided_cells2_v a nthr tid e)
  ensures a |-> Frac 1.0R e
{
  forevery_map #(natlt nthr)
    (fun tid -> strided_cells2_v a nthr tid e)
    (fun tid ->
      forall+ (ij : stride_index2 rows cols nthr tid).
        Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R (acc2 e ij._1 ij._2))
    fn tid {
      unfold strided_cells2_v a nthr tid e;
    };
  forevery_flatten_dep
    (fun (tid : natlt nthr) (ij : stride_index2 rows cols nthr tid) ->
      Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R (acc2 e ij._1 ij._2));
  forevery_iso (stride_partition_bij rows cols nthr)
    (fun x ->
      Cell a (idx2 x._2._1 x._2._2) |-> Frac 1.0R (acc2 e x._2._1 x._2._2));
  forevery_map #(natlt rows & natlt cols)
    (fun ij ->
      Cell a
        (idx2
          ((stride_partition_bij rows cols nthr).gg ij)._2._1
          ((stride_partition_bij rows cols nthr).gg ij)._2._2)
        |-> Frac 1.0R
          (acc2 e
            ((stride_partition_bij rows cols nthr).gg ij)._2._1
            ((stride_partition_bij rows cols nthr).gg ij)._2._2))
    (fun ij -> Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R (acc2 e ij._1 ij._2))
    fn ij {
      assert pure (
        ((stride_partition_bij rows cols nthr).gg ij)._2 == ij);
      rewrite
        (tensor_pts_to_cell a #1.0R
          (idx2
            ((stride_partition_bij rows cols nthr).gg ij)._2._1
            ((stride_partition_bij rows cols nthr).gg ij)._2._2)
          (acc2 e
            ((stride_partition_bij rows cols nthr).gg ij)._2._1
            ((stride_partition_bij rows cols nthr).gg ij)._2._2))
        as
        (tensor_pts_to_cell a #1.0R (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2));
    };
  forevery_unflatten'
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R (acc2 e ij._1 ij._2));
  cells2_gather_v a e;
}

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
{
  forevery_factor' (block_threads nw) (SZ.v nw) BW.warp_size
    (fun w lane ->
      strided_cells2
        (array2_subtile shO 16 (SZ.v d <: pos) w 0)
        BW.warp_size lane);
  forevery_map #(natlt (SZ.v nw))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        strided_cells2
          (array2_subtile shO 16 (SZ.v d <: pos) w 0)
          BW.warp_size lane)
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile
            (array2_subtile shO 16 (SZ.v d <: pos) w 0)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    fn w {
      strided_cells2_gather
        (array2_subtile shO 16 (SZ.v d <: pos) w 0)
        BW.warp_size;
      with e. assert (
        array2_subtile shO 16 (SZ.v d <: pos) w 0 |-> Frac 1.0R e);
      array2_stride_tile
        (array2_subtile shO 16 (SZ.v d <: pos) w 0)
        warp_row_span 16;
      forevery_map_2
        (fun (tr : natlt warp_row_span) (tc : natlt 16) ->
          array2_stride_subtile
            (array2_subtile shO 16 (SZ.v d <: pos) w 0)
            warp_row_span 16 tr tc
            |-> Frac 1.0R (ematrix_stride_subtile e warp_row_span 16 tr tc))
        (fun (tr : natlt warp_row_span) (tc : natlt 16) ->
          exists* (x : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
            array2_stride_subtile
              (array2_subtile shO 16 (SZ.v d <: pos) w 0)
              warp_row_span 16 tr tc
              |-> Frac 1.0R x)
        fn tr tc { () };
      forevery_unfactor' BW.warp_size warp_row_span 16
        (fun tr tc ->
          exists* (x : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
            array2_stride_subtile
              (array2_subtile shO 16 (SZ.v d <: pos) w 0)
              warp_row_span 16 tr tc
              |-> Frac 1.0R x);
    };
  forevery_unfactor' (block_threads nw) (SZ.v nw) BW.warp_size
    (fun w lane ->
      exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile
          (array2_subtile shO 16 (SZ.v d <: pos) w 0)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e);
}

ghost
fn row_cell_to_matrix
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (r : natlt rows) (c : natlt cols)
  requires cell_full (row a r) c
  ensures
    exists* (v : et). Cell a (idx2 r c) |-> Frac 1.0R v
{
  unfold cell_full (row a r) c;
  with v. assert (tensor_pts_to_cell (row a r) #1.0R (idx1 c) v);
  rewrite each (row a r) as (sliceof a 0 r);
  tensor_slice_cell_eq a 0 r (idx1 c) 1.0R v;
  rewrite
    (tensor_pts_to_cell (sliceof a 0 r) #1.0R (idx1 c) v)
    as
    (tensor_pts_to_cell a #1.0R
      ((abs_bring_forward_bij 0 (rows @| cols @| INil)).gg (r, idx1 c))
      v);
  assert pure (
    (abs_bring_forward_bij 0 (rows @| cols @| INil)).gg (r, idx1 c)
      == idx2 r c);
  rewrite
    (tensor_pts_to_cell a #1.0R
      ((abs_bring_forward_bij 0 (rows @| cols @| INil)).gg (r, idx1 c))
      v)
    as
    (tensor_pts_to_cell a #1.0R (idx2 r c) v);
}

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
{
  forevery_map #(natlt (block_threads nw))
    (fun tid -> b1_pre_one nw a tid)
    (fun tid ->
      when__ (thread_lane nw tid < 16) (fun _ ->
        cell_full (row a (thread_w nw tid))
          (thread_lane nw tid <: natlt 16)))
    fn tid {
      unfold b1_pre_one nw a tid;
    };
  forevery_factor' (block_threads nw) (SZ.v nw) BW.warp_size
    (fun w lane ->
      when__ (lane < 16) (fun _ ->
        cell_full (row a w) (lane <: natlt 16)));
  forevery_map #(natlt (SZ.v nw))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          cell_full (row a w) (lane <: natlt 16)))
    (fun w ->
      forall+ (c : natlt 16).
        exists* (v : et). Cell a (idx2 w c) |-> Frac 1.0R v)
    fn w {
      forevery_map #(natlt BW.warp_size)
        (fun lane ->
          when__ (lane < 16) (fun _ ->
            cell_full (row a w) (lane <: natlt 16)))
        (fun lane ->
          when_ (lane < 16)
            (cell_full (row a w) (clamp_nat_lt 16 lane)))
        fn lane {
          rewrite
            (when__ (lane < 16) (fun _ ->
              cell_full (row a w) (lane <: natlt 16)))
            as
            (when_ (lane < 16)
              (cell_full (row a w) (clamp_nat_lt 16 lane)));
        };
      forevery_refine_pred
        (fun (lane : natlt BW.warp_size) ->
          cell_full (row a w) (clamp_nat_lt 16 lane))
        (fun (lane : natlt BW.warp_size) -> lane < 16);
      forevery_map #(lane : natlt BW.warp_size { lane < 16 })
        (fun lane -> cell_full (row a w) (clamp_nat_lt 16 lane))
        (fun lane -> cell_full (row a w) (lane <: natlt 16))
        fn lane {
          rewrite
            (cell_full (row a w) (clamp_nat_lt 16 lane))
            as
            (cell_full (row a w) (lane <: natlt 16));
        };
      forevery_natlt_restrict BW.warp_size
        (fun (c : natlt 16) -> cell_full (row a w) c);
      forevery_map #(natlt 16)
        (fun c -> cell_full (row a w) c)
        (fun c ->
          exists* (v : et). Cell a (idx2 w c) |-> Frac 1.0R v)
        fn c {
          row_cell_to_matrix a w c;
        };
    };
  cells2_gather a;
  with e. assert (a |-> Frac 1.0R e);
  tensor_share_n a (block_threads nw);
  forevery_map #(natlt (block_threads nw))
    (fun _ -> a |-> Frac (1.0R /. (block_threads nw)) e)
    (fun _ ->
      exists* (x : chest2 et (SZ.v nw) 16).
        a |-> Frac (1.0R /. (block_threads nw)) x)
    fn tid { () };
}

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
{
  forevery_map #(natlt (block_threads nw))
    (fun tid -> b2_o_pre nw d shO tid)
    (fun tid ->
      exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile
          (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
          warp_row_span 16
          (thread_lane nw tid / 16) (thread_lane nw tid % 16)
          |-> Frac 1.0R e)
    fn tid {
      unfold b2_o_pre nw d shO tid;
    };
  forevery_factor' (block_threads nw) (SZ.v nw) BW.warp_size
    (fun w lane ->
      exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile
          (array2_subtile shO 16 (SZ.v d <: pos) w 0)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e);
  forevery_map #(natlt (SZ.v nw))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile
            (array2_subtile shO 16 (SZ.v d <: pos) w 0)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    (fun w ->
      exists* (e : chest2 et 16 (SZ.v d)).
        array2_subtile shO 16 (SZ.v d <: pos) w 0 |-> Frac 1.0R e)
    fn w {
      forevery_factor' BW.warp_size warp_row_span 16
        (fun tr tc ->
          exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
            array2_stride_subtile
              (array2_subtile shO 16 (SZ.v d <: pos) w 0)
              warp_row_span 16 tr tc
              |-> Frac 1.0R e);
      let tf = forevery_exists_2
        (fun (tr : natlt warp_row_span) (tc : natlt 16)
          (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)) ->
            array2_stride_subtile
              (array2_subtile shO 16 (SZ.v d <: pos) w 0)
              warp_row_span 16 tr tc
              |-> Frac 1.0R e);
      array2_stride_untile'
        (array2_subtile shO 16 (SZ.v d <: pos) w 0)
        warp_row_span 16 tf #1.0R;
    };
  let rf = forevery_exists
    (fun (w : natlt (SZ.v nw)) (e : chest2 et 16 (SZ.v d)) ->
      array2_subtile shO 16 (SZ.v d <: pos) w 0 |-> Frac 1.0R e);
  forevery_map #(natlt (SZ.v nw))
    (fun w ->
      array2_subtile shO 16 (SZ.v d <: pos) w 0 |-> Frac 1.0R (rf w))
    (fun w ->
      forall+ (tc : natlt 1).
        array2_subtile shO 16 (SZ.v d <: pos) w tc |-> Frac 1.0R (rf w))
    fn w {
      forevery_singleton_intro
        (fun (tc : natlt 1) ->
          array2_subtile shO 16 (SZ.v d <: pos) w tc |-> Frac 1.0R (rf w));
    };
  forevery_rw_size2
    (SZ.v nw) (SZ.v nw * 16 / 16)
    1 (SZ.v d / SZ.v d)
    #(fun tr tc ->
      array2_subtile shO 16 (SZ.v d <: pos) tr tc |-> Frac 1.0R (rf tr));
  array2_untile' shO 16 (SZ.v d <: pos) (fun w _tc -> rf w) #1.0R;
  with e. assert (shO |-> Frac 1.0R e);
  tensor_share_n shO (block_threads nw);
  forevery_map #(natlt (block_threads nw))
    (fun _ -> shO |-> Frac (1.0R /. (block_threads nw)) e)
    (fun _ ->
      exists* (x : chest2 et (SZ.v nw * 16) (SZ.v d)).
        shO |-> Frac (1.0R /. (block_threads nw)) x)
    fn tid { () };
}

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
{
  forevery_map #(natlt (block_threads nw))
    (fun tid -> b2_scale_pre nw shscale shgl tid)
    (fun tid ->
      if_ (b2_active nw tid)
        (scale_tile nw shscale
          (clamp_nat_lt 16 (thread_lane nw tid)) **
         cell_full shgl (clamp_nat_lt 16 (thread_lane nw tid))))
    fn tid {
      unfold b2_scale_pre nw shscale shgl tid;
    };
  forevery_factor' (block_threads nw) (SZ.v nw) BW.warp_size
    (fun w lane ->
      if_ (t2b (w = 0 /\ lane < 16))
        (scale_tile nw shscale (clamp_nat_lt 16 lane) **
         cell_full shgl (clamp_nat_lt 16 lane)));
  forevery_flatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      if_ (t2b (w = 0 /\ lane < 16))
        (scale_tile nw shscale (clamp_nat_lt 16 lane) **
         cell_full shgl (clamp_nat_lt 16 lane)));
  forevery_map #(natlt (SZ.v nw) & natlt BW.warp_size)
    (fun wl ->
      if_ (t2b (wl._1 = 0 /\ wl._2 < 16))
        (scale_tile nw shscale (clamp_nat_lt 16 wl._2) **
         cell_full shgl (clamp_nat_lt 16 wl._2)))
    (fun wl ->
      when_ (wl._1 = 0 /\ wl._2 < 16)
        (scale_tile nw shscale (clamp_nat_lt 16 wl._2) **
         cell_full shgl (clamp_nat_lt 16 wl._2)))
    fn wl {
      rewrite
        (if_ (t2b (wl._1 = 0 /\ wl._2 < 16))
          (scale_tile nw shscale (clamp_nat_lt 16 wl._2) **
           cell_full shgl (clamp_nat_lt 16 wl._2)))
        as
        (when_ (wl._1 = 0 /\ wl._2 < 16)
          (scale_tile nw shscale (clamp_nat_lt 16 wl._2) **
           cell_full shgl (clamp_nat_lt 16 wl._2)));
    };
  forevery_refine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      scale_tile nw shscale (clamp_nat_lt 16 wl._2) **
      cell_full shgl (clamp_nat_lt 16 wl._2))
    (fun wl -> wl._1 = 0 /\ wl._2 < 16);
  forevery_rw_type
    (wl : (natlt (SZ.v nw) & natlt BW.warp_size) {
      wl._1 = 0 /\ wl._2 < 16})
    (b2_active_idx nw)
    (fun wl ->
      scale_tile nw shscale (clamp_nat_lt 16 wl._2) **
      cell_full shgl (clamp_nat_lt 16 wl._2));
  forevery_iso (b2_active_lane_bij nw)
    (fun wl ->
      scale_tile nw shscale (clamp_nat_lt 16 wl._2) **
      cell_full shgl (clamp_nat_lt 16 wl._2));
  forevery_map #(natlt 16)
    (fun lane ->
      scale_tile nw shscale
        (clamp_nat_lt 16 ((b2_active_lane_bij nw).gg lane)._2) **
      cell_full shgl
        (clamp_nat_lt 16 ((b2_active_lane_bij nw).gg lane)._2))
    (fun lane ->
      scale_tile nw shscale lane **
      cell_full shgl lane)
    fn lane {
      rewrite
        (scale_tile nw shscale
          (clamp_nat_lt 16 ((b2_active_lane_bij nw).gg lane)._2) **
         cell_full shgl
          (clamp_nat_lt 16 ((b2_active_lane_bij nw).gg lane)._2))
        as
        (scale_tile nw shscale lane **
         cell_full shgl lane);
    };
  forevery_unzip #(natlt 16)
    (fun lane -> scale_tile nw shscale lane)
    (fun lane -> cell_full shgl lane);

  forevery_map #(natlt 16)
    (fun tc -> scale_tile nw shscale tc)
    (fun tc ->
      forall+ (_tr : natlt 1).
        scale_tile nw shscale tc)
    fn tc {
      forevery_singleton_intro
        (fun (_tr : natlt 1) -> scale_tile nw shscale tc);
    };
  forevery_commute
    (fun (_tc : natlt 16) (_tr : natlt 1) ->
      scale_tile nw shscale _tc);
  forevery_map_2
    (fun (_tr : natlt 1) (tc : natlt 16) ->
      scale_tile nw shscale tc)
    (fun (tr : natlt 1) (tc : natlt 16) ->
      exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to (array2_stride_subtile shscale 1 16 tr tc) e)
    fn tr tc {
      unfold scale_tile nw shscale tc;
      with e. assert (
        tensor_pts_to (array2_stride_subtile shscale 1 16 0 tc) e);
      rewrite
        (tensor_pts_to (array2_stride_subtile shscale 1 16 0 tc) e)
        as
        (tensor_pts_to (array2_stride_subtile shscale 1 16 tr tc) e);
    };
  let sf = forevery_exists_2
    (fun (tr : natlt 1) (tc : natlt 16)
      (e : chest2 et (SZ.v nw) 1) ->
        tensor_pts_to (array2_stride_subtile shscale 1 16 tr tc) e);
  array2_stride_untile' shscale 1 16 sf #1.0R;

  forevery_map #(natlt 16)
    (fun lane -> cell_full shgl lane)
    (fun lane ->
      exists* (v : et). Cell shgl (idx1 lane) |-> Frac 1.0R v)
    fn lane {
      unfold cell_full shgl lane;
    };
  cells1_gather shgl;

  with es. assert (shscale |-> Frac 1.0R es);
  tensor_share_n shscale (block_threads nw);
  forevery_map #(natlt (block_threads nw))
    (fun _ -> shscale |-> Frac (1.0R /. (block_threads nw)) es)
    (fun _ ->
      exists* (e : chest2 et (SZ.v nw) 16).
        shscale |-> Frac (1.0R /. (block_threads nw)) e)
    fn tid { () };
  with egl. assert (shgl |-> Frac 1.0R egl);
  tensor_share_n shgl (block_threads nw);
  forevery_map #(natlt (block_threads nw))
    (fun _ -> shgl |-> Frac (1.0R /. (block_threads nw)) egl)
    (fun _ ->
      exists* (e : chest1 et 16).
        shgl |-> Frac (1.0R /. (block_threads nw)) e)
    fn tid { () };
}

ghost
fn barrier_ok
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (eQsh : chest2 et_ab 16 (SZ.v d))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (it : nat)
  requires
    forall+ (tid : natlt (block_threads nw)).
      barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid
  ensures
    forall+ (tid : natlt (block_threads nw)).
      barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid
{
  if (it = 0) {
    forevery_map #(natlt (block_threads nw))
      (fun tid -> barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid)
      (fun tid -> b0_pre nw d shQ eQsh shO tid)
      fn tid {
        rewrite
          (barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid)
          as
          (b0_pre nw d shQ eQsh shO tid);
      };
    forevery_map #(natlt (block_threads nw))
      (fun tid -> b0_pre nw d shQ eQsh shO tid)
      (fun tid ->
        strided_cells2_v shQ (block_threads nw) tid eQsh **
        strided_cells2
          (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
          BW.warp_size (thread_lane nw tid))
      fn tid {
        unfold b0_pre nw d shQ eQsh shO tid;
      };
    forevery_unzip #(natlt (block_threads nw))
      (fun tid -> strided_cells2_v shQ (block_threads nw) tid eQsh)
      (fun tid ->
        strided_cells2
          (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
          BW.warp_size (thread_lane nw tid));

    strided_cells2_gather_v shQ (block_threads nw) eQsh;
    tensor_share_n shQ (block_threads nw);

    b0_o_transform nw d shO;
    forevery_zip #(natlt (block_threads nw))
      (fun _ -> shQ |-> Frac (1.0R /. (block_threads nw)) eQsh)
      (fun tid ->
        exists* (e : chest2 et_acc (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile
            (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
            warp_row_span 16
            (thread_lane nw tid / 16) (thread_lane nw tid % 16)
            |-> Frac 1.0R e);
    forevery_map #(natlt (block_threads nw))
      (fun tid ->
        (shQ |-> Frac (1.0R /. (block_threads nw)) eQsh) **
        (exists* (e : chest2 et_acc (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile
            (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
            warp_row_span 16
            (thread_lane nw tid / 16) (thread_lane nw tid % 16)
            |-> Frac 1.0R e))
      (fun tid -> barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid)
      fn tid {
        rewrite
          ((shQ |-> Frac (1.0R /. (block_threads nw)) eQsh) **
           (exists* (e : chest2 et_acc (16 / warp_row_span) (SZ.v d / 16)).
              array2_stride_subtile
                (array2_subtile shO 16 (SZ.v d <: pos) (thread_w nw tid) 0)
                warp_row_span 16
                (thread_lane nw tid / 16) (thread_lane nw tid % 16)
                |-> Frac 1.0R e))
          as
          (b0_post nw d shQ eQsh shO tid);
        rewrite
          (b0_post nw d shQ eQsh shO tid)
          as
          (barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid);
      };
  } else if (it = 1) {
    forevery_map #(natlt (block_threads nw))
      (fun tid -> barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid)
      (fun tid -> b1_pre nw shM shL tid)
      fn tid {
        rewrite
          (barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid)
          as
          (b1_pre nw shM shL tid);
      };
    forevery_map #(natlt (block_threads nw))
      (fun tid -> b1_pre nw shM shL tid)
      (fun tid -> b1_pre_one nw shM tid ** b1_pre_one nw shL tid)
      fn tid {
        unfold b1_pre nw shM shL tid;
      };
    forevery_unzip #(natlt (block_threads nw))
      (fun tid -> b1_pre_one nw shM tid)
      (fun tid -> b1_pre_one nw shL tid);
    b1_matrix_transform nw shM;
    b1_matrix_transform nw shL;
    forevery_zip #(natlt (block_threads nw))
      (fun _ ->
        exists* (e : chest2 et_acc (SZ.v nw) 16).
          shM |-> Frac (1.0R /. (block_threads nw)) e)
      (fun _ ->
        exists* (e : chest2 et_acc (SZ.v nw) 16).
          shL |-> Frac (1.0R /. (block_threads nw)) e);
    forevery_map #(natlt (block_threads nw))
      (fun _ ->
        (exists* (e : chest2 et_acc (SZ.v nw) 16).
          shM |-> Frac (1.0R /. (block_threads nw)) e) **
        (exists* (e : chest2 et_acc (SZ.v nw) 16).
          shL |-> Frac (1.0R /. (block_threads nw)) e))
      (fun tid -> barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid)
      fn tid {
        rewrite
          ((exists* (e : chest2 et_acc (SZ.v nw) 16).
              shM |-> Frac (1.0R /. (block_threads nw)) e) **
           (exists* (e : chest2 et_acc (SZ.v nw) 16).
              shL |-> Frac (1.0R /. (block_threads nw)) e))
          as
          (b1_post nw shM shL tid);
        rewrite
          (b1_post nw shM shL tid)
          as
          (barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid);
      };
  } else if (it = 2) {
    forevery_map #(natlt (block_threads nw))
      (fun tid -> barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid)
      (fun tid -> b2_pre nw d shscale shO shgl tid)
      fn tid {
        rewrite
          (barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid)
          as
          (b2_pre nw d shscale shO shgl tid);
      };
    forevery_map #(natlt (block_threads nw))
      (fun tid -> b2_pre nw d shscale shO shgl tid)
      (fun tid ->
        b2_o_pre nw d shO tid ** b2_scale_pre nw shscale shgl tid)
      fn tid {
        unfold b2_pre nw d shscale shO shgl tid;
      };
    forevery_unzip #(natlt (block_threads nw))
      (fun tid -> b2_o_pre nw d shO tid)
      (fun tid -> b2_scale_pre nw shscale shgl tid);
    b2_o_transform nw d shO;
    b2_scale_transform nw shscale shgl;
    forevery_zip #(natlt (block_threads nw))
      (fun _ ->
        exists* (e : chest2 et_acc (SZ.v nw * 16) (SZ.v d)).
          shO |-> Frac (1.0R /. (block_threads nw)) e)
      (fun _ ->
        exists* (e : chest1 et_acc 16).
          shgl |-> Frac (1.0R /. (block_threads nw)) e);
    forevery_zip #(natlt (block_threads nw))
      (fun _ ->
        exists* (e : chest2 et_acc (SZ.v nw) 16).
          shscale |-> Frac (1.0R /. (block_threads nw)) e)
      (fun _ ->
        (exists* (e : chest2 et_acc (SZ.v nw * 16) (SZ.v d)).
          shO |-> Frac (1.0R /. (block_threads nw)) e) **
        (exists* (e : chest1 et_acc 16).
          shgl |-> Frac (1.0R /. (block_threads nw)) e));
    forevery_map #(natlt (block_threads nw))
      (fun _ ->
        (exists* (e : chest2 et_acc (SZ.v nw) 16).
          shscale |-> Frac (1.0R /. (block_threads nw)) e) **
        ((exists* (e : chest2 et_acc (SZ.v nw * 16) (SZ.v d)).
          shO |-> Frac (1.0R /. (block_threads nw)) e) **
         (exists* (e : chest1 et_acc 16).
          shgl |-> Frac (1.0R /. (block_threads nw)) e)))
      (fun tid -> barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid)
      fn tid {
        rewrite
          ((exists* (e : chest2 et_acc (SZ.v nw) 16).
              shscale |-> Frac (1.0R /. (block_threads nw)) e) **
           ((exists* (e : chest2 et_acc (SZ.v nw * 16) (SZ.v d)).
              shO |-> Frac (1.0R /. (block_threads nw)) e) **
            (exists* (e : chest1 et_acc 16).
              shgl |-> Frac (1.0R /. (block_threads nw)) e)))
          as
          (b2_post nw d shscale shO shgl tid);
        rewrite
          (b2_post nw d shscale shO shgl tid)
          as
          (barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid);
      };
  } else {
    forevery_map #(natlt (block_threads nw))
      (fun tid -> barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid)
      (fun tid -> barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid)
      fn tid {
        rewrite (barrier_rin nw d shQ eQsh shM shL shscale shO shgl it tid) as emp;
        rewrite emp as
          (barrier_rout nw d shQ eQsh shM shL shscale shO shgl it tid);
      };
  }
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
      cell_full (row m w) lane) **
    when__ (lane < 16) (fun _ ->
      cell_full (row l w) lane)
  ensures
    if_ (lane_active 16sz ls)
      (ml_cells 16sz (row m (SZ.v ws))
        (row l (SZ.v ws)) ls)
{
  rewrite each w as (SZ.v ws);
  rewrite each lane as (SZ.v ls);
  rewrite
    (when__ (SZ.v ls < 16) (fun _ ->
      cell_full (row m (SZ.v ws)) (SZ.v ls)))
    as
    (when__ (lane_active 16sz ls) (fun _ ->
      cell_full (row m (SZ.v ws)) (SZ.v ls)));
  rewrite
    (when__ (SZ.v ls < 16) (fun _ ->
      cell_full (row l (SZ.v ws)) (SZ.v ls)))
    as
    (when__ (lane_active 16sz ls) (fun _ ->
      cell_full (row l (SZ.v ws)) (SZ.v ls)));
  if lane_active 16sz ls {
    rewrite
      (when__ l_True (fun _ ->
        cell_full (row m (SZ.v ws)) (SZ.v ls)))
      as
      (cell_full (row m (SZ.v ws)) (SZ.v ls));
    rewrite
      (when__ l_True (fun _ ->
        cell_full (row l (SZ.v ws)) (SZ.v ls)))
      as
      (cell_full (row l (SZ.v ws)) (SZ.v ls));
    flash_cell_full_to_n
      (row m (SZ.v ws)) (SZ.v ls);
    flash_cell_full_to_n
      (row l (SZ.v ws)) (SZ.v ls);
    assert pure (
      SZ.v (clamp_lt 16sz ls) == SZ.v ls);
    rewrite each (SZ.v ls) as
      (SZ.v (clamp_lt 16sz ls));
    fold ml_cells 16sz (row m (SZ.v ws))
      (row l (SZ.v ws)) ls;
    if_intro_true (
      ml_cells 16sz (row m (SZ.v ws))
        (row l (SZ.v ws)) ls);
  } else {
    rewrite
      (when__ l_False (fun _ ->
        cell_full (row m (SZ.v ws)) (SZ.v ls)))
      as emp;
    rewrite
      (when__ l_False (fun _ ->
        cell_full (row l (SZ.v ws)) (SZ.v ls)))
      as emp;
    if_intro_false (
      ml_cells 16sz (row m (SZ.v ws))
        (row l (SZ.v ws)) ls);
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
            (clamp_nat_lt 16 (SZ.v ls))) e)
       ** cell_full gm (clamp_nat_lt 16 (SZ.v ls))
       ** cell_full gl (clamp_nat_lt 16 (SZ.v ls))))
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
      (cell_full gm (clamp_nat_lt 16 lane))
{
  if combine_active 16sz ws ls {
    if_elim_true (
      cell_full gm (SZ.v (clamp_lt 16sz ls)));
    assert pure (w = 0 /\ lane < 16);
    assert pure (
      SZ.v (clamp_lt 16sz ls) ==
      clamp_nat_lt 16 lane);
    rewrite
      (cell_full gm (SZ.v (clamp_lt 16sz ls)))
      as
      (cell_full gm (clamp_nat_lt 16 lane));
    flash_when_prop_intro_true
      (w = 0 /\ lane < 16)
      (t2b (w = 0 /\ lane < 16))
      (cell_full gm (clamp_nat_lt 16 lane));
  } else {
    if_elim_false (
      cell_full gm (SZ.v (clamp_lt 16sz ls)));
    assert pure (~(w = 0 /\ lane < 16));
    flash_when_prop_intro_false
      (w = 0 /\ lane < 16)
      (t2b (w = 0 /\ lane < 16))
      (cell_full gm (clamp_nat_lt 16 lane));
  }
}
