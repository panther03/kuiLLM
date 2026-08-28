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
open Kuiops.Array2.Strided
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
module SF = Kuiops.Sdpa.Flash.Spec.Float
module Trade = Pulse.Lib.Trade

let natlt1_singleton () : Lemma (forall (x : natlt 1). (0 <: natlt 1) == x)
= introduce forall (x : natlt 1). (0 <: natlt 1) == x with ()


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
    (forall+ (tid : natlt nthr). strided_cells2 a nthr tid)
  ensures live a
{
  forevery_map #(natlt nthr)
    (fun tid -> strided_cells2 a nthr tid)
    (fun tid ->
      forall+ (ij : stride_index2 rows cols nthr tid).
        exists* (v : et).
          Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v)
    fn tid {
      unfold strided_cells2 a nthr tid;
    };
  forevery_flatten_dep
    (fun (tid : natlt nthr)
      (ij : stride_index2 rows cols nthr tid) ->
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
      strided_cells2 a nthr tid
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
      (tid : natlt nthr & stride_index2 rows cols nthr tid)) ->
      Cell a (idx2 x._2._1 x._2._2) |-> Frac 1.0R
        (acc e (idx2 x._2._1 x._2._2)));
  forevery_unflatten_dep
    (fun (tid : natlt nthr)
      (ij : stride_index2 rows cols nthr tid) ->
      Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
        (acc e (idx2 ij._1 ij._2)));
  forevery_map #(natlt nthr)
    (fun tid ->
      forall+ (ij : stride_index2 rows cols nthr tid).
        Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
          (acc e (idx2 ij._1 ij._2)))
    (fun tid -> strided_cells2 a nthr tid)
    fn tid {
      forevery_map
        (fun (ij : stride_index2 rows cols nthr tid) ->
          Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R
            (acc e (idx2 ij._1 ij._2)))
        (fun (ij : stride_index2 rows cols nthr tid) ->
          exists* (v : et).
            Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v)
        fn ij { () };
      fold strided_cells2 a nthr tid;
    };
}

ghost
fn flash_gather_strided_cells2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (tid : natlt nthr).
      strided_cells2 a nthr tid)
  ensures live a
{
  flash_strided_cells2_gather a nthr;
}

ghost
fn flash_warp_split_stride
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#_ : squash (warp_row_span /? rows))
  (#_ : squash (16 /? cols))
  requires live a
  ensures
    forall+ (lane : natlt BW.warp_size).
      exists* (r : chest2 et (rows / warp_row_span) (cols / 16)).
        array2_stride_subtile a warp_row_span 16
          (lane / 16) (lane % 16) |-> Frac 1.0R r
{
  unfold live a;
  with e. assert (a |-> e);
  array2_stride_tile a warp_row_span 16;
  forevery_unfactor' BW.warp_size warp_row_span 16
    (fun (tr : natlt warp_row_span) (tc : natlt 16) ->
      array2_stride_subtile a warp_row_span 16 tr tc
        |-> Frac 1.0R
          (ematrix_stride_subtile e warp_row_span 16 tr tc));
  forevery_map #(natlt BW.warp_size)
    (fun lane ->
      array2_stride_subtile a warp_row_span 16
        (lane / 16) (lane % 16) |-> Frac 1.0R
          (ematrix_stride_subtile e warp_row_span 16
            (lane / 16) (lane % 16)))
    (fun lane ->
      exists* (r : chest2 et
        (rows / warp_row_span) (cols / 16)).
        array2_stride_subtile a warp_row_span 16
          (lane / 16) (lane % 16) |-> Frac 1.0R r)
    fn lane { () };
}

ghost
fn flash_warp_gather_stride
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#_ : squash (warp_row_span /? rows))
  (#_ : squash (16 /? cols))
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (lane : natlt BW.warp_size).
      exists* (r : chest2 et (rows / warp_row_span) (cols / 16)).
        array2_stride_subtile a warp_row_span 16
          (lane / 16) (lane % 16) |-> Frac 1.0R r)
  ensures live a
{
  let rf = forevery_exists
    (fun (lane : natlt BW.warp_size)
      (r : chest2 et (rows / warp_row_span) (cols / 16)) ->
      array2_stride_subtile a warp_row_span 16
        (lane / 16) (lane % 16) |-> Frac 1.0R r);
  forevery_ext #(natlt BW.warp_size)
    (fun lane ->
      array2_stride_subtile a warp_row_span 16
        (lane / 16) (lane % 16) |-> Frac 1.0R (rf lane))
    (fun lane ->
      array2_stride_subtile a warp_row_span 16
        (lane / 16) (lane % 16) |-> Frac 1.0R
          (rf ((lane / 16) * 16 + (lane % 16))));
  forevery_factor' BW.warp_size warp_row_span 16
    (fun (tr : natlt warp_row_span) (tc : natlt 16) ->
      array2_stride_subtile a warp_row_span 16 tr tc
        |-> Frac 1.0R (rf (tr * 16 + tc)));
  array2_stride_untile' a warp_row_span 16
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
        (row_subtile a (clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> row_subtile a lane))
        as
        (when_ (lane < 16)
          (row_subtile a (clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      row_subtile a (clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> row_subtile a (clamp_nat_lt 16 lane))
    (fun lane -> row_subtile a (lane <: natlt 16))
    fn lane {
      rewrite
        (row_subtile a (clamp_nat_lt 16 lane))
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
        (cell_full a (clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> cell_full a lane))
        as
        (when_ (lane < 16)
          (cell_full a (clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      cell_full a (clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> cell_full a (clamp_nat_lt 16 lane))
    (fun lane -> cell_full a (lane <: natlt 16))
    fn lane {
      rewrite
        (cell_full a (clamp_nat_lt 16 lane))
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
        cell_full (row a w) lane)
{
  unfold live a;
  with e. assert (a |-> e);
  tensor_ilower2 a;
  forevery_map_2
    (fun (w : natlt (SZ.v nw)) (lane : natlt 16) ->
      Cell a (idx2 w lane) |-> Frac 1.0R (acc e (idx2 w lane)))
    (fun (w : natlt (SZ.v nw)) (lane : natlt 16) ->
      cell_full (row a w) lane)
    fn w lane {
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
        cell_full (row a w) lane)
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          cell_full (row a w) lane))
    fn w {
      forevery_natlt_extend BW.warp_size
        (fun (lane : natlt 16) ->
          cell_full (row a w) lane);
      forevery_unrefine_pred'
        (fun (lane : natlt BW.warp_size) -> lane < 16)
        (fun lane _ ->
          cell_full (row a w) (lane <: natlt 16));
      forevery_map #(natlt BW.warp_size)
        (fun lane ->
          when__ (lane < 16) (fun _ ->
            cell_full (row a w) (lane <: natlt 16)))
        (fun lane ->
          when__ (lane < 16) (fun _ ->
            cell_full (row a w) lane))
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
        cell_full (row a w) lane)
  ensures live a
{
  forevery_map #(natlt (SZ.v nw))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          cell_full (row a w) lane))
    (fun w ->
      forall+ (lane : natlt 16).
        exists* (v : et).
          Cell a (idx2 w lane) |-> Frac 1.0R v)
    fn w {
      forevery_map #(natlt BW.warp_size)
        (fun lane ->
          when__ (lane < 16) (fun _ ->
            cell_full (row a w) lane))
        (fun lane ->
          when_ (lane < 16)
            (cell_full (row a w) (clamp_nat_lt 16 lane)))
        fn lane {
          rewrite
            (when__ (lane < 16) (fun _ ->
              cell_full (row a w) lane))
            as
            (when_ (lane < 16)
              (cell_full (row a w)
                (clamp_nat_lt 16 lane)));
        };
      forevery_refine_pred
        (fun (lane : natlt BW.warp_size) ->
          cell_full (row a w) (clamp_nat_lt 16 lane))
        (fun lane -> lane < 16);
      forevery_map #(lane : natlt BW.warp_size { lane < 16 })
        (fun lane ->
          cell_full (row a w) (clamp_nat_lt 16 lane))
        (fun lane ->
          cell_full (row a w) (lane <: natlt 16))
        fn lane {
          rewrite
            (cell_full (row a w) (clamp_nat_lt 16 lane))
            as
            (cell_full (row a w) (lane <: natlt 16));
        };
      forevery_natlt_restrict BW.warp_size
        (fun (lane : natlt 16) ->
          cell_full (row a w) lane);
      forevery_map #(natlt 16)
        (fun lane ->
          cell_full (row a w) lane)
        (fun lane ->
          exists* (v : et).
            Cell a (idx2 w lane) |-> Frac 1.0R v)
        fn lane {
          unfold cell_full (row a w) lane;
          with v. assert (
            tensor_pts_to_cell (row a w) #1.0R
              (idx1 lane) v);
          rewrite each (row a w) as (sliceof a 0 w);
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
  natlt1_singleton ();
  forevery_singleton_elim'
    (fun (tr : natlt 1) ->
      forall+ (lane : natlt 16).
        array2_stride_subtile vscale 1 16 tr lane
          |-> Frac 1.0R
            (ematrix_stride_subtile escale 1 16 tr lane))
    (0 <: natlt 1);
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
        (cell_full vgm (clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> cell_full vgm lane))
        as
        (when_ (lane < 16)
          (cell_full vgm (clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      cell_full vgm (clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> cell_full vgm (clamp_nat_lt 16 lane))
    (fun lane -> cell_full vgm (lane <: natlt 16))
    fn lane {
      rewrite
        (cell_full vgm (clamp_nat_lt 16 lane))
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
        (cell_full vgl (clamp_nat_lt 16 lane)))
    fn lane {
      rewrite
        (when__ (lane < 16) (fun _ -> cell_full vgl lane))
        as
        (when_ (lane < 16)
          (cell_full vgl (clamp_nat_lt 16 lane)));
    };
  forevery_refine_pred
    (fun (lane : natlt BW.warp_size) ->
      cell_full vgl (clamp_nat_lt 16 lane))
    (fun (lane : natlt BW.warp_size) -> lane < 16)
    ;
  forevery_map #(lane : natlt BW.warp_size { lane < 16 })
    (fun lane -> cell_full vgl (clamp_nat_lt 16 lane))
    (fun lane -> cell_full vgl (lane <: natlt 16))
    fn lane {
      rewrite
        (cell_full vgl (clamp_nat_lt 16 lane))
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
            (clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (clamp_nat_lt 16 wl._2)
      ** cell_full vgl (clamp_nat_lt 16 wl._2))
    fn wl {
      rewrite each (wl._2 <: natlt 16)
        as (clamp_nat_lt 16 wl._2);
    };
  forevery_unrefine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (clamp_nat_lt 16 wl._2)
      ** cell_full vgl (clamp_nat_lt 16 wl._2))
    (fun wl -> wl._1 = 0 /\ wl._2 < 16);
  forevery_map #(natlt (SZ.v nw) & natlt BW.warp_size)
    (fun wl ->
      when_ (wl._1 = 0 /\ wl._2 < 16)
        ((exists* (e : chest2 et (SZ.v nw) 1).
           tensor_pts_to
             (array2_stride_subtile vscale 1 16 0
               (clamp_nat_lt 16 wl._2)) e)
         ** cell_full vgm (clamp_nat_lt 16 wl._2)
         ** cell_full vgl (clamp_nat_lt 16 wl._2)))
    (fun wl ->
      flash_combine_local nw vscale vgm vgl wl._1 wl._2)
    fn wl {
      rewrite
        (when_ (wl._1 = 0 /\ wl._2 < 16)
          ((exists* (e : chest2 et (SZ.v nw) 1).
             tensor_pts_to
               (array2_stride_subtile vscale 1 16 0
                  (clamp_nat_lt 16 wl._2)) e)
           ** cell_full vgm (clamp_nat_lt 16 wl._2)
           ** cell_full vgl (clamp_nat_lt 16 wl._2)))
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
               (clamp_nat_lt 16 wl._2)) e)
         ** cell_full vgm (clamp_nat_lt 16 wl._2)
         ** cell_full vgl (clamp_nat_lt 16 wl._2)))
    fn wl {
      unfold flash_combine_local nw vscale vgm vgl wl._1 wl._2;
    };
  forevery_refine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (clamp_nat_lt 16 wl._2)
      ** cell_full vgl (clamp_nat_lt 16 wl._2))
    (fun wl -> wl._1 = 0 /\ wl._2 < 16);
  forevery_map #(flash_combine_idx nw)
    (fun wl ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (clamp_nat_lt 16 wl._2)) e)
      ** cell_full vgm (clamp_nat_lt 16 wl._2)
      ** cell_full vgl (clamp_nat_lt 16 wl._2))
    (fun wl ->
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile vscale 1 16 0
            (wl._2 <: natlt 16)) e)
      ** cell_full vgm (wl._2 <: natlt 16)
      ** cell_full vgl (wl._2 <: natlt 16))
    fn wl {
      rewrite each (clamp_nat_lt 16 wl._2)
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
      if active {
        flash_when_prop_elim_true
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          let y = flash_padded_logical (SZ.v rows) x in
          out_cell b hq sq d gout
            y._1 y._2 (SZ.v group) y._3 y._4);
        assert pure (clamp_nat_lt (SZ.v rows) (rt * 16 + i)
          == rt * 16 + i);
        rewrite
          (let y = flash_padded_logical (SZ.v rows) x in
           out_cell b hq sq d gout
             y._1 y._2 (SZ.v group) y._3 y._4)
          as
          (out_cell b hq sq d gout bi kvh (SZ.v group)
            (rt * 16 + i) dd);
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
fn flash_output_remove_warps_v
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nw : szp)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et_ab lout)
  (escale : chest2 et_acc (SZ.v nw) 16)
  (eO : chest2 et_acc (SZ.v nw * 16) (SZ.v d))
  (egl : chest1 et_acc 16)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  requires
    forall+ (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size).
      when_ (w = 0)
        (out_store_cells_v nw b hq sq 16sz d rows gout escale eO egl
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane)
  ensures flash_block_output_v nw
    b hq hkv group sq rows tiles d gout escale eO egl bid

  {
  forevery_flatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      when_ (w = 0)
        (out_store_cells_v nw b hq sq 16sz d rows gout escale eO egl
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane));
  forevery_refine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      out_store_cells_v nw b hq sq 16sz d rows gout escale eO egl
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
      out_store_cells_v nw b hq sq 16sz d rows gout escale eO egl
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        wl._2);
  forevery_map
    (fun lane ->
      out_store_cells_v nw b hq sq 16sz d rows gout escale eO egl
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        ((flash_w0_bij nw).gg lane)._2)
    (fun lane ->
      out_store_cells_v nw b hq sq 16sz d rows gout escale eO egl
        (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
        (SZ.v group)
        (flash_bid_rt
          (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
        lane)
    fn lane {
      rewrite each ((flash_w0_bij nw).gg lane)._2 as lane;
    };
  fold (flash_block_output_v nw
    b hq hkv group sq rows tiles d gout escale eO egl bid);
}

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
{
  forevery_flatten
    (fun (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) ->
      when_ (w = 0 /\ lane < 16)
        (cell_full gm (clamp_nat_lt 16 lane)));
  forevery_refine_pred
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      cell_full gm (clamp_nat_lt 16 wl._2))
    (fun (wl : natlt (SZ.v nw) & natlt BW.warp_size) ->
      wl._1 = 0 /\ wl._2 < 16);
  forevery_rw_type
    (wl : (natlt (SZ.v nw) & natlt BW.warp_size) {
      wl._1 = 0 /\ wl._2 < 16})
    (flash_combine_idx nw)
    (fun wl -> cell_full gm (clamp_nat_lt 16 wl._2));
  forevery_iso (flash_combine_bij nw)
    (fun wl -> cell_full gm (clamp_nat_lt 16 wl._2));
  forevery_map #(natlt 16)
    (fun lane ->
      cell_full gm
        (clamp_nat_lt 16 ((flash_combine_bij nw).gg lane)._2))
    (fun lane -> cell_full gm lane)
    fn lane {
      rewrite each
        (clamp_nat_lt 16 ((flash_combine_bij nw).gg lane)._2)
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
  TRO.tensor_gather_n a (SZ.v nw * BW.warp_size) #f;
}

let flash_out_chest
  (#et : Type0)
  (b hq hkv group sq rows d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq })
  (vfun : (natlt (SZ.v b) & natlt (SZ.v hkv) &
    natlt (SZ.v rows) & natlt (SZ.v d)) -> GTot et)
  : GTot (chest (b @| hq @| sq @| d @| INil) et)
= mk (b @| hq @| sq @| d @| INil)
    (fun idx -> vfun ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).ff idx))

let flash_out_chest_acc
  (#et : Type0)
  (b hq hkv group sq rows d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq })
  (vfun : (natlt (SZ.v b) & natlt (SZ.v hkv) &
    natlt (SZ.v rows) & natlt (SZ.v d)) -> GTot et)
  : Lemma (forall (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
             natlt (SZ.v rows) & natlt (SZ.v d)).
      acc (flash_out_chest b hq hkv group sq rows d vfun)
        ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
      == vfun y)
= ()

(* Per-cell payload of the gather chain: the output cell at
   (bi, kvh, r, dd) pinned to [vfun]'s value there. *)
unfold
let out_cell_vf
  (#et : Type0) (b hq hkv sq rows d : szp)
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (vfun : (natlt (SZ.v b) & natlt (SZ.v hkv) &
    natlt (SZ.v rows) & natlt (SZ.v d)) -> GTot et)
  (bi : natlt (SZ.v b)) (kvh : natlt (SZ.v hkv)) (group : pos)
  (r : nat) (dd : natlt (SZ.v d)) : slprop
= out_cell_v b hq sq d gout bi kvh group r dd
    (vfun (bi, kvh, clamp_nat_lt (SZ.v rows) r, dd))

ghost
fn flash_gather_output_v
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nw : szp)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et_ab lout)
  (escale : (natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) -> chest2 et_acc (SZ.v nw) 16)
  (eO : (natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) -> chest2 et_acc (SZ.v nw * 16) (SZ.v d))
  (egl : (natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) -> chest1 et_acc 16)
  (vfun : (natlt (SZ.v b) & natlt (SZ.v hkv) &
    natlt (SZ.v rows) & natlt (SZ.v d)) -> GTot et_ab)
  requires
    pure (SZ.fits (tlayout_ulen lout)) **
    pure (forall (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) (i : natlt 16) (dd : natlt (SZ.v d)).
      flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + i < SZ.v rows ==>
      vfun (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid,
            flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid,
            clamp_nat_lt (SZ.v rows) (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + i),
            dd)
      == FC.fcast (SF.out_val (escale bid) (eO bid) (egl bid) i dd)) **
    (forall+ (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)).
      flash_block_output_v nw b hq hkv group sq rows tiles d gout
        (escale bid) (eO bid) (egl bid) bid)
  ensures
    gout |-> Frac 1.0R (flash_out_chest b hq hkv group sq rows d vfun)
{
  forevery_map
    (fun (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      flash_block_output_v nw b hq hkv group sq rows tiles d gout
        (escale bid) (eO bid) (egl bid) bid)
    (fun bid ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells_v nw b hq sq 16sz d rows gout
          (escale bid) (eO bid) (egl bid)
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane)
    fn bid {
      unfold flash_block_output_v nw
        b hq hkv group sq rows tiles d gout
        (escale bid) (eO bid) (egl bid) bid;
    };
  forevery_map
    (fun (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells_v nw b hq sq 16sz d rows gout
          (escale bid) (eO bid) (egl bid)
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
          (SZ.v group)
          (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
          lane)
    (fun bid ->
      forall+ (lane : natlt BW.warp_size)
        (ij : out_stride_index2
          16 (SZ.v d) BW.warp_size lane).
        when_ (
          flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
            + ij._1 < SZ.v rows)
          (out_cell_vf b hq hkv sq rows d gout vfun
              (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
              (SZ.v group)
              (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                + ij._1)
              ij._2))
    fn bid {
      forevery_map
        (fun (lane : natlt BW.warp_size) ->
          out_store_cells_v nw b hq sq 16sz d rows gout
            (escale bid) (eO bid) (egl bid)
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
            lane)
        (fun lane ->
          forall+ (ij : out_stride_index2
            16 (SZ.v d) BW.warp_size lane).
            when_ (
              flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                + ij._1 < SZ.v rows)
              (out_cell_vf b hq hkv sq rows d gout vfun
                  (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (SZ.v group)
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16
                    + ij._1)
                  ij._2))
        fn lane {
          unfold out_store_cells_v nw b hq sq 16sz d rows gout
            (escale bid) (eO bid) (egl bid)
            (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
            (SZ.v group)
            (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
            lane;
          forevery_map
            #(out_stride_index2 16 (SZ.v d) BW.warp_size lane)
            (fun ij ->
              when_ (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows)
                (out_cell_v b hq sq d gout
                  (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (SZ.v group)
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                  ij._2
                  (FC.fcast (SF.out_val
                    (escale bid) (eO bid) (egl bid) ij._1 ij._2))))
            (fun ij ->
              when_ (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows)
                (out_cell_vf b hq hkv sq rows d gout vfun
                  (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                  (SZ.v group)
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                  ij._2))
            fn ij {
              let bb : bool =
                (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows);
              if bb {
                flash_when_elim_true
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows)
                  (out_cell_v b hq sq d gout
                    (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (SZ.v group)
                    (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                    ij._2
                    (FC.fcast (SF.out_val
                      (escale bid) (eO bid) (egl bid) ij._1 ij._2)));
                rewrite
                  (out_cell_v b hq sq d gout
                    (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (SZ.v group)
                    (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                    ij._2
                    (FC.fcast (SF.out_val
                      (escale bid) (eO bid) (egl bid) ij._1 ij._2)))
                as
                  (out_cell_vf b hq hkv sq rows d gout vfun
                    (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (SZ.v group)
                    (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                    ij._2);
                flash_when_intro_true
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows)
                  (out_cell_vf b hq hkv sq rows d gout vfun
                    (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (SZ.v group)
                    (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                    ij._2);
              } else {
                assert pure (
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows) == false);
                flash_when_elim_false
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows)
                  (out_cell_v b hq sq d gout
                    (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (SZ.v group)
                    (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                    ij._2
                    (FC.fcast (SF.out_val
                      (escale bid) (eO bid) (egl bid) ij._1 ij._2)));
                flash_when_intro_false
                  (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1 < SZ.v rows)
                  (out_cell_vf b hq hkv sq rows d gout vfun
                    (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
                    (SZ.v group)
                    (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16 + ij._1)
                    ij._2);
              }
            };
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
          (out_cell_vf b hq hkv sq rows d gout vfun
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
          (out_cell_vf b hq hkv sq rows d gout vfun
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
            (out_cell_vf b hq hkv sq rows d gout vfun
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
        (out_cell_vf b hq hkv sq rows d gout vfun
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
        (out_cell_vf b hq hkv sq rows d gout vfun
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
        out_cell_vf b hq hkv sq rows d gout vfun
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
          out_cell_vf b hq hkv sq rows d gout vfun
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
        assert pure (clamp_nat_lt (SZ.v rows) (rt * 16 + i)
          == rt * 16 + i);
        rewrite
          (out_cell_vf b hq hkv sq rows d gout vfun bi kvh (SZ.v group)
            (rt * 16 + i) dd)
          as
          (let y = flash_padded_logical (SZ.v rows) x in
           out_cell_vf b hq hkv sq rows d gout vfun
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
          out_cell_vf b hq hkv sq rows d gout vfun
            y._1 y._2 (SZ.v group) y._3 y._4);
      } else {
        flash_when_prop_elim_false
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles) owner._1 * 16
            + owner._2._2._1 < SZ.v rows)
          active (
          out_cell_vf b hq hkv sq rows d gout vfun
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
          out_cell_vf b hq hkv sq rows d gout vfun
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
        out_cell_vf b hq hkv sq rows d gout vfun
          y._1 y._2 (SZ.v group) y._3 y._4));
  forevery_refine_pred
    (fun x ->
      let y = flash_padded_logical (SZ.v rows) x in
      out_cell_vf b hq hkv sq rows d gout vfun
        y._1 y._2 (SZ.v group) y._3 y._4)
    (flash_padded_active (SZ.v rows));
  forevery_iso
    (flash_active_padded_bij
      (SZ.v b) (SZ.v hkv) (SZ.v rows) (SZ.v tiles) (SZ.v d))
    (fun (x : flash_active_padded_idx
      (SZ.v b) (SZ.v hkv) (SZ.v rows) (SZ.v tiles) (SZ.v d)) ->
      let y = flash_padded_logical (SZ.v rows) x in
      out_cell_vf b hq hkv sq rows d gout vfun
        y._1 y._2 (SZ.v group) y._3 y._4);
  forevery_map
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      out_cell_vf b hq hkv sq rows d gout vfun
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
      out_cell_vf b hq hkv sq rows d gout vfun
        y._1 y._2 (SZ.v group) y._3 y._4)
    fn y {
      bij_inv_bk
        (flash_active_padded_bij
          (SZ.v b) (SZ.v hkv) (SZ.v rows)
          (SZ.v tiles) (SZ.v d))
        y;
      rewrite
        (out_cell_vf b hq hkv sq rows d gout vfun
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
        (out_cell_vf b hq hkv sq rows d gout vfun
          y._1 y._2 (SZ.v group) y._3 y._4);
    };
  forevery_map
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      out_cell_vf b hq hkv sq rows d gout vfun
        y._1 y._2 (SZ.v group) y._3 y._4)
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      tensor_pts_to_cell gout #1.0R
        ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
        (vfun (y._1, y._2, y._3, y._4)))
    fn y {
      assert pure (clamp_nat_lt (SZ.v rows) y._3 == y._3);
      unfold out_cell_v b hq sq d gout
        y._1 y._2 (SZ.v group) y._3 y._4
        (vfun (y._1, y._2, clamp_nat_lt (SZ.v rows) y._3, y._4));
      rewrite
        (tensor_pts_to_cell gout #1.0R
          (idx4 y._1
            (out_qh (SZ.v hq) (SZ.v sq) y._2
              (SZ.v group) y._3)
            (out_qpos (SZ.v sq) y._3)
            y._4)
          (vfun (y._1, y._2, clamp_nat_lt (SZ.v rows) y._3, y._4)))
      as
        (tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
          (vfun (y._1, y._2, y._3, y._4)));
    };
  flash_out_chest_acc b hq hkv group sq rows d vfun;
  forevery_map
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      tensor_pts_to_cell gout #1.0R
        ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
        (vfun (y._1, y._2, y._3, y._4)))
    (fun (y : natlt (SZ.v b) & natlt (SZ.v hkv) &
      natlt (SZ.v rows) & natlt (SZ.v d)) ->
      tensor_pts_to_cell gout #1.0R
        ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
        (acc (flash_out_chest b hq hkv group sq rows d vfun)
          ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)))
    fn y {
      rewrite
        (tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
          (vfun (y._1, y._2, y._3, y._4)))
      as
        (tensor_pts_to_cell gout #1.0R
          ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)
          (acc (flash_out_chest b hq hkv group sq rows d vfun)
            ((flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d)).gg y)));
    };
  forevery_iso_back
    (flash_output_logical_bij
          (SZ.v b) (SZ.v hkv) (SZ.v group) (SZ.v sq)
          (SZ.v hq) (SZ.v rows) (SZ.v d))
    (fun idx ->
      tensor_pts_to_cell gout #1.0R idx
        (acc (flash_out_chest b hq hkv group sq rows d vfun) idx));
  tensor_implode gout;
}
