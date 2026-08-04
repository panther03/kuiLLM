module Kuiops.Sdpa.Flash.Shmem

(* Opening and closing the block's shared-memory allocation, and splitting it
   into the per-warp tiles used by the kernel body. *)

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
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module Trade = Pulse.Lib.Trade


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

ghost
fn flash_split_warp_tiles
  (#et : Type0) (nw cols : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v cols)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v cols)))
  requires live a
  ensures
    forall+ (w : natlt (SZ.v nw)).
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0)

ghost
fn flash_gather_warp_tiles
  (#et : Type0) (nw cols : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v cols)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v cols)))
  requires
    forall+ (w : natlt (SZ.v nw)).
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0)
  ensures live a

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
