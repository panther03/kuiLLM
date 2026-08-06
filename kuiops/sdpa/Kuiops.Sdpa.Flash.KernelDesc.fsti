module Kuiops.Sdpa.Flash.KernelDesc

(* The [kernel_desc] obligations: block setup and teardown, grid setup and
   teardown, and the descriptor's view of the barrier's initial state. *)

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

open Kuiper.Bijection
open Kuiper.ForEvery
open Kuiper.Floating
open Kuiper.TensorCore
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module FSp = Kuiops.Sdpa.Flash.Split



ghost
fn flash_setup
  (#et_ab : Type0)
  (nblk : szp)
  (b hq hkv group sq rows tiles sk d : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  ()
  norewrite
  requires
    (gQ |-> Frac fQ eQ) **
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (gmask |-> Frac fmask emask) **
    live gout
  ensures
    (forall+ (bid : natlt (SZ.v nblk)).
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid) **
    emp

ghost
fn flash_block_setup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| scalar et_acc |}
  (nblk : szp)
  (nw nthr : szp {
    SZ.v nthr == block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  (bid : natlt (SZ.v nblk))
  ()
  norewrite
  requires
    live_c_shmems sh **
    flash_block_state nblk b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask bid
  ensures
    (forall+ (tid : natlt (SZ.v nthr)).
      flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout (flash_views_of nw d sh)
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        (tid / BW.warp_size) (tid % BW.warp_size)) **
    emp

ghost
fn flash_block_teardown
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nblk : szp)
  (nw nthr : szp {
    SZ.v nthr == block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (escale : (natlt (SZ.v nblk)) -> chest2 et_acc (SZ.v nw) 16)
  (eO : (natlt (SZ.v nblk)) -> chest2 et_acc (SZ.v nw * 16) (SZ.v d))
  (egl : (natlt (SZ.v nblk)) -> chest1 et_acc 16)
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  (bid : natlt (SZ.v nblk))
  ()
  norewrite
  requires
    (forall+ (tid : natlt (SZ.v nthr)).
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout (flash_views_of nw d sh)
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (escale bid) (eO bid) (egl bid)
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        (tid / BW.warp_size) (tid % BW.warp_size)) **
    emp
  ensures
    live_c_shmems sh **
    flash_block_state_v nw nblk b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask
      (escale bid) (eO bid) (egl bid) bid

ghost
fn flash_b0_to_descriptor
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size)
  (tid : natlt (block_threads nw))
  requires
    pure (thread_w nw tid == w /\
          thread_lane nw tid == lane) **
    flash_b0_local nw d v w lane
  ensures b0_raw nw d v.shQv v.shOv tid

(* Valued grid teardown: reassembles the whole output tensor, with every
   logical cell pinned to the value [flash_out_vfun] gives it. *)
ghost
fn flash_teardown_v
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (nblk nw : szp)
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (d : szp { 16 /?+ SZ.v d })
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lout |}
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (has_mask causal : bool) (scale : et_acc)
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (SZ.v nblk)).
      flash_block_state_v nw nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask
        (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
        bid) **
    emp
  ensures
    (gQ |-> Frac fQ eQ) **
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (gmask |-> Frac fmask emask) **
    (gout |-> Frac 1.0R
      (FSp.flash_out_chest b hq hkv group sq rows d (flash_out_vfun nw d b hq hkv group sq rows sk eQ eK eV emask has_mask causal scale)))
