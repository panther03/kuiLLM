module Kuiops.Sdpa.Flash

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
open Kuiops.Sdpa.Flash.KfSub
open Kuiops.Sdpa.Flash.KfBlock
open Kuiops.Sdpa.Flash.Types
open Kuiops.Sdpa.Flash.Split
open Kuiops.Sdpa.Flash.Shmem
open Kuiops.Sdpa.Flash.Thread

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module FD = Kuiops.Sdpa.Flash.KernelDesc
module FSp = Kuiops.Sdpa.Flash.Split
module FB = Kuiops.Sdpa.Flash.Barrier
module Trade = Pulse.Lib.Trade


(* The kernel_desc record is one large, shallow VC: a single query needs a
   little over the default rlimit.  Splitting it costs 200+ subqueries. *)
#push-options "--z3rlimit 10"
inline_for_extraction noextract
let sdpa_flash_kd
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |} {| floating_real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (nblk : szp { SZ.v nblk <= max_blocks })
  (nw nthr : szp {
    SZ.v nthr == block_threads nw /\
    SZ.v nthr <= max_threads })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.fits (SZ.v rows + 15) /\
    SZ.v tiles == (SZ.v rows + 15) / 16 /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| TRO.cvtlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : TRO.roarray4 et_ab lgmask { TRO.is_global gmask })
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (has_mask : bool) (scale : et_acc)
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (SZ.fits (16 * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (16 * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (#_ : squash (SZ.fits (SZ.v sk + 32)))
  (#_ : squash (SZ.fits (SZ.v sk + 32 + SZ.v nw)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  : kernel_desc
      ((gQ |-> Frac fQ eQ) **
       (gK |-> Frac fK eK) **
       (gV |-> Frac fV eV) **
       (gmask |-> Frac fmask emask) **
       live gout)
      ((gQ |-> Frac fQ eQ) **
       (gK |-> Frac fK eK) **
       (gV |-> Frac fV eV) **
       (gmask |-> Frac fmask emask) **
       (gout |-> Frac 1.0R
          (FSp.flash_out_chest b hq hkv group sq rows d
             (flash_out_vfun nw d b hq hkv group sq rows sk eQ eK eV emask has_mask causal scale))))
= {
  nblk;
  nthr;
  shmems_desc = flash_shmems et_ab et_acc nw d;
  barrier_contract = (fun _bid sh ->
    let v = flash_views_of nw d sh in
    barrier_contract nw d v.shQv
      (flash_eQsh d b hq hkv group sq rows tiles eQ _bid)
      v.shMv v.shLv
      (flash_eM nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid)
      (flash_eL nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid)
      v.shscalev v.shOv v.shglv
      (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid)
      (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale _bid)
      (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid));
  barrier_count = (fun _ -> 3);
  barrier_ok = (fun _bid sh ->
    let v = flash_views_of nw d sh in
    FB.barrier_ok nw d v.shQv
      (flash_eQsh d b hq hkv group sq rows tiles eQ _bid)
      v.shMv v.shLv
      (flash_eM nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid)
      (flash_eL nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid)
      v.shscalev v.shOv v.shglv
      (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid)
      (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale _bid)
      (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid));
  frame = emp;
  block_pre = (fun bid ->
    flash_block_state nblk
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask bid);
  block_post = (fun bid ->
    flash_block_state_v nw nblk
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask
      (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
        (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
      (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale
        (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
      (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale
        (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles)))
      bid);
  setup = FD.flash_setup nblk
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout
    #fQ #fK #fV #fmask #eQ #eK #eV #emask;
  teardown = FD.flash_teardown_v nblk nw
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout has_mask causal scale
    #fQ #fK #fV #fmask #eQ #eK #eV #emask;
  block_frame = (fun _sh _bid -> emp);
  block_setup = FD.flash_block_setup nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout
    #fQ #fK #fV #fmask #eQ #eK #eV #emask;
  block_teardown = FD.flash_block_teardown nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout
    #fQ #fK #fV #fmask #eQ #eK #eV #emask
    (fun _bid -> flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid)
    (fun _bid -> flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale _bid)
    (fun _bid -> flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale _bid);
  kpre = (fun sh bid tid ->
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
      (tid / BW.warp_size) (tid % BW.warp_size));
  kpost = (fun sh bid tid ->
    flash_thread_post nw nthr
      b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout (flash_views_of nw d sh)
      #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
      #(fK /. (SZ.v nblk) /. (SZ.v nthr))
      #(fV /. (SZ.v nblk) /. (SZ.v nthr))
      #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
      #eQ #eK #eV #emask
      (flash_escale nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale bid)
      (flash_eO nw d b hq hkv group sq rows tiles sk eQ eK eV emask has_mask causal scale bid)
      (flash_egl nw d b hq hkv group sq rows tiles sk eQ eK emask has_mask causal scale bid)
      (bid <: natlt
        (SZ.v b * SZ.v hkv * SZ.v tiles))
      (tid / BW.warp_size) (tid % BW.warp_size));
  f = sdpa_flash_thread nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout causal has_mask scale;
  block_pre_sendable = (fun _ -> magic());
  block_post_sendable = (fun _ -> magic());
  kpre_sendable = (fun _ _ _ _ -> magic());
  kpost_sendable = (fun _ _ _ _ -> magic());
}

#pop-options

inline_for_extraction noextract
fn sdpa_flash_async
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |} {| floating_real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  // TODO: why do we have an nblk? why is it not just b * hkv * tiles etc.?
  (nblk : szp { SZ.v nblk <= max_blocks })
  // And an nthr?
  (nw nthr : szp {
    SZ.v nthr == block_threads nw /\
    SZ.v nthr <= max_threads })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.fits (SZ.v rows + 15) /\
    SZ.v tiles == (SZ.v rows + 15) / 16 /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| TRO.cvtlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : TRO.roarray4 et_ab lgmask { TRO.is_global gmask })
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (has_mask : bool) (scale : et_acc)
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (SZ.fits (16 * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (16 * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (#_ : squash (SZ.fits (SZ.v sk + 32)))
  (#_ : squash (SZ.fits (SZ.v sk + 32 + SZ.v nw)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (s : stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (
      (gQ |-> Frac fQ eQ) **
      (gK |-> Frac fK eK) **
      (gV |-> Frac fV eV) **
      (gmask |-> Frac fmask emask) **
      live gout)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (
        (gQ |-> Frac fQ eQ) **
        (gK |-> Frac fK eK) **
        (gV |-> Frac fV eV) **
        (gmask |-> Frac fmask emask) **
        (gout |-> Frac 1.0R
           (FSp.flash_out_chest b hq hkv group sq rows d
              (flash_out_vfun nw d b hq hkv group sq rows sk eQ eK eV emask has_mask causal scale)))))
{
  launch (sdpa_flash_kd nblk nw nthr
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout causal has_mask scale) s;
}
