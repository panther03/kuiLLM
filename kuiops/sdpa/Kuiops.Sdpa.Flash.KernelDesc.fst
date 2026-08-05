module Kuiops.Sdpa.Flash.KernelDesc

(* The [kernel_desc] obligations: block setup and teardown, grid setup and
   teardown, and the descriptor's view of the barrier's initial state. *)

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
open Kuiops.Sdpa.Flash.Shmem

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module FC = Kuiper.Float.Casts
module Trade = Pulse.Lib.Trade
module BW = Kuiper.Barrier.Warp


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
{
  tensor_share_n gQ (SZ.v nblk);
  tensor_share_n gK (SZ.v nblk);
  tensor_share_n gV (SZ.v nblk);
  TRO.tensor_share_n gmask (SZ.v nblk);
  flash_split_output b hq hkv group sq rows tiles d gout;
  forevery_rw_size
    (SZ.v b * SZ.v hkv * SZ.v tiles) (SZ.v nblk)
    #(fun bid ->
      flash_block_output b hq hkv group sq rows tiles d gout bid);
  forevery_map
    (fun (bid : natlt (SZ.v nblk)) ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane)
    (fun (bid : natlt (SZ.v nblk)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)))
    fn bid {
      fold flash_block_output
        b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles));
    };
  forevery_zip
    (fun _ -> gmask |-> Frac (fmask /. (SZ.v nblk)) emask)
    (fun (bid : natlt (SZ.v nblk)) ->
      forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane);
  forevery_zip
    (fun _ -> gV |-> Frac (fV /. (SZ.v nblk)) eV)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_zip
    (fun _ -> gK |-> Frac (fK /. (SZ.v nblk)) eK)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_zip
    (fun _ -> gQ |-> Frac (fQ /. (SZ.v nblk)) eQ)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_map #(natlt (SZ.v nblk))
    (fun bid ->
      (gQ |-> Frac (fQ /. (SZ.v nblk)) eQ) **
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      (forall+ (lane : natlt BW.warp_size).
        out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane))
    (fun bid ->
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid)
    fn bid {
      fold flash_block_output
        b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles));
      fold flash_block_state nblk
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid;
    };
}

ghost
fn flash_teardown
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
  {| ctlayout lout |}
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
    (forall+ (bid : natlt (SZ.v nblk)).
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid) **
    emp
  ensures
    (gQ |-> Frac fQ eQ) **
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (gmask |-> Frac fmask emask) **
    live gout
{
  assert pure (SZ.fits (tlayout_ulen lout));
  forevery_map #(natlt (SZ.v nblk))
    (fun bid ->
      flash_block_state nblk b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid)
    (fun bid ->
      (gQ |-> Frac (fQ /. (SZ.v nblk)) eQ) **
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)))
    fn bid {
      unfold flash_block_state nblk
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout #fQ #fK #fV #fmask
        #eQ #eK #eV #emask bid;
    };
  forevery_unzip
    (fun _ -> gQ |-> Frac (fQ /. (SZ.v nblk)) eQ)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_unzip
    (fun _ -> gK |-> Frac (fK /. (SZ.v nblk)) eK)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_unzip
    (fun _ -> gV |-> Frac (fV /. (SZ.v nblk)) eV)
    (fun (bid : natlt (SZ.v nblk)) ->
      (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_unzip
    (fun _ -> gmask |-> Frac (fmask /. (SZ.v nblk)) emask)
    (fun (bid : natlt (SZ.v nblk)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  tensor_gather_n gQ (SZ.v nblk) #fQ;
  tensor_gather_n gK (SZ.v nblk) #fK;
  tensor_gather_n gV (SZ.v nblk) #fV;
  TRO.tensor_gather_n gmask (SZ.v nblk) #fmask;
  forevery_rw_size
    (SZ.v nblk) (SZ.v b * SZ.v hkv * SZ.v tiles)
    #(fun (bid : natlt (SZ.v nblk)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)));
  forevery_map
    (fun (bid : natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles)) ->
      flash_block_output b hq hkv group sq rows tiles d gout
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles)))
    (fun bid ->
      flash_block_output b hq hkv group sq rows tiles d gout bid)
    fn bid {
      rewrite
        (flash_block_output b hq hkv group sq rows tiles d gout
          (bid <: natlt
            (SZ.v b * SZ.v hkv * SZ.v tiles)))
        as
        (flash_block_output b hq hkv group sq rows tiles d gout bid);
    };
  flash_gather_output b hq hkv group sq rows tiles d gout;
}

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
{
  unfold flash_block_state nblk
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout #fQ #fK #fV #fmask
    #eQ #eK #eV #emask bid;
  tensor_share_n gQ (SZ.v nthr);
  tensor_share_n gK (SZ.v nthr);
  tensor_share_n gV (SZ.v nthr);
  TRO.tensor_share_n gmask (SZ.v nthr);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV);
  forevery_factor (SZ.v nthr) (SZ.v nw) BW.warp_size
    (fun _ ->
      gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask);
  forevery_zip4_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask);

  flash_output_add_warps nw
    b hq hkv group sq rows tiles d gout
    (bid <: natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles));
  flash_open_shmems nw d sh;
  let v = flash_views_of nw d sh;
  rewrite each (flash_views_of nw d sh) as v;
  unfold flash_views_live v;
  flash_split_b0 nw d v;
  flash_split_ml nw v.shMv;
  flash_split_ml nw v.shLv;
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (row v.shMv w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (row v.shLv w) lane));
  flash_split_jt nw d v;
  flash_split_combine nw v.shscalev v.shgmv v.shglv;
  forevery_zip4_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_b0_local nw d v w lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (row v.shMv w) lane) **
      when__ (lane < 16) (fun _ ->
        cell_full (row v.shLv w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_combine_local nw
        v.shscalev v.shgmv v.shglv w lane);
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
      (gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
      (gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
      (gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_b0_local nw d v w lane **
      (when__ (lane < 16) (fun _ ->
        cell_full (row v.shMv w) lane) **
       when__ (lane < 16) (fun _ ->
        cell_full (row v.shLv w) lane)) **
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane **
      flash_combine_local nw
        v.shscalev v.shgmv v.shglv w lane);
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      ((gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
       (gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
       (gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
       (gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask)) **
      (flash_b0_local nw d v w lane **
       (when__ (lane < 16) (fun _ ->
          cell_full (row v.shMv w) lane) **
        when__ (lane < 16) (fun _ ->
          cell_full (row v.shLv w) lane)) **
       flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane **
       flash_combine_local nw
        v.shscalev v.shgmv v.shglv w lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane));
  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (((gQ |-> Frac
          (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
        (gK |-> Frac
          (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
        (gV |-> Frac
          (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
        (gmask |-> Frac
          (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask)) **
       (flash_b0_local nw d v w lane **
        (when__ (lane < 16) (fun _ ->
           cell_full (row v.shMv w) lane) **
         when__ (lane < 16) (fun _ ->
           cell_full (row v.shLv w) lane)) **
        flash_jt_local d
          (flash_warp_k v w) (flash_warp_v v w)
          (flash_warp_s v w) (flash_warp_p v w)
          (flash_warp_pv v w) (flash_warp_cw v w) lane **
        flash_combine_local nw
          v.shscalev v.shgmv v.shglv w lane)) **
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane)
    fn w lane {
      fold flash_thread_pre nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane;
    };
  rewrite each v as (flash_views_of nw d sh);
  flash_unfactor_threads nw
    (fun w lane ->
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
        w lane);
  forevery_rw_size
    (SZ.v nw * BW.warp_size) (SZ.v nthr)
    #(fun (tid : natlt
      (SZ.v nw * BW.warp_size)) ->
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
}

ghost
fn flash_block_teardown
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
    (forall+ (tid : natlt (SZ.v nthr)).
      flash_thread_post nw nthr
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
  ensures
    live_c_shmems sh **
    flash_block_state nblk b hq hkv group sq rows tiles sk d
      gQ gK gV gmask gout #fQ #fK #fV #fmask
      #eQ #eK #eV #emask bid
{
  let v = flash_views_of nw d sh;
  rewrite each (flash_views_of nw d sh) as v;
  forevery_rw_size
    (SZ.v nthr) (SZ.v nw * BW.warp_size)
    #(fun (tid : natlt (SZ.v nthr)) ->
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        (tid / BW.warp_size) (tid % BW.warp_size));
  flash_factor_threads nw
    (fun w lane ->
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane);
  forevery_map_2
    (fun w lane ->
      flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ) **
      (gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK) **
      (gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV) **
      (gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask) **
      (exists* (e : chest2 et_ab 16 (SZ.v d)).
        v.shQv |-> Frac
          (1.0R /. (block_threads nw)) e) **
      b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw)) **
      b2_post nw d v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw)) **
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane **
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane) **
      when_ (w = 0 /\ lane < 16)
        (cell_full v.shgmv (clamp_nat_lt 16 lane)))
    fn w lane {
      unfold flash_thread_post nw nthr
        b hq hkv group sq rows tiles sk d
        gQ gK gV gmask gout v
        #(fQ /. (SZ.v nblk) /. (SZ.v nthr))
        #(fK /. (SZ.v nblk) /. (SZ.v nthr))
        #(fV /. (SZ.v nblk) /. (SZ.v nthr))
        #(fmask /. (SZ.v nblk) /. (SZ.v nthr))
        #eQ #eK #eV #emask
        (bid <: natlt
          (SZ.v b * SZ.v hkv * SZ.v tiles))
        w lane;
    };
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gQ |-> Frac
        (fQ /. (SZ.v nblk) /. (SZ.v nthr)) eQ)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gK |-> Frac
        (fK /. (SZ.v nblk) /. (SZ.v nthr)) eK)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gV |-> Frac
        (fV /. (SZ.v nblk) /. (SZ.v nthr)) eV)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      gmask |-> Frac
        (fmask /. (SZ.v nblk) /. (SZ.v nthr)) emask)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab 16 (SZ.v d)).
        v.shQv |-> Frac
          (1.0R /. (block_threads nw)) e)
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw)))
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      b2_post nw d v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw)))
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    _;
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when_ (w = 0)
        (out_store_cells b hq sq 16sz d rows gout
          (flash_bid_bi
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (flash_bid_kvh
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)))
          (SZ.v group)
          (flash_bid_rt
            (SZ.v b) (SZ.v hkv) (SZ.v tiles)
            (bid <: natlt
              (SZ.v b * SZ.v hkv * SZ.v tiles)) * 16)
          lane))
    _;

  flash_gather_thread_tensor nw nthr gQ
    #(fQ /. (SZ.v nblk)) #eQ;
  flash_gather_thread_tensor nw nthr gK
    #(fK /. (SZ.v nblk)) #eK;
  flash_gather_thread_tensor nw nthr gV
    #(fV /. (SZ.v nblk)) #eV;
  flash_gather_thread_rotensor nw nthr gmask
    #(fmask /. (SZ.v nblk)) #emask;

  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_ab 16 (SZ.v d)).
        v.shQv |-> Frac
          (1.0R /. (block_threads nw)) e);
  flash_gather_tensor v.shQv (block_threads nw);

  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw)))
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shMv |-> Frac
          (1.0R /. (block_threads nw)) e) **
      (exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shLv |-> Frac
          (1.0R /. (block_threads nw)) e))
    fn w lane {
      unfold b1_post nw v.shMv v.shLv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw));
    };
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shMv |-> Frac
          (1.0R /. (block_threads nw)) e)
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shLv |-> Frac
          (1.0R /. (block_threads nw)) e);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shMv |-> Frac
          (1.0R /. (block_threads nw)) e);
  flash_gather_tensor v.shMv (block_threads nw);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shLv |-> Frac
          (1.0R /. (block_threads nw)) e);
  flash_gather_tensor v.shLv (block_threads nw);

  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      b2_post nw d v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw)))
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shscalev |-> Frac
          (1.0R /. (block_threads nw)) e) **
      (exists* (e : chest2 et_acc
        (SZ.v nw * 16) (SZ.v d)).
        v.shOv |-> Frac
          (1.0R /. (block_threads nw)) e) **
      (exists* (e : chest1 et_acc 16).
        v.shglv |-> Frac
          (1.0R /. (block_threads nw)) e))
    fn w lane {
      unfold b2_post nw d
        v.shscalev v.shOv v.shglv
        (w * BW.warp_size + lane
          <: natlt (block_threads nw));
    };
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shscalev |-> Frac
          (1.0R /. (block_threads nw)) e)
    _;
  forevery_unzip_2
    (fun (_w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc
        (SZ.v nw * 16) (SZ.v d)).
        v.shOv |-> Frac
          (1.0R /. (block_threads nw)) e)
    _;
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc (SZ.v nw) 16).
        v.shscalev |-> Frac
          (1.0R /. (block_threads nw)) e);
  flash_gather_tensor v.shscalev (block_threads nw);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest2 et_acc
        (SZ.v nw * 16) (SZ.v d)).
        v.shOv |-> Frac
          (1.0R /. (block_threads nw)) e);
  flash_gather_tensor v.shOv (block_threads nw);
  flash_unfactor_threads nw
    (fun _w _lane ->
      exists* (e : chest1 et_acc 16).
        v.shglv |-> Frac
          (1.0R /. (block_threads nw)) e);
  flash_gather_tensor v.shglv (block_threads nw);

  flash_gather_jt nw d v;
  flash_output_remove_warps nw
    b hq hkv group sq rows tiles d gout
    (bid <: natlt
      (SZ.v b * SZ.v hkv * SZ.v tiles));
  flash_gather_gm nw v.shgmv;

  fold flash_views_live v;
  rewrite each v as (flash_views_of nw d sh);
  flash_close_shmems nw d sh;
  fold flash_block_state nblk
    b hq hkv group sq rows tiles sk d
    gQ gK gV gmask gout #fQ #fK #fV #fmask
    #eQ #eK #eV #emask bid;
}

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
  ensures b0_pre nw d v.shQv v.shOv tid
{
  unfold flash_b0_local nw d v w lane;
  rewrite each w as (thread_w nw tid);
  rewrite each lane as (thread_lane nw tid);
  FStar.Math.Lemmas.lemma_div_mod
    (tid <: nat) BW.warp_size;
  rewrite each
    (thread_w nw tid * BW.warp_size +
      thread_lane nw tid)
    as tid;
  fold b0_pre nw d v.shQv v.shOv tid;
}

