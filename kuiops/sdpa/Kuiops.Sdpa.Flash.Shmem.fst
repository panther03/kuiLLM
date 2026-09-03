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
open Kuiops.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiper.Bijection
open Kuiper.EMatrix
open Kuiper.ForEvery
open Kuiper.Shape
open Pulse.Lib.Pledge
open Kuiops.Sdpa.Flash.Split
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
{
  unfold_c_shmems sh (`%flash_shmems);
  let
    (q, (k, (v, (s, (p, (pv, (cw,
    (m, (l, (scale, (o, (gm, (gl, _))))))))))))) = sh;
  gpu_pts_to_ref q;
  gpu_pts_to_ref k;
  gpu_pts_to_ref v;
  gpu_pts_to_ref s;
  gpu_pts_to_ref p;
  gpu_pts_to_ref pv;
  gpu_pts_to_ref cw;
  gpu_pts_to_ref m;
  gpu_pts_to_ref l;
  gpu_pts_to_ref scale;
  gpu_pts_to_ref o;
  gpu_pts_to_ref gm;
  gpu_pts_to_ref gl;
  tensor_abs' (l2_row_major 16 (SZ.v d)) q;
  tensor_abs' (l2_row_major (SZ.v nw * 16) (SZ.v d)) k;
  tensor_abs' (l2_row_major (SZ.v nw * 16) (SZ.v d)) v;
  tensor_abs' (l2_row_major (SZ.v nw * 16) 16) s;
  tensor_abs' (l2_row_major (SZ.v nw * 16) 16) p;
  tensor_abs' (l2_row_major (SZ.v nw * 16) 16) pv;
  tensor_abs' (l2_row_major (SZ.v nw) 16) cw;
  tensor_abs' (l2_row_major (SZ.v nw) 16) m;
  tensor_abs' (l2_row_major (SZ.v nw) 16) l;
  tensor_abs' (l2_row_major (SZ.v nw) 16) scale;
  tensor_abs' (l2_row_major (SZ.v nw * 16) (SZ.v d)) o;
  tensor_abs' (l1_forward 16) gm;
  tensor_abs' (l1_forward 16) gl;
  let views = flash_views_of nw d sh;
  with eQ. assert (
    from_array (l2_row_major 16 (SZ.v d)) q |-> eQ);
  rewrite (from_array (l2_row_major 16 (SZ.v d)) q |-> eQ)
    as (views.shQv |-> eQ);
  fold live views.shQv;
  with eK. assert (
    from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) k |-> eK);
  rewrite
    (from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) k |-> eK)
    as (views.shKv |-> eK);
  fold live views.shKv;
  with eV. assert (
    from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) v |-> eV);
  rewrite
    (from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) v |-> eV)
    as (views.shVv |-> eV);
  fold live views.shVv;
  with eS. assert (
    from_array (l2_row_major (SZ.v nw * 16) 16) s |-> eS);
  rewrite (from_array (l2_row_major (SZ.v nw * 16) 16) s |-> eS)
    as (views.shSv |-> eS);
  fold live views.shSv;
  with eP. assert (
    from_array (l2_row_major (SZ.v nw * 16) 16) p |-> eP);
  rewrite (from_array (l2_row_major (SZ.v nw * 16) 16) p |-> eP)
    as (views.shPv |-> eP);
  fold live views.shPv;
  with ePV. assert (
    from_array (l2_row_major (SZ.v nw * 16) 16) pv |-> ePV);
  rewrite (from_array (l2_row_major (SZ.v nw * 16) 16) pv |-> ePV)
    as (views.shPVv |-> ePV);
  fold live views.shPVv;
  with ecw. assert (
    from_array (l2_row_major (SZ.v nw) 16) cw |-> ecw);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) cw |-> ecw)
    as (views.shcwv |-> ecw);
  fold live views.shcwv;
  with eM. assert (
    from_array (l2_row_major (SZ.v nw) 16) m |-> eM);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) m |-> eM)
    as (views.shMv |-> eM);
  fold live views.shMv;
  with eL. assert (
    from_array (l2_row_major (SZ.v nw) 16) l |-> eL);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) l |-> eL)
    as (views.shLv |-> eL);
  fold live views.shLv;
  with escale. assert (
    from_array (l2_row_major (SZ.v nw) 16) scale |-> escale);
  rewrite (from_array (l2_row_major (SZ.v nw) 16) scale |-> escale)
    as (views.shscalev |-> escale);
  fold live views.shscalev;
  with eO. assert (
    from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) o |-> eO);
  rewrite
    (from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) o |-> eO)
    as (views.shOv |-> eO);
  fold live views.shOv;
  with egm. assert (from_array (l1_forward 16) gm |-> egm);
  rewrite (from_array (l1_forward 16) gm |-> egm)
    as (views.shgmv |-> egm);
  fold live views.shgmv;
  with egl. assert (from_array (l1_forward 16) gl |-> egl);
  rewrite (from_array (l1_forward 16) gl |-> egl)
    as (views.shglv |-> egl);
  fold live views.shglv;
  rewrite each views as (flash_views_of nw d sh);
  fold flash_views_live (flash_views_of nw d sh);
}

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
{
  unfold flash_views_live (flash_views_of nw d sh);
  unfold live (flash_views_of nw d sh).shQv;
  unfold live (flash_views_of nw d sh).shKv;
  unfold live (flash_views_of nw d sh).shVv;
  unfold live (flash_views_of nw d sh).shSv;
  unfold live (flash_views_of nw d sh).shPv;
  unfold live (flash_views_of nw d sh).shPVv;
  unfold live (flash_views_of nw d sh).shcwv;
  unfold live (flash_views_of nw d sh).shMv;
  unfold live (flash_views_of nw d sh).shLv;
  unfold live (flash_views_of nw d sh).shscalev;
  unfold live (flash_views_of nw d sh).shOv;
  unfold live (flash_views_of nw d sh).shgmv;
  unfold live (flash_views_of nw d sh).shglv;
  tensor_concr (flash_views_of nw d sh).shQv;
  tensor_concr (flash_views_of nw d sh).shKv;
  tensor_concr (flash_views_of nw d sh).shVv;
  tensor_concr (flash_views_of nw d sh).shSv;
  tensor_concr (flash_views_of nw d sh).shPv;
  tensor_concr (flash_views_of nw d sh).shPVv;
  tensor_concr (flash_views_of nw d sh).shcwv;
  tensor_concr (flash_views_of nw d sh).shMv;
  tensor_concr (flash_views_of nw d sh).shLv;
  tensor_concr (flash_views_of nw d sh).shscalev;
  tensor_concr (flash_views_of nw d sh).shOv;
  tensor_concr (flash_views_of nw d sh).shgmv;
  tensor_concr (flash_views_of nw d sh).shglv;
  let q = fst sh;
  let k = fst (snd sh);
  let v = fst (snd (snd sh));
  let s = fst (snd (snd (snd sh)));
  let p = fst (snd (snd (snd (snd sh))));
  let pv = fst (snd (snd (snd (snd (snd sh)))));
  let cw = fst (snd (snd (snd (snd (snd (snd sh))))));
  let m = fst (snd (snd (snd (snd (snd (snd (snd sh)))))));
  let l = fst (snd (snd (snd (snd (snd (snd (snd (snd sh))))))));
  let scale = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))));
  let o = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))));
  let gm = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))))));
  let gl = fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))))));
  lem_from_array_core #et_ab #2
    #(16 @| SZ.v d @| INil) #(l2_row_major 16 (SZ.v d)) q;
  lem_from_array_core #et_ab #2
    #(SZ.v nw * 16 @| SZ.v d @| INil)
    #(l2_row_major (SZ.v nw * 16) (SZ.v d)) k;
  lem_from_array_core #et_ab #2
    #(SZ.v nw * 16 @| SZ.v d @| INil)
    #(l2_row_major (SZ.v nw * 16) (SZ.v d)) v;
  lem_from_array_core #et_acc #2
    #(SZ.v nw * 16 @| 16 @| INil)
    #(l2_row_major (SZ.v nw * 16) 16) s;
  lem_from_array_core #et_ab #2
    #(SZ.v nw * 16 @| 16 @| INil)
    #(l2_row_major (SZ.v nw * 16) 16) p;
  lem_from_array_core #et_acc #2
    #(SZ.v nw * 16 @| 16 @| INil)
    #(l2_row_major (SZ.v nw * 16) 16) pv;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) cw;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) m;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) l;
  lem_from_array_core #et_acc #2
    #(SZ.v nw @| 16 @| INil)
    #(l2_row_major (SZ.v nw) 16) scale;
  lem_from_array_core #et_acc #2
    #(SZ.v nw * 16 @| SZ.v d @| INil)
    #(l2_row_major (SZ.v nw * 16) (SZ.v d)) o;
  lem_from_array_core #et_acc #1
    #(16 @| INil) #(l1_forward 16) gm;
  lem_from_array_core #et_acc #1
    #(16 @| INil) #(l1_forward 16) gl;
  with sq. assert (core (flash_views_of nw d sh).shQv |-> sq);
  rewrite (core (flash_views_of nw d sh).shQv |-> sq) as (q |-> sq);
  with sk. assert (core (flash_views_of nw d sh).shKv |-> sk);
  rewrite (core (flash_views_of nw d sh).shKv |-> sk) as (k |-> sk);
  with sv. assert (core (flash_views_of nw d sh).shVv |-> sv);
  rewrite (core (flash_views_of nw d sh).shVv |-> sv) as (v |-> sv);
  with ss. assert (core (flash_views_of nw d sh).shSv |-> ss);
  rewrite (core (flash_views_of nw d sh).shSv |-> ss) as (s |-> ss);
  with sp. assert (core (flash_views_of nw d sh).shPv |-> sp);
  rewrite (core (flash_views_of nw d sh).shPv |-> sp) as (p |-> sp);
  with spv. assert (core (flash_views_of nw d sh).shPVv |-> spv);
  rewrite (core (flash_views_of nw d sh).shPVv |-> spv)
    as (pv |-> spv);
  with scw. assert (core (flash_views_of nw d sh).shcwv |-> scw);
  rewrite (core (flash_views_of nw d sh).shcwv |-> scw)
    as (cw |-> scw);
  with sm. assert (core (flash_views_of nw d sh).shMv |-> sm);
  rewrite (core (flash_views_of nw d sh).shMv |-> sm) as (m |-> sm);
  with sl. assert (core (flash_views_of nw d sh).shLv |-> sl);
  rewrite (core (flash_views_of nw d sh).shLv |-> sl) as (l |-> sl);
  with sscale. assert (
    core (flash_views_of nw d sh).shscalev |-> sscale);
  rewrite (core (flash_views_of nw d sh).shscalev |-> sscale)
    as (scale |-> sscale);
  with so. assert (core (flash_views_of nw d sh).shOv |-> so);
  rewrite (core (flash_views_of nw d sh).shOv |-> so) as (o |-> so);
  with sgm. assert (core (flash_views_of nw d sh).shgmv |-> sgm);
  rewrite (core (flash_views_of nw d sh).shgmv |-> sgm)
    as (gm |-> sgm);
  with sgl. assert (core (flash_views_of nw d sh).shglv |-> sgl);
  rewrite (core (flash_views_of nw d sh).shglv |-> sgl)
    as (gl |-> sgl);
  rewrite each q as (fst sh);
  rewrite each k as (fst (snd sh));
  rewrite each v as (fst (snd (snd sh)));
  rewrite each s as (fst (snd (snd (snd sh))));
  rewrite each p as (fst (snd (snd (snd (snd sh)))));
  rewrite each pv as (fst (snd (snd (snd (snd (snd sh))))));
  rewrite each cw as
    (fst (snd (snd (snd (snd (snd (snd sh)))))));
  rewrite each m as
    (fst (snd (snd (snd (snd (snd (snd (snd sh))))))));
  rewrite each l as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))));
  rewrite each scale as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))));
  rewrite each o as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))))));
  rewrite each gm as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh))))))))))));
  rewrite each gl as
    (fst (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd (snd sh)))))))))))));
  fold_c_shmems sh (`%flash_shmems);
}

ghost
fn flash_split_warp_tiles
  (#et : Type0) (nw cols : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v cols)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v cols)))
  requires live a
  ensures
    forall+ (w : natlt (SZ.v nw)).
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0)
{
  unfold live a;
  with e. assert (a |-> e);
  array2_tile a 16 (SZ.v cols);
  forevery_rw_size2
    (SZ.v nw * 16 / 16) (SZ.v nw)
    (SZ.v cols / SZ.v cols) 1
    #(fun (w : natlt (SZ.v nw)) (tc : natlt 1) ->
      array2_subtile a 16 (SZ.v cols) w tc
        |-> Frac 1.0R
          (ematrix_subtile e 16 (SZ.v cols) w tc));
  forevery_map
    (fun (w : natlt (SZ.v nw)) ->
      forall+ (tc : natlt 1).
        array2_subtile a 16 (SZ.v cols) w tc
          |-> Frac 1.0R
            (ematrix_subtile e 16 (SZ.v cols) w tc))
    (fun w ->
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0))
    fn w {
      forevery_singleton_elim
        (fun (tc : natlt 1) ->
          array2_subtile a 16 (SZ.v cols) w tc
            |-> Frac 1.0R
              (ematrix_subtile e 16 (SZ.v cols) w tc));
      fold live (array2_subtile a 16 (SZ.v cols <: pos) w 0);
    };
}

ghost
fn flash_gather_warp_tiles
  (#et : Type0) (nw cols : szp)
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v cols)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v cols)))
  requires
    forall+ (w : natlt (SZ.v nw)).
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0)
  ensures live a
{
  forevery_map
    (fun (w : natlt (SZ.v nw)) ->
      live (array2_subtile a 16 (SZ.v cols <: pos) w 0))
    (fun w ->
      forall+ (tc : natlt 1).
        exists* (e : chest2 et 16 (SZ.v cols)).
          array2_subtile a 16 (SZ.v cols) w tc |-> Frac 1.0R e)
    fn w {
      unfold live (array2_subtile a 16 (SZ.v cols <: pos) w 0);
      with e. assert (
        array2_subtile a 16 (SZ.v cols) w 0 |-> Frac 1.0R e);
      forevery_singleton_intro
        (fun (tc : natlt 1) ->
          exists* (x : chest2 et 16 (SZ.v cols)).
            array2_subtile a 16 (SZ.v cols) w tc |-> Frac 1.0R x);
    };
  let ef = forevery_exists_2
    (fun (w : natlt (SZ.v nw)) (tc : natlt 1)
      (e : chest2 et 16 (SZ.v cols)) ->
      array2_subtile a 16 (SZ.v cols) w tc |-> Frac 1.0R e);
  forevery_rw_size2
    (SZ.v nw) (SZ.v nw * 16 / 16)
    1 (SZ.v cols / SZ.v cols)
    #(fun (w : natlt (SZ.v nw)) (tc : natlt 1) ->
      array2_subtile a 16 (SZ.v cols) w tc
        |-> Frac 1.0R (ef w tc));
  array2_untile' a 16 (SZ.v cols <: pos) ef;
  fold live a;
}

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
{
  flash_split_warp_tiles nw d v.shKv;
  forevery_map
    (fun w -> live (flash_warp_k v w))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_k v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    fn w { flash_warp_split_stride (flash_warp_k v w); };

  flash_split_warp_tiles nw d v.shVv;
  forevery_map
    (fun w -> live (flash_warp_v v w))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_v v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    fn w { flash_warp_split_stride (flash_warp_v v w); };

  flash_split_warp_tiles nw 16sz v.shSv;
  forevery_map
    (fun w -> live (flash_warp_s v w))
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_s v w |-> Frac
            (1.0R /. BW.warp_size) e)
    fn w {
      flash_share_tensor (flash_warp_s v w) BW.warp_size;
    };

  flash_split_warp_tiles nw 16sz v.shPv;
  forevery_map
    (fun w -> live (flash_warp_p v w))
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          row_subtile (flash_warp_p v w) lane))
    fn w { flash_split_rows16 (flash_warp_p v w); };

  flash_split_warp_tiles nw 16sz v.shPVv;
  forevery_map
    (fun w -> live (flash_warp_pv v w))
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_pv v w |-> Frac
            (1.0R /. BW.warp_size) e)
    fn w {
      flash_share_tensor (flash_warp_pv v w) BW.warp_size;
    };

  flash_split_ml nw v.shcwv;

  forevery_zip4_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane));
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane))
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e);
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      ((exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane))
      **
      (when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e)))
    (fun w lane ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    fn w lane {
      fold flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane;
    };
}

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
{
  forevery_map_2
    (fun w lane ->
      flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e))
    fn w lane {
      unfold flash_jt_local d
        (flash_warp_k v w) (flash_warp_v v w)
        (flash_warp_s v w) (flash_warp_p v w)
        (flash_warp_pv v w) (flash_warp_cw v w) lane;
    };
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_k v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_ab
        (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile (flash_warp_v v w)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      (exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
      ** when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_s v w |-> Frac
          (1.0R /. BW.warp_size) e)
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane)
      ** when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        row_subtile (flash_warp_p v w) lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane)
      ** (exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e));
  forevery_unzip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      when__ (lane < 16) (fun _ ->
        cell_full (flash_warp_cw v w) lane))
    (fun (w : natlt (SZ.v nw))
      (_lane : natlt BW.warp_size) ->
      exists* (e : chest2 et_acc 16 16).
        flash_warp_pv v w |-> Frac
          (1.0R /. BW.warp_size) e);

  forevery_map
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_k v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    (fun w -> live (flash_warp_k v w))
    fn w { flash_warp_gather_stride (flash_warp_k v w); };
  flash_gather_warp_tiles nw d v.shKv;

  forevery_map
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        exists* (e : chest2 et_ab
          (16 / warp_row_span) (SZ.v d / 16)).
          array2_stride_subtile (flash_warp_v v w)
            warp_row_span 16 (lane / 16) (lane % 16)
            |-> Frac 1.0R e)
    (fun w -> live (flash_warp_v v w))
    fn w { flash_warp_gather_stride (flash_warp_v v w); };
  flash_gather_warp_tiles nw d v.shVv;

  forevery_map
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_s v w |-> Frac
            (1.0R /. BW.warp_size) e)
    (fun w -> live (flash_warp_s v w))
    fn w {
      flash_gather_tensor (flash_warp_s v w) BW.warp_size;
    };
  flash_gather_warp_tiles nw 16sz v.shSv;

  forevery_map
    (fun w ->
      forall+ (lane : natlt BW.warp_size).
        when__ (lane < 16) (fun _ ->
          row_subtile (flash_warp_p v w) lane))
    (fun w -> live (flash_warp_p v w))
    fn w { flash_gather_rows16 (flash_warp_p v w); };
  flash_gather_warp_tiles nw 16sz v.shPv;

  forevery_map
    (fun w ->
      forall+ (_ : natlt BW.warp_size).
        exists* (e : chest2 et_acc 16 16).
          flash_warp_pv v w |-> Frac
            (1.0R /. BW.warp_size) e)
    (fun w -> live (flash_warp_pv v w))
    fn w {
      flash_gather_tensor (flash_warp_pv v w) BW.warp_size;
    };
  flash_gather_warp_tiles nw 16sz v.shPVv;

  flash_gather_ml nw v.shcwv;
}

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
{
  flash_split_strided_cells2 v.shQv (block_threads nw);
  forevery_factor (block_threads nw)
    (SZ.v nw) BW.warp_size
    (fun tid ->
      strided_cells2 v.shQv (block_threads nw)
        tid);
  flash_split_warp_tiles nw d v.shOv;
  forevery_map
    (fun (w : natlt (SZ.v nw)) ->
      live (array2_subtile v.shOv
        16 (SZ.v d <: pos) w 0))
    (fun (w : natlt (SZ.v nw)) ->
      forall+ (lane : natlt BW.warp_size).
        strided_cells2
          (array2_subtile v.shOv
            16 (SZ.v d <: pos) w 0)
          BW.warp_size lane)
    fn w {
      flash_split_strided_cells2
        (array2_subtile v.shOv
          16 (SZ.v d <: pos) w 0)
        BW.warp_size;
    };
  forevery_zip_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      strided_cells2 v.shQv (block_threads nw)
        (w * BW.warp_size + lane))
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      strided_cells2
        (array2_subtile v.shOv
          16 (SZ.v d <: pos) w 0)
        BW.warp_size lane);
  forevery_map_2
    (fun (w : natlt (SZ.v nw))
      (lane : natlt BW.warp_size) ->
      strided_cells2 v.shQv (block_threads nw)
        (w * BW.warp_size + lane) **
      strided_cells2
        (array2_subtile v.shOv
          16 (SZ.v d <: pos) w 0)
        BW.warp_size lane)
    (fun w lane -> flash_b0_local nw d v w lane)
    fn w lane {
      fold flash_b0_local nw d v w lane;
    };
}
