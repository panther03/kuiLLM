module Kuiops.Sdpa.Flash.KfBlock

(* Memory-safety-only helpers for the bf16 tensor-core flash-attention kernel
   in [flash_attn_fa1.cu].  This module contains the prologue, key-tile
   loop, epilogue, and ownership bridges composed by [sdpa_flash_kf].  It has
   no functional specification; it verifies bounds and resource ownership. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Floating
open Kuiper.Shape
open Kuiper.Bijection
open Kuiper.Tensor.Layout.Bijection
open Pulse.Lib.Trade
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiper.ForEvery
open Kuiper.Ghost.TensorTranspose
open Kuiper.EMatrix
open Kuiops.Sdpa.Flash.KfSub

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module Trade = Pulse.Lib.Trade
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float
module FT = Kuiops.Sdpa.Flash.Types
(* Ownership of the row-major cells visited by
   [for (idx = tid; idx < rows*cols; idx += nthr)]. *)
inline_for_extraction noextract
let optional_inc (b : bool) (x : sz { SZ.fits (SZ.v x + 1) }) : sz =
  if b then x +^ 1sz else 0sz

inline_for_extraction noextract
let flash_scale_ctlayout
  (nw bm : szp) (lane : szlt bm)
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (#uf : squash (SZ.fits (
    stride_subtile_layout
      (l2_row_major (SZ.v nw) (SZ.v bm))
      1 (SZ.v bm) 0 (SZ.v lane)).ulen))
  (#af : squash (all_fit (
    (SZ.v nw / 1) @| (SZ.v bm / SZ.v bm) @| INil)))
  : ctlayout (
      stride_subtile_layout
        (l2_row_major (SZ.v nw) (SZ.v bm))
        1 (SZ.v bm) 0 (SZ.v lane)) =
{
  ulen_fits = uf;
  all_fit = af;
  cimap = flash_scale_cimap nw bm lane;
}

inline_for_extraction noextract
fn sdpa_flash_ml_init_active
  (#et #et_q : Type0) {| floating et |}
  (bm : szp)
  (b hq sq d : szp)
  (#lq : layout4 b hq sq d)
  (#lm #ll : layout1 bm) {| ctlayout lm |} {| ctlayout ll |}
  (shm : array1 et lm) (shl : array1 et ll)
  (gQ : array4 et_q lq)
  (lane : szlt bm)
  (#fQ : perm) (#eQ : chest (b @| hq @| sq @| d @| INil) et_q)
  preserves (gQ |-> Frac fQ eQ)
  requires cell_full_n shm (SZ.v lane) ** cell_full_n shl (SZ.v lane)
  ensures  cell_full_n_v shm (SZ.v lane) (neg infinity)
        ** cell_full_n_v shl (SZ.v lane) zero
{
  unfold (cell_full_n shm (SZ.v lane));
  unfold (cell_full_n shl (SZ.v lane));
  with vm. assert (tensor_pts_to_cell shm (idx1 (SZ.v lane)) vm);
  rewrite (tensor_pts_to_cell shm (idx1 (SZ.v lane)) vm)
       as (tensor_pts_to_cell shm (up (cidx1 lane)) vm);
  tensor_write_cell shm (cidx1 lane) (neg infinity);
  rewrite (tensor_pts_to_cell shm (up (cidx1 lane)) (neg infinity))
       as (tensor_pts_to_cell shm (idx1 (SZ.v lane)) (neg infinity));
  with vl. assert (tensor_pts_to_cell shl (idx1 (SZ.v lane)) vl);
  rewrite (tensor_pts_to_cell shl (idx1 (SZ.v lane)) vl)
       as (tensor_pts_to_cell shl (up (cidx1 lane)) vl);
  tensor_write_cell shl (cidx1 lane) zero;
  rewrite (tensor_pts_to_cell shl (up (cidx1 lane)) zero)
       as (tensor_pts_to_cell shl (idx1 (SZ.v lane)) zero);
  fold (cell_full_n_v shm (SZ.v lane) (neg infinity));
  fold (cell_full_n_v shl (SZ.v lane) (zero #et));
}

inline_for_extraction noextract
fn sdpa_flash_ml_init_maybe
  (#et #et_q : Type0) {| floating et |}
  (bm : szp)
  (b hq sq d : szp)
  (#lq : layout4 b hq sq d)
  (#lm #ll : layout1 bm) {| ctlayout lm |} {| ctlayout ll |}
  (shm : array1 et lm) (shl : array1 et ll)
  (gQ : array4 et_q lq)
  (lane : szlt warp_size)
  (#fQ : perm) (#eQ : chest (b @| hq @| sq @| d @| INil) et_q)
  preserves (gQ |-> Frac fQ eQ)
  requires if_ (lane_active bm lane) (ml_cells bm shm shl lane)
  ensures  if_ (lane_active bm lane)
             (ml_cells_v bm shm shl lane (neg infinity) zero)
{
  let active = lane_active bm lane;
  if active {
    if_elim_true _;
    unfold (ml_cells bm shm shl lane);
    sdpa_flash_ml_init_active bm b hq sq d shm shl gQ (clamp_lt bm lane);
    fold (ml_cells_v bm shm shl lane (neg infinity) (zero #et));
    if_intro_true (ml_cells_v bm shm shl lane (neg infinity) (zero #et));
  } else {
    if_elim_false (ml_cells bm shm shl lane);
    if_intro_false (ml_cells_v bm shm shl lane (neg infinity) (zero #et));
  }
}

(* Block-strided Q cache load and per-warp M/L/O initialization
   (flash_attn_fa1.cu, lines 104-114). *)
inline_for_extraction noextract
fn sdpa_flash_q_load
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |}
  {| floating et_acc |}
  (bm d nthr : szp)
  (b hq sq : szp)
  (#lgQ : layout4 b hq sq d) {| ctlayout lgQ |}
  (#lm #ll : layout1 bm) (#lO : layout2 (SZ.v bm) (SZ.v d))
  {| ctlayout lm |} {| ctlayout ll |} {| ctlayout lO |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (shQ : array2 et_ab (l2_row_major bm d))
  (shm : array1 et_acc lm) (shl : array1 et_acc ll)
  (shO : array2 et_acc lO)
  (tid : szlt nthr) (lane : szlt warp_size)
  (bi : szlt b) (r0 : sz) (rows : szp) (group : szp) (kvh : sz)
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fQ : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  requires
    (gQ |-> Frac fQ eQ) **
    strided_cells2 shQ (SZ.v nthr) (SZ.v tid) **
    strided_cells2 shO BW.warp_size (SZ.v lane) **
    if_ (lane_active bm lane) (ml_cells bm shm shl lane)
  ensures
    (gQ |-> Frac fQ eQ) **
    strided_cells2_v shQ (SZ.v nthr) (SZ.v tid)
      (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group)
         eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) **
    strided_cells2_v shO BW.warp_size (SZ.v lane) (ozero (SZ.v bm) (SZ.v d)) **
    if_ (lane_active bm lane)
      (ml_cells_v bm shm shl lane (neg infinity) zero)
{
  let ncells : sz = bm *^ d;
  unfold strided_cells2 shQ (SZ.v nthr) (SZ.v tid);
  forevery_map #(stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid))
    (fun ij -> exists* (v : et_ab). tensor_pts_to_cell shQ (idx2 ij._1 ij._2) v)
    (fun ij -> q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v tid) ij)
    fn ij {
      stride_ge_tid (SZ.v nthr) (SZ.v tid) (ij._1 * SZ.v d + ij._2);
      q_cell_intro_todo shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v tid) ij;
    };

  let mut idx : sz = tid;
  let mut iter : sz = 0sz;
  while (!idx <^ ncells)
    invariant
      live idx ** live iter **
      (gQ |-> Frac fQ eQ) **
      (forall+ (ij : stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid)).
        q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v !idx) ij) **
      strided_cells2 shO BW.warp_size (SZ.v lane) **
      if_ (lane_active bm lane) (ml_cells bm shm shl lane) **
      pure (SZ.v !idx % SZ.v nthr == SZ.v tid) **
      pure (SZ.v !idx < SZ.v bm * SZ.v d + SZ.v nthr) **
      pure (SZ.v !iter <= SZ.v !idx)
    decreases (SZ.v ncells - SZ.v !iter)
  {
    let flat = !idx;
    let next = flat +^ nthr;
    assert pure (SZ.v next == SZ.v flat + SZ.v nthr);
    let i : szlt bm = flat /^ d;
    let dd : szlt d = flat %^ d;
    let r = r0 +^ i;
    let rr : szlt rows = clamp_lt rows r;
    let qh0 = kvh *^ group +^ (rr /^ sq);
    let qh1 : szlt hq = clamp_lt hq qh0;
    let qpos : szlt sq = rr %^ sq;
    let qread = tensor_read gQ (cidx4 bi qh1 qpos dd);
    let qv : et_ab = q_sel (r <^ rows) qread;
    FStar.Math.Lemmas.euclidean_division_definition (SZ.v flat) (SZ.v d);
    assert pure (SZ.v i * SZ.v d + SZ.v dd == SZ.v flat);
    assert pure (acc2 (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v i) (SZ.v dd) == qv);
    forevery_remove
      #(stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid))
      (fun ij -> q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v flat) ij)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    q_cell_elim_todo shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v flat)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    with oldq. assert (tensor_pts_to_cell shQ (idx2 (SZ.v i) (SZ.v dd)) oldq);
    rewrite (tensor_pts_to_cell shQ (idx2 (SZ.v i) (SZ.v dd)) oldq)
         as (tensor_pts_to_cell shQ (up (cidx2 i dd)) oldq);
    tensor_write_cell shQ (cidx2 i dd) qv;
    rewrite (tensor_pts_to_cell shQ (up (cidx2 i dd)) qv)
         as (tensor_pts_to_cell shQ (idx2 (SZ.v i) (SZ.v dd)) qv);
    FStar.Math.Lemmas.lemma_mod_plus (SZ.v flat) 1 (SZ.v nthr);
    q_cell_intro_done shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v next)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    forevery_map
      #(ij : stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid) {
          ij =!= (((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)))
                   <: stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid)) })
      (fun ij -> q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v flat) ij)
      (fun ij -> q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v next) ij)
      fn ij {
        stride_next (SZ.v nthr) (SZ.v tid) (SZ.v bm) (SZ.v d)
          (SZ.v i) (SZ.v dd) ij;
        q_cell_bump shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v flat) (SZ.v next) ij;
      };
    forevery_insert
      #(stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid))
      (fun ij -> q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v next) ij)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    forevery_unrefine
      #(stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid))
      (fun ij -> q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v next) ij);
    assert pure (SZ.v !idx < SZ.v next);
    idx := next;
    iter := !iter +^ 1sz;
  };

  let fin = !idx;
  forevery_map #(stride_index2 (SZ.v bm) (SZ.v d) (SZ.v nthr) (SZ.v tid))
    (fun ij -> q_cell shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v fin) ij)
    (fun ij -> tensor_pts_to_cell shQ (idx2 ij._1 ij._2)
                 (acc2 (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) ij._1 ij._2))
    fn ij {
      flat_lt (SZ.v bm) (SZ.v d) ij._1 ij._2;
      q_cell_elim_done shQ (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) (SZ.v fin) ij;
    };
  fold strided_cells2_v shQ (SZ.v nthr) (SZ.v tid) (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0));

  sdpa_flash_ml_init_maybe bm b hq sq d shm shl gQ lane;

  unfold strided_cells2 shO BW.warp_size (SZ.v lane);
  forevery_map #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
    (fun ij -> exists* (v : et_acc). tensor_pts_to_cell shO (idx2 ij._1 ij._2) v)
    (fun ij -> q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v lane) ij)
    fn ij {
      stride_ge_tid BW.warp_size (SZ.v lane) (ij._1 * SZ.v d + ij._2);
      q_cell_intro_todo shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v lane) ij;
    };

  idx := lane;
  iter := 0sz;
  while (!idx <^ ncells)
    invariant
      live idx ** live iter **
      (gQ |-> Frac fQ eQ) **
      strided_cells2_v shQ (SZ.v nthr) (SZ.v tid) (SF.q_tile (SZ.v bm) (SZ.v rows) (SZ.v group) eQ (SZ.v bi) (SZ.v kvh) (SZ.v r0)) **
      (forall+ (ij : stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane)).
        q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v !idx) ij) **
      if_ (lane_active bm lane)
        (ml_cells_v bm shm shl lane (neg infinity) zero) **
      pure (SZ.v !idx % BW.warp_size == SZ.v lane) **
      pure (SZ.v !idx < SZ.v bm * SZ.v d + BW.warp_size) **
      pure (SZ.v !iter <= SZ.v !idx)
    decreases (SZ.v ncells - SZ.v !iter)
  {
    let flat = !idx;
    let next = flat +^ 32sz;
    assert pure (SZ.v next == SZ.v flat + BW.warp_size);
    let i : szlt bm = flat /^ d;
    let dd : szlt d = flat %^ d;
    FStar.Math.Lemmas.euclidean_division_definition (SZ.v flat) (SZ.v d);
    assert pure (SZ.v i * SZ.v d + SZ.v dd == SZ.v flat);
    forevery_remove
      #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
      (fun ij -> q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v flat) ij)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    q_cell_elim_todo shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v flat)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    with oldo. assert (tensor_pts_to_cell shO (idx2 (SZ.v i) (SZ.v dd)) oldo);
    rewrite (tensor_pts_to_cell shO (idx2 (SZ.v i) (SZ.v dd)) oldo)
         as (tensor_pts_to_cell shO (up (cidx2 i dd)) oldo);
    tensor_write_cell shO (cidx2 i dd) (zero #et_acc #_);
    rewrite (tensor_pts_to_cell shO (up (cidx2 i dd)) (zero #et_acc #_))
         as (tensor_pts_to_cell shO (idx2 (SZ.v i) (SZ.v dd))
               (acc2 (ozero (SZ.v bm) (SZ.v d)) (SZ.v i) (SZ.v dd)));
    FStar.Math.Lemmas.lemma_mod_plus (SZ.v flat) 1 BW.warp_size;
    q_cell_intro_done shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v next)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    forevery_map
      #(ij : stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane) {
          ij =!= (((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)))
                   <: stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane)) })
      (fun ij -> q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v flat) ij)
      (fun ij -> q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v next) ij)
      fn ij {
        stride_next BW.warp_size (SZ.v lane) (SZ.v bm) (SZ.v d)
          (SZ.v i) (SZ.v dd) ij;
        q_cell_bump shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v flat) (SZ.v next) ij;
      };
    forevery_insert
      #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
      (fun ij -> q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v next) ij)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    forevery_unrefine
      #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
      (fun ij -> q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v next) ij);
    assert pure (SZ.v !idx < SZ.v next);
    idx := next;
    iter := !iter +^ 1sz;
  };

  let ofin = !idx;
  forevery_map #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
    (fun ij -> q_cell shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v ofin) ij)
    (fun ij -> tensor_pts_to_cell shO (idx2 ij._1 ij._2)
                 (acc2 (ozero (SZ.v bm) (SZ.v d)) ij._1 ij._2))
    fn ij {
      flat_lt (SZ.v bm) (SZ.v d) ij._1 ij._2;
      q_cell_elim_done shO (ozero (SZ.v bm) (SZ.v d)) (SZ.v ofin) ij;
    };
  fold strided_cells2_v shO BW.warp_size (SZ.v lane) (ozero (SZ.v bm) (SZ.v d));
  ()
}

(* Causal early-exit bound (flash_attn_fa1.cu, lines 118-127).  The CUDA
   initializes [maxpos] to -1; [found] keeps that case representable with [sz]:
   no valid row leaves [kmax = sk - sq], while a valid maximum contributes the
   additional [maxpos + 1]. *)
inline_for_extraction noextract
fn sdpa_flash_causal_active
  (bm sk sq rows : szp)
  (r0 : sz)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v sk + SZ.v bm + 1)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  returns kmax : sz
  ensures pure (SZ.v kmax <= SZ.v sk /\
                SZ.v kmax == SF.causal_kmax (SZ.v bm) (SZ.v sq) (SZ.v sk)
                               (SZ.v rows) (SZ.v r0))
{
  let mut maxpos : sz = 0sz;
  let mut found : bool = false;
  let mut i : sz = 0sz;
  while (!i <^ bm)
    invariant
      live i ** live maxpos ** live found **
      pure (SZ.v !i <= SZ.v bm) **
      pure (SZ.v !maxpos < SZ.v sq) **
      pure ((SZ.v !maxpos, !found)
              == SF.tile_maxpos (SZ.v sq) (SZ.v rows) (SZ.v r0) (SZ.v !i))
    decreases (bm - !i)
  {
    let vi = !i;
    let r = r0 +^ vi;
    let valid = r <^ rows;
    let pos : szlt sq = r %^ sq;
    let take = valid && ((not !found) || (pos >^ !maxpos));
    let nextmax : sz = if take { (pos <: sz) } else { !maxpos };
    maxpos := nextmax;
    found := !found || valid;
    i := !i +^ 1sz;
  };
  let base = sk -^ sq;
  let extra = optional_inc !found !maxpos;
  SZ.smin sk (base +^ extra)
}

inline_for_extraction noextract
fn sdpa_flash_causal_mask
  (bm bn sk sq rows : szp)
  (r0 : sz) (causal : bool)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v sk + SZ.v bm + 1)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  returns nkt : sz
  ensures pure (SZ.v nkt <= SZ.v sk / SZ.v bn + 1 /\
                SZ.v nkt == SF.key_tiles (SZ.v bn) (SZ.v bm) (SZ.v sq) (SZ.v sk)
                              (SZ.v rows) (SZ.v r0) causal)
{
  let kmax : sz =
    if causal { sdpa_flash_causal_active bm sk sq rows r0 } else { (sk <: sz) };
  let r = SZ.sdivup kmax bn;
  SZ.lem_sdivup kmax bn;
  r
}

inline_for_extraction noextract
fn sdpa_flash_combine_partials_active
  (#et : Type0) {| floating et |}
  (nw bm : szp)
  (#lgm #lgl : layout1 bm) {| ctlayout lgm |} {| ctlayout lgl |}
  (shM shL shscale : array2 et (l2_row_major nw bm))
  (shgm : array1 et lgm) (shgl : array1 et lgl)
  (lane : szlt bm)
  (#fM #fL : perm)
  (#eM #eL : chest2 et (SZ.v nw) (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  preserves
    (shM |-> Frac fM eM) ** (shL |-> Frac fL eL)
  requires
    (exists* (e : chest2 et (SZ.v nw) 1).
       tensor_pts_to (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane)) e)
    ** cell_full_n shgm (SZ.v lane)
    ** cell_full_n shgl (SZ.v lane)
  ensures
    tensor_pts_to (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane))
      (SF.gscale_col eM (SZ.v lane))
    ** cell_full_n_v shgm (SZ.v lane) (SF.gmax eM (SZ.v lane) (SZ.v nw))
    ** cell_full_n_v shgl (SZ.v lane)
         (SF.gsum eM eL (SF.gmax eM (SZ.v lane) (SZ.v nw)) (SZ.v lane) (SZ.v nw))
{
  let mut gm : et = neg infinity;
  let mut ww : sz = 0sz;
  while (!ww <^ nw)
    invariant
      live ww **
      (shM |-> Frac fM eM) ** (shL |-> Frac fL eL) **
      (exists* (n : nat { n <= SZ.v nw }).
         (gm |-> SF.gmax eM (SZ.v lane) n) ** pure (n == SZ.v !ww)) **
      (exists* (e : chest2 et (SZ.v nw) 1).
         tensor_pts_to (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane)) e) **
      cell_full_n shgm (SZ.v lane) **
      cell_full_n shgl (SZ.v lane) **
      pure (SZ.v !ww <= SZ.v nw)
    decreases (nw - !ww)
  {
    let vww = !ww;
    let iw : szlt nw = vww;
    let mv = tensor_read shM (cidx2 iw lane);
    gm := fmax !gm mv;
    ww := !ww +^ 1sz;
  };

  let gmv = !gm;
  let zer : et = zero;
  let mut gl : et = zer;
  rewrite (gl |-> zer) as (gl |-> SF.gsum eM eL gmv (SZ.v lane) 0);
  ww := 0sz;
  while (!ww <^ nw)
    invariant
      live gm ** live ww **
      (shM |-> Frac fM eM) ** (shL |-> Frac fL eL) **
      (exists* (n : nat { n <= SZ.v nw }) (e : chest2 et (SZ.v nw) 1).
         (gl |-> SF.gsum eM eL gmv (SZ.v lane) n) **
         tensor_pts_to (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane)) e **
         pure (n == SZ.v !ww /\
               (forall (w : natlt (SZ.v nw)). w < n ==>
                  acc2 e w 0 == SF.gscale eM gmv w (SZ.v lane)))) **
      cell_full_n shgm (SZ.v lane) **
      cell_full_n shgl (SZ.v lane) **
      pure (SZ.v !ww <= SZ.v nw)
    decreases (nw - !ww)
  {
    let vww = !ww;
    let iw : szlt nw = vww;
    with n0 e0. assert (
      (gl |-> SF.gsum eM eL gmv (SZ.v lane) n0) **
      tensor_pts_to (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane)) e0);
    let mv = tensor_read shM (cidx2 iw lane);
    let lv = tensor_read shL (cidx2 iw lane);
    let sc = fexp (mv `sub` gmv);
    (* TODO(line 223): clamp [sc] to zero when not finite once Kuiper has an
       extractable [isfinite].  The current [kind] test is ghost-only. *)
    tensor_write #_ #_ #_ #_ #(flash_scale_ctlayout nw bm lane)
      (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane))
      (cidx2 iw 0sz) sc;
    with e1. assert (
      tensor_pts_to (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane)) e1);
    let cur = !gl;
    gl := cur `add` (sc `mul` lv);
    rewrite (gl |-> (cur `add` (sc `mul` lv)))
         as (gl |-> SF.gsum eM eL gmv (SZ.v lane) (n0 + 1));
    assert pure (forall (w : natlt (SZ.v nw)). w < n0 + 1 ==>
      acc2 e1 w 0 == SF.gscale eM gmv w (SZ.v lane));
    ww := !ww +^ 1sz;
  };

  with escale. assert (
    tensor_pts_to (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v lane)) escale);
  SF.gscale_col_ext eM (SZ.v lane) escale;

  unfold (cell_full_n shgm (SZ.v lane));
  with oldgm. assert (tensor_pts_to_cell shgm (idx1 (SZ.v lane)) oldgm);
  rewrite (tensor_pts_to_cell shgm (idx1 (SZ.v lane)) oldgm)
       as (tensor_pts_to_cell shgm (up (cidx1 lane)) oldgm);
  tensor_write_cell shgm (cidx1 lane) gmv;
  fold (cell_full_n_v shgm (SZ.v lane) gmv);

  unfold (cell_full_n shgl (SZ.v lane));
  with oldgl. assert (tensor_pts_to_cell shgl (idx1 (SZ.v lane)) oldgl);
  rewrite (tensor_pts_to_cell shgl (idx1 (SZ.v lane)) oldgl)
       as (tensor_pts_to_cell shgl (up (cidx1 lane)) oldgl);
  let glv = !gl;
  tensor_write_cell shgl (cidx1 lane) glv;
  fold (cell_full_n_v shgl (SZ.v lane) glv);
}

inline_for_extraction noextract
fn sdpa_flash_combine_partials
  (#et : Type0) {| floating et |}
  (nw bm : szp)
  (#lgm #lgl : layout1 bm) {| ctlayout lgm |} {| ctlayout lgl |}
  (shM shL shscale : array2 et (l2_row_major nw bm))
  (shgm : array1 et lgm) (shgl : array1 et lgl)
  (w : szlt nw) (lane : szlt warp_size)
  (#fM #fL : perm)
  (#eM #eL : chest2 et (SZ.v nw) (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  preserves
    (shM |-> Frac fM eM) ** (shL |-> Frac fL eL)
  requires
    if_ (combine_active bm w lane)
      (combine_cells nw bm shscale shgm shgl lane)
  ensures
    if_ (combine_active bm w lane)
      (combine_cells_v nw bm shscale shgm shgl lane
        (SF.gscale_col eM (SZ.v (clamp_lt bm lane)))
        (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
        (SF.gsum eM eL (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
           (SZ.v (clamp_lt bm lane)) (SZ.v nw)))
{
  let active = combine_active bm w lane;
  if active {
    if_elim_true (combine_cells nw bm shscale shgm shgl lane);
    unfold (combine_cells nw bm shscale shgm shgl lane);
    sdpa_flash_combine_partials_active nw bm shM shL shscale shgm shgl
      (clamp_lt bm lane);
    fold (combine_cells_v nw bm shscale shgm shgl lane
      (SF.gscale_col eM (SZ.v (clamp_lt bm lane)))
      (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
      (SF.gsum eM eL (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
         (SZ.v (clamp_lt bm lane)) (SZ.v nw)));
    if_intro_true (combine_cells_v nw bm shscale shgm shgl lane
      (SF.gscale_col eM (SZ.v (clamp_lt bm lane)))
      (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
      (SF.gsum eM eL (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
         (SZ.v (clamp_lt bm lane)) (SZ.v nw)));
  } else {
    if_elim_false (combine_cells nw bm shscale shgm shgl lane);
    if_intro_false (combine_cells_v nw bm shscale shgm shgl lane
      (SF.gscale_col eM (SZ.v (clamp_lt bm lane)))
      (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
      (SF.gsum eM eL (SF.gmax eM (SZ.v (clamp_lt bm lane)) (SZ.v nw))
         (SZ.v (clamp_lt bm lane)) (SZ.v nw)));
  }
}

inline_for_extraction noextract
fn sdpa_flash_o_store_cell_active
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nw bm d rows b hq sq : szp)
  (#lscale : layout2 nw bm) (#lgl : layout1 bm)
  (#lout : layout4 b hq sq d)
  {| ctlayout lscale |} {| ctlayout lgl |} {| ctlayout lout |}
  (shscale : array2 et_acc lscale)
  (shO : array2 et_acc (l2_row_major (SZ.v nw * SZ.v bm) d))
  (shgl : array1 et_acc lgl)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (bi : szlt b) (kvh : sz) (group : szp)
  (i : szlt bm) (dd : szlt d) (r : szlt rows)
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fscale #fO #fgl : perm)
  (#escale : chest2 et_acc (SZ.v nw) (SZ.v bm))
  (#eO : chest2 et_acc (SZ.v nw * SZ.v bm) (SZ.v d))
  (#egl : chest1 et_acc (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm * SZ.v d)))
  preserves
    (shscale |-> Frac fscale escale) **
    (shO |-> Frac fO eO) **
    (shgl |-> Frac fgl egl)
  requires
    out_cell b hq sq d gout (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r) (SZ.v dd)
  ensures
    out_cell_v b hq sq d gout (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r)
      (SZ.v dd) (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd)))
{
  let zer : et_acc = zero;
  let mut acc : et_acc = zer;
  let mut ww : sz = 0sz;
  rewrite (acc |-> zer) as (acc |-> SF.ocomb escale eO (SZ.v i) (SZ.v dd) 0);
  while (!ww <^ nw)
    invariant
      live ww **
      (shscale |-> Frac fscale escale) **
      (shO |-> Frac fO eO) **
      (shgl |-> Frac fgl egl) **
      out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r) (SZ.v dd) **
      (exists* (n : (m:nat { m <= SZ.v nw })).
        (acc |-> SF.ocomb escale eO (SZ.v i) (SZ.v dd) n) **
        pure (n == SZ.v !ww))
    decreases (nw - !ww)
  {
    let vww = !ww;
    let iw : szlt nw = vww;
    tile_idx_lem (SZ.v bm) (SZ.v iw) (SZ.v i) (SZ.v nw * SZ.v bm);
    let orow : szlt (SZ.v nw * SZ.v bm) = bm *^ iw +^ i;
    let sv = tensor_read shscale (cidx2 iw i);
    let ov = tensor_read shO (cidx2 orow dd);
    with n0. assert (acc |-> SF.ocomb escale eO (SZ.v i) (SZ.v dd) n0);
    let a0 = !acc;
    acc := a0 `add` (sv `mul` ov);
    SF.ocomb_step escale eO (SZ.v i) (SZ.v dd) (SZ.v iw);
    rewrite (acc |-> (a0 `add` (sv `mul` ov)))
      as (acc |-> SF.ocomb escale eO (SZ.v i) (SZ.v dd) (SZ.v vww + 1));
    ww := vww +^ 1sz;
  };
  with nend. assert (acc |-> SF.ocomb escale eO (SZ.v i) (SZ.v dd) nend);
  rewrite (acc |-> SF.ocomb escale eO (SZ.v i) (SZ.v dd) nend)
    as (acc |-> SF.ocomb escale eO (SZ.v i) (SZ.v dd) (SZ.v nw));
  let lv = tensor_read shgl (cidx1 i);
  let inv : et_acc = SF.onorm lv;
  let qh0 = kvh *^ group +^ (r /^ sq);
  let qh1 : szlt hq = clamp_lt hq qh0;
  let qpos : szlt sq = r %^ sq;
  assert pure (
    SZ.v qh1 ==
      out_qh (SZ.v hq) (SZ.v sq) (SZ.v kvh) (SZ.v group) (SZ.v r));
  assert pure (SZ.v qpos == out_qpos (SZ.v sq) (SZ.v r));
  unfold (out_cell b hq sq d gout
    (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r) (SZ.v dd));
  with old. assert (
    tensor_pts_to_cell gout
      (idx4 (SZ.v bi)
        (out_qh (SZ.v hq) (SZ.v sq) (SZ.v kvh) (SZ.v group) (SZ.v r))
        (out_qpos (SZ.v sq) (SZ.v r))
        (SZ.v dd))
      old);
  rewrite
    (tensor_pts_to_cell gout
      (idx4 (SZ.v bi)
        (out_qh (SZ.v hq) (SZ.v sq) (SZ.v kvh) (SZ.v group) (SZ.v r))
        (out_qpos (SZ.v sq) (SZ.v r))
        (SZ.v dd))
      old)
  as (tensor_pts_to_cell gout (up (cidx4 bi qh1 qpos dd)) old);
  let av = !acc;
  let res : et_ab = FC.fcast (av `mul` inv);
  tensor_write_cell gout (cidx4 bi qh1 qpos dd) res;
  rewrite
    (tensor_pts_to_cell gout (up (cidx4 bi qh1 qpos dd)) res)
  as
    (tensor_pts_to_cell gout
      (idx4 (SZ.v bi)
        (out_qh (SZ.v hq) (SZ.v sq) (SZ.v kvh) (SZ.v group) (SZ.v r))
        (out_qpos (SZ.v sq) (SZ.v r))
        (SZ.v dd))
      (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd))));
  fold (out_cell_v b hq sq d gout
    (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r) (SZ.v dd)
    (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd))));
}

inline_for_extraction noextract
fn sdpa_flash_o_store_cell_maybe
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nw bm d rows b hq sq : szp)
  (#lscale : layout2 nw bm) (#lgl : layout1 bm)
  (#lout : layout4 b hq sq d)
  {| ctlayout lscale |} {| ctlayout lgl |} {| ctlayout lout |}
  (shscale : array2 et_acc lscale)
  (shO : array2 et_acc (l2_row_major (SZ.v nw * SZ.v bm) d))
  (shgl : array1 et_acc lgl)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (bi : szlt b) (kvh : sz) (group : szp)
  (i : szlt bm) (dd : szlt d) (r0 : sz)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fscale #fO #fgl : perm)
  (#escale : chest2 et_acc (SZ.v nw) (SZ.v bm))
  (#eO : chest2 et_acc (SZ.v nw * SZ.v bm) (SZ.v d))
  (#egl : chest1 et_acc (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm * SZ.v d)))
  preserves
    (shscale |-> Frac fscale escale) **
    (shO |-> Frac fO eO) **
    (shgl |-> Frac fgl egl)
  requires
    when_ (SZ.v r0 + SZ.v i < SZ.v rows)
      (out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd))
  ensures
    when_ (SZ.v r0 + SZ.v i < SZ.v rows)
      (out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd)
        (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd))))
{
  let r = r0 +^ i;
  let valid = r <^ rows;
  if valid {
    when_elim_true (SZ.v r0 + SZ.v i < SZ.v rows) (
      out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd));
    let rr : szlt rows = clamp_lt rows r;
    rewrite
      (out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd))
    as
      (out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v rr) (SZ.v dd));
    sdpa_flash_o_store_cell_active nw bm d rows b hq sq
      shscale shO shgl gout bi kvh group i dd rr;
    rewrite
      (out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v rr) (SZ.v dd) (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd))))
    as
      (out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd)
        (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd))));
    when_intro_true (SZ.v r0 + SZ.v i < SZ.v rows) (
      out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd)
        (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd))));
  } else {
    assert pure ((SZ.v r0 + SZ.v i < SZ.v rows) == false);
    when_elim_false (SZ.v r0 + SZ.v i < SZ.v rows) (
      out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd));
    when_intro_false (SZ.v r0 + SZ.v i < SZ.v rows) (
      out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + SZ.v i) (SZ.v dd)
        (FC.fcast (SF.out_val escale eO egl (SZ.v i) (SZ.v dd))));
  }
}

#push-options "--ifuel 1 --fuel 0 --z3rlimit 20"
let mod_gap (n : pos) (a k : nat)
  : Lemma (requires a % n == k % n /\ a > k) (ensures a >= k + n)
= FStar.Math.Lemmas.euclidean_division_definition a n;
  FStar.Math.Lemmas.euclidean_division_definition k n;
  FStar.Math.Lemmas.distributivity_add_right n (k / n) 1;
  if a / n <= k / n
  then FStar.Math.Lemmas.lemma_mult_le_left n (a / n) (k / n)
  else FStar.Math.Lemmas.lemma_mult_le_left n (k / n + 1) (a / n)

let flat_inj (bm d : pos) (x y : natlt bm & natlt d)
  : Lemma (requires x._1 * d + x._2 == y._1 * d + y._2) (ensures x == y)
= FStar.Math.Lemmas.lemma_mod_plus x._2 x._1 d;
  FStar.Math.Lemmas.lemma_mod_plus y._2 y._1 d;
  FStar.Math.Lemmas.division_addition_lemma x._2 d x._1;
  FStar.Math.Lemmas.division_addition_lemma y._2 d y._1

let stride_ge_lane (bm d : pos) (lane : natlt BW.warp_size)
  : Lemma (forall (x : stride_index2 bm d BW.warp_size lane).
             x._1 * d + x._2 >= lane)
= introduce forall (x : stride_index2 bm d BW.warp_size lane).
      x._1 * d + x._2 >= lane
  with FStar.Math.Lemmas.euclidean_division_definition
         (x._1 * d + x._2) BW.warp_size

let stride_lt_ncells (bm d : pos) (lane : natlt BW.warp_size)
  : Lemma (forall (x : stride_index2 bm d BW.warp_size lane).
             x._1 * d + x._2 < bm * d)
= introduce forall (x : stride_index2 bm d BW.warp_size lane).
      x._1 * d + x._2 < bm * d
  with FStar.Math.Lemmas.lemma_mult_le_right d (x._1 + 1) bm

let stride_step_lem (bm d : pos) (lane : natlt BW.warp_size) (k : nat)
  (y : stride_index2 bm d BW.warp_size lane)
  : Lemma (requires y._1 * d + y._2 == k)
    (ensures forall (x : stride_index2 bm d BW.warp_size lane).
      ((~(x._1 * d + x._2 < k) /\ x =!= y)
         <==> ~(x._1 * d + x._2 < k + BW.warp_size)) /\
      ((x._1 * d + x._2 < k \/ y == x)
         <==> x._1 * d + x._2 < k + BW.warp_size))
= introduce forall (x : stride_index2 bm d BW.warp_size lane).
      ((~(x._1 * d + x._2 < k) /\ x =!= y)
         <==> ~(x._1 * d + x._2 < k + BW.warp_size)) /\
      ((x._1 * d + x._2 < k \/ y == x)
         <==> x._1 * d + x._2 < k + BW.warp_size)
  with (
    if x._1 * d + x._2 > k
    then mod_gap BW.warp_size (x._1 * d + x._2) k
    else if x._1 * d + x._2 = k
    then flat_inj bm d x y
    else ()
  )
#pop-options

inline_for_extraction noextract
fn sdpa_flash_o_store_active
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nw bm d rows b hq sq : szp)
  (#lscale : layout2 nw bm) (#lgl : layout1 bm)
  (#lout : layout4 b hq sq d)
  {| ctlayout lscale |} {| ctlayout lgl |} {| ctlayout lout |}
  (shscale : array2 et_acc lscale)
  (shO : array2 et_acc (l2_row_major (SZ.v nw * SZ.v bm) d))
  (shgl : array1 et_acc lgl)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (bi : szlt b) (kvh : sz) (group : szp)
  (r0 : sz) (lane : szlt warp_size)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fscale #fO #fgl : perm)
  (#escale : chest2 et_acc (SZ.v nw) (SZ.v bm))
  (#eO : chest2 et_acc (SZ.v nw * SZ.v bm) (SZ.v d))
  (#egl : chest1 et_acc (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d + BW.warp_size)))
  preserves
    (shscale |-> Frac fscale escale) **
    (shO |-> Frac fO eO) **
    (shgl |-> Frac fgl egl)
  requires out_store_cells b hq sq bm d rows gout
    (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane)
  ensures  out_store_cells_v nw b hq sq bm d rows gout escale eO egl
    (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane)
{
  let ncells : sz = bm *^ d;
  let mut idx : sz = lane;
  let mut iter : sz = 0sz;
  stride_ge_lane (SZ.v bm) (SZ.v d) (SZ.v lane);
  unfold (out_store_cells b hq sq bm d rows gout
    (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane));
  forevery_refine_split
    #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
    (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
      (out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2))
    (fun ij -> ij._1 * SZ.v d + ij._2 < SZ.v lane);
  forevery_elim_empty
    #(ij : stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane)
        { ij._1 * SZ.v d + ij._2 < SZ.v lane })
    (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
      (out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2));
  forevery_intro_false
    #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
    (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
      (out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2
        (FC.fcast (SF.out_val escale eO egl ij._1 ij._2))));
  forevery_refine_ext
    #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
    #(fun _ -> False)
    (fun ij -> ij._1 * SZ.v d + ij._2 < SZ.v lane)
    (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
      (out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2
        (FC.fcast (SF.out_val escale eO egl ij._1 ij._2))));
  while (!idx <^ ncells)
    invariant
      live idx ** live iter **
      (shscale |-> Frac fscale escale) **
      (shO |-> Frac fO eO) **
      (shgl |-> Frac fgl egl) **
      (forall+ (ij : stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane)
                  { ij._1 * SZ.v d + ij._2 < SZ.v !idx }).
        when_ (SZ.v r0 + ij._1 < SZ.v rows)
          (out_cell_v b hq sq d gout
            (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2
            (FC.fcast (SF.out_val escale eO egl ij._1 ij._2)))) **
      (forall+ (ij : stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane)
                  { ~(ij._1 * SZ.v d + ij._2 < SZ.v !idx) }).
        when_ (SZ.v r0 + ij._1 < SZ.v rows)
          (out_cell b hq sq d gout
            (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2)) **
      pure (SZ.v !idx % BW.warp_size == SZ.v lane) **
      pure (SZ.v !idx < SZ.v bm * SZ.v d + BW.warp_size) **
      pure (SZ.v !iter <= SZ.v !idx)
    decreases (SZ.v ncells - SZ.v !iter)
  {
    let flat = !idx;
    let next = flat +^ 32sz;
    assert pure (SZ.v next == SZ.v flat + BW.warp_size);
    let i : szlt bm = flat /^ d;
    let dd : szlt d = flat %^ d;
    FStar.Math.Lemmas.euclidean_division_definition (SZ.v flat) (SZ.v d);
    stride_step_lem (SZ.v bm) (SZ.v d) (SZ.v lane) (SZ.v flat)
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    forevery_remove'
      #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
      (fun ij -> ~(ij._1 * SZ.v d + ij._2 < SZ.v flat))
      (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
        (out_cell b hq sq d gout
          (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2))
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    sdpa_flash_o_store_cell_maybe nw bm d rows b hq sq
      shscale shO shgl gout bi kvh group i dd r0;
    forevery_refine_ext
      #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
      #(fun ij -> ~(ij._1 * SZ.v d + ij._2 < SZ.v flat) /\
                  ij =!= ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d))))
      (fun ij -> ~(ij._1 * SZ.v d + ij._2 < SZ.v next))
      (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
        (out_cell b hq sq d gout
          (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2));
    forevery_insert
      #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
      #(fun ij -> ij._1 * SZ.v d + ij._2 < SZ.v flat)
      (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
        (out_cell_v b hq sq d gout
          (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2
          (FC.fcast (SF.out_val escale eO egl ij._1 ij._2))))
      ((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)));
    forevery_refine_ext
      #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
      #(fun ij -> ij._1 * SZ.v d + ij._2 < SZ.v flat \/
                  (((SZ.v i <: natlt (SZ.v bm)), (SZ.v dd <: natlt (SZ.v d)))
                     <: stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane)) == ij)
      (fun ij -> ij._1 * SZ.v d + ij._2 < SZ.v next)
      (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
        (out_cell_v b hq sq d gout
          (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2
          (FC.fcast (SF.out_val escale eO egl ij._1 ij._2))));
    FStar.Math.Lemmas.lemma_mod_plus (SZ.v flat) 1 BW.warp_size;
    idx := next;
    iter := !iter +^ 1sz;
  };
  stride_lt_ncells (SZ.v bm) (SZ.v d) (SZ.v lane);
  forevery_refine_ext
    #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
    (fun _ -> False)
    (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
      (out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2));
  forevery_elim_empty
    #(ij : stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane) { False })
    (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
      (out_cell b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2));
  forevery_unrefine
    #(stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size (SZ.v lane))
    (fun ij -> when_ (SZ.v r0 + ij._1 < SZ.v rows)
      (out_cell_v b hq sq d gout
        (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0 + ij._1) ij._2
        (FC.fcast (SF.out_val escale eO egl ij._1 ij._2))));
  fold (out_store_cells_v nw b hq sq bm d rows gout escale eO egl
    (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane));
}

inline_for_extraction noextract
fn sdpa_flash_o_store
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_acc et_ab |}
  (nw bm d rows b hq sq : szp)
  (#lscale : layout2 nw bm) (#lgl : layout1 bm)
  (#lout : layout4 b hq sq d)
  {| ctlayout lscale |} {| ctlayout lgl |} {| ctlayout lout |}
  (shscale : array2 et_acc lscale)
  (shO : array2 et_acc (l2_row_major (SZ.v nw * SZ.v bm) d))
  (shgl : array1 et_acc lgl)
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (bi : szlt b) (kvh : sz) (group : szp)
  (r0 : sz) (w : szlt nw) (lane : szlt warp_size)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fscale #fO #fgl : perm)
  (#escale : chest2 et_acc (SZ.v nw) (SZ.v bm))
  (#eO : chest2 et_acc (SZ.v nw * SZ.v bm) (SZ.v d))
  (#egl : chest1 et_acc (SZ.v bm))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v d + BW.warp_size)))
  preserves
    (shscale |-> Frac fscale escale) **
    (shO |-> Frac fO eO) **
    (shgl |-> Frac fgl egl)
  requires if_ (w = 0sz)
    (out_store_cells b hq sq bm d rows gout
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane))
  ensures  if_ (w = 0sz)
    (out_store_cells_v nw b hq sq bm d rows gout escale eO egl
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane))
{
  let active = w = 0sz;
  if active {
    if_elim_true (out_store_cells b hq sq bm d rows gout
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane));
    sdpa_flash_o_store_active nw bm d rows b hq sq
      shscale shO shgl gout bi kvh group r0 lane;
    if_intro_true (out_store_cells_v nw b hq sq bm d rows gout escale eO egl
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane));
  } else {
    if_elim_false (out_store_cells b hq sq bm d rows gout
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane));
    if_intro_false (out_store_cells_v nw b hq sq bm d rows gout escale eO egl
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane));
  }
}
