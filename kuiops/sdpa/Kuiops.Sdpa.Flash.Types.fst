module Kuiops.Sdpa.Flash.Types

(* Shared pure definitions for the flash-attention kernel: index algebra and
   bijections, the shared-memory view record and its per-warp/per-thread
   projections, and the slprops of the three-phase barrier contract. *)

#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg
open Kuiper.Bijection
open Kuiper.ForEvery
open Kuiper.Kernel.FlashAttention.KernelDesc

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module BW = Kuiper.Barrier.Warp
module B = Kuiper.Barrier

inline_for_extraction noextract
let flash_scale_cimap
  (nw bm : szp) (lane : szlt bm)
  (#_ : squash (SZ.fits (SZ.v nw * SZ.v bm)))
  (x : conc ((SZ.v nw / 1) @| (SZ.v bm / SZ.v bm) @| INil))
  : r:sz {
      SZ.v r ==
        (stride_subtile_layout
          (l2_row_major (SZ.v nw) (SZ.v bm))
          1 (SZ.v bm) 0 (SZ.v lane)).imap.f (up x) } =
  match x with
  | (i, (j, ())) ->
    assert (SZ.v j == 0);
    i *^ bm +^ j *^ bm +^ lane

noeq inline_for_extraction noextract
type flash_views
  (et_ab et_acc : Type0) (nw d : nat) =
{
  shQv : array2 et_ab (l2_row_major 16 d);
  shKv : array2 et_ab (l2_row_major (nw * 16) d);
  shVv : array2 et_ab (l2_row_major (nw * 16) d);
  shSv : array2 et_acc (l2_row_major (nw * 16) 16);
  shPv : array2 et_ab (l2_row_major (nw * 16) 16);
  shPVv : array2 et_acc (l2_row_major (nw * 16) 16);
  shcwv : array2 et_acc (l2_row_major nw 16);
  shMv : array2 et_acc (l2_row_major nw 16);
  shLv : array2 et_acc (l2_row_major nw 16);
  shscalev : array2 et_acc (l2_row_major nw 16);
  shOv : array2 et_acc (l2_row_major (nw * 16) d);
  shgmv : array1 et_acc (l1_forward 16);
  shglv : array1 et_acc (l1_forward 16);
}


(* ---- shared index algebra and cell-ownership slprops ---- *)

unfold let warp_row_span : nat = 2

(* Concrete, content-free ownership states for the three block barriers in
   the flash-attention block.  Values are existential throughout: the
   description proves only permission transfer and bounds safety. *)

let block_threads (nw : szp) : nat = SZ.v nw * BW.warp_size

let thread_w (nw : szp) (tid : natlt (block_threads nw)) : natlt (SZ.v nw) =
  tid / BW.warp_size

let thread_lane (nw : szp) (tid : natlt (block_threads nw)) : natlt BW.warp_size =
  tid % BW.warp_size

let stride_index2 (rows cols : nat) (nthr : pos) (tid : natlt nthr) : Type0 =
  ij:(natlt rows & natlt cols) { (ij._1 * cols + ij._2) % nthr == tid }

unfold
let strided_cells2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (nthr : pos) (tid : natlt nthr) : slprop
= forall+ (ij : stride_index2 rows cols nthr tid).
    exists* (v : et). Cell a (idx2 ij._1 ij._2) |-> Frac 1.0R v

unfold
let cell_full
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l) (i : natlt len) : slprop
= exists* (v : et). Cell a (idx1 i) |-> Frac 1.0R v

let row_layout
  (#et : Type0) (#rows #cols : erased nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : erased nat { i < rows }) : layout1 cols
= tlayout_slice l 0 i

inline_for_extraction noextract
let row
  (#et : Type0) (#rows #cols : erased nat) (#l : layout2 rows cols)
  (a : array2 et l) (i : erased nat { i < rows })
  : array1 et (row_layout a i)
= sliceof a 0 i

let clamp_nat_lt (n : pos) (x : nat) : natlt n =
  if x < n then x else 0

unfold
let row_subtile
  (#et:Type0) (#l : layout2 16 16)
  (shA : array2 et l)
  (i : natlt 16) : slprop
= exists* (r : chest2 et 1 16).
    array2_subtile shA 1 16 i 0 |-> Frac 1.0R r

inline_for_extraction noextract
let clamp_lt (sk : szp) (x : sz) : szlt sk =
  if x <^ sk then x else 0sz

unfold
let out_stride_index2 (rows cols : nat) (nthr : pos) (tid : natlt nthr) : Type0 =
  ij:(natlt rows & natlt cols) {
    (ij._1 * cols + ij._2) % nthr == tid}

unfold
let cell_full_n
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (shA : array1 et l) (i : natlt len) : slprop
= exists* (v : et). tensor_pts_to_cell shA (idx1 i) v

inline_for_extraction noextract
let lane_active (bm : szp) (lane : szlt warp_size) : bool =
  lane <^ bm

let ml_cells
  (#et : Type0) (bm : szp)
  (#lm #ll : layout1 bm)
  (shm : array1 et lm) (shl : array1 et ll)
  (lane : szlt warp_size) : slprop
= cell_full_n shm (SZ.v (clamp_lt bm lane)) **
  cell_full_n shl (SZ.v (clamp_lt bm lane))

inline_for_extraction noextract
let combine_active (bm : szp) (w : sz) (lane : szlt warp_size) : bool =
  (w = 0sz) && (lane <^ bm)

let combine_cells
  (#et : Type0) (nw bm : szp)
  (#lgm #lgl : layout1 bm)
  (shscale : array2 et (l2_row_major nw bm))
  (shgm : array1 et lgm) (shgl : array1 et lgl)
  (lane : szlt warp_size) : slprop
= (exists* (e : chest2 et (SZ.v nw) 1).
     tensor_pts_to
       (array2_stride_subtile shscale 1 (SZ.v bm) 0 (SZ.v (clamp_lt bm lane))) e)
  ** cell_full_n shgm (SZ.v (clamp_lt bm lane))
  ** cell_full_n shgl (SZ.v (clamp_lt bm lane))

let out_qh
  (hq sq : pos) (kvh : nat) (group : pos) (r : nat) : natlt hq
= clamp_nat_lt hq (kvh * group + r / sq)

let out_qpos (sq : pos) (r : nat) : natlt sq =
  r % sq

let out_cell
  (#et : Type0) (b hq sq d : szp)
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bi : natlt (SZ.v b)) (kvh : nat) (group : pos)
  (r : nat) (dd : natlt (SZ.v d)) : slprop
= exists* (v : et).
    tensor_pts_to_cell gout
      (idx4 bi
        (out_qh (SZ.v hq) (SZ.v sq) kvh group r)
        (out_qpos (SZ.v sq) r)
        dd)
      v

let out_store_cells
  (#et : Type0) (b hq sq bm d rows : szp)
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bi : natlt (SZ.v b)) (kvh : nat) (group : pos)
  (r0 : nat) (lane : natlt BW.warp_size) : slprop
= forall+ (ij : out_stride_index2 (SZ.v bm) (SZ.v d) BW.warp_size lane).
    when_ (r0 + ij._1 < SZ.v rows)
      (out_cell b hq sq d gout bi kvh group (r0 + ij._1) ij._2)

unfold let flash_nwarps : nat = 4

unfold
let flash_scale_tile
  (#et : Type0) (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (lane : natlt 16) : slprop =
  exists* (e : chest2 et (SZ.v nw) 1).
    tensor_pts_to
      (array2_stride_subtile shscale 1 16 0 lane) e

inline_for_extraction noextract
let flash_shmems
  (et_ab et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (nw d : szp)
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  : list shmem_desc =
  [ SHArray et_ab (16sz *^ d)
  ; SHArray et_ab (nw *^ 16sz *^ d)
  ; SHArray et_ab (nw *^ 16sz *^ d)
  ; SHArray et_acc (nw *^ 16sz *^ 16sz)
  ; SHArray et_ab (nw *^ 16sz *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz)
  ; SHArray et_acc (nw *^ 16sz *^ d)
  ; SHArray et_acc 16sz
  ; SHArray et_acc 16sz
  ]

inline_for_extraction noextract
let flash_views_of
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (nw d : szp)
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (sh : c_shmems (flash_shmems et_ab et_acc nw d))
  : flash_views et_ab et_acc (SZ.v nw) (SZ.v d) =
  let
    (q, (k, (v, (s, (p, (pv, (cw,
    (m, (l, (scale, (o, (gm, (gl, _))))))))))))) = sh in
  {
    shQv = from_array (l2_row_major 16 (SZ.v d)) q;
    shKv = from_array
      (l2_row_major (SZ.v nw * 16) (SZ.v d)) k;
    shVv = from_array
      (l2_row_major (SZ.v nw * 16) (SZ.v d)) v;
    shSv = from_array
      (l2_row_major (SZ.v nw * 16) 16) s;
    shPv = from_array
      (l2_row_major (SZ.v nw * 16) 16) p;
    shPVv = from_array
      (l2_row_major (SZ.v nw * 16) 16) pv;
    shcwv = from_array
      (l2_row_major (SZ.v nw) 16) cw;
    shMv = from_array (l2_row_major (SZ.v nw) 16) m;
    shLv = from_array (l2_row_major (SZ.v nw) 16) l;
    shscalev = from_array (l2_row_major (SZ.v nw) 16) scale;
    shOv = from_array (l2_row_major (SZ.v nw * 16) (SZ.v d)) o;
    shgmv = from_array (l1_forward 16) gm;
    shglv = from_array (l1_forward 16) gl;
  }

inline_for_extraction noextract
let flash_warp_k
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_ab
      (subtile_layout (l2_row_major (nw * 16) d) 16 (d <: pos) w 0) =
  array2_subtile v.shKv 16 (d <: pos) w 0

inline_for_extraction noextract
let flash_warp_v
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_ab
      (subtile_layout (l2_row_major (nw * 16) d) 16 (d <: pos) w 0) =
  array2_subtile v.shVv 16 (d <: pos) w 0

inline_for_extraction noextract
let flash_warp_s
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_acc
      (subtile_layout (l2_row_major (nw * 16) 16) 16 16 w 0) =
  array2_subtile v.shSv 16 16 w 0

inline_for_extraction noextract
let flash_warp_p
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_ab
      (subtile_layout (l2_row_major (nw * 16) 16) 16 16 w 0) =
  array2_subtile v.shPv 16 16 w 0

inline_for_extraction noextract
let flash_warp_pv
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array2 et_acc
      (subtile_layout (l2_row_major (nw * 16) 16) 16 16 w 0) =
  array2_subtile v.shPVv 16 16 w 0

inline_for_extraction noextract
let flash_warp_cw
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d)
  (w : natlt nw)
  : array1 et_acc
      (tlayout_slice (l2_row_major nw 16) 0 w) =
  row v.shcwv w

let flash_stride_partition_bij
  (rows cols : nat) (nthr : pos)
  : ((tid : natlt nthr & stride_index2 rows cols nthr tid)
      =~ (natlt rows & natlt cols)) =
{
  ff = (fun (x :
      (tid : natlt nthr & stride_index2 rows cols nthr tid)) ->
    x._2)
    <: ((tid : natlt nthr & stride_index2 rows cols nthr tid) ->
        GTot (natlt rows & natlt cols));
  gg = (fun (ij : natlt rows & natlt cols) ->
    (| ((ij._1 * cols + ij._2) % nthr), ij |))
    <: ((natlt rows & natlt cols) ->
        GTot (tid : natlt nthr &
          stride_index2 rows cols nthr tid));
}

unfold
let flash_abs1_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
{
  ff = (fun (i, ()) -> i);
  gg = (fun i -> idx1 i);
}

unfold
let flash_combine_local
  (#et : Type0) (nw : szp)
  (vscale : array2 et (l2_row_major (SZ.v nw) 16))
  (vgm vgl : array1 et (l1_forward 16))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size)
  : slprop =
  when_ (w = 0 /\ lane < 16)
    ((exists* (e : chest2 et (SZ.v nw) 1).
       tensor_pts_to
         (array2_stride_subtile vscale 1 16 0
           (clamp_nat_lt 16 lane)) e)
     ** cell_full vgm (clamp_nat_lt 16 lane)
     ** cell_full vgl (clamp_nat_lt 16 lane))

let flash_combine_idx (nw : szp) : Type0 =
  wl : (natlt (SZ.v nw) & natlt BW.warp_size) {
    wl._1 = 0 /\ wl._2 < 16}

let flash_combine_bij (nw : szp)
  : (flash_combine_idx nw =~ natlt 16) =
{
  ff = (fun (wl : flash_combine_idx nw) -> wl._2 <: natlt 16)
    <: (flash_combine_idx nw -> GTot (natlt 16));
  gg = (fun (lane : natlt 16) -> (0, lane))
    <: (natlt 16 -> GTot (flash_combine_idx nw));
}

let flash_query_bij
  (hkv group sq hq rows : pos {
    hq == hkv * group /\ rows == group * sq })
  : (natlt hq & natlt sq =~ natlt hkv & natlt rows) =
{
  ff = (fun (qh, qpos) ->
    let kg = (bij_nat_prod #hkv #group).gg
      (qh <: natlt (hkv * group)) in
    (kg._1,
      ((bij_nat_prod #group #sq).ff
        (kg._2, qpos) <: natlt rows)));
  gg = (fun (kvh, r) ->
    let gp = (bij_nat_prod #group #sq).gg
      (r <: natlt (group * sq)) in
    (((bij_nat_prod #hkv #group).ff
        (kvh, gp._1) <: natlt hq),
      gp._2));
  ff_gg = (fun (kvh, r) ->
    bij_inv_bk (bij_nat_prod #group #sq)
      (r <: natlt (group * sq));
    bij_inv_fwd (bij_nat_prod #hkv #group)
      (kvh,
        ((bij_nat_prod #group #sq).gg
          (r <: natlt (group * sq)))._1));
  gg_ff = (fun (qh, qpos) ->
    bij_inv_bk (bij_nat_prod #hkv #group)
      (qh <: natlt (hkv * group));
    bij_inv_fwd (bij_nat_prod #group #sq)
      (((bij_nat_prod #hkv #group).gg
        (qh <: natlt (hkv * group)))._2, qpos));
}

let flash_pair4_bij (#a #b #c #d : Type0)
  : (a & b & c & d =~ (a & b) & (c & d)) =
{
  ff = (fun (x, y, z, w) -> ((x, y), (z, w)));
  gg = (fun ((x, y), (z, w)) -> (x, y, z, w));
}

let flash_output_logical_bij
  (b hkv group sq hq rows d : pos {
    hq == hkv * group /\ rows == group * sq })
  : (abs (b @| hq @| sq @| d @| INil)
      =~ (natlt b & natlt hkv & natlt rows & natlt d)) =
{
  ff = (fun (bi, (qh, (qpos, (dd, ())))) ->
    let kr = (flash_query_bij hkv group sq hq rows).ff
      (qh, qpos) in
    (bi, kr._1, kr._2, dd));
  gg = (fun (bi, kvh, r, dd) ->
    let qq = (flash_query_bij hkv group sq hq rows).gg
      (kvh, r) in
    (bi, (qq._1, (qq._2, (dd, ())))));
  ff_gg = (fun (bi, kvh, r, dd) ->
    bij_inv_bk (flash_query_bij hkv group sq hq rows)
      (kvh, r));
  gg_ff = (fun (bi, (qh, (qpos, (dd, ())))) ->
    bij_inv_fwd (flash_query_bij hkv group sq hq rows)
      (qh, qpos));
}

unfold
let flash_padded_idx
  (b hkv tiles d : pos) : Type0 =
  natlt b & natlt hkv & natlt tiles & natlt 16 & natlt d

unfold
let flash_padded_active
  (rows : pos) (#b #hkv #tiles #d : pos)
  (x : flash_padded_idx b hkv tiles d) : prop =
  let (_, _, rt, i, _) = x in
  rt * 16 + i < rows

unfold
let flash_active_padded_idx
  (b hkv rows tiles d : pos) : Type0 =
  x : flash_padded_idx b hkv tiles d {
    flash_padded_active rows x }

let flash_padded_logical
  (rows : pos) (#b #hkv #tiles #d : pos)
  (x : flash_padded_idx b hkv tiles d)
  : natlt b & natlt hkv & natlt rows & natlt d =
  let (bi, kvh, rt, i, dd) = x in
  (bi, kvh, clamp_nat_lt rows (rt * 16 + i), dd)

unfold
let flash_active_padded_bij
  (b hkv rows tiles d : pos { rows <= tiles * 16 })
  : (flash_active_padded_idx b hkv rows tiles d
      =~ (natlt b & natlt hkv & natlt rows & natlt d)) =
{
  ff = (fun (x : flash_active_padded_idx b hkv rows tiles d) ->
    flash_padded_logical rows x);
  gg = (fun (bi, kvh, r, dd) ->
    (bi, kvh,
      (r / 16 <: natlt tiles),
      (r % 16 <: natlt 16),
      dd));
  ff_gg = (fun (bi, kvh, r, dd) ->
    FStar.Math.Lemmas.lemma_div_mod (r <: nat) 16);
  gg_ff = (fun (x : flash_active_padded_idx b hkv rows tiles d) ->
    let (bi, kvh, rt, i, dd) = x in
    assert (flash_padded_active rows x);
    assert (rt * 16 + i < rows);
    assert (clamp_nat_lt rows (rt * 16 + i) == rt * 16 + i);
    FStar.Math.Lemmas.lemma_div_plus
      (i <: nat) (rt <: nat) 16;
    FStar.Math.Lemmas.small_div (i <: nat) 16;
    FStar.Math.Lemmas.small_mod (i <: nat) 16);
}

unfold
let flash_block_bij
  (b hkv tiles : pos)
  : (natlt b & natlt hkv & natlt tiles
      =~ natlt (b * hkv * tiles)) =
{
  ff = (fun ((bi, kvh, rt) :
      natlt b & natlt hkv & natlt tiles) ->
    ((bi * hkv + kvh) * tiles + rt
      <: natlt (b * hkv * tiles)));
  gg = (fun bid ->
    let bh = bid / tiles in
    ((bh / hkv <: natlt b),
      (bh % hkv <: natlt hkv),
      (bid % tiles <: natlt tiles)));
  ff_gg = (fun bid ->
    FStar.Math.Lemmas.lemma_div_mod (bid <: nat) tiles;
    FStar.Math.Lemmas.lemma_div_mod
      ((bid / tiles) <: nat) hkv);
  gg_ff = (fun (bi, kvh, rt) ->
    FStar.Math.Lemmas.lemma_div_plus
      (rt <: nat) ((bi * hkv + kvh) <: nat) tiles;
    FStar.Math.Lemmas.small_div (rt <: nat) tiles;
    FStar.Math.Lemmas.small_mod (rt <: nat) tiles;
    FStar.Math.Lemmas.lemma_div_plus
      (kvh <: nat) (bi <: nat) hkv;
    FStar.Math.Lemmas.small_div (kvh <: nat) hkv;
    FStar.Math.Lemmas.small_mod (kvh <: nat) hkv);
}

let flash_owner_idx
  (b hkv tiles d : pos) : Type0 =
  bid : natlt (b * hkv * tiles) &
  (lane : natlt BW.warp_size &
    out_stride_index2 16 d BW.warp_size lane)

unfold
let flash_padded_owner_bij
  (b hkv tiles d : pos)
  : (flash_padded_idx b hkv tiles d
      =~ flash_owner_idx b hkv tiles d) =
{
  ff = (fun (bi, kvh, rt, i, dd) ->
    let bid = (flash_block_bij b hkv tiles).ff
      (bi, kvh, rt) in
    let lane_ij =
      (flash_stride_partition_bij 16 d BW.warp_size).gg
        (i, dd) in
    (| bid, lane_ij |));
  gg = (fun owner ->
    let (bi, kvh, rt) =
      (flash_block_bij b hkv tiles).gg owner._1 in
    let ij = owner._2._2 in
    (bi, kvh, rt, ij._1, ij._2));
  ff_gg = (fun owner ->
    bij_inv_bk (flash_block_bij b hkv tiles) owner._1;
    bij_inv_fwd
      (flash_stride_partition_bij 16 d BW.warp_size)
      owner._2);
  gg_ff = (fun (bi, kvh, rt, i, dd) ->
    bij_inv_fwd (flash_block_bij b hkv tiles)
      (bi, kvh, rt);
    bij_inv_bk
      (flash_stride_partition_bij 16 d BW.warp_size)
      (i, dd));
}

unfold
let flash_bid_rt
  (b hkv tiles : pos)
  (bid : natlt (b * hkv * tiles)) : natlt tiles =
  bid % tiles

unfold
let flash_bid_kvh
  (b hkv tiles : pos)
  (bid : natlt (b * hkv * tiles)) : natlt hkv =
  (bid / tiles) % hkv

unfold
let flash_bid_bi
  (b hkv tiles : pos)
  (bid : natlt (b * hkv * tiles)) : natlt b =
  (bid / tiles) / hkv

unfold
let flash_block_output
  (#et : Type0)
  (b hq hkv group sq rows tiles d : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (#lout : layout4 b hq sq d)
  (gout : array4 et lout)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  : slprop =
  forall+ (lane : natlt BW.warp_size).
    out_store_cells b hq sq 16sz d rows gout
      (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (SZ.v group)
      (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
      lane

unfold
let flash_views_live
  (#et_ab #et_acc : Type0) (#nw #d : pos)
  (v : flash_views et_ab et_acc nw d) : slprop =
  live v.shQv ** live v.shKv ** live v.shVv **
  live v.shSv ** live v.shPv ** live v.shPVv **
  live v.shcwv ** live v.shMv ** live v.shLv **
  live v.shscalev ** live v.shOv **
  live v.shgmv ** live v.shglv

unfold
let flash_jt_local
  (#et_ab #et_acc : Type0)
  (d : szp { 16 /?+ SZ.v d })
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPV : layout2 16 16)
  (#lcw : layout1 16)
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shPV : array2 et_acc lPV)
  (shcw : array1 et_acc lcw)
  (lane : natlt BW.warp_size) : slprop =
  (exists* (e : chest2 et_ab
      (16 / warp_row_span) (SZ.v d / 16)).
    array2_stride_subtile shK warp_row_span 16
      (lane / 16) (lane % 16) |-> Frac 1.0R e)
  ** (exists* (e : chest2 et_ab
      (16 / warp_row_span) (SZ.v d / 16)).
    array2_stride_subtile shV warp_row_span 16
      (lane / 16) (lane % 16) |-> Frac 1.0R e)
  ** (exists* (e : chest2 et_acc 16 16).
    shS |-> Frac (1.0R /. BW.warp_size) e)
  ** when__ (lane < 16) (fun _ -> row_subtile shP lane)
  ** when__ (lane < 16) (fun _ -> cell_full shcw lane)
  ** (exists* (e : chest2 et_acc 16 16).
    shPV |-> Frac (1.0R /. BW.warp_size) e)

unfold
let flash_b0_local
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (w : natlt (SZ.v nw)) (lane : natlt BW.warp_size) : slprop =
  strided_cells2 v.shQv (block_threads nw)
    (w * BW.warp_size + lane) **
  strided_cells2
    (array2_subtile v.shOv 16 (SZ.v d <: pos) w 0)
    BW.warp_size lane

let flash_w0_idx (nw : szp) : Type0 =
  wl : (natlt (SZ.v nw) & natlt BW.warp_size) { wl._1 = 0 }

let flash_w0_bij (nw : szp)
  : (flash_w0_idx nw =~ natlt BW.warp_size) =
{
  ff = (fun (wl : flash_w0_idx nw) -> wl._2)
    <: (flash_w0_idx nw -> GTot (natlt BW.warp_size));
  gg = (fun (lane : natlt BW.warp_size) -> (0, lane))
    <: (natlt BW.warp_size -> GTot (flash_w0_idx nw));
}

unfold
let b0_pre
  (#et_q #et_o : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_q (l2_row_major 16 (SZ.v d)))
  (shO : array2 et_o (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  strided_cells2 shQ (block_threads nw) tid **
  strided_cells2 (array2_subtile shO 16 (SZ.v d <: pos) w 0) BW.warp_size lane

unfold
let b0_post
  (#et_q #et_o : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_q (l2_row_major 16 (SZ.v d)))
  (shO : array2 et_o (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  (exists* (e : chest2 et_q 16 (SZ.v d)).
     shQ |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (o : chest2 et_o (16 / warp_row_span) (SZ.v d / 16)).
        array2_stride_subtile
          (array2_subtile shO 16 (SZ.v d <: pos) w 0)
          warp_row_span 16 (lane / 16) (lane % 16)
          |-> Frac 1.0R o)

unfold
let b1_pre_one
  (#et : Type0) (nw : szp)
  (#lM : layout2 (SZ.v nw) 16)
  (shM : array2 et lM)
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  when__ (lane < 16) (fun _ ->
    cell_full (row shM w) (lane <: natlt 16))

unfold
let b1_pre
  (#et : Type0) (nw : szp)
  (#lM #lL : layout2 (SZ.v nw) 16)
  (shM : array2 et lM) (shL : array2 et lL)
  (tid : natlt (block_threads nw)) : slprop
= b1_pre_one nw shM tid ** b1_pre_one nw shL tid

unfold
let b1_post
  (#et : Type0) (nw : szp)
  (#lM #lL : layout2 (SZ.v nw) 16)
  (shM : array2 et lM) (shL : array2 et lL)
  (_tid : natlt (block_threads nw)) : slprop
= (exists* (e : chest2 et (SZ.v nw) 16).
     shM |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (e : chest2 et (SZ.v nw) 16).
     shL |-> Frac (1.0R /. (block_threads nw)) e)

unfold
let b2_o_pre
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
    array2_stride_subtile
      (array2_subtile shO 16 (SZ.v d <: pos) w 0)
      warp_row_span 16 (lane / 16) (lane % 16)
      |-> Frac 1.0R e

unfold
let b2_active
  (nw : szp) (tid : natlt (block_threads nw)) : GTot bool
= t2b (thread_w nw tid = 0 /\ thread_lane nw tid < 16)


unfold
let b2_scale_pre
  (#et : Type0) (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (tid : natlt (block_threads nw)) : slprop
= let w = thread_w nw tid in
  let lane = thread_lane nw tid in
  if_ (b2_active nw tid)
    ((exists* (e : chest2 et (SZ.v nw) 1).
       tensor_pts_to (array2_stride_subtile shscale 1 16 0 (clamp_nat_lt 16 lane)) e)
     ** cell_full shgl (clamp_nat_lt 16 lane))

unfold
let b2_pre
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et (l1_forward 16))
  (tid : natlt (block_threads nw)) : slprop
= b2_o_pre nw d shO tid ** b2_scale_pre nw shscale shgl tid

unfold
let b2_post
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shO : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et (l1_forward 16))
  (_tid : natlt (block_threads nw)) : slprop
= (exists* (e : chest2 et (SZ.v nw) 16).
     shscale |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (e : chest2 et (SZ.v nw * 16) (SZ.v d)).
     shO |-> Frac (1.0R /. (block_threads nw)) e)
  ** (exists* (e : chest1 et 16).
     shgl |-> Frac (1.0R /. (block_threads nw)) e)

let barrier_rin
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  : B.barrier_side (block_threads nw)
= fun it tid ->
    if it = 0 then b0_pre nw d shQ shO tid
    else if it = 1 then b1_pre nw shM shL tid
    else if it = 2 then b2_pre nw d shscale shO shgl tid
    else emp

let barrier_rout
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  : B.barrier_side (block_threads nw)
= fun it tid ->
    if it = 0 then b0_post nw d shQ shO tid
    else if it = 1 then b1_post nw shM shL tid
    else if it = 2 then b2_post nw d shscale shO shgl tid
    else emp

let barrier_contract
  (#et_ab #et_acc : Type0)
  (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  : B.contract (block_threads nw)
= {
  rin = barrier_rin nw d shQ shM shL shscale shO shgl;
  rout = barrier_rout nw d shQ shM shL shscale shO shgl;
}

let barrier_count (_nw : szp) : GTot nat = 3
unfold
let flash_thread_pre
  (#et_ab #et_acc : Type0)
  (nw nthr : szp { SZ.v nthr == block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  (w : natlt (SZ.v nw))
  (lane : natlt BW.warp_size) : slprop =
  (gQ |-> Frac fQ eQ) **
  (gK |-> Frac fK eK) **
  (gV |-> Frac fV eV) **
  (gmask |-> Frac fmask emask) **
  flash_b0_local nw d v w lane **
  when__ (lane < 16) (fun _ ->
    cell_full (row v.shMv w) lane) **
  when__ (lane < 16) (fun _ ->
    cell_full (row v.shLv w) lane) **
  flash_jt_local d
    (flash_warp_k v w) (flash_warp_v v w)
    (flash_warp_s v w) (flash_warp_p v w)
    (flash_warp_pv v w) (flash_warp_cw v w) lane **
  flash_combine_local nw v.shscalev v.shgmv v.shglv w lane **
  when_ (w = 0)
    (out_store_cells b hq sq 16sz d rows gout
      (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (SZ.v group)
      (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
      lane)

unfold
let flash_thread_post
  (#et_ab #et_acc : Type0)
  (nw nthr : szp { SZ.v nthr == block_threads nw })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) })
  (d : szp { 16 /?+ SZ.v d })
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (gQ : array4 et_ab lgQ)
  (gK : array4 et_ab lgK)
  (gV : array4 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (v : flash_views et_ab et_acc (SZ.v nw) (SZ.v d))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (bid : natlt (SZ.v b * SZ.v hkv * SZ.v tiles))
  (w : natlt (SZ.v nw))
  (lane : natlt BW.warp_size) : slprop =
  (gQ |-> Frac fQ eQ) **
  (gK |-> Frac fK eK) **
  (gV |-> Frac fV eV) **
  (gmask |-> Frac fmask emask) **
  (exists* (e : chest2 et_ab 16 (SZ.v d)).
    v.shQv |-> Frac (1.0R /. (block_threads nw)) e) **
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
      (flash_bid_bi (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (flash_bid_kvh (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid)
      (SZ.v group)
      (flash_bid_rt (SZ.v b) (SZ.v hkv) (SZ.v tiles) bid * 16)
      lane) **
  when_ (w = 0 /\ lane < 16)
    (cell_full v.shgmv (clamp_nat_lt 16 lane))

unfold
let flash_block_state
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
  (bid : natlt (SZ.v nblk)) : slprop =
  (gQ |-> Frac (fQ /. (SZ.v nblk)) eQ) **
  (gK |-> Frac (fK /. (SZ.v nblk)) eK) **
  (gV |-> Frac (fV /. (SZ.v nblk)) eV) **
  (gmask |-> Frac (fmask /. (SZ.v nblk)) emask) **
  flash_block_output b hq hkv group sq rows tiles d gout
    (bid <: natlt (SZ.v b * SZ.v hkv * SZ.v tiles))

inline_for_extraction noextract
let sdpa_flash_w
  (nw nthr : szp { SZ.v nthr == block_threads nw })
  (tid : szlt nthr) : szlt nw
= tid /^ 32sz

inline_for_extraction noextract
let sdpa_flash_lane
  (nw nthr : szp { SZ.v nthr == block_threads nw })
  (tid : szlt nthr) : szlt BW.warp_size
= tid %^ 32sz

unfold
let sdpa_flash_jt_frame
  (#et_ab #et_acc : Type0)
  (d sk : szp) (b hq sq : szp)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (gK : array2 et_ab lgK) (gV : array2 et_ab lgV) (gmask : TRO.roarray4 et_ab lgmask)
  (#fKg #fVg #fmask : perm)
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (#_ : squash (16 /?+ SZ.v d))
  (lane : natlt BW.warp_size) : slprop
= (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
      array2_stride_subtile shK warp_row_span 16 (lane / 16) (lane % 16)
        |-> Frac 1.0R r)
  ** (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
      array2_stride_subtile shV warp_row_span 16 (lane / 16) (lane % 16)
        |-> Frac 1.0R r)
  ** (exists* (e:chest2 et_acc 16 16). shS |-> Frac (1.0R /. BW.warp_size) e)
  ** when__ (lane < 16) (fun _ -> row_subtile shP lane)
  ** when__ (lane < 16) (fun _ -> cell_full shcw lane)
  ** (exists* (e:chest2 et_acc 16 16). shPVc |-> Frac (1.0R /. BW.warp_size) e)
  ** (gK |-> Frac fKg eKg) ** (gV |-> Frac fVg eVg)
  ** (gmask |-> Frac fmask emask)

unfold
let sdpa_flash_pre
  (#et_ab #et_acc : Type0)
  (nw nthr : szp)
  (d : szp { 16 /?+ SZ.v d })
  (sk : szp { SZ.v nthr == block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (gQ : array4 et_ab lgQ)
  (gK : array2 et_ab lgK)
  (gV : array2 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgm shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (bi : szlt b) (r0 : sz) (group : szp) (kvh : sz)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  : slprop
= gpu **
  thread_id (block_threads nw) tid **
  B.barrier_tok (barrier_contract nw d shQ shM shL shscale shO shgl) **
  B.barrier_state 0 **
  (gQ |-> Frac fQ eQ) **
  b0_pre nw d shQ shO tid **
  if_ (lane_active 16sz (sdpa_flash_lane nw nthr tid))
    (ml_cells 16sz
      (row shM (SZ.v (sdpa_flash_w nw nthr tid)))
      (row shL (SZ.v (sdpa_flash_w nw nthr tid)))
      (sdpa_flash_lane nw nthr tid))
  ** sdpa_flash_jt_frame d sk b hq sq
    shK shV shS shP shPVc shcw gK gV gmask
    #fKg #fVg #fmask #eKg #eVg #emask
    (SZ.v (sdpa_flash_lane nw nthr tid))
  ** if_ (combine_active 16sz (sdpa_flash_w nw nthr tid) (sdpa_flash_lane nw nthr tid))
    (combine_cells nw 16sz shscale shgm shgl (sdpa_flash_lane nw nthr tid))
  ** if_ (sdpa_flash_w nw nthr tid = 0sz)
    (out_store_cells b hq sq 16sz d rows gout
      bi (SZ.v kvh) (SZ.v group) (SZ.v r0)
      (SZ.v (sdpa_flash_lane nw nthr tid)))

unfold
let sdpa_flash_post
  (#et_ab #et_acc : Type0)
  (nw nthr : szp)
  (d : szp { 16 /?+ SZ.v d })
  (sk : szp { SZ.v nthr == block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (gQ : array4 et_ab lgQ)
  (gK : array2 et_ab lgK)
  (gV : array2 et_ab lgV)
  (gmask : TRO.roarray4 et_ab lgmask)
  (gout : array4 et_ab lout)
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (bi : szlt b) (r0 : sz) (group : szp) (kvh : sz)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  : slprop
= gpu **
  thread_id (block_threads nw) tid **
  B.barrier_tok (barrier_contract nw d shQ shM shL shscale shO shgl) **
  B.barrier_state 3 **
  (gQ |-> Frac fQ eQ) **
  (exists* (e : chest2 et_ab 16 (SZ.v d)).
    shQ |-> Frac (1.0R /. (block_threads nw)) e) **
  b1_post nw shM shL tid **
  b2_post nw d shscale shO shgl tid **
  sdpa_flash_jt_frame d sk b hq sq
    shK shV shS shP shPVc shcw gK gV gmask
    #fKg #fVg #fmask #eKg #eVg #emask
    (SZ.v (sdpa_flash_lane nw nthr tid))
  ** if_ (sdpa_flash_w nw nthr tid = 0sz)
    (out_store_cells b hq sq 16sz d rows gout
      bi (SZ.v kvh) (SZ.v group) (SZ.v r0)
      (SZ.v (sdpa_flash_lane nw nthr tid)))

(* Degenerate additive-mask layout for the no-mask case.

   Every index maps to cell 0, so the mask tensor is a broadcast view of a
   single element: the caller only has to own (and allocate) one cell rather
   than a dense [b x hq x sq x sk] buffer.  This is exactly the non-injective
   [vtlayout] that [rotensor] admits and an ordinary [tlayout] cannot express.
   The kernel is passed [has_mask = false] alongside it, so no read through
   this layout ever happens; it exists to give the ownership footprint of an
   absent mask a sound, constant-size denotation. *)
inline_for_extraction noextract
let vl4_broadcast0 (d0 d1 d2 d3 : nat)
  : TRO.vlayout4 d0 d1 d2 d3
= { ulen = 1; imap = (fun _ -> 0) }

inline_for_extraction noextract
instance cvl4_broadcast0 (d0 d1 d2 d3 : nat {
    SZ.fits d0 /\ SZ.fits d1 /\ SZ.fits d2 /\ SZ.fits d3 })
  : TRO.cvtlayout (vl4_broadcast0 d0 d1 d2 d3)
= {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun _ -> 0sz);
  }
