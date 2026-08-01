module Kuiops.Sdpa.Flash.KfSub

#lang-pulse

open Kuiper
open Kuiper.Floating
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Kernel.FlashAttention.KernelDesc

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module FD = Kuiops.Sdpa.Flash.KernelDesc

unfold let warp_row_span : nat = SZ.v warp_size / 16

unfold
let row_subtile
  (#et:Type0) (#l : layout2 16 16)
  (shA : array2 et l)
  (i : natlt 16) : slprop
= exists* (r : chest2 et 1 16).
    array2_subtile shA 1 16 i 0 |-> Frac 1.0R r

unfold
let cell_full
  (#et:Type0) (#lcw:layout1 16)
  (shcw : array1 et lcw) (i : natlt 16) : slprop
= exists* (v:et). Cell shcw (idx1 i) |-> Frac 1.0R v

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

let clamp_nat_lt (n : pos) (x : nat) : natlt n =
  if x < n then x else 0

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

unfold
let jt_rest
  (#et_ab #et_acc : Type0)
  (d sk : szp) (b hq sq : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : layout4 b hq sq sk)
  (#lcw #lm #ll : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (#lO : layout2 16 (SZ.v d))
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shO : array2 et_acc lO)
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (shm : array1 et_acc lm)
  (shl : array1 et_acc ll)
  (gK : array2 et_ab lgK) (gV : array2 et_ab lgV) (gmask : array4 et_ab lgmask)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest2 et_ab 16 (SZ.v d))
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (i : natlt BW.warp_size) : slprop
= (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shK warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ** (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shV warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ** (exists* (e:chest2 et_acc 16 16). shS |-> Frac (1.0R /. BW.warp_size) e)
  ** when__ (i < 16) (fun _ -> row_subtile shP i)
  ** when__ (i < 16) (fun _ -> cell_full shcw i)
  ** when__ (i < 16) (fun _ -> cell_full shm i)
  ** when__ (i < 16) (fun _ -> cell_full shl i)
  ** (exists* (e:chest2 et_acc (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shO warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R e)
  ** (shQ |-> Frac fQ eQ)
  ** (exists* (e:chest2 et_acc 16 16). shPVc |-> Frac (1.0R /. BW.warp_size) e)
  ** (gK |-> Frac fKg eKg) ** (gV |-> Frac fVg eVg) ** (gmask |-> Frac fmask emask)

inline_for_extraction noextract
let sdpa_flash_w
  (nw nthr : szp { SZ.v nthr == FD.block_threads nw })
  (tid : szlt nthr) : szlt nw
= tid /^ 32sz

inline_for_extraction noextract
let sdpa_flash_lane
  (nw nthr : szp { SZ.v nthr == FD.block_threads nw })
  (tid : szlt nthr) : szlt warp_size
= tid %^ 32sz

inline_for_extraction noextract
fn sdpa_flash_jt_body
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (d sk : szp) (b hq sq : szp)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : layout4 b hq sq sk)
  (#lcw #lm #ll : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (#lO : layout2 16 (SZ.v d))
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (tlayout_ulen lcw)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  {| ctlayout lgK |} {| ctlayout lgV |} {| ctlayout lgmask |}
  {| ctlayout lcw |} {| ctlayout lm |} {| ctlayout ll |}
  {| ctlayout lK |} {| ctlayout lV |} {| ctlayout lS |}
  {| ctlayout lP |} {| ctlayout lPVc |} {| ctlayout lO |}
  {| strided_row_major lK |} {| strided_row_major lV |}
  {| strided_row_major lS |} {| strided_row_major lP |}
  {| strided_row_major lPVc |}
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shO : array2 et_acc lO)
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (shm : array1 et_acc lm)
  (shl : array1 et_acc ll)
  (gK : array2 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array2 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : array4 et_ab lgmask)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq)
  (k0 : sz) (#_ : squash (SZ.fits (SZ.v k0 + 16)))
  (cbound : sz) (row_active : bool) (causal : bool) (scale : et_acc)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest2 et_ab 16 (SZ.v d))
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires
    jt_rest #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask (SZ.v lane)
  ensures
    jt_rest #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask (SZ.v lane)

inline_for_extraction noextract
fn sdpa_flash_causal_mask
  (bm bn sk sq rows : szp)
  (r0 : sz) (causal : bool)
  (#_ : squash (SZ.fits (SZ.v r0 + SZ.v bm)))
  (#_ : squash (SZ.fits (SZ.v sk + SZ.v bm + 1)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  returns nkt : sz
  ensures pure (SZ.v nkt <= SZ.v sk / SZ.v bn + 1)

inline_for_extraction noextract
fn sdpa_flash_combine_partials
  (#et : Type0) {| scalar et |} {| floating et |}
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
      (combine_cells nw bm shscale shgm shgl lane)

inline_for_extraction noextract
fn sdpa_flash_o_store
  (#et_acc #et_ab : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
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
  ensures if_ (w = 0sz)
    (out_store_cells b hq sq bm d rows gout
      (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v r0) (SZ.v lane))

inline_for_extraction noextract
fn sdpa_flash_block_prologue
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| scalar et_acc |} {| floating et_acc |}
  (nw nthr d : szp { SZ.v nthr == FD.block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  {| ctlayout lgQ |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (bi : szlt b) (r0 : sz) (group : szp) (kvh : sz)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (16 /?+ (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (16 * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (16 * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.fits (SZ.v r0 + 16)))
  (#_ : squash (SZ.fits (SZ.v kvh * SZ.v group + SZ.v rows)))
  (#fQ : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  requires
    gpu **
    thread_id (FD.block_threads nw) tid **
    B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
    B.barrier_state 0 **
    (gQ |-> Frac fQ eQ) **
    FD.b0_pre nw d shQ shO (SZ.v tid) **
    if_ (lane_active 16sz (tid %^ 32sz))
      (ml_cells 16sz
        (FD.row shM (SZ.v (tid /^ 32sz)))
        (FD.row shL (SZ.v (tid /^ 32sz)))
        (tid %^ 32sz))
  ensures
    gpu **
    thread_id (FD.block_threads nw) tid **
    B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
    B.barrier_state 1 **
    (gQ |-> Frac fQ eQ) **
    FD.b0_post nw d shQ shO (SZ.v tid) **
    when__ (SZ.v (tid %^ 32sz) < 16) (fun _ ->
      cell_full (FD.row shM (SZ.v (tid /^ 32sz))) (SZ.v (tid %^ 32sz)))
    ** when__ (SZ.v (tid %^ 32sz) < 16) (fun _ ->
      cell_full (FD.row shL (SZ.v (tid /^ 32sz))) (SZ.v (tid %^ 32sz)))

inline_for_extraction noextract
fn sdpa_flash_block_barrier1
  (#et_ab #et_acc : Type0)
  (nw nthr d : szp { SZ.v nthr == FD.block_threads nw })
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (#_ : squash (16 /?+ SZ.v d))
  requires
    gpu **
    thread_id (FD.block_threads nw) tid **
    B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
    B.barrier_state 1 **
    FD.b1_pre nw shM shL (SZ.v tid)
  ensures
    gpu **
    thread_id (FD.block_threads nw) tid **
    B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
    B.barrier_state 2 **
    FD.b1_post nw shM shL (SZ.v tid)

inline_for_extraction noextract
fn sdpa_flash_block_barrier2
  (#et_ab #et_acc : Type0)
  (nw nthr d : szp { SZ.v nthr == FD.block_threads nw })
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shM shL : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shscale : array2 et_acc (l2_row_major (SZ.v nw) 16))
  (shO : array2 et_acc (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (shgl : array1 et_acc (l1_forward 16))
  (tid : szlt nthr)
  (#_ : squash (16 /?+ SZ.v d))
  requires
    gpu **
    thread_id (FD.block_threads nw) tid **
    B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
    B.barrier_state 2 **
    FD.b2_pre nw d shscale shO shgl (SZ.v tid)
  ensures
    gpu **
    thread_id (FD.block_threads nw) tid **
    B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
    B.barrier_state 3 **
    FD.b2_post nw d shscale shO shgl (SZ.v tid)

ghost
fn stride_reindex
  (#et:Type0) (#cols:nat) (#l:layout2 16 cols)
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (16 /?+ cols))
  (shX : array2 et l)
  (i j : natlt BW.warp_size)
  requires
    pure (i == j) **
    (exists* (r:chest2 et (16 / warp_row_span) (cols / 16)).
       array2_stride_subtile shX warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ensures
    (exists* (r:chest2 et (16 / warp_row_span) (cols / 16)).
       array2_stride_subtile shX warp_row_span 16 (j / 16) (j % 16) |-> Frac 1.0R r)

ghost
fn row_reindex
  (#et:Type0) (#l : layout2 16 16)
  (shA : array2 et l)
  (i j : natlt BW.warp_size)
  requires pure (i == j) ** when__ (i < 16) (fun _ -> row_subtile shA i)
  ensures when__ (j < 16) (fun _ -> row_subtile shA j)

ghost
fn cell_reindex
  (#et:Type0) (#l:layout1 16)
  (shA : array1 et l) (i j : natlt BW.warp_size)
  requires pure (i == j) ** when__ (i < 16) (fun _ -> cell_full shA i)
  ensures when__ (j < 16) (fun _ -> cell_full shA j)

ghost
fn block_row_cell_reindex
  (#et : Type0) (#rows : nat) (#l : layout2 rows 16)
  (a : array2 et l)
  (w1 w2 : natlt rows)
  (lane1 lane2 : natlt BW.warp_size)
  requires
    pure (w1 == w2 /\ lane1 == lane2) **
    when__ (lane1 < 16) (fun _ -> cell_full (FD.row a w1) lane1)
  ensures
    when__ (lane2 < 16) (fun _ -> cell_full (FD.row a w2) lane2)

ghost
fn combine_to_b2_local
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgm shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  requires
    if_ (combine_active 16sz w lane)
      (combine_cells nw 16sz shscale shgm shgl lane)
  ensures
    if_ (combine_active 16sz w lane)
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)))

ghost
fn b2_scale_to_descriptor
  (#et : Type0)
  (nw : szp)
  (shscale : array2 et (l2_row_major (SZ.v nw) 16))
  (shgl : array1 et (l1_forward 16))
  (w : szlt nw) (lane : szlt warp_size)
  (tid : szlt (FD.block_threads nw))
  requires
    pure (SZ.v w == FD.thread_w nw (SZ.v tid) /\
          SZ.v lane == FD.thread_lane nw (SZ.v tid)) **
    if_ (combine_active 16sz w lane)
      (exists* (e : chest2 et (SZ.v nw) 1).
        tensor_pts_to
          (array2_stride_subtile shscale 1 16 0 (SZ.v (clamp_lt 16sz lane))) e
        ** cell_full shgl (SZ.v (clamp_lt 16sz lane)))
  ensures
    FD.b2_scale_pre nw shscale shgl (SZ.v tid)

ghost
fn block_o_tile_reindex
  (#et : Type0) (nw d : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (16 /?+ (SZ.v nw * 16)))
  (a : array2 et (l2_row_major (SZ.v nw * 16) (SZ.v d)))
  (w1 w2 : natlt (SZ.v nw))
  (lane1 lane2 : natlt BW.warp_size)
  requires
    pure (w1 == w2 /\ lane1 == lane2) **
    (exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
      tensor_pts_to
        (array2_stride_subtile
          (array2_subtile a 16 (SZ.v d <: pos) w1 0)
          warp_row_span 16 (lane1 / 16) (lane1 % 16))
        e)
  ensures
    (exists* (e : chest2 et (16 / warp_row_span) (SZ.v d / 16)).
      tensor_pts_to
        (array2_stride_subtile
          (array2_subtile a 16 (SZ.v d <: pos) w2 0)
          warp_row_span 16 (lane2 / 16) (lane2 % 16))
        e)

unfold
let sdpa_flash_jt_frame
  (#et_ab #et_acc : Type0)
  (d sk : szp) (b hq sq : szp)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : layout4 b hq sq sk)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (gK : array2 et_ab lgK) (gV : array2 et_ab lgV) (gmask : array4 et_ab lgmask)
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
  (sk : szp { SZ.v nthr == FD.block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (gQ : array4 et_ab lgQ)
  (gK : array2 et_ab lgK)
  (gV : array2 et_ab lgV)
  (gmask : array4 et_ab lgmask)
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
  thread_id (FD.block_threads nw) tid **
  B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
  B.barrier_state 0 **
  (gQ |-> Frac fQ eQ) **
  FD.b0_pre nw d shQ shO tid **
  if_ (lane_active 16sz (sdpa_flash_lane nw nthr tid))
    (ml_cells 16sz
      (FD.row shM (SZ.v (sdpa_flash_w nw nthr tid)))
      (FD.row shL (SZ.v (sdpa_flash_w nw nthr tid)))
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
  (sk : szp { SZ.v nthr == FD.block_threads nw })
  (b hq sq rows : szp)
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  (#lcw : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (gQ : array4 et_ab lgQ)
  (gK : array2 et_ab lgK)
  (gV : array2 et_ab lgV)
  (gmask : array4 et_ab lgmask)
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
  thread_id (FD.block_threads nw) tid **
  B.barrier_tok (FD.barrier_contract nw d shQ shM shL shscale shO shgl) **
  B.barrier_state 3 **
  (gQ |-> Frac fQ eQ) **
  (exists* (e : chest2 et_ab 16 (SZ.v d)).
    shQ |-> Frac (1.0R /. (FD.block_threads nw)) e) **
  FD.b1_post nw shM shL tid **
  FD.b2_post nw d shscale shO shgl tid **
  sdpa_flash_jt_frame d sk b hq sq
    shK shV shS shP shPVc shcw gK gV gmask
    #fKg #fVg #fmask #eKg #eVg #emask
    (SZ.v (sdpa_flash_lane nw nthr tid))
  ** if_ (sdpa_flash_w nw nthr tid = 0sz)
    (out_store_cells b hq sq 16sz d rows gout
      bi (SZ.v kvh) (SZ.v group) (SZ.v r0)
      (SZ.v (sdpa_flash_lane nw nthr tid)))
