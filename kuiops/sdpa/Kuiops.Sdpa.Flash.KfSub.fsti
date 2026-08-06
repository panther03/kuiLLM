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
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float
module BM = Kuiops.Common.BlockMatmul
module SS = Kuiops.Sdpa.Flash.Spec.Step
open Kuiper.TensorRO { vtlayout_of_tlayout }

(* The score row lane [i] sees for the key tile at [k0]: row [i] of the
   tensor-core accumulation of [Q @ (K-tile)^T] over the [d/16] chunks of the
   head dimension. *)
unfold
let jt_score_row
  (#et_ab #et_acc : Type0) {| floating et_acc |} {| scalar et_ab |}
  (#sk : pos) (d : nat) (#_ : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (k0 : nat) (i : natlt 16)
  : chest1 et_acc 16
= subtile_row (ematrix_subtile
     (BM.emma_chain #et_ab #et_acc 16 eQ (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16))
     1 16 i 0)

(* [jt_score_row] is exactly the row of raw tile scores the bridge speaks
   about, so the two descriptions of one lane's tile can be interchanged. *)
let jt_score_row_eq
  (#et_ab #et_acc : Type0) {| floating et_acc |} {| scalar et_ab |}
  (#sk : pos) (d : pos) (#_ : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (k0 : nat) (i : natlt 16)
  : Lemma (jt_score_row #et_ab #et_acc d eQ eKg k0 i
           == SS.tile_score_row #et_ab #et_acc eQ eKg k0 i)
  = assert (equal (jt_score_row #et_ab #et_acc d eQ eKg k0 i)
                  (SS.tile_score_row #et_ab #et_acc eQ eKg k0 i))

(* The same, for an unrefined lane index. *)
let jt_score_row_eq_g
  (#et_ab #et_acc : Type0) {| floating et_acc |} {| scalar et_ab |}
  (#sk : pos) (d : pos) (#_ : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (k0 : nat) (i : nat)
  : Lemma (i < 16 ==>
             jt_score_row #et_ab #et_acc d eQ eKg k0 i
             == SS.tile_score_row #et_ab #et_acc eQ eKg k0 i)
  = if i < 16 then jt_score_row_eq #et_ab #et_acc d eQ eKg k0 i else ()

(* The raw score tile the warp computes for the key tile at [k0]. *)
unfold
let jt_stile
  (#et_ab #et_acc : Type0) {| floating et_acc |} {| scalar et_ab |}
  (#sk : pos) (d : nat) (#_ : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (k0 : nat)
  : chest2 et_acc 16 16
= BM.emma_chain #et_ab #et_acc 16 eQ (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16)

(* Lane [i]'s online-softmax update, read off the whole-warp descriptions. *)
let jt_upd_post_g
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (evm evl : chest1 et_acc 16) (i : nat)
  : Lemma (i < 16 ==>
      SF.softmax_upd_post emask has_mask row_active causal bi qh qpos k0 cbound scale
        (jt_score_row #et_ab #et_acc d eQ eKg k0 i) (acc16 evm i) (acc16 evl i)
        (SF.erow (SF.score_tile emask has_mask row_active causal bi qh qpos k0 cbound
                    scale (jt_stile #et_ab #et_acc d eQ eKg k0)) i)
        (SF.erow (SF.prob_tile emask has_mask row_active causal bi qh qpos k0 cbound
                    scale (jt_stile #et_ab #et_acc d eQ eKg k0) evm) i)
        (acc16 (SF.m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                  scale (jt_stile #et_ab #et_acc d eQ eKg k0) evm) i)
        (acc16 (SF.l_vec emask has_mask row_active causal bi qh qpos k0 cbound
                  scale (jt_stile #et_ab #et_acc d eQ eKg k0) evm evl) i)
        (acc16 (SF.cw_vec emask has_mask row_active causal bi qh qpos k0 cbound
                  scale (jt_stile #et_ab #et_acc d eQ eKg k0) evm) i))
  = if i < 16
    then (SF.tile_upd_post emask has_mask row_active causal bi qh qpos k0 cbound
            scale (jt_stile #et_ab #et_acc d eQ eKg k0) evm evl i;
          subtile_row_sub (jt_stile #et_ab #et_acc d eQ eKg k0) i)
    else ()

(* Chunk [b] of the [P@V] product: the [16 x 16] tile [P @ V[:, 16b:16b+16]],
   accumulated by the tensor-core [emma] chain over its single 16-wide chunk. *)
unfold
let jt_pv_chunk
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#hd : nat) (#_ : squash (16 /?+ hd))
  (eP : chest2 et_ab 16 16) (eV : chest2 et_ab 16 hd) (b : natlt (hd / 16))
  : GTot (chest2 et_acc 16 16)
= BM.emma_chain #et_ab #et_acc 16 eP (ematrix_subtile eV 16 16 0 b) 1

(* The [P@V] accumulation into one lane's [(span, 16)] stride sub-tile of the
   output tile, after the first [n] head-dimension chunks: lane [(tr, tc)] adds
   cell [(span*a + tr, tc)] of chunk [b] into its own output cell [(a, b)]. *)
let jt_pv_acc
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#hd : nat) (#_ : squash (16 /?+ hd)) (#nr : nat)
  (eO0 : chest2 et_acc nr (hd / 16))
  (eP : chest2 et_ab 16 16) (eV : chest2 et_ab 16 hd)
  (span tr tc n : nat)
  : GTot (chest2 et_acc nr (hd / 16))
= mk2 (fun a b ->
    if b < n
    then add (acc2 eO0 a b)
             (acc2 (jt_pv_chunk eP eV b) (SF.clamp_nat 16 (span * a + tr)) (SF.clamp_nat 16 tc))
    else acc2 eO0 a b)

(* Partial progress of that accumulation: chunks before [j] are done, and chunk
   [j] is done for the first [n] rows of the lane's sub-tile. *)
let jt_pv_part
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#hd : nat) (#_ : squash (16 /?+ hd)) (#nr : nat)
  (eO0 eO : chest2 et_acc nr (hd / 16))
  (eP : chest2 et_ab 16 16) (eV : chest2 et_ab 16 hd)
  (span tr tc j n : nat)
  : prop
= forall (a : natlt nr) (b : natlt (hd / 16)).
    acc2 eO a b ==
      (if b < j || (b = j && a < n)
       then add (acc2 eO0 a b)
                (acc2 (jt_pv_chunk eP eV b) (SF.clamp_nat 16 (span * a + tr)) (SF.clamp_nat 16 tc))
       else acc2 eO0 a b)

unfold
let jt_rest_v
  (#et_ab #et_acc : Type0)
  (d sk : szp) (b hq sq : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
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
  (gK : array2 et_ab lgK) (gV : array2 et_ab lgV) (gmask : TRO.roarray4 et_ab lgmask)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest2 et_ab 16 (SZ.v d))
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (vm vl : et_acc)
  (i : natlt BW.warp_size) : slprop
= (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shK warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ** (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shV warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ** (exists* (e:chest2 et_acc 16 16). shS |-> Frac (1.0R /. BW.warp_size) e)
  ** when__ (i < 16) (fun _ -> row_subtile shP i)
  ** when__ (i < 16) (fun _ -> cell_full shcw i)
  ** when__ (i < 16) (fun _ -> cell_full_v shm i vm)
  ** when__ (i < 16) (fun _ -> cell_full_v shl i vl)
  ** (exists* (e:chest2 et_acc (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shO warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R e)
  ** (shQ |-> Frac fQ eQ)
  ** (exists* (e:chest2 et_acc 16 16). shPVc |-> Frac (1.0R /. BW.warp_size) e)
  ** (gK |-> Frac fKg eKg) ** (gV |-> Frac fVg eVg) ** (gmask |-> Frac fmask emask)

unfold
let jt_rest
  (#et_ab #et_acc : Type0)
  (d sk : szp) (b hq sq : szp)
  (#_ : squash (16 /?+ SZ.v d))
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
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
  (gK : array2 et_ab lgK) (gV : array2 et_ab lgV) (gmask : TRO.roarray4 et_ab lgmask)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest2 et_ab 16 (SZ.v d))
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (i : natlt BW.warp_size) : slprop
= exists* (vm vl : et_acc).
    jt_rest_v #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask vm vl i

inline_for_extraction noextract
fn sdpa_flash_jt_body
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (d sk : szp) (b hq sq : szp)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
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
  {| ctlayout lgK |} {| ctlayout lgV |} {| TRO.cvtlayout lgmask |}
  {| ctlayout lcw |} {| ctlayout lm |} {| ctlayout ll |}
  {| ctlayout lK |} {| ctlayout lV |} {| ctlayout lS |}
  {| ctlayout lP |} {| ctlayout lPVc |} {| ctlayout lO |}
  {| strided_row_major (vtlayout_of_tlayout lK) |} {| strided_row_major (vtlayout_of_tlayout lV) |}
  {| strided_row_major (vtlayout_of_tlayout lS) |} {| strided_row_major (vtlayout_of_tlayout lP) |}
  {| strided_row_major (vtlayout_of_tlayout lPVc) |}
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
  (gmask : TRO.roarray4 et_ab lgmask)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq)
  (k0 : sz) (#_ : squash (SZ.fits (SZ.v k0 + 16)))
  (cbound : sz) (row_active : bool) (causal : bool) (has_mask : bool) (scale : et_acc)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest2 et_ab 16 (SZ.v d))
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (vm vl : Ghost.erased et_acc)
  (evm evl : Ghost.erased (chest1 et_acc 16))
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires pure (SZ.v lane < 16 ==>
                   reveal vm == acc1 evm (SZ.v lane) /\ reveal vl == acc1 evl (SZ.v lane))
  requires
    jt_rest_v #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask vm vl (SZ.v lane)
  ensures
    exists* (m' l' : et_acc).
    jt_rest_v #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask m' l' (SZ.v lane)
    ** pure (SZ.v lane < 16 ==>
              (exists (sr' : chest1 et_acc 16) (pr' : chest1 et_ab 16) (cw' : et_acc).
                 SF.softmax_upd_post emask has_mask row_active causal
                   (SZ.v bi) (SZ.v qh) (SZ.v qpos) (SZ.v k0) (SZ.v cbound) scale
                   (jt_score_row (SZ.v d) eQ eKg (SZ.v k0) (SZ.v lane))
                   vm vl sr' pr' m' l' cw' /\ Finite? (kind cw')))

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


val tile_idx_lem (s i r n : nat)
  : Lemma (requires s > 0 /\ (s /? n) /\ i < n / s /\ r < s) (ensures s * i + r < n)

ghost
fn when__elim_true (b:bool{b == true}) (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures q ()

ghost
fn when__elim_false (b:bool{b == false}) (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures emp

ghost
fn when__intro_true (b:bool{b == true}) (q : squash (b2t b) -> slprop)
  requires q ()
  ensures when__ b q

ghost
fn when__intro_false (b:bool{b == false}) (q : squash (b2t b) -> slprop)
  requires emp
  ensures when__ b q

ghost
fn when__pin_cell16 (#et:Type0) (#l:layout1 16)
  (shA : array1 et l) (lane : natlt BW.warp_size) (dflt : et)
  requires when__ (lane < 16) (fun _ -> cell_full shA lane)
  returns v : Ghost.erased et
  ensures  when__ (lane < 16) (fun _ -> cell_full_v shA lane (reveal v))

ghost
fn when__forget_cell16 (#et:Type0) (#l:layout1 16)
  (shA : array1 et l) (lane : natlt BW.warp_size) (v : et)
  requires when__ (lane < 16) (fun _ -> cell_full_v shA lane v)
  ensures  when__ (lane < 16) (fun _ -> cell_full shA lane)

ghost
fn when_elim_true (b:bool{b == true}) (q : slprop)
  requires when_ b q
  ensures q

ghost
fn when_elim_false (b:bool{b == false}) (q : slprop)
  requires when_ b q
  ensures emp

ghost
fn when_intro_true (b:bool{b == true}) (q : slprop)
  requires q
  ensures when_ b q

ghost
fn when_intro_false (b:bool{b == false}) (q : slprop)
  ensures when_ b q
