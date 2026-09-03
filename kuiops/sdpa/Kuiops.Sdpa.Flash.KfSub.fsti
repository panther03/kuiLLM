module Kuiops.Sdpa.Flash.KfSub

#lang-pulse

open Kuiper
open Kuiper.Floating
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiops.Array2.Strided
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

(* The lane-uniform tile descriptions agree with lane [i]'s own on row [i].
   A warp barrier's predicate family has to be the same expression for every
   lane, so everything that crosses one is phrased with the [_t] versions. *)
#push-options "--fuel 1 --ifuel 2"

let jt_t_row_eq
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (kvh group rows r0 k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm evl : chest1 et_acc 16) (i : natlt 16)
  : Lemma
      (requires SF.lane_params_ok hq sq sk kvh group rows r0 i row_active
                  qh qpos cbound)
      (ensures
        ematrix_subtile (SF.score_tile_t emask has_mask causal bi kvh group rows r0
                           k0 scale eS) 1 16 i 0
        == ematrix_subtile (SF.score_tile emask has_mask row_active causal bi qh qpos
                              k0 cbound scale eS) 1 16 i 0 /\
        ematrix_subtile (SF.prob_tile_t emask has_mask causal bi kvh group rows r0
                           k0 scale eS evm) 1 16 i 0
        == ematrix_subtile (SF.prob_tile emask has_mask row_active causal bi qh qpos
                              k0 cbound scale eS evm) 1 16 i 0 /\
        acc1 (SF.cw_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS evm) i
        == acc1 (SF.cw_vec emask has_mask row_active causal bi qh qpos k0 cbound scale
                   eS evm) i /\
        acc1 (SF.m_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS evm) i
        == acc1 (SF.m_vec emask has_mask row_active causal bi qh qpos k0 cbound scale
                   eS evm) i /\
        acc1 (SF.l_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS evm evl) i
        == acc1 (SF.l_vec emask has_mask row_active causal bi qh qpos k0 cbound scale
                   eS evm evl) i)
= SF.tile_row_eq
    (SF.score_tile_t emask has_mask causal bi kvh group rows r0 k0 scale eS)
    (SF.score_tile emask has_mask row_active causal bi qh qpos k0 cbound scale eS) i;
  SF.tile_row_eq
    (SF.prob_tile_t emask has_mask causal bi kvh group rows r0 k0 scale eS evm)
    (SF.prob_tile emask has_mask row_active causal bi qh qpos k0 cbound scale eS evm) i

let jt_t_row_eq_g
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (kvh group rows r0 k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm evl : chest1 et_acc 16) (i : nat)
  : Lemma
      (requires i < 16 ==>
        SF.lane_params_ok hq sq sk kvh group rows r0 i row_active qh qpos cbound)
      (ensures i < 16 ==>
        (acc1 (SF.cw_vec_t emask has_mask causal bi kvh group rows r0 k0 scale
                 eS evm) i
         == acc1 (SF.cw_vec emask has_mask row_active causal bi qh qpos k0 cbound
                    scale eS evm) i /\
         acc1 (SF.m_vec_t emask has_mask causal bi kvh group rows r0 k0 scale
                 eS evm) i
         == acc1 (SF.m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                    scale eS evm) i /\
         acc1 (SF.l_vec_t emask has_mask causal bi kvh group rows r0 k0 scale
                 eS evm evl) i
         == acc1 (SF.l_vec emask has_mask row_active causal bi qh qpos k0 cbound
                    scale eS evm evl) i))
= if i < 16
  then jt_t_row_eq #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2 #b #hq #sq #sk
         emask has_mask row_active causal bi qh qpos kvh group rows r0 k0 cbound
         scale eS evm evl i
  else ()

#pop-options

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
             (acc2 (SF.pv_chunk eP eV b) (SF.clamp_nat 16 (span * a + tr)) (SF.clamp_nat 16 tc))
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
                (acc2 (SF.pv_chunk eP eV b) (SF.clamp_nat 16 (span * a + tr)) (SF.clamp_nat 16 tc))
       else acc2 eO0 a b)

let div_mod_16 (q r : nat)
  : Lemma (requires r < 16) (ensures (q * 16 + r) / 16 == q /\ (q * 16 + r) % 16 == r)
= FStar.Math.Lemmas.lemma_div_plus r q 16;
  FStar.Math.Lemmas.lemma_mod_plus r q 16;
  FStar.Math.Lemmas.small_div r 16;
  FStar.Math.Lemmas.small_mod r 16

#push-options "--fuel 1 --ifuel 2 --z3rlimit 30"

(* One lane's view of that update: rescale its stride sub-tile, then add the
   [P@V] chunks -- exactly the composition [sdpa_flash_scale] followed by
   [sdpa_flash_pv_mm] performs. *)
let jt_out_subtile
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#d : nat) (#_ : squash (16 /?+ d)) (#_ : squash (warp_row_span /?+ 16))
  (eO : chest2 et_acc 16 d) (ecw : chest1 et_acc 16)
  (eP : chest2 et_ab 16 16) (eV : chest2 et_ab 16 d)
  (tr : natlt warp_row_span) (tc : natlt 16)
  : Lemma (jt_pv_acc (SF.scale_subtile (ematrix_stride_subtile eO warp_row_span 16 tr tc)
                        ecw warp_row_span tr)
             eP eV warp_row_span tr tc (d / 16)
           == ematrix_stride_subtile (SF.out_tile eO ecw eP eV) warp_row_span 16 tr tc)
= let lhs = jt_pv_acc (SF.scale_subtile (ematrix_stride_subtile eO warp_row_span 16 tr tc)
                         ecw warp_row_span tr)
              eP eV warp_row_span tr tc (d / 16) in
  let rhs = ematrix_stride_subtile (SF.out_tile eO ecw eP eV) warp_row_span 16 tr tc in
  introduce forall (a : natlt (16 / warp_row_span)) (bb : natlt (d / 16)).
    acc2 lhs a bb == acc2 rhs a bb
  with div_mod_16 bb tc;
  assert (equal lhs rhs)

#pop-options

(* The output tile after the key-tile step at [k0]. *)
unfold
let jt_out_step
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat)
  (k0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg eVg : chest2 et_ab sk d)
  (evm : chest1 et_acc 16) (eOw : chest2 et_acc 16 d)
  : GTot (chest2 et_acc 16 d)
= SF.out_tile eOw
    (SF.cw_vec_t emask has_mask causal bi kvh group rows r0 k0 scale
       (jt_stile #et_ab #et_acc d eQ eKg k0) evm)
    (SF.prob_tile_t emask has_mask causal bi kvh group rows r0 k0 scale
       (jt_stile #et_ab #et_acc d eQ eKg k0) evm)
    (SF.kv_tile 16 eVg k0)

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
  (eOw : chest2 et_acc 16 (SZ.v d))
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
  ** (array2_stride_subtile shO warp_row_span 16 (i / 16) (i % 16)
        |-> Frac 1.0R (ematrix_stride_subtile eOw warp_row_span 16 (i / 16) (i % 16)))
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
= exists* (vm vl : et_acc) (eOw : chest2 et_acc 16 (SZ.v d)).
    jt_rest_v #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask vm vl eOw i


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

(* Re-express a pinned row cell as a cell of a whole-matrix value, at the same
   time renaming the warp/lane indices. *)
ghost
fn when__cell16_setval (#et:Type0) (#rows:nat) (#l:layout2 rows 16)
  (a : array2 et l) (w1 w2 : natlt rows) (lane1 lane2 : natlt BW.warp_size)
  (e : chest2 et rows 16) (v : et)
  requires
    when__ (lane1 < 16) (fun _ -> cell_full_v (row a w1) lane1 v)
    ** pure (w1 == w2 /\ lane1 == lane2 /\
             (lane1 < 16 ==> v == acc2 e w1 (clamp_nat_lt 16 lane1)))
  ensures
    when__ (lane2 < 16) (fun _ -> cell_full_v (row a w2) lane2 (acc2 e w2 lane2))

ghost
fn cell_reindex
  (#et:Type0) (#l:layout1 16)
  (shA : array1 et l) (i j : natlt BW.warp_size)
  requires pure (i == j) ** when__ (i < 16) (fun _ -> cell_full shA i)
  ensures when__ (j < 16) (fun _ -> cell_full shA j)

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
  (kvh group rows r0 : sz)
  (k0 : sz) (#_ : squash (SZ.fits (SZ.v k0 + 16)))
  (cbound : sz) (row_active : bool) (causal : bool) (has_mask : bool) (scale : et_acc)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest2 et_ab 16 (SZ.v d))
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (vm vl : Ghost.erased et_acc)
  (evm evl : Ghost.erased (chest1 et_acc 16))
  (eOw : Ghost.erased (chest2 et_acc 16 (SZ.v d)))
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires pure (SZ.v lane < 16 ==>
                   reveal vm == acc1 evm (SZ.v lane) /\ reveal vl == acc1 evl (SZ.v lane))
  requires pure (SZ.v lane < 16 ==>
                   SF.lane_params_ok (SZ.v hq) (SZ.v sq) (SZ.v sk) (SZ.v kvh)
                     (SZ.v group) (SZ.v rows) (SZ.v r0) (SZ.v lane)
                     row_active (SZ.v qh) (SZ.v qpos) (SZ.v cbound))
  requires
    jt_rest_v #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask vm vl eOw (SZ.v lane)
  ensures
    exists* (m' l' : et_acc).
    jt_rest_v #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask m' l'
      (jt_out_step #et_ab #et_acc (SZ.v d) emask has_mask causal
         (SZ.v bi) (SZ.v kvh) (SZ.v group) (SZ.v rows) (SZ.v r0) (SZ.v k0) scale
         eQ eKg eVg evm eOw)
      (SZ.v lane)
    ** pure (SZ.v lane < 16 ==>
              (exists (sr' : chest1 et_acc 16) (pr' : chest1 et_ab 16) (cw' : et_acc).
                 SF.softmax_upd_post emask has_mask row_active causal
                   (SZ.v bi) (SZ.v qh) (SZ.v qpos) (SZ.v k0) (SZ.v cbound) scale
                   (jt_score_row (SZ.v d) eQ eKg (SZ.v k0) (SZ.v lane))
                   vm vl sr' pr' m' l' cw' /\ Finite? (kind cw')))
