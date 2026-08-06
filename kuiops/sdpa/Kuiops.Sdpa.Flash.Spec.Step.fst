module Kuiops.Sdpa.Flash.Spec.Step

(* One lane, one key tile: from the float-level description of the kernel's
   softmax update ([Kuiops.Sdpa.Flash.Spec.Float]) to the real-valued flash
   invariant ([Kuiops.Sdpa.Flash.Spec.Bridge.ml_state]).

   [Kuiops.Sdpa.Flash.Spec.Bridge.ml_step] already does the online-softmax
   algebra; what is left, and what this module supplies, is the identification
   of the kernel's tile-local data with the global real quantities:

   - the raw tensor-core scores of the tile are the corresponding block of the
     exact real [Q K^T] (via [Kuiops.Common.BlockMatmul.emma_chain_approx]);
   - scaling and biasing them yields the real pre-softmax score;
   - the tile's key-admission test is the restriction of the global one.

   Overflow of a score is out of scope: whether a float multiply-add stays
   finite is not derivable from the [floating] laws, so [scores_finite] is a
   hypothesis carried into the theorem rather than something proved here. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Shape
open Kuiper.Floating
open Kuiper.Approximates
open Kuiper.EMatrix

module FC = Kuiper.Float.Casts
module MS = Kuiper.Spec.GEMM
module BM = Kuiops.Common.BlockMatmul
module SF = Kuiops.Sdpa.Flash.Spec.Float
module SO = Kuiops.Sdpa.Flash.Spec.Online
module SB = Kuiops.Sdpa.Flash.Spec.Bridge

(* ------------------------------------------------------------------ *)
(* A matmul cell depends on the right operand only through its column. *)
(* ------------------------------------------------------------------ *)

let rec matmul_single_col_ext
  (#et : Type) {| scalar et |}
  (#rows #shared #cols1 #cols2 : nat)
  (m : chest2 et rows shared)
  (a : chest2 et shared cols1) (b : chest2 et shared cols2)
  (row : natlt rows) (c1 : natlt cols1) (c2 : natlt cols2) (to : natle shared)
  : Lemma (requires forall (k : natlt shared). acc2 a k c1 == acc2 b k c2)
          (ensures MS.__matmul_single m a row c1 to
                   == MS.__matmul_single m b row c2 to)
          (decreases to)
  = if to = 0 then ()
    else matmul_single_col_ext m a b row c1 c2 (to - 1)

let matmul_col_ext
  (#et : Type) {| scalar et |}
  (#rows #shared #cols1 #cols2 : nat)
  (m : chest2 et rows shared)
  (a : chest2 et shared cols1) (b : chest2 et shared cols2)
  (row : natlt rows) (c1 : natlt cols1) (c2 : natlt cols2)
  : Lemma (requires forall (k : natlt shared). acc2 a k c1 == acc2 b k c2)
          (ensures acc2 (MS.matmul m a) row c1 == acc2 (MS.matmul m b) row c2)
  = MS.lemma_matmul_index m a row c1;
    MS.lemma_matmul_index m b row c2;
    matmul_single_col_ext m a b row c1 c2 shared

(* ------------------------------------------------------------------ *)
(* Approximation is pointwise, so it survives the tile constructions.  *)
(* ------------------------------------------------------------------ *)

let kv_tile_approx
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#sk : pos) (#d : nat) (bn : nat)
  (e : chest2 et sk d) (r : chest2 real sk d) (k0 : nat)
  : Lemma (requires e %~ r)
          (ensures SF.kv_tile bn e k0 %~ SF.kv_tile bn r k0)
  = introduce forall (ij : Kuiper.Shape.abs (bn @| d @| INil)).
                acc (SF.kv_tile bn e k0) ij %~ acc (SF.kv_tile bn r k0) ij
    with (let (i, (j, ())) = ij in
          assert (acc2 (SF.kv_tile bn e k0) i j
                  == acc2 e (SF.clamp_nat sk (k0 + i)) j))

let mtranspose_approx
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#rows #cols : nat)
  (e : chest2 et rows cols) (r : chest2 real rows cols)
  : Lemma (requires e %~ r) (ensures mtranspose e %~ mtranspose r)
  = introduce forall (ij : Kuiper.Shape.abs (cols @| rows @| INil)).
                acc (mtranspose e) ij %~ acc (mtranspose r) ij
    with (let (i, (j, ())) = ij in
          assert (acc2 (mtranspose e) i j == acc2 e j i))

(* ------------------------------------------------------------------ *)
(* The raw tile scores are a block of the exact real [Q K^T].          *)
(* ------------------------------------------------------------------ *)

(* Row [i], column [j] of the tensor-core accumulation the kernel performs for
   the key tile starting at [k0]. *)
let raw_score
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (k0 : nat)
  (i j : natlt 16)
  : GTot et_acc
  = acc2 (BM.emma_chain #et_ab #et_acc 16 eQ
            (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16)) i j

let raw_score_approx
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| real_like et_ab |}
  {| scalar et_acc |} {| real_like et_acc |}
  (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (rQ : chest2 real 16 d) (rK : chest2 real sk d)
  (k0 : nat) (i j : natlt 16)
  : Lemma (requires eQ %~ rQ /\ eKg %~ rK /\ k0 + j < sk)
          (ensures raw_score #et_ab #et_acc eQ eKg k0 i j
                   %~ acc2 (MS.matmul rQ (mtranspose rK)) i (k0 + j))
  = kv_tile_approx 16 eKg rK k0;
    mtranspose_approx (SF.kv_tile 16 eKg k0) (SF.kv_tile 16 rK k0);
    BM.emma_chain_approx #et_ab #et_acc 16 eQ (mtranspose (SF.kv_tile 16 eKg k0))
      rQ (mtranspose (SF.kv_tile 16 rK k0));
    assert (forall (k : natlt d).
              acc2 (mtranspose (SF.kv_tile 16 rK k0)) k j
              == acc2 rK (SF.clamp_nat sk (k0 + j)) k);
    matmul_col_ext rQ (mtranspose (SF.kv_tile 16 rK k0)) (mtranspose rK) i j (k0 + j);
    assert (acc (BM.emma_chain #et_ab #et_acc 16 eQ
                   (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16))
                ((i, (j, ())) <: Kuiper.Shape.abs (16 @| 16 @| INil))
            %~ acc (MS.matmul rQ (mtranspose (SF.kv_tile 16 rK k0)))
                   ((i, (j, ())) <: Kuiper.Shape.abs (16 @| 16 @| INil)))

(* ------------------------------------------------------------------ *)
(* The tile's slice of the real problem.                               *)
(* ------------------------------------------------------------------ *)

(* The real pre-softmax score of key [j] for query row [i] of the tile: the
   additive bias plus the scaled exact dot product.  This is
   [Kuiops.Sdpa.Flash.Spec.attn_scores] restricted to one row. *)
let real_row
  (#sk #d : pos)
  (rQ : chest2 real 16 d) (rK : chest2 real sk d)
  (rbias : chest1 real sk) (rscale : real) (i : natlt 16)
  (j : natlt sk) : GTot real
  = acc1 rbias j +. (acc2 (MS.matmul rQ (mtranspose rK)) i j *. rscale)

(* The keys of the tile at [k0] that this query row admits, globally... *)
let tile_pred
  (row_active causal : bool) (sk : pos) (cbound k0 bn : nat) : SO.pred sk
  = fun j -> k0 <= j && j < k0 + bn && SF.key_ok row_active causal sk cbound j

(* ... and tile-locally. *)
let local_pred
  (row_active causal : bool) (sk cbound k0 bn : nat) : SO.pred bn
  = fun i -> SF.key_ok row_active causal sk cbound (k0 + i)

(* The real scores of the tile, with the out-of-range slots of the last tile
   filled by an arbitrary value that the predicate never selects. *)
let local_real
  (#sk #d : pos)
  (rQ : chest2 real 16 d) (rK : chest2 real sk d)
  (rbias : chest1 real sk) (rscale : real) (i : natlt 16) (k0 : nat)
  (t : natlt 16) : GTot real
  = if k0 + t < sk then real_row rQ rK rbias rscale i (k0 + t) else 0.0R

(* ------------------------------------------------------------------ *)
(* Finiteness of the tile scores: the overflow hypothesis.             *)
(* ------------------------------------------------------------------ *)

let scores_finite
  (#et : Type0) {| floating et |}
  (row_active causal : bool) (sk cbound k0 : nat) (es : chest1 et 16) : prop
  = forall (t : natlt 16).
      SF.key_ok row_active causal sk cbound (k0 + t)
      ==> Finite? (kind (acc1 es t))

(* ------------------------------------------------------------------ *)
(* One admitted score approximates the real one.                       *)
(* ------------------------------------------------------------------ *)

let masked_score_approx
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (rQ : chest2 real 16 d) (rK : chest2 real sk d)
  (rbias : chest1 real sk) (rscale : real)
  (i t : natlt 16)
  : Lemma (requires
             eQ %~ rQ /\ eKg %~ rK /\ scale %~ rscale /\
             (forall (j : natlt sk).
                SF.mask_bias emask has_mask bi qh qpos j %~ acc1 rbias j) /\
             SF.key_ok row_active causal sk cbound (k0 + t))
          (ensures
             acc1 (SF.tile_scores emask has_mask row_active causal bi qh qpos
                     k0 cbound scale
                     (mk1 (fun j -> raw_score #et_ab #et_acc eQ eKg k0 i j))) t
             %~ local_real rQ rK rbias rscale i k0 t)
  = let kj : natlt sk = k0 + t in
    let raw = raw_score #et_ab #et_acc eQ eKg k0 i t in
    let rawr = acc2 (MS.matmul rQ (mtranspose rK)) i kj in
    raw_score_approx #et_ab #et_acc eQ eKg rQ rK k0 i t;
    a_mul raw scale rawr rscale;
    assert (SF.clamp_nat sk (k0 + t) == kj);
    FC.fcast_approx #et_ab #et_acc (SF.mask_bias emask has_mask bi qh qpos kj) (acc1 rbias kj);
    a_add (raw `mul` scale)
          (FC.fcast #et_ab #et_acc (SF.mask_bias emask has_mask bi qh qpos kj))
          (rawr *. rscale) (acc1 rbias kj)

(* ------------------------------------------------------------------ *)
(* The step.                                                           *)
(* ------------------------------------------------------------------ *)

(* Everything the kernel and the caller must supply for one lane to absorb one
   key tile: the shared operands approximate the real ones, the tile's scores
   do not overflow, and the keys of the tile have not been absorbed before. *)
let step_ok
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (rQ : chest2 real 16 d) (rK : chest2 real sk d)
  (rbias : chest1 real sk) (rscale : real)
  (i : natlt 16) (p : SO.pred sk) : prop
  = k0 <= sk /\
    eQ %~ rQ /\ eKg %~ rK /\ scale %~ rscale /\
    (forall (j : natlt sk).
       SF.mask_bias emask has_mask bi qh qpos j %~ acc1 rbias j) /\
    scores_finite row_active causal sk cbound k0
      (SF.tile_scores emask has_mask row_active causal bi qh qpos k0 cbound scale
         (mk1 (fun j -> raw_score #et_ab #et_acc eQ eKg k0 i j))) /\
    (forall (j : natlt sk). p j ==> j < k0)

#push-options "--z3rlimit 30 --fuel 0 --ifuel 1"

(* One lane absorbs one key tile: the running maximum and denominator move from
   the key set [p] to [p] together with the tile's admitted keys. *)
let ml_step_tile
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |} {| floating_real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (rQ : chest2 real 16 d) (rK : chest2 real sk d)
  (rbias : chest1 real sk) (rscale : real)
  (i : natlt 16) (p : SO.pred sk)
  (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16) (vm vl m' l' cw' : et_acc)
  : Lemma
      (requires
        step_ok emask has_mask row_active causal bi qh qpos k0 cbound scale
          eQ eKg rQ rK rbias rscale i p /\
        SF.softmax_upd_post emask has_mask row_active causal bi qh qpos
          k0 cbound scale
          (mk1 (fun j -> raw_score #et_ab #et_acc eQ eKg k0 i j)) vm vl
          es' ep' m' l' cw' /\
        Finite? (kind cw') /\
        SB.ml_state (real_row rQ rK rbias rscale i) p vm vl)
      (ensures
        SB.ml_state (real_row rQ rK rbias rscale i)
          (SO.por p (tile_pred row_active causal sk cbound k0 16)) m' l')
  = let x = real_row rQ rK rbias rscale i in
    let t = tile_pred row_active causal sk cbound k0 16 in
    let q = local_pred row_active causal sk cbound k0 16 in
    let xt = local_real rQ rK rbias rscale i k0 in
    let es = mk1 (fun j -> raw_score #et_ab #et_acc eQ eKg k0 i j) in
    SF.scores_post_det emask has_mask row_active causal bi qh qpos k0 cbound
      scale es es';
    SB.ninf_not_nan #et_acc ();
    introduce forall (u : natlt 16). q u ==>
                (acc1 es' u %~ xt u /\ k0 + u < sk /\ xt u == x (k0 + u))
    with introduce _ ==> _
    with _. masked_score_approx emask has_mask row_active causal bi qh qpos
              k0 cbound scale eQ eKg rQ rK rbias rscale i u;
    SB.ml_step x p t (SO.por p t) q xt k0 es' vm vl m' l' cw'

#pop-options
