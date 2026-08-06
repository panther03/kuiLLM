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

(* The whole score row lane [i] computes for the tile at [k0]. *)
let tile_score_row
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (k0 : nat) (i : natlt 16)
  : GTot (chest1 et_acc 16)
  = mk1 (fun j -> raw_score #et_ab #et_acc eQ eKg k0 i j)

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

(* ------------------------------------------------------------------ *)
(* The whole key-tile loop of one warp.                                *)
(* ------------------------------------------------------------------ *)

(* Two indices in the same residue class that are less than one stride apart
   are equal. *)
let mod_eq_close (nw : pos) (a c : nat)
  : Lemma (requires a % nw == c % nw /\ c <= a /\ a < c + nw) (ensures a == c)
  = FStar.Math.Lemmas.euclidean_division_definition a nw;
    FStar.Math.Lemmas.euclidean_division_definition c nw;
    if a > c then FStar.Math.Lemmas.lemma_mult_lt_right nw (c / nw) (a / nw)
    else ()

(* A key belongs to the tile named by its own quotient. *)
let key_tile (bn : pos) (j : nat)
  : Lemma ((j / bn) * bn <= j /\ j < (j / bn) * bn + bn)
  = FStar.Math.Lemmas.euclidean_division_definition j bn

let tile_key (bn : pos) (t j : nat)
  : Lemma (requires t * bn <= j /\ j < t * bn + bn) (ensures j / bn == t)
  = FStar.Math.Lemmas.lemma_div_plus (j - t * bn) t bn;
    FStar.Math.Lemmas.small_div (j - t * bn) bn

(* The keys warp [w] of [nw] has absorbed once it has run every tile [t] with
   [t % nw == w] and [t < vjt]. *)
let absorbed_pred
  (row_active causal : bool) (sk : pos) (cbound : nat)
  (nw : pos) (w vjt : nat) : SO.pred sk
  = fun j -> j / 16 < vjt && (j / 16) % nw = w
             && SF.key_ok row_active causal sk cbound j

let absorbed_lt
  (row_active causal : bool) (sk : pos) (cbound : nat)
  (nw : pos) (w vjt : nat) (j : natlt sk)
  : Lemma (requires absorbed_pred row_active causal sk cbound nw w vjt j)
          (ensures j < vjt * 16)
  = key_tile 16 j;
    FStar.Math.Lemmas.lemma_mult_le_right 16 (j / 16 + 1) vjt

(* Running the tile at [vjt] extends the absorbed set by exactly one stride. *)
let absorbed_step
  (row_active causal : bool) (sk : pos) (cbound : nat)
  (nw : pos) (w vjt : nat)
  : Lemma (requires vjt % nw == w /\ w < nw)
          (ensures forall (j : natlt sk).
             SO.por (absorbed_pred row_active causal sk cbound nw w vjt)
                    (tile_pred row_active causal sk cbound (vjt * 16) 16) j
             == absorbed_pred row_active causal sk cbound nw w (vjt + nw) j)
  = introduce forall (j : natlt sk).
      SO.por (absorbed_pred row_active causal sk cbound nw w vjt)
             (tile_pred row_active causal sk cbound (vjt * 16) 16) j
      == absorbed_pred row_active causal sk cbound nw w (vjt + nw) j
    with (key_tile 16 j;
          if vjt * 16 <= j && j < vjt * 16 + 16 then tile_key 16 vjt j else ();
          if vjt <= j / 16 && j / 16 < vjt + nw && (j / 16) % nw = w
          then mod_eq_close nw (j / 16) vjt else ())

(* The overflow hypothesis for every tile the lane will ever see. *)
let all_finite
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16) : prop
  = forall (k0 : nat). k0 <= sk ==>
      scores_finite row_active causal sk cbound k0
        (SF.tile_scores emask has_mask row_active causal bi qh qpos
           k0 cbound scale (tile_score_row #et_ab #et_acc eQ eKg k0 i))

(* The real problem one lane solves, read off the shared tiles. *)
let lane_real
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| real_like et_ab |}
  {| scalar et_acc |} {| real_like et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask : bool) (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  : GTot (natlt sk -> GTot real)
  = real_row (to_real_chest eQ) (to_real_chest eKg)
      (mk1 (fun j -> to_real (SF.mask_bias emask has_mask bi qh qpos j)))
      (to_real scale) i

(* The loop invariant of one warp's key-tile loop. *)
let loop_state
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw : pos) (w vjt : nat) (vm vl : et_acc) : prop
  = SB.ml_state (lane_real emask has_mask bi qh qpos scale eQ eKg i)
      (absorbed_pred row_active causal sk cbound nw w vjt) vm vl

#push-options "--z3rlimit 30 --fuel 1 --ifuel 2"

(* One iteration of the loop preserves [loop_state]. *)
let ml_step_loop
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |} {| floating_real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw : pos) (w vjt : nat) (vm vl : et_acc)
  (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16) (m' l' cw' : et_acc)
  : Lemma
      (requires
        w < nw /\ vjt % nw == w /\ vjt * 16 <= sk /\
        all_finite emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i /\
        SF.softmax_upd_post emask has_mask row_active causal bi qh qpos
          (vjt * 16) cbound scale
          (tile_score_row #et_ab #et_acc eQ eKg (vjt * 16) i) vm vl
          es' ep' m' l' cw' /\
        Finite? (kind cw') /\
        loop_state emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i nw w vjt vm vl)
      (ensures
        loop_state emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i nw w (vjt + nw) m' l')
  = let k0 = vjt * 16 in
    let p = absorbed_pred row_active causal sk cbound nw w vjt in
    let rbias : chest1 real sk =
      mk1 (fun j -> to_real (SF.mask_bias emask has_mask bi qh qpos j)) in
    assert (forall (j : natlt sk). acc1 rbias j
              == to_real (SF.mask_bias emask has_mask bi qh qpos j));
    introduce forall (j : natlt sk). p j ==> j < k0
    with introduce _ ==> _
    with _. absorbed_lt row_active causal sk cbound nw w vjt j;
    assert (step_ok emask has_mask row_active causal bi qh qpos k0 cbound scale
      eQ eKg (to_real_chest eQ) (to_real_chest eKg) rbias (to_real scale) i p);
    ml_step_tile emask has_mask row_active causal bi qh qpos k0 cbound scale
      eQ eKg (to_real_chest eQ) (to_real_chest eKg) rbias (to_real scale)
      i p es' ep' vm vl m' l' cw';
    absorbed_step row_active causal sk cbound nw w vjt;
    SB.ml_state_ext (lane_real emask has_mask bi qh qpos scale eQ eKg i)
      (SO.por p (tile_pred row_active causal sk cbound k0 16))
      (absorbed_pred row_active causal sk cbound nw w (vjt + nw))
      m' l'

#pop-options

(* Before its first tile a warp has absorbed nothing: the tiles it will visit
   all lie at or past its own index. *)
let absorbed_init
  (row_active causal : bool) (sk : pos) (cbound : nat) (nw : pos) (w : nat)
  : Lemma (requires w < nw)
          (ensures forall (j : natlt sk).
             ~(absorbed_pred row_active causal sk cbound nw w w j))
  = introduce forall (j : natlt sk).
      ~(absorbed_pred row_active causal sk cbound nw w w j)
    with (if j / 16 < w then FStar.Math.Lemmas.small_mod (j / 16) nw else ())

let loop_state_init
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw : pos) (w : nat)
  : Lemma (requires w < nw)
          (ensures loop_state emask has_mask row_active causal bi qh qpos
                     cbound scale eQ eKg i nw w w (neg infinity) zero)
  = absorbed_init row_active causal sk cbound nw w;
    SB.pnone_spec (absorbed_pred row_active causal sk cbound nw w w) sk;
    SB.ninf_not_nan #et_acc ()

(* The loop invariant as the Pulse loop states it: [i] is an unrefined lane
   index, and the bridge only says anything about the lanes that own a row. *)
let lane_state
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : nat)
  (nw : pos) (w vjt : nat) (vm vl : et_acc) : prop
  = i < 16 ==>
      (all_finite emask has_mask row_active causal bi qh qpos cbound scale
         eQ eKg i
       ==> loop_state emask has_mask row_active causal bi qh qpos cbound scale
             eQ eKg i nw w vjt vm vl)

let lane_state_init
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : nat)
  (nw : pos) (w : nat)
  : Lemma (requires w < nw)
          (ensures lane_state emask has_mask row_active causal bi qh qpos
                     cbound scale eQ eKg i nw w w (neg infinity) zero)
  = if i < 16
    then loop_state_init emask has_mask row_active causal bi qh qpos cbound
           scale eQ eKg i nw w
    else ()

(* One lane's tile update, as exposed by the kernel's postcondition. *)
unfold
let step_out
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (vm vl : et_acc)
  (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16) (m' l' cw' : et_acc) : prop
  = SF.softmax_upd_post emask has_mask row_active causal bi qh qpos k0 cbound
      scale (tile_score_row #et_ab #et_acc eQ eKg k0 i) vm vl es' ep' m' l' cw'
    /\ Finite? (kind cw')

#push-options "--z3rlimit 30 --fuel 1 --ifuel 2"

(* [ml_step_loop] with the primed registers universally quantified. *)
let ml_step_loop_fa
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _fr : floating_real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw : pos) (w vjt : nat) (vm vl : et_acc)
  : Lemma
      (requires
        w < nw /\ vjt % nw == w /\ vjt * 16 <= sk /\
        all_finite #et_ab #et_acc #_f #_r #_s #_rb #_c1
          #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi qh qpos
          cbound scale eQ eKg i /\
        loop_state #et_ab #et_acc #_f #_r #_s #_rb
          #b #hq #sq #sk #d emask has_mask row_active causal bi qh qpos
          cbound scale eQ eKg i nw w vjt vm vl)
      (ensures
        forall (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16)
               (m' l' cw' : et_acc).
          step_out #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
            #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
            bi qh qpos (vjt * 16) cbound scale eQ eKg i vm vl
            es' ep' m' l' cw'
          ==> loop_state #et_ab #et_acc #_f #_r #_s #_rb
                #b #hq #sq #sk #d emask has_mask row_active causal
                bi qh qpos cbound scale eQ eKg i nw w (vjt + nw) m' l')
  = introduce forall (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16)
                     (m' l' cw' : et_acc).
      step_out #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
        #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
        bi qh qpos (vjt * 16) cbound scale eQ eKg i vm vl es' ep' m' l' cw'
      ==> loop_state #et_ab #et_acc #_f #_r #_s #_rb
            #b #hq #sq #sk #d emask has_mask row_active causal
            bi qh qpos cbound scale eQ eKg i nw w (vjt + nw) m' l'
    with introduce
      step_out #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
        #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
        bi qh qpos (vjt * 16) cbound scale eQ eKg i vm vl es' ep' m' l' cw'
      ==> loop_state #et_ab #et_acc #_f #_r #_s #_rb
            #b #hq #sq #sk #d emask has_mask row_active causal
            bi qh qpos cbound scale eQ eKg i nw w (vjt + nw) m' l'
    with _.
      ml_step_loop #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
        #b #hq #sq #sk #d #sq16
        emask has_mask row_active causal bi qh qpos cbound
        scale eQ eKg i nw w vjt vm vl es' ep' m' l' cw'

(* One step of the Pulse key-tile loop: the kernel's postcondition only
   exposes the updated registers existentially, and only for lanes that own a
   row of the query tile. *)
let lane_step
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _fr : floating_real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : nat)
  (nw : pos) (w vjt : nat) (vm vl m' l' : et_acc)
  : Lemma
      (requires
        w < nw /\ vjt % nw == w /\ vjt * 16 <= sk /\
        lane_state #et_ab #et_acc #_f #_r #_s #_rb #_c1
          #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi qh qpos
          cbound scale eQ eKg i nw w vjt vm vl /\
        (i < 16 ==>
          (exists (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16)
                  (cw' : et_acc).
             step_out #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
               #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
               bi qh qpos (vjt * 16) cbound scale eQ eKg i vm vl
               es' ep' m' l' cw')))
      (ensures
        lane_state #et_ab #et_acc #_f #_r #_s #_rb #_c1
          #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi qh qpos
          cbound scale eQ eKg i nw w (vjt + nw) m' l')
  = if i < 16
    then introduce
           all_finite #et_ab #et_acc #_f #_r #_s #_rb #_c1
             #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
             bi qh qpos cbound scale eQ eKg i
           ==> loop_state #et_ab #et_acc #_f #_r #_s #_rb
                 #b #hq #sq #sk #d emask has_mask row_active causal
                 bi qh qpos cbound scale eQ eKg i nw w (vjt + nw) m' l'
         with _.
           ml_step_loop_fa #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
             #b #hq #sq #sk #d #sq16
             emask has_mask row_active causal bi qh qpos cbound
             scale eQ eKg i nw w vjt vm vl
    else ()

#pop-options

(* ------------------------------------------------------------------ *)
(* Determinacy of the running registers.                               *)
(*                                                                     *)
(* The whole warp must agree on the value a shared tile holds, so the  *)
(* barriers around [scale]/[pv_mm] can only pin the probability tile   *)
(* if the per-lane running maximum is a function of data every lane    *)
(* knows.  It is: the registers are determined by the tiles the warp   *)
(* has consumed.                                                       *)
(* ------------------------------------------------------------------ *)

let step_ml
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (k0 : nat) (vm vl : et_acc) : GTot (et_acc & et_acc)
  = let es' = SF.tile_scores emask has_mask row_active causal bi qh qpos
                k0 cbound scale (tile_score_row #et_ab #et_acc eQ eKg k0 i) in
    let m' = fmax vm (SF.row_max es' 16) in
    (m', (vl `mul` fexp (vm `sub` m')) `add` SF.row_sum es' m' 16)

(* The registers after the warp's first [t] tiles: it owns tile [w], then
   every [nw]-th one after it. *)
let rec run_ml
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw w t : nat) : GTot (et_acc & et_acc) (decreases t)
  = if t = 0 then (neg infinity, zero)
    else
      let ml = run_ml emask has_mask row_active causal bi qh qpos cbound scale
                 eQ eKg i nw w (t - 1) in
      step_ml emask has_mask row_active causal bi qh qpos cbound scale
        eQ eKg i ((w + (t - 1) * nw) * 16) (fst ml) (snd ml)

let lane_ml
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : nat)
  (nw w t : nat) (vm vl : et_acc) : prop
  = i < 16 ==>
      (vm == fst (run_ml emask has_mask row_active causal bi qh qpos cbound
                    scale eQ eKg i nw w t) /\
       vl == snd (run_ml emask has_mask row_active causal bi qh qpos cbound
                    scale eQ eKg i nw w t))

let lane_ml_init
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : nat)
  (nw w : nat)
  : Lemma (lane_ml emask has_mask row_active causal bi qh qpos cbound scale
             eQ eKg i nw w 0 (neg infinity) zero)
  = ()

(* [softmax_upd_post] pins the primed registers exactly. *)
let softmax_upd_det
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#bn : nat)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (es : chest1 et_acc bn) (vm vl m' l' : et_acc)
  : Lemma
      (requires
        exists (es' : chest1 et_acc bn) (ep' : chest1 et_ab bn) (cw' : et_acc).
          SF.softmax_upd_post emask has_mask row_active causal bi qh qpos
            k0 cbound scale es vm vl es' ep' m' l' cw')
      (ensures
        (let ts = SF.tile_scores emask has_mask row_active causal bi qh qpos
                    k0 cbound scale es in
         m' == fmax vm (SF.row_max ts bn) /\
         l' == (vl `mul` fexp (vm `sub` m')) `add` SF.row_sum ts m' bn))
  = FStar.Classical.forall_intro
      (FStar.Classical.move_requires
         (SF.scores_post_det #et_acc #et_ab #_f #_r #_s #_rb #_c1
            #b #hq #sq #sk #bn emask has_mask row_active causal bi qh qpos
            k0 cbound scale es))

#push-options "--z3rlimit 20 --fuel 2 --ifuel 2"

let lane_ml_step
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : nat)
  (nw w t : nat) (vm vl m' l' : et_acc)
  : Lemma
      (requires
        lane_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
          #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi qh qpos
          cbound scale eQ eKg i nw w t vm vl /\
        (i < 16 ==>
          (exists (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16)
                  (cw' : et_acc).
             step_out #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
               #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
               bi qh qpos ((w + t * nw) * 16) cbound scale eQ eKg i vm vl
               es' ep' m' l' cw')))
      (ensures
        lane_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
          #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi qh qpos
          cbound scale eQ eKg i nw w (t + 1) m' l')
  = if i < 16
    then softmax_upd_det #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
           #b #hq #sq #sk #16 emask has_mask row_active causal bi qh qpos
           ((w + t * nw) * 16) cbound scale
           (tile_score_row #et_ab #et_acc eQ eKg ((w + t * nw) * 16) i)
           vm vl m' l'
    else ()

#pop-options
