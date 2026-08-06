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

(* ------------------------------------------------------------------ *)
(* The warp's registers as vectors.                                    *)
(* ------------------------------------------------------------------ *)

let escore_erow
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (k0 : nat) (i : natlt 16)
  : Lemma (SF.erow (BM.emma_chain #et_ab #et_acc 16 eQ
                      (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16)) i
           == tile_score_row #et_ab #et_acc eQ eKg k0 i)
  = assert (equal (SF.erow (BM.emma_chain #et_ab #et_acc 16 eQ
                              (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16)) i)
                  (tile_score_row #et_ab #et_acc eQ eKg k0 i))

let run_mv
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw w t : nat) : GTot (chest1 et_acc 16)
  = mk1 (fun i -> fst (run_ml emask has_mask row_active causal bi qh qpos cbound
                         scale eQ eKg i nw w t))

let run_lv
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw w t : nat) : GTot (chest1 et_acc 16)
  = mk1 (fun i -> snd (run_ml emask has_mask row_active causal bi qh qpos cbound
                         scale eQ eKg i nw w t))

#push-options "--fuel 1 --ifuel 2"

(* The register vectors read at a lane are that lane's registers. *)
let run_mlv_acc
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
  (nw w t : nat) (vm vl : et_acc)
  : Lemma
      (requires lane_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                  #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
                  bi qh qpos cbound scale eQ eKg i nw w t vm vl)
      (ensures i < 16 ==>
        (vm == acc1 (run_mv emask has_mask row_active causal bi qh qpos cbound
                       scale eQ eKg nw w t) i /\
         vl == acc1 (run_lv emask has_mask row_active causal bi qh qpos cbound
                       scale eQ eKg nw w t) i))
  = ()

#pop-options

#push-options "--fuel 2 --ifuel 2 --z3rlimit 20"

(* One key tile advances the register vectors by exactly one [run_ml] step. *)
let run_mlv_step
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw w t : nat)
  : Lemma
      (let k0 = (w + t * nw) * 16 in
       let eS = BM.emma_chain #et_ab #et_acc 16 eQ
                  (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
       SF.m_vec emask has_mask row_active causal bi qh qpos k0 cbound scale eS
         (run_mv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w t)
       == run_mv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w (t + 1) /\
       SF.l_vec emask has_mask row_active causal bi qh qpos k0 cbound scale eS
         (run_mv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w t)
         (run_lv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w t)
       == run_lv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w (t + 1))
  = let k0 = (w + t * nw) * 16 in
    let eS = BM.emma_chain #et_ab #et_acc 16 eQ
               (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
    FStar.Classical.forall_intro
      (escore_erow #et_ab #et_acc #_ #_ #sk #d #sq16 eQ eKg k0);
    assert (equal
      (SF.m_vec emask has_mask row_active causal bi qh qpos k0 cbound scale eS
         (run_mv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w t))
      (run_mv emask has_mask row_active causal bi qh qpos cbound scale
         eQ eKg nw w (t + 1)));
    assert (equal
      (SF.l_vec emask has_mask row_active causal bi qh qpos k0 cbound scale eS
         (run_mv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w t)
         (run_lv emask has_mask row_active causal bi qh qpos cbound scale
            eQ eKg nw w t))
      (run_lv emask has_mask row_active causal bi qh qpos cbound scale
         eQ eKg nw w (t + 1)))

#pop-options

(* ------------------------------------------------------------------ *)
(* Whole-tile descriptions.                                            *)
(*                                                                     *)
(* [run_ml] applies one row's parameters to all 16 rows; these versions *)
(* give row [i] its own query head, position and causal bound, so they  *)
(* are the same expression for every lane of the warp.                  *)
(* ------------------------------------------------------------------ *)

let run_ml_t
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw w t : nat) : GTot (et_acc & et_acc)
= run_ml emask has_mask (SF.lane_active_row rows r0 i) causal bi
    (SF.lane_qh hq sq kvh group rows r0 i) (SF.lane_qpos sq rows r0 i)
    (SF.lane_cbound sq sk rows r0 i) scale eQ eKg i nw w t

let run_mv_t
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw w t : nat) : GTot (chest1 et_acc 16)
= mk1 (fun i -> fst (run_ml_t emask has_mask causal bi kvh group rows r0 scale
                       eQ eKg i nw w t))

let run_lv_t
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw w t : nat) : GTot (chest1 et_acc 16)
= mk1 (fun i -> snd (run_ml_t emask has_mask causal bi kvh group rows r0 scale
                       eQ eKg i nw w t))

#push-options "--fuel 1 --ifuel 2"

(* Row [i] of the lane-uniform vectors is that row's whole-tile register. *)
let run_mlv_t_acc
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw w t : nat)
  : Lemma
      (acc1 (run_mv_t emask has_mask causal bi kvh group rows r0 scale
               eQ eKg nw w t) i
       == fst (run_ml_t emask has_mask causal bi kvh group rows r0 scale
                 eQ eKg i nw w t) /\
       acc1 (run_lv_t emask has_mask causal bi kvh group rows r0 scale
               eQ eKg nw w t) i
       == snd (run_ml_t emask has_mask causal bi kvh group rows r0 scale
                 eQ eKg i nw w t))
  = ()

(* At row [i] the lane-uniform vectors are that row's own. *)
let run_mlv_t_eq
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (kvh group rows r0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : nat)
  (nw w t : nat)
  : Lemma
      (requires i < 16 ==>
        SF.lane_params_ok hq sq sk kvh group rows r0 i row_active qh qpos cbound)
      (ensures i < 16 ==>
        (acc1 (run_mv_t emask has_mask causal bi kvh group rows r0 scale
                 eQ eKg nw w t) i
         == acc1 (run_mv emask has_mask row_active causal bi qh qpos cbound
                    scale eQ eKg nw w t) i /\
         acc1 (run_lv_t emask has_mask causal bi kvh group rows r0 scale
                 eQ eKg nw w t) i
         == acc1 (run_lv emask has_mask row_active causal bi qh qpos cbound
                    scale eQ eKg nw w t) i))
  = ()

#pop-options

#push-options "--fuel 2 --ifuel 2 --z3rlimit 30"

(* One key tile advances row [i] of the lane-uniform register vectors. *)
let mlv_step_t_row
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw w t : nat) (i : natlt 16)
  : Lemma
      (let k0 = (w + t * nw) * 16 in
       let eS = BM.emma_chain #et_ab #et_acc 16 eQ
                  (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
       acc1 (SF.m_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS
               (run_mv_t emask has_mask causal bi kvh group rows r0 scale
                  eQ eKg nw w t)) i
       == acc1 (run_mv_t emask has_mask causal bi kvh group rows r0 scale
                  eQ eKg nw w (t + 1)) i /\
       acc1 (SF.l_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS
               (run_mv_t emask has_mask causal bi kvh group rows r0 scale
                  eQ eKg nw w t)
               (run_lv_t emask has_mask causal bi kvh group rows r0 scale
                  eQ eKg nw w t)) i
       == acc1 (run_lv_t emask has_mask causal bi kvh group rows r0 scale
                  eQ eKg nw w (t + 1)) i)
  = let k0 = (w + t * nw) * 16 in
    let eS = BM.emma_chain #et_ab #et_acc 16 eQ
               (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
    let ra = SF.lane_active_row rows r0 i in
    let qh = SF.lane_qh hq sq kvh group rows r0 i in
    let qp = SF.lane_qpos sq rows r0 i in
    let cb = SF.lane_cbound sq sk rows r0 i in
    run_mlv_step emask has_mask ra causal bi qh qp cb scale eQ eKg nw w t;
    SF.vec_local emask has_mask ra causal bi qh qp k0 cb scale eS
      (run_mv_t emask has_mask causal bi kvh group rows r0 scale eQ eKg nw w t)
      (run_mv emask has_mask ra causal bi qh qp cb scale eQ eKg nw w t)
      (run_lv_t emask has_mask causal bi kvh group rows r0 scale eQ eKg nw w t)
      (run_lv emask has_mask ra causal bi qh qp cb scale eQ eKg nw w t) i

(* One key tile advances the lane-uniform register vectors by one step. *)
let run_mlv_step_t
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw w t : nat)
  : Lemma
      (let k0 = (w + t * nw) * 16 in
       let eS = BM.emma_chain #et_ab #et_acc 16 eQ
                  (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
       SF.m_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS
         (run_mv_t emask has_mask causal bi kvh group rows r0 scale eQ eKg nw w t)
       == run_mv_t emask has_mask causal bi kvh group rows r0 scale
            eQ eKg nw w (t + 1) /\
       SF.l_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS
         (run_mv_t emask has_mask causal bi kvh group rows r0 scale eQ eKg nw w t)
         (run_lv_t emask has_mask causal bi kvh group rows r0 scale eQ eKg nw w t)
       == run_lv_t emask has_mask causal bi kvh group rows r0 scale
            eQ eKg nw w (t + 1))
  = let k0 = (w + t * nw) * 16 in
    let eS = BM.emma_chain #et_ab #et_acc 16 eQ
               (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
    let evm = run_mv_t emask has_mask causal bi kvh group rows r0 scale
                eQ eKg nw w t in
    let evl = run_lv_t emask has_mask causal bi kvh group rows r0 scale
                eQ eKg nw w t in
    FStar.Classical.forall_intro
      (mlv_step_t_row #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
         #b #hq #sq #sk #d #sq16 emask has_mask causal bi kvh group
         rows r0 scale eQ eKg nw w t);
    assert (equal
      (SF.m_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS evm)
      (run_mv_t emask has_mask causal bi kvh group rows r0 scale
         eQ eKg nw w (t + 1)));
    assert (equal
      (SF.l_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS evm evl)
      (run_lv_t emask has_mask causal bi kvh group rows r0 scale
         eQ eKg nw w (t + 1)))

#pop-options

#push-options "--fuel 1 --ifuel 1"

(* The output tile a warp holds after [t] key tiles: the zero fill, then one
   [SF.out_tile] step per tile the warp visits. *)
let rec run_O
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg eVg : chest2 et_ab sk d)
  (nw w t : nat) : GTot (chest2 et_acc 16 d) (decreases t)
= if t = 0 then const (16 @| d @| INil) zero
  else
    let k0 = (w + (t - 1) * nw) * 16 in
    let eS = BM.emma_chain #et_ab #et_acc 16 eQ
               (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
    let evm = run_mv_t emask has_mask causal bi kvh group rows r0 scale
                eQ eKg nw w (t - 1) in
    SF.out_tile
      (run_O emask has_mask causal bi kvh group rows r0 scale
         eQ eKg eVg nw w (t - 1))
      (SF.cw_vec_t emask has_mask causal bi kvh group rows r0 k0 scale eS evm)
      (SF.prob_tile_t emask has_mask causal bi kvh group rows r0 k0 scale
         eS evm)
      (SF.kv_tile 16 eVg k0)

(* One key tile advances the output tile by exactly one [run_O] step. *)
let run_O_step
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg eVg : chest2 et_ab sk d)
  (nw w t : nat)
  : Lemma
      (let k0 = (w + t * nw) * 16 in
       let eS = BM.emma_chain #et_ab #et_acc 16 eQ
                  (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16) in
       let evm = run_mv_t emask has_mask causal bi kvh group rows r0 scale
                   eQ eKg nw w t in
       SF.out_tile
         (run_O emask has_mask causal bi kvh group rows r0 scale
            eQ eKg eVg nw w t)
         (SF.cw_vec_t emask has_mask causal bi kvh group rows r0 k0 scale
            eS evm)
         (SF.prob_tile_t emask has_mask causal bi kvh group rows r0 k0 scale
            eS evm)
         (SF.kv_tile 16 eVg k0)
       == run_O emask has_mask causal bi kvh group rows r0 scale
            eQ eKg eVg nw w (t + 1))
  = ()

#pop-options

(* ------------------------------------------------------------------ *)
(* What the block holds when every warp has finished its sweep.        *)
(* ------------------------------------------------------------------ *)

let block_m
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw : pos) (nkt : nat)
  : GTot (chest2 et_acc nw 16)
= mk2 (fun w i -> acc1 (run_mv_t emask has_mask causal bi kvh group rows r0
                          scale eQ eKg nw w (SF.warp_iters nw nkt w)) i)

let block_l
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw : pos) (nkt : nat)
  : GTot (chest2 et_acc nw 16)
= mk2 (fun w i -> acc1 (run_lv_t emask has_mask causal bi kvh group rows r0
                          scale eQ eKg nw w (SF.warp_iters nw nkt w)) i)

#push-options "--fuel 1 --ifuel 2"

(* Cell [(w, i)] of the block's published vectors is row [i] of warp [w]'s
   register vector after its last key tile. *)
let block_ml_acc
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw : pos) (nkt : nat) (w : natlt nw) (i : natlt 16)
  : Lemma
      (acc2 (block_m emask has_mask causal bi kvh group rows r0 scale
               eQ eKg nw nkt) w i
       == acc1 (run_mv_t emask has_mask causal bi kvh group rows r0 scale
                  eQ eKg nw w (SF.warp_iters nw nkt w)) i /\
       acc2 (block_l emask has_mask causal bi kvh group rows r0 scale
               eQ eKg nw nkt) w i
       == acc1 (run_lv_t emask has_mask causal bi kvh group rows r0 scale
                  eQ eKg nw w (SF.warp_iters nw nkt w)) i)
  = ()

#pop-options

(* The block's output tile: warp [w] owns rows [16w .. 16w+15]. *)
let block_O
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg eVg : chest2 et_ab sk d)
  (nw : pos) (nkt : nat) (w : natlt nw)
  : GTot (chest2 et_acc 16 d)
= run_O emask has_mask causal bi kvh group rows r0 scale eQ eKg eVg nw w
    (SF.warp_iters nw nkt w)


(* ------------------------------------------------------------------ *)
(* The warp's registers satisfy the real flash invariant.              *)
(* ------------------------------------------------------------------ *)

(* The correction weight the lane applies when it absorbs its [t]th tile. *)
let cw_at
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
  (nw w t : nat) : GTot et_acc
  = fexp (fst (run_ml emask has_mask row_active causal bi qh qpos cbound scale
                 eQ eKg i nw w t)
          `sub`
          fst (run_ml emask has_mask row_active causal bi qh qpos cbound scale
                 eQ eKg i nw w (t + 1)))

(* Every correction weight of the first [t] steps is finite.  Like [all_finite]
   this is an overflow hypothesis and is not derivable from the [floating]
   laws; the kernel exports it (see the [kind cw' == Finite] conjunct of
   [Kuiops.Sdpa.Flash.KfSub.sdpa_flash_softmax_upd]'s postcondition). *)
let cw_upto
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
  (nw w t : nat) : prop
  = forall (u : nat).
      {:pattern (cw_at #et_ab #et_acc #_f #_r #_s #_rb #_c1
                   #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
                   bi qh qpos cbound scale eQ eKg i nw w u)}
      u < t ==>
      Finite? (kind (cw_at #et_ab #et_acc #_f #_r #_s #_rb #_c1
                       #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
                       bi qh qpos cbound scale eQ eKg i nw w u))

#push-options "--fuel 1 --ifuel 2"

let acc1_mk1 (#et : Type) (#d0 : nat) (f : natlt d0 -> GTot et)
  : Lemma (forall (j : natlt d0). acc1 (mk1 f) j == f j)
  = ()

(* The canonical witnesses of [softmax_upd_post]. *)
let softmax_upd_canon
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#bn : nat)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (es : chest1 et_acc bn) (vm vl : et_acc)
  : Lemma
      (ensures
        (let es' = SF.tile_scores emask has_mask row_active causal bi qh qpos
                     k0 cbound scale es in
         let m' = fmax vm (SF.row_max es' bn) in
         SF.softmax_upd_post emask has_mask row_active causal bi qh qpos
           k0 cbound scale es vm vl es'
           (mk1 (fun j -> FC.fcast (SF.sel_prob (acc1 es' j) m')))
           m'
           ((vl `mul` fexp (vm `sub` m')) `add` SF.row_sum es' m' bn)
           (fexp (vm `sub` m'))))
  = ()

let run_ml_zero
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
  (nw w : nat)
  : Lemma (run_ml emask has_mask row_active causal bi qh qpos cbound scale
             eQ eKg i nw w 0 == (neg infinity, zero))
  = ()

let run_ml_unfold
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
  (nw w t : nat)
  : Lemma
      (ensures
        (let ml = run_ml emask has_mask row_active causal bi qh qpos cbound
                    scale eQ eKg i nw w t in
         run_ml emask has_mask row_active causal bi qh qpos cbound scale
           eQ eKg i nw w (t + 1)
         == step_ml emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i ((w + t * nw) * 16) (fst ml) (snd ml)))
  = ()

#pop-options

(* The first tile the warp has NOT yet visited after [t] steps. *)
let warp_tile (nw : pos) (w t : nat) : nat = w + t * nw

let warp_tile_succ (nw : pos) (w t : nat)
  : Lemma (warp_tile nw w t + nw == warp_tile nw w (t + 1))
  = FStar.Math.Lemmas.distributivity_add_left t 1 nw

#push-options "--z3rlimit 60 --fuel 0 --ifuel 1 --split_queries always"

let rec run_ml_state
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
  (nw : pos) (w t : nat)
  : Lemma
      (requires
        w < nw /\
        (forall (u : nat). u < t ==> (w + u * nw) * 16 <= sk) /\
        all_finite #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
          emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i /\
        cw_upto #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
          emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i nw w t)
      (ensures
        (let ml = run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                    #b #hq #sq #sk #d #sq16
                    emask has_mask row_active causal bi qh qpos cbound scale
                    eQ eKg i nw w t in
         loop_state #et_ab #et_acc #_f #_r #_s #_rb #b #hq #sq #sk #d
           emask has_mask row_active causal bi qh qpos cbound scale
           eQ eKg i nw w (warp_tile nw w t) (fst ml) (snd ml)))
      (decreases t)
= if t = 0
  then begin
    run_ml_zero #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
      emask has_mask row_active causal bi qh qpos cbound scale eQ eKg i nw w;
    loop_state_init #et_ab #et_acc #_f #_r #_s #_rb #b #hq #sq #sk #d
      emask has_mask row_active causal bi qh qpos cbound
      scale eQ eKg i nw w
  end
  else begin
    let t1 : nat = t - 1 in
    let vjt : nat = w + t1 * nw in
    warp_tile_succ nw w t1;
    assert (vjt + nw == warp_tile nw w t);
    run_ml_state #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
      #b #hq #sq #sk #d #sq16
      emask has_mask row_active causal bi qh qpos cbound scale
      eQ eKg i nw w t1;
    let ml = run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
               emask has_mask row_active causal bi qh qpos cbound scale
               eQ eKg i nw w t1 in
    let vm = fst ml in
    let vl = snd ml in
    let es = tile_score_row #et_ab #et_acc eQ eKg (vjt * 16) i in
    let es' = SF.tile_scores emask has_mask row_active causal bi qh qpos
                (vjt * 16) cbound scale es in
    let m' = fmax vm (SF.row_max es' 16) in
    let cw' = fexp (vm `sub` m') in
    let l' = (vl `mul` cw') `add` SF.row_sum es' m' 16 in
    let ep' : chest1 et_ab 16 =
      mk1 (fun j -> FC.fcast (SF.sel_prob (acc1 es' j) m')) in
    run_ml_unfold #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
      emask has_mask row_active causal bi qh qpos cbound scale
      eQ eKg i nw w t1;
    assert (run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
              emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i nw w t == (m', l'));
    assert (cw' == cw_at #et_ab #et_acc #_f #_r #_s #_rb #_c1
                     #b #hq #sq #sk #d #sq16
                     emask has_mask row_active causal bi qh qpos cbound scale
                     eQ eKg i nw w t1);
    softmax_upd_canon #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
      #b #hq #sq #sk #16 emask has_mask row_active causal bi qh qpos
      (vjt * 16) cbound scale es vm vl;
    FStar.Math.Lemmas.lemma_mod_plus w t1 nw;
    FStar.Math.Lemmas.small_mod w nw;
    assert (SF.scores_post emask has_mask row_active causal bi qh qpos
              (vjt * 16) cbound scale
              (tile_score_row #et_ab #et_acc eQ eKg (vjt * 16) i) es');
    acc1_mk1 #et_ab #16 (fun j -> FC.fcast (SF.sel_prob (acc1 es' j) m'));
    assert (forall (j : natlt 16).
              acc1 ep' j == FC.fcast (SF.sel_prob (acc1 es' j) m'));
    assert (SF.softmax_upd_post emask has_mask row_active causal bi qh qpos
              (vjt * 16) cbound scale
              (tile_score_row #et_ab #et_acc eQ eKg (vjt * 16) i) vm vl
              es' ep' m' l' cw');
    assert (Finite? (kind cw'));
    assert (w < nw);
    assert (vjt % nw == w);
    assert (vjt * 16 <= sk);
    assert (all_finite #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
              emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i);
    assert (loop_state #et_ab #et_acc #_f #_r #_s #_rb #b #hq #sq #sk #d
              emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i nw w vjt vm vl);
    assert (w < nw /\ vjt % nw == w /\ vjt * 16 <= sk /\
            all_finite emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i /\
            SF.softmax_upd_post emask has_mask row_active causal bi qh qpos
              (vjt * 16) cbound scale
              (tile_score_row #et_ab #et_acc eQ eKg (vjt * 16) i) vm vl
              es' ep' m' l' cw' /\
            Finite? (kind cw') /\
            loop_state emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i nw w vjt vm vl);
    assert (step_out #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
              #b #hq #sq #sk #d #sq16 emask has_mask row_active causal
              bi qh qpos (vjt * 16) cbound scale eQ eKg i vm vl
              es' ep' m' l' cw');
    FStar.Math.Lemmas.distributivity_add_left t1 1 nw;
    assert (vjt + nw == warp_tile nw w t);
    ml_step_loop_fa #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
      #b #hq #sq #sk #d #sq16
      emask has_mask row_active causal bi qh qpos cbound scale
      eQ eKg i nw w vjt vm vl
  end

#pop-options

(* ------------------------------------------------------------------ *)
(* Coverage: the block's tile count reaches every key a lane admits.   *)
(* ------------------------------------------------------------------ *)

(* The causal scan's running maximum dominates every active row's position. *)
let rec tile_maxpos_ge (sq : pos) (rows r0 n i : nat)
  : Lemma (requires i < n /\ r0 + i < rows)
          (ensures (let mf = SF.tile_maxpos sq rows r0 n in
                    snd mf /\ SF.lane_qpos sq rows r0 i <= fst mf))
          (decreases n)
  = if i < n - 1 then tile_maxpos_ge sq rows r0 (n - 1) i else ()

let key_ok_lt_kmax
  (sq : pos) (sk : nat { sq <= sk }) (rows r0 : nat) (causal : bool)
  (i : natlt 16) (j : nat)
  : Lemma (requires SF.key_ok (SF.lane_active_row rows r0 i) causal sk
                      (SF.lane_cbound sq sk rows r0 i) j)
          (ensures j < (if causal then SF.causal_kmax 16 sq sk rows r0 else sk))
  = if causal then tile_maxpos_ge sq rows r0 16 i else ()

let div_lt_divup (e j : nat)
  : Lemma (requires j < e) (ensures j / 16 < Kuiper.Divides.divup e 16)
  = FStar.Math.Lemmas.euclidean_division_definition j 16;
    if j / 16 >= Kuiper.Divides.divup e 16
    then FStar.Math.Lemmas.lemma_mult_le_left 16
           (Kuiper.Divides.divup e 16) (j / 16)
    else ()

let divup_pred_lt (e : nat) (k : pos)
  : Lemma (requires e > 0) (ensures (Kuiper.Divides.divup e k - 1) * k < e)
  = FStar.Math.Lemmas.euclidean_division_definition (e + (k - 1)) k;
    FStar.Math.Lemmas.distributivity_sub_left (Kuiper.Divides.divup e k) 1 k

(* Every key a lane admits lies in one of the block's key tiles. *)
let key_ok_tile_lt
  (sq : pos) (sk : nat { sq <= sk }) (rows r0 : nat) (causal : bool)
  (i : natlt 16) (j : nat)
  : Lemma (requires SF.key_ok (SF.lane_active_row rows r0 i) causal sk
                      (SF.lane_cbound sq sk rows r0 i) j)
          (ensures j / 16 < SF.key_tiles 16 16 sq sk rows r0 causal)
  = key_ok_lt_kmax sq sk rows r0 causal i j;
    div_lt_divup (if causal then SF.causal_kmax 16 sq sk rows r0 else sk) j

(* The last key tile starts inside the key range. *)
let key_tiles_last (sq : pos) (sk : nat { sq <= sk }) (rows r0 : nat)
  (causal : bool)
  : Lemma (ensures (let nkt = SF.key_tiles 16 16 sq sk rows r0 causal in
                    nkt == 0 \/ (nkt - 1) * 16 < sk))
  = let e = if causal then SF.causal_kmax 16 sq sk rows r0 else sk in
    if e = 0 then () else divup_pred_lt e 16

(* The warp's step count reaches the last tile and no further. *)
let warp_iters_bounds (nw : pos) (nkt w : nat)
  : Lemma (ensures (let t = SF.warp_iters nw nkt w in
                    w + t * nw >= nkt /\
                    (forall (u : nat). u < t ==> w + u * nw < nkt)))
  = if w >= nkt then ()
    else begin
      let t = SF.warp_iters nw nkt w in
      Kuiper.Divides.lem_divup_back (nkt - w) nw;
      FStar.Math.Lemmas.euclidean_division_definition (nkt - w + nw - 1) nw;
      FStar.Math.Lemmas.distributivity_sub_left t 1 nw;
      introduce forall (u : nat). u < t ==> w + u * nw < nkt
      with introduce _ ==> _
      with _. FStar.Math.Lemmas.lemma_mult_le_right nw u (t - 1)
    end

(* ------------------------------------------------------------------ *)
(* The warp's final registers as a real online-softmax state.          *)
(* ------------------------------------------------------------------ *)

(* The keys warp [w] of [nw] is responsible for. *)
let warp_keys
  (row_active causal : bool) (sk : pos) (cbound : nat) (nw : pos) (w : nat)
  : SO.pred sk
  = fun j -> SF.key_ok row_active causal sk cbound j && (j / 16) % nw = w

#push-options "--z3rlimit 20 --fuel 0 --ifuel 1"

let warp_ml_state
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _fr : floating_real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (kvh group rows r0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw : pos) (w : natlt nw) (nkt : nat)
  : Lemma
      (requires
        sq <= sk /\
        SF.lane_params_ok hq sq sk kvh group rows r0 i
          row_active qh qpos cbound /\
        nkt == SF.key_tiles 16 16 sq sk rows r0 causal /\
        all_finite #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
          emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i /\
        cw_upto #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
          emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i nw w (SF.warp_iters nw nkt w))
      (ensures
        (let ml = run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                    #b #hq #sq #sk #d #sq16
                    emask has_mask row_active causal bi qh qpos cbound scale
                    eQ eKg i nw w (SF.warp_iters nw nkt w) in
         SB.ml_state (lane_real emask has_mask bi qh qpos scale eQ eKg i)
           (warp_keys row_active causal sk cbound nw w) (fst ml) (snd ml)))
  = let t = SF.warp_iters nw nkt w in
    warp_iters_bounds nw nkt w;
    key_tiles_last sq sk rows r0 causal;
    introduce forall (u : nat). u < t ==> (w + u * nw) * 16 <= sk
    with introduce _ ==> _
    with _. FStar.Math.Lemmas.lemma_mult_le_right 16 (w + u * nw) (nkt - 1);
    run_ml_state #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
      #b #hq #sq #sk #d #sq16
      emask has_mask row_active causal bi qh qpos cbound scale
      eQ eKg i nw w t;
    introduce forall (j : natlt sk).
      absorbed_pred row_active causal sk cbound nw w (warp_tile nw w t) j
      == warp_keys row_active causal sk cbound nw w j
    with (if SF.key_ok row_active causal sk cbound j
          then key_ok_tile_lt sq sk rows r0 causal i j
          else ());
    SB.ml_state_ext (lane_real emask has_mask bi qh qpos scale eQ eKg i)
      (absorbed_pred row_active causal sk cbound nw w (warp_tile nw w t))
      (warp_keys row_active causal sk cbound nw w)
      (fst (run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
              #b #hq #sq #sk #d #sq16
              emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i nw w t))
      (snd (run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
              #b #hq #sq #sk #d #sq16
              emask has_mask row_active causal bi qh qpos cbound scale
              eQ eKg i nw w t))

#pop-options

(* ------------------------------------------------------------------ *)
(* The block's published (m, l) vectors as real online-softmax states.  *)
(* ------------------------------------------------------------------ *)

#push-options "--z3rlimit 60 --fuel 0 --ifuel 1 --split_queries always"

let block_ml_state
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _fr : floating_real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (kvh group rows r0 cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw : pos) (w : natlt nw) (nkt : nat)
  : Lemma
      (requires
        sq <= sk /\
        SF.lane_params_ok hq sq sk kvh group rows r0 i
          row_active qh qpos cbound /\
        nkt == SF.key_tiles 16 16 sq sk rows r0 causal /\
        all_finite #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
          emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i /\
        cw_upto #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
          emask has_mask row_active causal bi qh qpos cbound scale
          eQ eKg i nw w (SF.warp_iters nw nkt w))
      (ensures
        SB.ml_state (lane_real emask has_mask bi qh qpos scale eQ eKg i)
          (warp_keys row_active causal sk cbound nw w)
          (acc2 (block_m #et_ab #et_acc #_f #_r #_s #_rb #_c1
                   #b #hq #sq #sk #d #sq16
                   emask has_mask causal bi kvh group rows r0 scale
                   eQ eKg nw nkt) w i)
          (acc2 (block_l #et_ab #et_acc #_f #_r #_s #_rb #_c1
                   #b #hq #sq #sk #d #sq16
                   emask has_mask causal bi kvh group rows r0 scale
                   eQ eKg nw nkt) w i))
  = let t = SF.warp_iters nw nkt w in
    block_ml_acc #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16
      emask has_mask causal bi kvh group rows r0 scale eQ eKg nw nkt w i;
    run_mlv_t_acc #et_ab #et_acc #_f #_r #_s #_rb #_c1
      #b #hq #sq #sk #d #sq16
      emask has_mask causal bi kvh group rows r0 scale eQ eKg i nw w t;
    assert (run_ml_t #et_ab #et_acc #_f #_r #_s #_rb #_c1
              #b #hq #sq #sk #d #sq16
              emask has_mask causal bi kvh group rows r0 scale eQ eKg i nw w t
            == run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                 #b #hq #sq #sk #d #sq16
                 emask has_mask row_active causal bi qh qpos cbound scale
                 eQ eKg i nw w t);
    assert (acc2 (block_m #et_ab #et_acc #_f #_r #_s #_rb #_c1
                    #b #hq #sq #sk #d #sq16
                    emask has_mask causal bi kvh group rows r0 scale
                    eQ eKg nw nkt) w i
            == fst (run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                      #b #hq #sq #sk #d #sq16
                      emask has_mask row_active causal bi qh qpos cbound scale
                      eQ eKg i nw w t));
    assert (acc2 (block_l #et_ab #et_acc #_f #_r #_s #_rb #_c1
                    #b #hq #sq #sk #d #sq16
                    emask has_mask causal bi kvh group rows r0 scale
                    eQ eKg nw nkt) w i
            == snd (run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                      #b #hq #sq #sk #d #sq16
                      emask has_mask row_active causal bi qh qpos cbound scale
                      eQ eKg i nw w t));
    warp_ml_state #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
      #b #hq #sq #sk #d #sq16
      emask has_mask row_active causal bi qh qpos kvh group rows r0 cbound
      scale eQ eKg i nw w nkt

#pop-options

(* ------------------------------------------------------------------ *)
(* The [P@V] tile cell is an exact real sum over the tile's keys.       *)
(* ------------------------------------------------------------------ *)

let subtile_approx
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#rows #cols : nat)
  (e : chest2 et rows cols) (r : chest2 real rows cols)
  (trows : pos { trows /?+ rows }) (tcols : pos { tcols /?+ cols })
  (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
  : Lemma (requires e %~ r)
          (ensures Kuiper.EMatrix.Tiling.ematrix_subtile e trows tcols tr tc
                   %~ Kuiper.EMatrix.Tiling.ematrix_subtile r trows tcols tr tc)
  = ()

(* The real matmul's inner accumulation is a plain masked sum. *)
let rec matmul_single_sum
  (#rows #shared #cols : nat)
  (m1 : chest2 real rows shared) (m2 : chest2 real shared cols)
  (row : natlt rows) (col : natlt cols) (to : natle shared)
  : Lemma (ensures MS.__matmul_single m1 m2 row col to
                   == SO.sum_upto #shared
                        (fun k -> acc2 m1 row k *. acc2 m2 k col) SO.ptrue to)
          (decreases to)
  = if to = 0 then assert (MS.__matmul_single m1 m2 row col 0 == 0.0R)
    else begin
      matmul_single_sum m1 m2 row col (to - 1);
      assert (MS.__matmul_single m1 m2 row col to
              == MS.__matmul_single m1 m2 row col (to - 1)
                 +. (acc2 m1 row (to - 1) *. acc2 m2 (to - 1) col))
    end

let matmul_cell_sum
  (#rows #shared #cols : nat)
  (m1 : chest2 real rows shared) (m2 : chest2 real shared cols)
  (row : natlt rows) (col : natlt cols)
  : Lemma (acc2 (MS.matmul m1 m2) row col
           == SO.sum_where #shared
                (fun k -> acc2 m1 row k *. acc2 m2 k col) SO.ptrue)
  = MS.lemma_matmul_index m1 m2 row col;
    matmul_single_sum m1 m2 row col shared

(* Column [c] of a [16]-wide output chunk lives in chunk [c / 16]. *)
let chunk_of_col (d : pos) (#dd16 : squash (16 /?+ d)) (c : natlt d)
  : Lemma (SF.clamp_nat (d / 16) (c / 16) == c / 16 /\
           (c / 16) * 16 + c % 16 == c)
  = FStar.Math.Lemmas.lemma_div_mod c 16;
    FStar.Math.Lemmas.lemma_div_mod d 16;
    if c / 16 >= d / 16
    then FStar.Math.Lemmas.lemma_mult_le_left 16 (d / 16) (c / 16)
    else ()

(* Cell [(r, c)] of the tile's [P@V] product approximates the exact real dot
   product of probability row [r] with value column [c] of the tile. *)
let pv_cell_approx
  (#et_ab #et_acc : Type0)
  {| _sa : scalar et_ab |} {| _ra : real_like et_ab |}
  {| _sc : scalar et_acc |} {| _rc : real_like et_acc |}
  (#d : pos) (#dd16 : squash (16 /?+ d))
  (eP : chest2 et_ab 16 16) (eV : chest2 et_ab 16 d)
  (rP : chest2 real 16 16) (rV : chest2 real 16 d)
  (r : natlt 16) (c : natlt d)
  : Lemma (requires eP %~ rP /\ eV %~ rV)
          (ensures acc2 (SF.pv_chunk #et_ab #et_acc eP eV
                           (SF.clamp_nat (d / 16) (c / 16))) r (c % 16)
                   %~ SO.sum_where #16
                        (fun k -> acc2 rP r k *. acc2 rV k c) SO.ptrue)
  = chunk_of_col d #dd16 c;
    let cb : natlt (d / 16) = c / 16 in
    let cc : natlt 16 = c % 16 in
    subtile_approx eV rV 16 16 0 cb;
    BM.emma_chain_approx #et_ab #et_acc #_sa #_ra #_sc #_rc #16 #16 #16 16
      eP (Kuiper.EMatrix.Tiling.ematrix_subtile eV 16 16 0 cb)
      rP (Kuiper.EMatrix.Tiling.ematrix_subtile rV 16 16 0 cb);
    matmul_cell_sum rP (Kuiper.EMatrix.Tiling.ematrix_subtile rV 16 16 0 cb)
      r cc;
    SO.sum_where_ext #16
      (fun k -> acc2 rP r k
                *. acc2 (Kuiper.EMatrix.Tiling.ematrix_subtile rV 16 16 0 cb)
                     k cc)
      (fun k -> acc2 rP r k *. acc2 rV k c) SO.ptrue SO.ptrue;
    assert (acc (BM.emma_chain #et_ab #et_acc 16 eP
                   (Kuiper.EMatrix.Tiling.ematrix_subtile eV 16 16 0 cb) 1)
                ((r, (cc, ())) <: Kuiper.Shape.abs (16 @| 16 @| INil))
            %~ acc (MS.matmul rP
                      (Kuiper.EMatrix.Tiling.ematrix_subtile rV 16 16 0 cb))
                   ((r, (cc, ())) <: Kuiper.Shape.abs (16 @| 16 @| INil)))
