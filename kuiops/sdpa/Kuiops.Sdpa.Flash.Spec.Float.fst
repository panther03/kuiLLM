module Kuiops.Sdpa.Flash.Spec.Float

(* Float-level description of the per-lane online-softmax update performed by
   [Kuiops.Sdpa.Flash.KfSub.sdpa_flash_softmax_upd].

   Everything here is a pure function of the values the kernel reads, phrased
   in the element type of the accumulator rather than in reals.  The Pulse
   postconditions are stated against these definitions, so the separation-logic
   development never mentions reals; the bridge to the real-valued spec
   ([Kuiops.Sdpa.Flash.Spec] via [Kuiops.Sdpa.Flash.Spec.Online]) is a separate,
   purely functional argument. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Floating
open Kuiper.Shape

module FC = Kuiper.Float.Casts

(* Clamp a key index into [0, n) so a mask read is unconditionally in bounds.
   The clamped value is only ever used when it is the identity. *)
let clamp_nat (n : pos) (x : nat) : natlt n = if x < n then x else 0

(* Select-to-zero probability: a masked score carries the [-inf] sentinel and
   maps to the literal zero, never to [exp (-inf - m)].  See the header of
   kuipy/unverified/flash_attn_fa1.cu. *)
inline_for_extraction noextract
let sel_prob (#et : Type0) {| floating et |} (sv mnew : et) : et =
  if eq sv (neg infinity) then zero else fexp (sv `sub` mnew)

(* Whether key [kj] contributes to query row [qpos]: the row must exist, the
   key must be in range, and under [causal] it must not lie past [cbound]. *)
let key_ok (row_active causal : bool) (sk cbound kj : nat) : bool =
  row_active && not (causal && kj > cbound) && kj < sk

(* The score the kernel writes into the tile for one key. *)
inline_for_extraction noextract
let score_upd
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (scale sc : et_acc) (mv : et_ab) (ok : bool)
  : et_acc
  = if ok then (sc `mul` scale) `add` (FC.fcast mv) else neg infinity

(* The additive mask bias for one (row, key) pair, or [zero] when the caller
   supplied no mask tensor. *)
let mask_bias
  (#et_ab : Type0) {| scalar et_ab |}
  (#b #hq #sq #sk : nat)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq) (kjb : natlt sk)
  : GTot et_ab
  = if has_mask then acc4 emask bi qh qpos kjb else zero

(* ------------------------------------------------------------------ *)
(* Row folds, in the exact shape of the kernel's two loops.            *)
(* ------------------------------------------------------------------ *)

(* [fmax] folded left from [-inf] over the first [k] scores. *)
let rec row_max
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (k : nat { k <= bn })
  : GTot et (decreases k)
  = if k = 0 then neg infinity
    else fmax (row_max s (k - 1)) (acc1 s (k - 1))

(* The select-to-zero probabilities summed left from [zero]. *)
let rec row_sum
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (mnew : et) (k : nat { k <= bn })
  : GTot et (decreases k)
  = if k = 0 then zero
    else (row_sum s mnew (k - 1)) `add` (sel_prob (acc1 s (k - 1)) mnew)

let rec row_max_ext
  (#et : Type0) {| floating et |} (#bn : nat)
  (s1 s2 : chest1 et bn) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). t < k ==> acc1 s1 t == acc1 s2 t)
          (ensures row_max s1 k == row_max s2 k)
          (decreases k)
  = if k = 0 then () else row_max_ext s1 s2 (k - 1)

let rec row_sum_ext
  (#et : Type0) {| floating et |} (#bn : nat)
  (s1 s2 : chest1 et bn) (mnew : et) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). t < k ==> acc1 s1 t == acc1 s2 t)
          (ensures row_sum s1 mnew k == row_sum s2 mnew k)
          (decreases k)
  = if k = 0 then () else row_sum_ext s1 s2 mnew (k - 1)

(* ------------------------------------------------------------------ *)
(* The update, packaged.                                               *)
(* ------------------------------------------------------------------ *)

(* The score row after masking and biasing.  [k0] is the first key of the tile;
   [es] holds the raw Q@K^T dot products. *)
let scores_post
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#bn : nat)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat)
  (scale : et_acc)
  (es es' : chest1 et_acc bn)
  : prop
  = forall (j : natlt bn).
      acc1 es' j ==
        score_upd scale (acc1 es j)
          (mask_bias emask has_mask bi qh qpos (clamp_nat sk (k0 + j)))
          (key_ok row_active causal sk cbound (k0 + j))

(* The complete effect of one lane's online-softmax update: primed values are
   the post-state of the score tile, the probability tile, and the running
   max / denominator / correction registers. *)
let softmax_upd_post
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos) (#bn : nat)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat)
  (scale : et_acc)
  (es : chest1 et_acc bn) (vm vl : et_acc)
  (es' : chest1 et_acc bn) (ep' : chest1 et_ab bn) (m' l' cw' : et_acc)
  : prop
  = scores_post emask has_mask row_active causal bi qh qpos k0 cbound scale es es' /\
    m' == fmax vm (row_max es' bn) /\
    cw' == fexp (vm `sub` m') /\
    (forall (j : natlt bn). acc1 ep' j == FC.fcast (sel_prob (acc1 es' j) m')) /\
    l' == (vl `mul` cw') `add` (row_sum es' m' bn)

(* The score row after masking and biasing, as a function.  [scores_post] says
   exactly that the kernel wrote this row. *)
let tile_scores
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#bn : nat)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat)
  (scale : et_acc)
  (es : chest1 et_acc bn)
  : GTot (chest1 et_acc bn)
  = mk1 (fun j ->
      score_upd scale (acc1 es j)
        (mask_bias emask has_mask bi qh qpos (clamp_nat sk (k0 + j)))
        (key_ok row_active causal sk cbound (k0 + j)))

let scores_post_det
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#bn : nat)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat)
  (scale : et_acc)
  (es es' : chest1 et_acc bn)
  : Lemma (requires scores_post emask has_mask row_active causal bi qh qpos
                      k0 cbound scale es es')
          (ensures es' == tile_scores emask has_mask row_active causal bi qh qpos
                            k0 cbound scale es)
  = assert (equal es'
              (tile_scores emask has_mask row_active causal bi qh qpos
                 k0 cbound scale es))

(* The [bn x d] slice of a [sk x d] key/value matrix starting at key [k0], with
   out-of-range rows clamped to row 0 (the kernel's bounds clamp; those rows are
   masked out of the softmax and never contribute). *)
let kv_tile
  (#et : Type0) (#sk : pos) (#d : nat) (bn : nat)
  (e : chest2 et sk d) (k0 : nat)
  : chest2 et bn d
  = mk2 fun r c -> acc2 e (clamp_nat sk (k0 + r)) c
