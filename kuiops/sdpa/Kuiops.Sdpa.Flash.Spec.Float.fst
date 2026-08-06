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
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling

module FC = Kuiper.Float.Casts
module BM = Kuiops.Common.BlockMatmul

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

(* [scale]'s in-place row-broadcast multiply on one lane's [(span, 16)] stride
   sub-tile of the output tile: cell [(a, b)] lives on tile row [span * a + tr],
   so it is scaled by that row's correction weight. *)
let scale_subtile
  (#et : Type0) {| scalar et |} (#nr #nc : nat)
  (eO : chest2 et nr nc) (ecw : chest1 et 16) (span tr : nat)
  : GTot (chest2 et nr nc)
  = mk2 (fun a bb -> mul (acc2 eO a bb) (acc1 ecw (clamp_nat 16 (span * a + tr))))

(* Partial progress of that multiply: rows before [r] are done, row [r] is done
   up to column [n], the rest is untouched. *)
let scale_part
  (#et : Type0) {| scalar et |} (#nr #nc : nat)
  (eO0 eO : chest2 et nr nc) (ecw : chest1 et 16) (span tr : nat) (r n : nat)
  : prop
  = forall (a : natlt nr) (bb : natlt nc).
      acc2 eO a bb ==
        (if a < r || (a = r && bb < n)
         then acc2 (scale_subtile eO0 ecw span tr) a bb
         else acc2 eO0 a bb)

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

(* The [bm x d] block of Q the block loads into shared memory, exactly as
   [Kuiops.Sdpa.Flash.KfBlock.sdpa_flash_q_load] writes it: row [i] is query
   [r0 + i] of the flattened (head, position) axis, and rows past the end of
   the sequence are zeroed. *)
let q_tile
  (#et : Type0) {| scalar et |}
  (#b : nat) (#hq #sq : pos) (#d : nat)
  (bm : nat) (rows : pos) (group : nat)
  (eQ : chest (b @| hq @| sq @| d @| INil) et)
  (bi : natlt b) (kvh : nat) (r0 : nat)
  : GTot (chest2 et bm d)
  = mk2 fun i dd ->
      let r = r0 + i in
      let rr = clamp_nat rows r in
      let qh = clamp_nat hq (kvh * group + rr / sq) in
      let qpos : natlt sq = rr % sq in
      if r < rows then acc4 eQ bi qh qpos dd else zero

(* ------------------------------------------------------------------ *)
(* The whole warp's view of one tile update.                           *)
(*                                                                     *)
(* [shS], [shP] and [shcw] are collectively owned between barriers, so *)
(* their values have to be named by expressions every lane can write   *)
(* down: the incoming register vectors [evm]/[evl] plus the raw score  *)
(* tile.                                                               *)
(* ------------------------------------------------------------------ *)

let erow (#et : Type0) (#r #c : nat) (e : chest2 et r c) (i : natlt r)
  : GTot (chest1 et c)
  = mk1 (fun j -> acc2 e i j)

let m_vec
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm : chest1 et_acc 16)
  : GTot (chest1 et_acc 16)
  = mk1 (fun i ->
      fmax (acc1 evm i)
        (row_max (tile_scores emask has_mask row_active causal bi qh qpos
                    k0 cbound scale (erow eS i)) 16))

let cw_vec
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm : chest1 et_acc 16)
  : GTot (chest1 et_acc 16)
  = mk1 (fun i ->
      fexp (acc1 evm i `sub`
            acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                    scale eS evm) i))

let l_vec
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm evl : chest1 et_acc 16)
  : GTot (chest1 et_acc 16)
  = mk1 (fun i ->
      (acc1 evl i `mul`
       acc1 (cw_vec emask has_mask row_active causal bi qh qpos k0 cbound
               scale eS evm) i)
      `add`
      row_sum (tile_scores emask has_mask row_active causal bi qh qpos
                 k0 cbound scale (erow eS i))
        (acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                 scale eS evm) i) 16)

let score_tile
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16)
  : GTot (chest2 et_acc 16 16)
  = mk2 (fun i j ->
      acc1 (tile_scores emask has_mask row_active causal bi qh qpos
              k0 cbound scale (erow eS i)) j)

let prob_tile
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm : chest1 et_acc 16)
  : GTot (chest2 et_ab 16 16)
  = mk2 (fun i j ->
      FC.fcast
        (sel_prob
          (acc1 (tile_scores emask has_mask row_active causal bi qh qpos
                   k0 cbound scale (erow eS i)) j)
          (acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                   scale eS evm) i)))

#push-options "--fuel 1 --ifuel 2 --z3rlimit 20"

(* [softmax_upd_post] for lane [i], read off the whole-warp descriptions. *)
let tile_upd_post
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm evl : chest1 et_acc 16) (i : natlt 16)
  : Lemma
      (softmax_upd_post emask has_mask row_active causal bi qh qpos k0 cbound
         scale (erow eS i) (acc1 evm i) (acc1 evl i)
         (erow (score_tile emask has_mask row_active causal bi qh qpos k0
                  cbound scale eS) i)
         (erow (prob_tile emask has_mask row_active causal bi qh qpos k0
                  cbound scale eS evm) i)
         (acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                  scale eS evm) i)
         (acc1 (l_vec emask has_mask row_active causal bi qh qpos k0 cbound
                  scale eS evm evl) i)
         (acc1 (cw_vec emask has_mask row_active causal bi qh qpos k0 cbound
                  scale eS evm) i))
  = row_max_ext
      (erow (score_tile emask has_mask row_active causal bi qh qpos k0 cbound
               scale eS) i)
      (tile_scores emask has_mask row_active causal bi qh qpos k0 cbound scale
         (erow eS i)) 16;
    row_sum_ext
      (erow (score_tile emask has_mask row_active causal bi qh qpos k0 cbound
               scale eS) i)
      (tile_scores emask has_mask row_active causal bi qh qpos k0 cbound scale
         (erow eS i))
      (acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
               scale eS evm) i) 16


(* The whole-warp descriptions are the *only* possible outcome of lane [i]'s
   update, so a [softmax_upd_post] hypothesis pins every primed value. *)
let tile_upd_det
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm evl : chest1 et_acc 16) (i : natlt 16)
  (es' : chest1 et_acc 16) (ep' : chest1 et_ab 16) (m' l' cw' : et_acc)
  : Lemma
      (requires softmax_upd_post emask has_mask row_active causal bi qh qpos
                  k0 cbound scale (erow eS i) (acc1 evm i) (acc1 evl i)
                  es' ep' m' l' cw')
      (ensures
        es' == erow (score_tile emask has_mask row_active causal bi qh qpos k0
                       cbound scale eS) i /\
        ep' == erow (prob_tile emask has_mask row_active causal bi qh qpos k0
                       cbound scale eS evm) i /\
        m' == acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                      scale eS evm) i /\
        l' == acc1 (l_vec emask has_mask row_active causal bi qh qpos k0 cbound
                      scale eS evm evl) i /\
        cw' == acc1 (cw_vec emask has_mask row_active causal bi qh qpos k0
                       cbound scale eS evm) i)
  = tile_upd_post emask has_mask row_active causal bi qh qpos k0 cbound scale
      eS evm evl i;
    scores_post_det emask has_mask row_active causal bi qh qpos k0 cbound scale
      (erow eS i) es';
    scores_post_det emask has_mask row_active causal bi qh qpos k0 cbound scale
      (erow eS i)
      (erow (score_tile emask has_mask row_active causal bi qh qpos k0 cbound
               scale eS) i);
    assert (equal ep'
              (erow (prob_tile emask has_mask row_active causal bi qh qpos k0
                       cbound scale eS evm) i))

#pop-options

(* Chunk [b] of the [P@V] product: the [16 x 16] tile [P @ V[:, 16b:16b+16]],
   accumulated by the tensor-core [emma] chain over its single 16-wide chunk. *)
unfold
let pv_chunk
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#hd : nat) (#_ : squash (16 /?+ hd))
  (eP : chest2 et_ab 16 16) (eV : chest2 et_ab 16 hd) (b : natlt (hd / 16))
  : GTot (chest2 et_acc 16 16)
= BM.emma_chain #et_ab #et_acc 16 eP (ematrix_subtile eV 16 16 0 b) 1

(* The whole [16 x d] output tile after one key-tile step: every row is
   rescaled by its correction weight, and the [P@V] product is added to it. *)
let out_tile
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#d : nat) (#_ : squash (16 /?+ d))
  (eO : chest2 et_acc 16 d) (ecw : chest1 et_acc 16)
  (eP : chest2 et_ab 16 16) (eV : chest2 et_ab 16 d)
  : GTot (chest2 et_acc 16 d)
= mk2 (fun r c -> add (mul (acc2 eO r c) (acc1 ecw r))
                      (acc2 (pv_chunk eP eV (clamp_nat (d / 16) (c / 16))) r (c % 16)))

(* Warp [w] of [nw] visits the key tiles [w], [w + nw], ... below [nkt], so it
   runs this many online-softmax steps. *)
let warp_iters (nw : pos) (nkt w : nat) : nat
= if w >= nkt then 0 else (nkt - w + nw - 1) / nw

(* The loop exit condition pins the step count to [warp_iters]. *)
let warp_iters_uniq (nw : pos) (nkt w t : nat)
  : Lemma (requires w + t * nw >= nkt /\ (t == 0 \/ w + (t - 1) * nw < nkt))
          (ensures t == warp_iters nw nkt w)
= if t = 0 then ()
  else begin
    let m = nkt - w in
    FStar.Math.Lemmas.distributivity_sub_left t 1 nw;
    FStar.Math.Lemmas.division_definition (m + nw - 1) nw t
  end

(* The [(maxpos, found)] pair the causal-bound scan holds after [n] rows of the
   query tile starting at [r0]. *)
let rec tile_maxpos (sq : pos) (rows r0 n : nat) : Tot (nat & bool) (decreases n)
= if n = 0 then (0, false)
  else
    let mf = tile_maxpos sq rows r0 (n - 1) in
    let r = r0 + n - 1 in
    let valid = r < rows in
    let take = valid && ((not (snd mf)) || r % sq > fst mf) in
    ((if take then r % sq else fst mf), snd mf || valid)

(* One past the largest key index the causal mask admits for the tile. *)
let causal_kmax (bm : nat) (sq : pos) (sk : nat { sq <= sk }) (rows r0 : nat) : nat
= let mf = tile_maxpos sq rows r0 bm in
  let e = (sk - sq) + (if snd mf then fst mf + 1 else 0) in
  if sk < e then sk else e

(* Number of key tiles the block sweeps. *)
let key_tiles
  (bn : pos) (bm : nat) (sq : pos) (sk : nat { sq <= sk })
  (rows r0 : nat) (causal : bool) : nat
= Kuiper.Divides.divup (if causal then causal_kmax bm sq sk rows r0 else sk) bn

(* Per-row parameters of the query tile starting at [r0]: row [i] of the tile is
   query row [r0 + i], which the kernel clamps into range before splitting it
   into a query head and a position. *)
let lane_active_row (rows r0 i : nat) : bool = r0 + i < rows

let lane_rr (rows r0 i : nat) : nat = if r0 + i < rows then r0 + i else 0

let lane_qpos (sq : pos) (rows r0 i : nat) : natlt sq = lane_rr rows r0 i % sq

let lane_qh (hq sq : pos) (kvh group rows r0 i : nat) : natlt hq
= let q = kvh * group + lane_rr rows r0 i / sq in
  if q < hq then q else 0

let lane_cbound (sq : pos) (sk rows r0 i : nat) : nat
= lane_qpos sq rows r0 i + (if sk >= sq then sk - sq else 0)


(* The row parameters a lane computes for its own row of the query tile. *)
let lane_params_ok (hq sq : pos) (sk kvh group rows r0 i : nat)
  (row_active : bool) (qh qpos cbound : nat) : prop
= row_active == lane_active_row rows r0 i /\
  qh == lane_qh hq sq kvh group rows r0 i /\
  qpos == lane_qpos sq rows r0 i /\
  cbound == lane_cbound sq sk rows r0 i

(* Entry [i] of the register vectors reads only entry [i] of the incoming
   registers. *)
#push-options "--fuel 1 --ifuel 2"

let vec_local
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (k0 cbound : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16)
  (evm1 evm2 evl1 evl2 : chest1 et_acc 16) (i : natlt 16)
  : Lemma (requires acc1 evm1 i == acc1 evm2 i /\ acc1 evl1 i == acc1 evl2 i)
          (ensures
            acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                    scale eS evm1) i
            == acc1 (m_vec emask has_mask row_active causal bi qh qpos k0 cbound
                       scale eS evm2) i /\
            acc1 (cw_vec emask has_mask row_active causal bi qh qpos k0 cbound
                    scale eS evm1) i
            == acc1 (cw_vec emask has_mask row_active causal bi qh qpos k0 cbound
                       scale eS evm2) i /\
            acc1 (l_vec emask has_mask row_active causal bi qh qpos k0 cbound
                    scale eS evm1 evl1) i
            == acc1 (l_vec emask has_mask row_active causal bi qh qpos k0 cbound
                       scale eS evm2 evl2) i)
= ()

#pop-options

(* ------------------------------------------------------------------ *)
(* Whole-tile descriptions.                                            *)
(*                                                                     *)
(* [m_vec] and friends apply one row's parameters to all 16 rows, which *)
(* is only meaningful at that row.  A warp barrier's predicate family   *)
(* must be the same for every lane, so anything pinned across one is    *)
(* phrased with these row-indexed versions instead: row [i] uses row    *)
(* [i]'s own query head, position and causal bound.                     *)
(* ------------------------------------------------------------------ *)

let m_vec_t
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat)
  (k0 : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm : chest1 et_acc 16)
  : GTot (chest1 et_acc 16)
= mk1 (fun i ->
    acc1 (m_vec emask has_mask (lane_active_row rows r0 i) causal bi
            (lane_qh hq sq kvh group rows r0 i) (lane_qpos sq rows r0 i)
            k0 (lane_cbound sq sk rows r0 i) scale eS evm) i)

let cw_vec_t
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat)
  (k0 : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm : chest1 et_acc 16)
  : GTot (chest1 et_acc 16)
= mk1 (fun i ->
    acc1 (cw_vec emask has_mask (lane_active_row rows r0 i) causal bi
            (lane_qh hq sq kvh group rows r0 i) (lane_qpos sq rows r0 i)
            k0 (lane_cbound sq sk rows r0 i) scale eS evm) i)

let l_vec_t
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat)
  (k0 : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm evl : chest1 et_acc 16)
  : GTot (chest1 et_acc 16)
= mk1 (fun i ->
    acc1 (l_vec emask has_mask (lane_active_row rows r0 i) causal bi
            (lane_qh hq sq kvh group rows r0 i) (lane_qpos sq rows r0 i)
            k0 (lane_cbound sq sk rows r0 i) scale eS evm evl) i)

let score_tile_t
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat)
  (k0 : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16)
  : GTot (chest2 et_acc 16 16)
= mk2 (fun i j ->
    acc2 (score_tile emask has_mask (lane_active_row rows r0 i) causal bi
            (lane_qh hq sq kvh group rows r0 i) (lane_qpos sq rows r0 i)
            k0 (lane_cbound sq sk rows r0 i) scale eS) i j)

let prob_tile_t
  (#et_acc #et_ab : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (#b : nat) (#hq #sq #sk : pos)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat)
  (k0 : nat) (scale : et_acc)
  (eS : chest2 et_acc 16 16) (evm : chest1 et_acc 16)
  : GTot (chest2 et_ab 16 16)
= mk2 (fun i j ->
    acc2 (prob_tile emask has_mask (lane_active_row rows r0 i) causal bi
            (lane_qh hq sq kvh group rows r0 i) (lane_qpos sq rows r0 i)
            k0 (lane_cbound sq sk rows r0 i) scale eS evm) i j)

(* Two tiles that agree on row [i] have the same row [i], in either of the two
   shapes the kernel uses for a row. *)
let tile_row_eq
  (#et : Type0) (#c : pos) (e1 e2 : chest2 et 16 c) (i : natlt 16)
  : Lemma (requires forall (j : natlt c). acc2 e1 i j == acc2 e2 i j)
          (ensures ematrix_subtile e1 1 c i 0 == ematrix_subtile e2 1 c i 0 /\
                   erow e1 i == erow e2 i)
= assert (equal (ematrix_subtile e1 1 c i 0) (ematrix_subtile e2 1 c i 0));
  assert (equal (erow e1 i) (erow e2 i))

(* ---------------------------------------------------------------------- *)
(* Cross-warp combine.                                                     *)
(*                                                                         *)
(* Each warp published a partial maximum and denominator for every row of  *)
(* the query tile.  Warp 0 folds them into a block-wide maximum, rescales  *)
(* each warp's denominator to that maximum, and sums the result.           *)
(* ---------------------------------------------------------------------- *)

(* Block-wide maximum for row [i] after folding the first [n] warps.  The
   kernel seeds the fold with [neg infinity], which is safe here because it is
   only ever compared, never exponentiated. *)
let rec gmax
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (i : natlt bm) (n : nat { n <= nw })
  : GTot et (decreases n)
= if n = 0 then neg infinity
  else fmax (gmax eM i (n - 1)) (acc2 eM (n - 1) i)

(* The factor that rescales warp [w]'s partial state for row [i] from its own
   maximum to the block-wide one [gm]. *)
let gscale
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (gm : et) (w : natlt nw) (i : natlt bm) : GTot et
= fexp (acc2 eM w i `sub` gm)

(* Block-wide denominator for row [i] after folding the first [n] warps. *)
let rec gsum
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM eL : chest2 et nw bm) (gm : et) (i : natlt bm) (n : nat { n <= nw })
  : GTot et (decreases n)
= if n = 0 then zero
  else add (gsum eM eL gm i (n - 1))
           (mul (gscale eM gm (n - 1) i) (acc2 eL (n - 1) i))

(* The [nw x 1] column of rescaling factors the combine writes for row [i]. *)
let gscale_col
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (i : natlt bm) : GTot (chest2 et nw 1)
= mk2 (fun w _ -> gscale eM (gmax eM i nw) w i)

#push-options "--fuel 1 --ifuel 2"
let gscale_col_ext
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (i : natlt bm) (e : chest2 et nw 1)
  : Lemma
      (requires forall (w : natlt nw). acc2 e w 0 == gscale eM (gmax eM i nw) w i)
      (ensures e == gscale_col eM i)
= assert (equal e (gscale_col eM i))
#pop-options

(* ---------------------------------------------------------------------- *)
(* Epilogue.                                                              *)
(*                                                                        *)
(* Warp 0 folds the per-warp output tiles into the block's output tile,    *)
(* weighting warp [w]'s row [i] by the rescaling factor published for it,  *)
(* and normalises by the block-wide denominator. *)
(* ---------------------------------------------------------------------- *)

let ocomb_row (bm : pos) (w i : nat) : nat = bm * w + i

let ocomb_row_lt (nw bm : pos) (w : natlt nw) (i : natlt bm)
  : Lemma (ocomb_row bm w i < nw * bm)
= FStar.Math.Lemmas.lemma_mult_le_right bm (w + 1) nw

(* The escale-weighted sum of the first [n] warps' contributions to column
   [dd] of row [i] of the block output tile. *)
let rec ocomb
  (#et : Type0) {| floating et |} (#nw #bm #d : pos)
  (escale : chest2 et nw bm) (eO : chest2 et (nw * bm) d)
  (i : natlt bm) (dd : natlt d) (n : nat { n <= nw })
  : GTot et (decreases n)
= if n = 0 then zero
  else
    (ocomb_row_lt nw bm (n - 1) i;
     add (ocomb escale eO i dd (n - 1))
         (mul (acc2 escale (n - 1) i) (acc2 eO (ocomb_row bm (n - 1) i) dd)))

(* The reciprocal of the block-wide denominator, with the empty-row case
   (every key masked out) mapped to zero rather than to a division by zero. *)
inline_for_extraction noextract
let onorm (#et : Type0) {| floating et |} (l : et) : et
= if l `gt` zero then one `div` l else zero

(* The accumulator-typed value the epilogue writes for row [i], column [dd]. *)
let out_val
  (#et : Type0) {| floating et |} (#nw #bm #d : pos)
  (escale : chest2 et nw bm) (eO : chest2 et (nw * bm) d)
  (egl : chest1 et bm)
  (i : natlt bm) (dd : natlt d) : GTot et
= mul (ocomb escale eO i dd nw) (onorm (acc1 egl i))

let ocomb_step
  (#et : Type0) {| floating et |} (#nw #bm #d : pos)
  (escale : chest2 et nw bm) (eO : chest2 et (nw * bm) d)
  (i : natlt bm) (dd : natlt d) (w : natlt nw)
  : Lemma
      (ocomb_row bm w i < nw * bm /\
       ocomb escale eO i dd (w + 1)
       == add (ocomb escale eO i dd w)
              (mul (acc2 escale w i) (acc2 eO (ocomb_row bm w i) dd)))
= ocomb_row_lt nw bm w i
