module Kuiops.Sdpa.Flash.Spec

(* Real-valued functional specification of the fused SDPA kernel in
   [Kuiops.Sdpa.Flash].

   The kernel computes, for every batch [bi], query head [qh] and query
   position [i]:

     out[bi, qh, i, :] = softmax_j (scale * <Q[bi, qh, i, :], K[bi, kvh, j, :]>
                                    + bias[bi, qh, i, j])  @  V[bi, kvh, :, :]

   where [kvh = qh / group] is the grouped-query-attention key/value head,
   [bias] is the additive mask when [has_mask] and zero otherwise, and the
   softmax runs only over the keys [j] admitted by the causal mask.

   This mirrors [Kuiper.Spec.Attention], with three differences:

   - K is stored non-transposed ([sk x d]), so the score matmul transposes it
     here rather than taking a pre-transposed argument.
   - Query and key/value heads differ (GQA): [hq = hkv * group].
   - Keys can be masked out.  Kuiper has no extended reals, so a masked key is
     not modelled by a [-inf] score; instead the softmax is taken over the
     subset of admitted keys, and the probability of a masked key is exactly
     [0.0R].  This is the mathematical limit of what the kernel computes, which
     writes the [-inf] float sentinel into the score and thus [exp(-inf - m) =
     0] into the probability.

   The masked softmax divides by a sum over the admitted keys only, so its
   well-definedness rests on every row admitting at least one key.  That is
   enforced in the types: [valid_pred] carries the non-emptiness proof and
   [masked_denom] returns a strictly positive real.

   This module also states the kernel's PRECONDITION, [sdpa_flash_finite],
   at the bottom.  Together with [Kuiops.Sdpa.Flash.fsti] and
   [Kuiops.FloatAxioms] it is everything one has to read to know what
   [Kuiops.Sdpa.Flash.sdpa_flash_async] assumes and guarantees. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Shape
open Kuiper.Floating
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Seq.Common

module MS = Kuiper.Spec.GEMM
module SM = Kuiper.Spec.Softmax
module FC = Kuiper.Float.Casts
module BM = Kuiops.Common.BlockMatmul
module SF = Kuiops.Sdpa.Flash.Spec.Float

(* ------------------------------------------------------------------ *)
(* Positivity of a sum of non-negative reals with one positive entry.  *)
(* ------------------------------------------------------------------ *)

let nonneg_seq = s:Seq.seq real { seq_forall (fun x -> x >=. 0.0R) s }

let rec rsum_ge_acc (s : nonneg_seq) (acc : real)
  : Lemma (ensures seq_fold_left (+.) acc s >=. acc)
          (decreases Seq.length s)
  = match view_seq s with
    | SNil -> ()
    | SCons hd tl ->
      assert (Seq.length tl == Seq.length s - 1);
      assert (forall (i : natlt (Seq.length tl)). tl @! i == s @! (i + 1));
      rsum_ge_acc tl (acc +. hd)

let rec rsum_gt_acc (s : nonneg_seq) (acc : real) (k : natlt (Seq.length s))
  : Lemma (requires s @! k >. 0.0R)
          (ensures seq_fold_left (+.) acc s >. acc)
          (decreases Seq.length s)
  = let SCons hd tl = view_seq s in
    assert (Seq.length tl == Seq.length s - 1);
    assert (forall (i : natlt (Seq.length tl)). tl @! i == s @! (i + 1));
    if k = 0
    then rsum_ge_acc tl (acc +. hd)
    else rsum_gt_acc tl (acc +. hd) (k - 1)

let chest1_rsum_pos (#n : pos) (c : chest1 real n) (k : natlt n)
  : Lemma (requires (forall (j : natlt n). acc1 c j >=. 0.0R) /\ acc1 c k >. 0.0R)
          (ensures chest1_rsum c >. 0.0R)
  = rsum_gt_acc (chest1_to_seq c) 0.0R k

(* ------------------------------------------------------------------ *)
(* Masked softmax.                                                     *)
(* ------------------------------------------------------------------ *)

(* A key-admission predicate over [n] keys, with at least one admitted key --
   exactly the side condition that makes the masked softmax well defined. *)
let valid_pred (n : pos) = f:(natlt n -> bool) { exists (j : natlt n). f j }

let valid_all (n : pos) : valid_pred n =
  let f : natlt n -> bool = fun _ -> true in
  assert (f 0);
  f

(* Unnormalized probabilities: [exp] of the admitted scores, [0] elsewhere. *)
let masked_exps (#n : pos) (valid : natlt n -> bool) (r : chest1 real n)
  : chest1 real n
  = mk1 (fun j -> if valid j then exp (acc1 r j) else 0.0R)

let masked_denom_pos (#n : pos) (valid : valid_pred n) (r : chest1 real n)
  : Lemma (chest1_rsum (masked_exps valid r) >. 0.0R)
  = eliminate exists (j : natlt n). valid j
    returns chest1_rsum (masked_exps valid r) >. 0.0R
    with _. chest1_rsum_pos (masked_exps valid r) j

let masked_denom (#n : pos) (valid : valid_pred n) (r : chest1 real n)
  : GTot (z:real { z >. 0.0R })
  = masked_denom_pos valid r;
    chest1_rsum (masked_exps valid r)

(* Softmax over the admitted keys of a row; masked-out keys get probability
   [0].  With [valid] constantly true this is [Kuiper.Spec.Softmax.softmax_real]. *)
let masked_softmax_real (#n : pos) (valid : valid_pred n) (r : chest1 real n)
  : GTot (chest1 real n)
  = let z = masked_denom valid r in
    mk1 (fun j -> if valid j then exp (acc1 r j) /. z else 0.0R)

let masked_softmax_real_all (#n : pos) (r : chest1 real n)
  : Lemma (masked_softmax_real (valid_all n) r == SM.softmax_real r)
  = assert (equal (masked_exps (valid_all n) r) (chest_map exp r));
    assert (equal (masked_softmax_real (valid_all n) r) (SM.softmax_real r))

(* ------------------------------------------------------------------ *)
(* The causal mask.                                                    *)
(* ------------------------------------------------------------------ *)

(* Bottom-right aligned causal mask (the kernel's [cbound = qpos + (sk - sq)]):
   query position [qpos] attends to keys [0 .. qpos + (sk - sq)].  Stated
   without subtraction, so it is well formed for any [sq], [sk]. *)
let key_valid (causal : bool) (sq sk : pos) (qpos : natlt sq) (j : natlt sk) : bool
  = not causal || j + sq <= qpos + sk

(* When [sq <= sk] every query position admits at least key [0]. *)
let row_keys (causal : bool) (sq : pos) (sk : pos { sq <= sk }) (qpos : natlt sq)
  : valid_pred sk
  = assert (key_valid causal sq sk qpos 0);
    key_valid causal sq sk qpos

(* ------------------------------------------------------------------ *)
(* Attention on a single (batch, query head) page.                     *)
(* ------------------------------------------------------------------ *)

(* Pre-softmax scores: [scale * (Q K^T) + bias].  K is [sk x d], so it is
   transposed here; the [bias] is added after scaling, as the kernel does. *)
let attn_scores
  (#sq #sk #d : pos)
  (rQ : chest2 real sq d)
  (rK : chest2 real sk d)
  (bias : chest2 real sq sk)
  (scale : real)
  : chest2 real sq sk
  = chest_comb (fun bias_qk score -> bias_qk +. (score *. scale))
      bias (MS.matmul rQ (mtranspose rK))

(* Row-wise masked softmax of the scores. *)
let attn_probs
  (#sq : pos) (#sk : pos { sq <= sk })
  (causal : bool)
  (scores : chest2 real sq sk)
  : GTot (chest2 real sq sk)
  = mk2 (fun i j ->
      acc1 (masked_softmax_real (row_keys causal sq sk i) (chest2_row scores i)) j)

let attention_page_real
  (#sq : pos) (#sk : pos { sq <= sk }) (#d #dv : pos)
  (rQ : chest2 real sq d)
  (rK : chest2 real sk d)
  (rV : chest2 real sk dv)
  (bias : chest2 real sq sk)
  (scale : real)
  (causal : bool)
  : GTot (chest2 real sq dv)
  = MS.matmul (attn_probs causal (attn_scores rQ rK bias scale)) rV

(* ------------------------------------------------------------------ *)
(* Top-level batched, grouped-query spec.                              *)
(* ------------------------------------------------------------------ *)

(* The additive bias: the mask tensor when [has_mask], zero otherwise (the
   kernel skips the mask read and biases by [zero]). *)
let attn_bias
  (#b #hq #sq #sk : pos)
  (has_mask : bool)
  (rmask : chest4 real b hq sq sk)
  : chest4 real b hq sq sk
  = if has_mask then rmask else mk4 (fun _ _ _ _ -> 0.0R)

(* Query head [qh] reads the key/value head [qh / group]. *)
let kv_head (#hkv : pos) (group : pos) (qh : natlt (hkv * group)) : natlt hkv
  = let q = qh / group in
    FStar.Math.Lemmas.lemma_div_mod qh group;
    if q >= hkv then FStar.Math.Lemmas.lemma_mult_le_left group hkv q else ();
    q

let sdpa_flash_real
  (#b #hq #hkv : pos)
  (group : pos { hq == hkv * group })
  (#sq : pos) (#sk : pos { sq <= sk }) (#d : pos)
  (rQ : chest4 real b hq sq d)
  (rK : chest4 real b hkv sk d)
  (rV : chest4 real b hkv sk d)
  (rmask : chest4 real b hq sq sk)
  (scale : real)
  (causal : bool)
  (has_mask : bool)
  : GTot (chest4 real b hq sq d)
  = mk4 (fun bi qh i j ->
      let kvh : natlt hkv = kv_head #hkv group qh in
      acc2 (attention_page_real
              (slice_page4 rQ bi qh)
              (slice_page4 rK bi kvh)
              (slice_page4 rV bi kvh)
              (slice_page4 (attn_bias has_mask rmask) bi qh)
              scale causal)
           i j)

(* ------------------------------------------------------------------ *)
(* The precondition.                                                   *)
(* ------------------------------------------------------------------ *)

(* The key/value page for batch [bi] and key/value head [kvh]. *)
let page_kv
  (#et : Type0) (#b #hkv #sk #d : nat)
  (eK : chest (b @| hkv @| sk @| d @| INil) et)
  (bi : natlt b) (kvh : natlt hkv)
  : chest2 et sk d
  = chest_slice 0 kvh (chest_slice 0 bi eK)

(* The float score the kernel computes for query row [(bi, qh, qpos)] --
   row [i] of the query tile [eQt] -- and key [k0 + j] of the key tile at
   [k0]: the tensor-core [Q K^T] entry, scaled, plus the additive mask.

   This is the kernel's own arithmetic, not the exact real score: the dot
   product is accumulated over [d / 16] tensor-core fragments, in that order,
   at [et_acc] precision. *)
let flash_score
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#_ : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab) (has_mask : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq) (scale : et_acc)
  (eQt : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (i : natlt 16) (k0 : nat) (j : natlt 16)
  : GTot et_acc
  = (acc2 (BM.emma_chain #et_ab #et_acc 16 eQt
             (mtranspose (SF.kv_tile 16 eKg k0)) (d / 16)) i j `mul` scale)
    `add` FC.fcast (SF.mask_bias emask has_mask bi qh qpos
                      (SF.clamp_nat sk (k0 + j)))

(* Row [r] of the block that serves key/value head [kvh] belongs to query
   head [kvh * group + r / sq]. *)
let query_head
  (#hq #hkv #sq : pos) (group : pos { hq == hkv * group })
  (rows : pos { rows == group * sq })
  (kvh : natlt hkv) (r : natlt rows) : natlt hq
  = FStar.Math.Lemmas.euclidean_division_definition r sq;
    if r / sq >= group
    then FStar.Math.Lemmas.lemma_mult_le_right sq group (r / sq) else ();
    FStar.Math.Lemmas.distributivity_add_left kvh 1 group;
    FStar.Math.Lemmas.lemma_mult_le_right group (kvh + 1) hkv;
    kvh * group + r / sq

(* THE PRECONDITION of [Kuiops.Sdpa.Flash.sdpa_flash_async].

   Every score the kernel computes for a query row and a key that row attends
   to is a finite float: neither [Q K^T], nor the scaling, nor the additive
   mask overflows.  That is the whole assumption.  In particular nothing is
   assumed about the kernel's running state, and nothing about the softmax
   denominator: it is positive by construction, because every query row
   attends to at least key [0].

   The block that serves batch [bi] and key/value head [kvh] holds [rows]
   query rows; row [r] of that block is query head [kvh * group + r / sq] at
   query position [r % sq].  It reaches key [k] through the query tile
   starting at [r / 16 * 16], as lane [r % 16], and the key tile starting at
   [k0]; under [causal] it attends to keys up to [r % sq + (sk - sq)].

   Everything this mentions is a function of the kernel's arguments alone.
   The helpers it borrows -- [Spec.Float.q_tile], [Spec.Float.kv_tile],
   [Spec.Float.mask_bias], [Spec.Float.clamp_nat] and
   [Common.BlockMatmul.emma_chain] -- are all transparent one-liners over
   those arguments. *)
let sdpa_flash_finite
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (#b #hq #hkv : pos) (group : pos { hq == hkv * group })
  (#sq : pos) (#sk : pos { sq <= sk }) (#d : pos) (#_ : squash (16 /?+ d))
  (rows : pos { rows == group * sq })
  (eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (eK : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  : prop
  = forall (bi : natlt b) (kvh : natlt hkv) (r : natlt rows)
      (k0 : nat { k0 <= sk }) (j : natlt 16).
      (let qh : natlt hq = query_head #hq #hkv #sq group rows kvh r in
       let qpos : natlt sq = r % sq in
       let k : nat = k0 + j in
       k < sk /\ (causal ==> k <= qpos + (sk - sq)) ==>
       Finite? (kind (flash_score emask has_mask bi qh qpos scale
                        (SF.q_tile 16 rows group eQ bi kvh (r / 16 * 16))
                        (page_kv eK bi kvh) (r % 16) k0 j)))
