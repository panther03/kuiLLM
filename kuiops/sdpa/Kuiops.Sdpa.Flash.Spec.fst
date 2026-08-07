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
   [masked_denom] returns a strictly positive real. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Seq.Common

module MS = Kuiper.Spec.GEMM
module SM = Kuiper.Spec.Softmax

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
