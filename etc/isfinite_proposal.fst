module Isfinite_proposal

(* PROPOSED: an extractable [isfinite] on [Kuiper.Floating.Base.floating].

   This is the answer to "what's the pre/postcondition you want for that
   function, and what changes once I have it?".

   Part 1 is the proposed addition, written in the exact shape it would take
   as a field of [floating].
   Part 2 machine-checks that it discharges the obligation, unconditionally.
   Part 3 records what it does NOT fix.

   Check with:  ./fstar.sh etc/isfinite_proposal.fst

   WHY A MOCK CLASS AND NOT [floating] ITSELF.  Two reasons.  [floating]'s
   ordering axioms are currently inconsistent (see
   [etc/floating_ord_unsound.fst]), so a proof conducted against the real
   class would prove nothing.  And a deliberately MINIMAL mock makes the
   result sharp: [flt] below has no ordering relation at all, no [fexp], and
   no laws beyond the four used here, so whatever is proved in Part 2 is
   proved from those four and nothing else.  Contrast
   [etc/floating_laws_proposal.fst], where the same obligation costs four
   axioms AND a side condition. *)

module T = FStar.Tactics.Typeclasses

type fkind = | Finite | Infinite | NaN

(* ------------------------------------------------------------------ *)
(* Part 1: the proposed addition.                                      *)
(* ------------------------------------------------------------------ *)

class flt (t : Type) = {
  (* --- the fragment of [floating] this argument uses, verbatim --- *)
  zero : t;
  mul  : t -> t -> t;
  kind : t -> fkind;

  kind_zero : squash (kind zero == Finite);

  mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures mul x y == mul y x)
          [SMTPat (mul x y)];

  mul_zero : (x : t) ->
    Lemma (requires Finite? (kind x))
          (ensures mul x zero == zero)
          [SMTPat (mul x zero)];

  (* ---------------------- THE PROPOSAL ---------------------------- *)

  (* Concrete, total, no precondition.  Extracts to CUDA [isfinite(x)],
     which is a standard device function for [float] and [double], and
     [__hisinf]/[__hisnan] or an f32 round-trip for [__half] / [__nv_bfloat16]
     (cf. the [KPR_BF16FALL1] fallback in [kuiper/math.h]).

     Note [Kuiper.Floating.is_finite] already exists as a GTot bool.  This
     one has to be a separate CONCRETE field, because the point of it is to
     drive control flow; [fkind] is [@@erasable], so [kind] cannot.  Suggest
     naming the field [fisfinite] to avoid the collision, and then
     strengthening the derived [is_finite] to
     [let is_finite x = fisfinite x] so there is only one notion. *)
  fisfinite : t -> bool;

  (* The postcondition.  An iff, and TOTAL -- no precondition.  That is the
     entire content of the proposal, and it is what makes Part 2 work with
     no side conditions.  The [SMTPat] means callers never cite it. *)
  fisfinite_spec : (x : t) ->
    Lemma (ensures fisfinite x <==> Finite? (kind x))
          [SMTPat (fisfinite x)];
}

(* ------------------------------------------------------------------ *)
(* Part 2: what it buys.                                               *)
(* ------------------------------------------------------------------ *)

(* The clamp, written exactly as flash_attn_fa1.cu l.173 (and l.222) writes
   it:

       float corr = __expf(Msh[w][i] - mnew);
       if (!isfinite(corr)) corr = 0.0f;

   In [Kuiops.Sdpa.Flash.KfSub] this replaces

       let corr : et_acc = corr0;
       assume pure (kind corr == Finite);          // the one permitted assume

   with

       let corr : et_acc = if fisfinite corr0 then corr0 else zero;

   and the assume is gone. *)
inline_for_extraction noextract
let clamp (#t : Type) {| flt t |} (c : t) : t =
  if fisfinite c then c else zero

(* Finiteness now holds BY CONSTRUCTION, for every input, with no hypothesis
   of any kind.  This is the whole point: it is not that the clamp makes the
   obligation easier to prove, it is that there is no longer an obligation. *)
let clamp_finite (#t : Type) {| flt t |} (c : t)
  : Lemma (ensures Finite? (kind (clamp c)))
  = kind_zero #t

(* THE CONSUMPTION SITE.  [Spec.Bridge.step_fresh] -- the step where a lane
   absorbs its FIRST key.  There the running denominator is still the
   literal [zero] and the proof must show that rescaling it by the
   correction weight leaves it at [zero].  That is [mul_zero], whose side
   condition is [Finite? (kind _)] -- i.e. [mul_zero]'s side condition IS
   CUDA line 173.  This is the single place the [cw_upto] conjunct of
   [Spec.Top.row_no_overflow] is used. *)
let step_fresh (#t : Type) {| flt t |} (c : t)
  : Lemma (ensures mul zero (clamp c) == zero)
  = kind_zero #t;
    clamp_finite c;
    mul_comm (zero <: t) (clamp c);
    mul_zero (clamp c)

(* And the control: without the clamp the same goal is NOT provable, which
   is exactly why the conjunct has to be carried today.  ([@@expect_failure]
   succeeds iff the definition below fails to typecheck.) *)
[@@expect_failure]
let step_fresh_unclamped (#t : Type) {| flt t |} (c : t)
  : Lemma (ensures mul zero c == zero)
  = kind_zero #t;
    mul_comm (zero <: t) c;
    mul_zero c

(* Matched positive control: the SAME body, differing from the failure above
   by exactly one hypothesis.  So the failure is about the missing
   [Finite?] and nothing else -- and that hypothesis is precisely what
   [cw_upto] carries today and what the clamp supplies for free. *)
let step_fresh_assumed (#t : Type) {| flt t |} (c : t)
  : Lemma (requires Finite? (kind c))
          (ensures mul zero c == zero)
  = kind_zero #t;
    mul_comm (zero <: t) c;
    mul_zero c

(* ------------------------------------------------------------------ *)
(* Part 3: the diff, and what is left over.                            *)
(* ------------------------------------------------------------------ *)

(* WHAT CHANGES, exactly.

   Upstream Kuiper (3 places):
     - [Kuiper.Floating.Base.floating]: the two fields above.
     - [Kuiper.{Float16,BFloat16,Float32,Float64}.Base]: a [val fisfinite]
       and a [val fisfinite_spec] each, plus the field in the four
       instance records in [Kuiper.<T>.fst].
     - [kuiper/math.h]: [#define kpr_f32isfinite(f) isfinite(f)] and the
       bf16/f16 analogues; [fexp] already lands as [expf], so the same
       wiring applies.

   This repo (4 places, all mechanical):
     - [Kuiops.Sdpa.Flash.KfSub.fst:476]: the two lines shown above.  THE
       ONE ASSUME IN THE PROOF DISAPPEARS.
     - [Kuiops.Sdpa.Flash.Spec.Float.cw_at]: gains the same clamp, so that
       the spec still mirrors the code.
     - [Kuiops.Sdpa.Flash.Spec.Step]: [cw_upto] becomes a lemma with proof
       [()] instead of a [prop] that is assumed; its [forall] and its
       [SMTPat] go away.
     - [Kuiops.Sdpa.Flash.Spec.Top.row_no_overflow]: conjunct (2), the
       [forall (w : natlt nwv). SS.cw_upto ...] clause, is DELETED.

   The same edit removes the identical [assume pure (is_finite y1)] at
   [Kuiper.Kernel.OnlineSoftmax.fst:235] and
   [Kuiper.Kernel.OnlineSoftmaxDotprod.fst:496], whose TODO asks for exactly
   this ("extend the scalar (or floating) class with a notion of the
   infinities that allows to prove this").

   WHY THE CLAMP AND NOT AN [fexp] LAW.  The alternative is
   [exp_nonpos] + [sub_nonpos] + [sub_not_nan] + [cmp_nan]
   (see [etc/floating_laws_proposal.fst], lemma [cw_finite]).  That route
   works, but it proves a WEAKER fact: it needs the side condition
   [~(m == -inf /\ s == -inf)], because when a whole key tile is masked out
   the shift really is [-inf - -inf = NaN].  Discharging that side condition
   means arguing that [SF.key_tiles] never hands a warp a fully masked tile
   -- an argument about the kernel's tiling, i.e. exactly the kind of
   internal reasoning we are trying to get OUT of the precondition.  The
   clamp has no side condition because the C has no side condition: it
   simply overwrites the NaN.  Four axioms and a tiling argument, versus one
   primitive and a branch.

   WHAT THIS DOES NOT FIX.  [row_no_overflow] has three conjuncts and this
   removes only the second.  Of the other two:

     (1) [all_finite] is a genuine, auditable precondition.  Unfolded, it
         says: for every key [k] admitted for this query row,
         [Finite? (kind (scale * <q,k> + bias))].  It mentions only the
         inputs Q, K, the mask, the scale and the masking predicate -- no
         kernel state, no warp, no tile contents.  It can really fail, and a
         caller can check it.  This one should stay.

     (3) [acc1 (SV.flash_egl_at ...) i `gt` zero] is the one that deserves
         the "assume dressed up as a precondition" charge, and [fisfinite]
         does not touch it.  It is the float denominator, an internal value.
         It cannot be derived from (1) and it cannot be routed through the
         reals: the kernel BRANCHES on it (l.241,
         [inv = (gl > 0) ? 1/gl : 0]) and [v_approximates] carries no error
         bound, so [gl %~ 0.7R] does not entail [gt gl zero].  The options
         are (a) keep it, (b) adopt the seven sign/[fexp] laws of
         [etc/floating_laws_proposal.fst] and replace it with the auditable
         input condition "query row [r] admits at least one key", which is
         what [gsum_pos] in that file proves, or (c) weaken the conclusion
         to a disjunction, which is degradable and should be rejected.
         Given that (3) is now the ONLY non-auditable conjunct left, (b) is
         worth the seven laws -- reversing the recommendation in
         [etc/Kuiops.Floating.Axioms.fsti], which assumed (2) and (3) would
         be carried together. *)
