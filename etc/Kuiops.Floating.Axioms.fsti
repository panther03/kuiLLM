module Kuiops.Floating.Axioms

(* THE PROPOSED TRUSTED BASE, AND NOTHING ELSE.

   Every declaration below is a [val] with no definition: each one is an
   axiom that would be added to Kuiper's trusted computing base.  There are
   eleven.  Nothing else in the SDPA flash-attention correctness development
   is trusted -- everything else is derived from these plus the existing
   [floating] laws.  See [etc/floating_laws_proposal.fst] for the
   machine-checked argument that these are sufficient; the mock class there
   states all eleven verbatim, so the argument transfers unchanged.

   They are stated as free-standing lemmas over the EXISTING
   [Kuiper.Floating.Base.floating] class, so adopting them requires no change
   to any typeclass and no re-instantiation of any backend.

   NO CHANGE IS NEEDED to [real_like] or [floating_real_like].

   THE ONE EXCEPTION -- a change rather than an addition -- is the correction
   to [lt_neg_flip] in [Kuiper.Floating.Base] and its four backing [val]s in
   [Kuiper.{Float16,BFloat16,Float32,Float64}.Base]:

     ensures lt x y <==> lte (zero `sub` y) (zero `sub` x)   (* current *)
     ensures lt x y <==> lt  (zero `sub` y) (zero `sub` x)   (* correct *)

   As it stands the class proves [False]; see [etc/floating_ord_unsound.fst].

   Check this file with:  ./fstar.sh etc/Kuiops.Floating.Axioms.fsti

   ------------------------------------------------------------------------
   RESPONSE TO REVIEW: "these exist on reals, argue by approximation"

   Six of the eleven were marked "EXISTS ON REALS - shouldn't be needed", on
   the grounds that any finite float has a [to_real], so the real-level
   lemma plus approximation ought to suffice.  I checked this mechanically
   rather than by argument; the experiment is [etc/floating_real_route.fst],
   which verifies clean.  The result is that the route does not close, for a
   structural reason.  In brief:

   1. THE APPROXIMATION CLASSES TRANSFER NOTHING BACK.  Part 1 of that file
      instantiates [real_like] + [floating_real_like] with

        v_approximates := fun _ _ -> True

      Every law of both classes -- [to_real_ok], [a0], [a1], [a_add],
      [a_mul], [fmax_approx], [sub_approx], [exp_approx], [div_approx] --
      holds trivially under it, for any float type and any [to_real].  So NO
      float fact whatsoever follows from those classes.  This is by design:
      [v_approximates] is an error-tolerant relation ([a_add] has no
      rounding side condition, so it must absorb accumulated error) and is
      deliberately abstract.  Information flows float -> real, never back.

   2. THE ONE LAW THAT WOULD RULE OUT THE TRIVIAL MODEL IS UNAVAILABLE FOR
      FLOATS.  That law is [precise_real_like.v_approximates_inj].  In the
      installed tree [precise_real_like] is instantiated for exactly four
      types -- [u8], [u16], [u32], [u64] -- and for no float type.  That is
      not an oversight.  It cannot hold for a float type: [a_add] forces
      [add x y] to approximate the EXACT sum, while [to_real (add x y)] is
      the ROUNDED value, so a float approximates many reals.  Part 4 of the
      experiment spells this out.

   3. THERE IS ALSO NO ORDER-TRANSFER LAW.  Nothing anywhere in Kuiper
      relates [lt]/[lte]/[eq] to [<.]/[<=.].  Even granting injectivity, a
      real-level fact cannot be converted into a float-level comparison
      without one.

   Part 2 of the experiment grants BOTH missing bridges anyway, as
   generously as possible, and finds that exactly three of the six do become
   derivable: [lte_trans], [sub_self], [exp_zero].  Part 3 shows the other
   three ([one_pos], [sub_nonpos], and the [add]/[mul] sign laws) still do
   not follow even then, and Part 4 replays each with one extra hypothesis
   to prove the failures are attributable to precisely the missing link and
   not to a defect in the attempts.

   So the concession is: three of the six COULD be dropped, but only in
   exchange for two new axioms, one of which ([precise_real_like] for
   floats) is FALSE.  Dropping them is therefore not actually available, and
   even if it were it would trade three axioms for two.  Each law's entry
   below records its individual verdict.

   The one change the review does buy: [sub_nan_spec] has been WEAKENED to
   the single implication that is actually consumed.  See its entry.        *)

open Kuiper.Floating

(* ------------------------------------------------------------------ *)
(* A. The float order.  Independent of flash attention.                *)
(* ------------------------------------------------------------------ *)

(* A: Looks good *)
(* Comparisons are false at NaN.

   Every ordering law in [floating] is guarded by [~(NaN? (kind x))], and
   nothing points the other way, so no proof can conclude "x is a number"
   from "x compares".  Without this, an ordering hypothesis carries no
   information about [kind] at all. *)
val cmp_nan (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires NaN? (kind x) \/ NaN? (kind y))
          (ensures ~(lt x y) /\ ~(lte x y) /\ ~(eq x y))

(* A: EXISTS ON REALS - Shouldn't be needed *)
(* VERDICT: derivable from reals, but only via a new order-transfer axiom,
   so it is a one-for-one swap and not a saving.

   Transitivity of [<=.] on reals is indeed free.  Getting from there to
   [lte x z] needs

     ord_transfer : ~(NaN? (kind x)) /\ ~(NaN? (kind y)) ==>
                    (lte x y <==> to_real x <=. to_real y)

   which does not exist anywhere in Kuiper.  With it, [lte_trans] follows
   (proved in Part 2a of the experiment) -- but [ord_transfer] is itself a
   new axiom, and it buys nothing else: it does not rescue [one_pos] or any
   of the sign laws, because those concern [to_real] of a COMPUTED result.

   I have kept [lte_trans] because it is the weaker of the two (implied by
   [ord_transfer]) and needs no [real_like] instance in scope.  Happy to
   swap if you prefer [ord_transfer] as the more fundamental statement.

   The class relates [lt], [lte] and [eq] to one another pointwise but never
   chains two comparisons, so [lte] is not currently known to be an order.
   Needed as soon as a running maximum is compared across iterations. *)
val lte_trans (#t : Type) {| floating t |} (x y z : t)
  : Lemma (requires lte x y /\ lte y z) (ensures lte x z)

(* ------------------------------------------------------------------ *)
(* B. Subtraction.                                                     *)
(* ------------------------------------------------------------------ *)

(* A: This is maybe fine, but I'd prefer it worded differently:
instead of saying conclusively a NaN difference can only result from these things (which I am skeptical is true),
you should say inf - inf = NaN, NaN - NaN = NaN. These would be perfectly acceptable. *)
(* VERDICT: weakened, but not in the suggested direction -- the suggested
   wording is the converse of the one the proof consumes.

   The old biconditional was used at four sites, and at every one of them it
   is used to establish the precondition of [sub_nonpos], i.e. to rule NaN
   OUT.  That is the [==>] direction (contrapositive: non-NaN operands that
   are not equal infinities give a non-NaN difference).  "[inf - inf] is
   NaN" and "[NaN - NaN] is NaN" are the [<==] direction; they say when a
   NaN APPEARS, which the proof never needs.

   So I have dropped the [<==] direction entirely and restated the axiom as
   the bare implication that is used.  This is strictly weaker than what was
   proposed before, and I re-verified [etc/floating_laws_proposal.fst]
   against this weakened form -- it still goes through unchanged.

   On the skepticism about the content: this is IEEE 754 subtraction, whose
   only NaN-producing cases on non-NaN operands are the two infinite ones
   with equal signs.  [kind] does not record a sign, so "equal signs" is
   expressed as [x == y].  ([+inf - -inf] is [+inf], correctly excluded.)
   [floating] says nothing about [kind (sub x y)] at all beyond
   [neg_kind]. *)
val sub_not_nan (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(Infinite? (kind x) /\ Infinite? (kind y) /\ x == y))
          (ensures ~(NaN? (kind (x `sub` y))))

(* A: EXISTS ON REALS - Shouldn't be needed
   because if something is finite you should have a real approximation for it *)
(* VERDICT: not available.  It is true that a finite float has a [to_real],
   and the real side is trivial: [sub_approx] gives
   [v_approximates (sub x x) (to_real x -. to_real x)], i.e.
   [v_approximates (sub x x) 0.0R], and [a0] gives
   [v_approximates zero 0.0R].  But concluding [sub x x == zero] from those
   two is exactly [v_approximates_inj], which is [precise_real_like] -- a
   class no float type has or can have (point 2 in the header).

   Part 2b of the experiment confirms the derivation works the moment
   injectivity is granted, and point 2 explains why it cannot be.

   Directly: [sub_is_add_neg] rewrites [x - y] to [x + (-y)], but no law
   then gives [x + (-x) == 0]; the class has no additive inverse. *)
val sub_self (#t : Type) {| floating t |} (x : t)
  : Lemma (requires Finite? (kind x)) (ensures x `sub` x == zero)

(* A: EXISTS ON REALS - Shouldn't be needed  *)
(* VERDICT: not available, and not even with both bridges granted.

   This concludes a float comparison about a COMPUTED result.  Via reals it
   would need to know where [to_real (sub x y)] sits.  What is available is
   [sub_approx], which yields [v_approximates (sub x y) (to_real x -.
   to_real y)] -- a fact about [v_approximates], not about [to_real] of the
   result.  Passing from the former to the latter is functionality of
   [v_approximates] in its real argument, which is precisely what rounding
   makes false.  Part 3 of the experiment shows the attempt failing with
   both bridges and every relevant hypothesis supplied; Part 4 shows it
   succeeding the instant that functionality is added, and then shows
   functionality is false.

   Sound under round-to-nearest: rounding is monotone and [0] is exact.
   [floating] has no monotonicity law of any kind. *)
val sub_nonpos (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ lte x y /\
                    ~(NaN? (kind (x `sub` y))))
          (ensures lte (x `sub` y) zero)

(* ------------------------------------------------------------------ *)
(* C. The exponential.  [fexp] is currently a bare field with no laws.  *)
(* ------------------------------------------------------------------ *)

(* A: This is fine. *)
(* THE CENTRAL AXIOM.  [exp] maps [[-inf, 0]] into [[0, 1]]: on a
   non-positive argument it cannot overflow, cannot be NaN, and cannot be
   negative.

   This is exactly why shifting by the running maximum makes online softmax
   safe, and it is the property the whole flash-attention kernel is built
   around.  It covers [x = -inf] (giving [0]), which is the case that arises
   when a key tile is entirely masked out.

   [floating_real_like.exp_approx] does not help: it relates [fexp] to the
   reals through the abstract [v_approximates], which says nothing about
   [kind]. *)
val exp_nonpos (#t : Type) {| floating t |} (x : t)
  : Lemma (requires ~(NaN? (kind x)) /\ lte x zero)
          (ensures Finite? (kind (fexp x)) /\
                   lte zero (fexp x) /\ lte (fexp x) one)

(* A: EXISTS ON REALS - Shouldn't be needed *)
(* VERDICT: not available, for the same reason as [sub_self].

   The real side is again clean -- [exp_approx] at [zero] with [a0] gives
   [v_approximates (fexp zero) (exp 0.0R)], and [FStar.Math.Exp.exp_base]
   gives [exp 0.0R == 1.0R], so [v_approximates (fexp zero) 1.0R]; with
   [a1] giving [v_approximates one 1.0R], the two agree on a real.  But
   concluding [fexp zero == one] is once more [v_approximates_inj].  Proved
   in Part 2c of the experiment, conditional on the same unavailable class.

   I also checked whether this axiom could be avoided by weakening it to
   "[fexp zero] is positive" and folding [one_pos] into it.  It cannot:
   [gsum_pos] rewrites [1 * l] to [l] via [mul_one], and with positivity
   alone it would instead need [0 < x /\ 0 < y ==> 0 < x * y], which is
   FALSE for floats (the product can underflow to zero).  The equality is
   load-bearing.

   [exp 0 == 1] is what gives the softmax denominator its one strictly
   positive summand: the key that attains the row maximum. *)
val exp_zero (#t : Type) {| floating t |} (_ : unit)
  : Lemma (fexp (zero <: t) == one)

(* ------------------------------------------------------------------ *)
(* D. Signs.                                                           *)
(* ------------------------------------------------------------------ *)

(* A: EXISTS ON REALS - Shouldn't be needed
   Zero and one approximate 0.0R and 1.0R, no? Do you really need this directly? *)
(* VERDICT: not available, and this one fails for an extra, independent
   reason worth flagging.

   Yes, [zero] and [one] approximate [0.0R] and [1.0R] -- that is exactly
   what [a0] and [a1] say.  But they constrain [v_approximates], NOT
   [to_real].  Nothing in [real_like] pins [to_real zero] to [0.0R] or
   [to_real one] to [1.0R]; the trivial model of point 1 has
   [to_real = fun _ -> 0.0R] and satisfies [a0] and [a1] happily.

   And injectivity does not close the gap either: it takes TWO FLOATS
   approximating ONE REAL and concludes the floats are equal.  Here we have
   one float and would need to extract its [to_real], which is the opposite
   shape.  So [one_pos] fails even with BOTH bridges granted -- see Part 3
   of the experiment, where it is the only entry that fails for this second
   reason rather than the rounding one.

   [floating] pins [kind zero] and [kind one] but never orders them, or even
   says they are distinct. *)
val one_pos (#t : Type) {| floating t |} (_ : unit)
  : Lemma (lt (zero <: t) one)

(* A: All of these should be provable on reals *)
(* VERDICT: not available.  Same obstruction as [sub_nonpos], and the
   sharpest illustration of it.

   On reals these are of course trivial.  The difficulty is not the real
   arithmetic; it is that the conclusion is a FLOAT comparison about
   [add x y], and the only route to [to_real (add x y)] runs through
   [a_add], which speaks about [v_approximates] and not [to_real].  Closing
   that step means asserting that the real a float approximates is unique --
   and [a_add] itself forces that to be false, since [add x y] approximates
   the exact sum while [to_real (add x y)] is the rounded one.  Concretely,
   for [x] one ulp below half an ulp of [one], [add one x == one], so
   uniqueness would force [to_real x == 0.0R] and hence [x == zero].

   Parts 3 and 4 of the experiment differ by exactly this one hypothesis:
   without it all three fail, with it all three succeed.

   A second, smaller gap turned up while checking: nothing says arithmetic
   on non-NaN operands yields a non-NaN result, so the real route would need
   a NaN-propagation law too -- which is float-level content by definition.

   Non-negatives are closed under [+] and [*], and a sum with a positive
   summand is positive.  Sound under round-to-nearest for the same reason as
   [sub_nonpos].

   Stated in this specialised form rather than as general monotonicity of
   [add] / [mul] because these hypotheses already exclude the [inf + -inf]
   and [0 * inf] NaN cases, so no side conditions are needed. *)
val add_nonneg (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (add x y))

val add_pos (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ (lt zero x \/ lt zero y))
          (ensures lt zero (add x y))

val mul_nonneg (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (mul x y))
