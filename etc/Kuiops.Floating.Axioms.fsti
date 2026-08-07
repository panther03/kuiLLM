module Kuiops.Floating.Axioms

(* WHERE FLOAT-LEVEL FACTS ARE ACTUALLY NEEDED IN FLASH ATTENTION.

   Revised after the review question "in the context you are using them, do
   you really need these facts directly on floats?".  The short answer is
   that you were right and my previous framing was wrong.  Concretely:

   THE APPROXIMATION CHAIN NEEDS ZERO NEW AXIOMS.  [a_add], [a_mul],
   [sub_approx], [exp_approx] and [fmax_approx] are unconditional
   congruences, so approximation propagates through the whole online-softmax
   recurrence with no float reasoning at all.  Your [x `sub` x %~ 0.0R]
   route is exactly what the committed proof already does everywhere -- and
   indeed the proof in [kuiops/sdpa/] is complete today with NO axioms.

   Float-level facts are needed in exactly one situation: when the KERNEL
   BRANCHES ON A FLOAT, or when a [floating] law's own side condition does.
   Approximation cannot decide a branch, because [v_approximates] carries no
   error bound -- [l %~ 0.7R] does not entail [gt l zero].

   There are four such sites in [kuipy/unverified/flash_attn_fa1.cu], and
   only two of them cost anything:

   +----------------------------------------+----------------+-----------+
   | CUDA site                              | float fact     | new axioms|
   +----------------------------------------+----------------+-----------+
   | l.169,171,218  fmaxf(-INFINITY, x)     | fmax(-inf,x)=x |     0     |
   | l.181  (sv == -INFINITY) ? 0 : expf()  | Finite sv =>   |     0     |
   |                                        |   sv =!= -inf  |           |
   | l.173,222  if(!isfinite(corr)) corr=0  | Finite corr    |     4     |
   | l.241  inv = (gl > 0) ? 1/gl : 0       | gt gl zero     |     7     |
   +----------------------------------------+----------------+-----------+

   The first two are the cases you already granted, and they cost nothing:
   both are discharged from the EXISTING laws ([fmax_spec],
   [negate_lt_is_lte], [eq_spec], [kind_infinity]).  See
   [Spec.Bridge.fmax_ninf_l] and [Spec.Bridge.sel_prob_admitted].

   ------------------------------------------------------------------------
   SITE 3 -- CUDA line 173.  Not an axiom problem; a missing primitive.

   The reference kernel writes

     float corr = __expf(Msh[w][i] - mnew);
     if (!isfinite(corr)) corr = 0.0f;          // line 173

   so [corr] is finite BY CONSTRUCTION, with no reasoning required.  The
   Kuiper port cannot express that clamp: [is_finite]/[kind] return the
   erasable [fkind], so finiteness cannot drive concrete control flow.  The
   port therefore assumes it ([KfSub.fst:476]) and propagates the obligation
   upward as the [cw_upto] conjunct of [flash_no_overflow].

   Why finiteness of [corr] is needed at all -- it is used at exactly ONE
   place, [Spec.Bridge.step_fresh], the step where the lane absorbs its
   first key.  There the old denominator is the literal [zero], and the
   proof needs [mul zero corr == zero].  That is [floating.mul_zero], whose
   side condition is [Finite? (kind x)].  So [mul_zero]'s side condition IS
   CUDA line 173; a NaN [corr] would poison the accumulator exactly as it
   would in C.

   CONCLUSION FOR SITE 3: the right fix is an extractable [is_finite] on the
   [floating] class, matching the reference kernel.  Then the clamp is code,
   the assume disappears, and axioms 1-4 below are not needed.  Adding the
   axioms instead would prove, at some cost, a fact the CUDA obtains with a
   branch.

   ------------------------------------------------------------------------
   SITE 4 -- CUDA line 241.  Genuinely float-level, but avoidable by
   keeping it as a precondition -- which is what the kernel's own header
   already prescribes.

     float inv = (gl_sh[i] > 0.0f) ? (1.0f / gl_sh[i]) : 0.0f;

   To know the kernel took the [1/gl] branch -- i.e. that the output is a
   normalised softmax at all -- the proof needs [gt gl zero] at the FLOAT
   level.  No amount of real reasoning supplies it, for the error-bound
   reason above.  This is what the seven sign/order axioms 5-11 are for, and
   I confirmed mechanically that they are used for nothing else: deleting
   all seven from the mock class in [etc/floating_laws_proposal.fst] leaves
   the site-3 obligation verifying unchanged.

   But the CUDA header already states the intended contract for the other
   branch:

     "the fully-masked row (max stays -inf) cannot occur for causal, and is
      anyway guarded by `inv = (l > 0) ? 1/l : 0` -> a defined finite 0
      output for which we simply claim no real-softmax approximation."

   That is precisely a CONDITIONAL theorem, which is what the current
   [flash_no_overflow] already gives.  So axioms 5-11 buy only cosmetics:
   they would let the [gt gl zero] conjunct be restated as the input-level
   condition "the row admits at least one key with a finite score"
   (automatic when [has_mask = false]) instead of being carried as is.

   ------------------------------------------------------------------------
   FINAL RECOMMENDATION.

   Adopt all eleven, AND add an extractable [isfinite], AND correct
   [lt_neg_flip].  Together these reduce the whole precondition of
   [sdpa_flash_async] to two statements about the kernel's arguments:

     for each query row, over the keys it attends to
     ([SF.key_ok]: k < sk, and under [causal] also k <= cbound),
     with  score(k) = dot(Q[bi,qh,qpos,:], K[bi,kvh,k,:]) * scale
                        + mask[bi,qh,qpos,k]

       1. every score(k) is a number or -inf, and
       2. at least one score(k) is a number.

   [etc/floating_laws_proposal.fst] machine-checks that reduction
   ([flash_denominator_pos]); [etc/isfinite_proposal.fst] does the same for
   [isfinite].  Roles:

     - [isfinite] retires conjunct (2), [cw_upto].  Strictly better than
       axioms 1-4 for that job: the clamp is unconditional, whereas
       [exp_nonpos] discharges it only under [~(m == -inf /\ s == -inf)],
       whose proof is an argument about [SF.key_tiles] -- internal
       reasoning again.  It also removes the one [assume] in the port and
       the identical one in [Kuiper.Kernel.OnlineSoftmax].
     - Axioms 5-11 retire conjunct (3), the denominator.  There is no
       alternative: the kernel branches on that float and [%~] carries no
       error bound, so no real-level fact can decide the branch.
     - Axioms 1-4 remain needed after all, but for a different reason than
       originally given: [cmp_nan], [sub_not_nan], [sub_nonpos] and
       [exp_nonpos] are what make [sel_prob] provably non-negative and the
       row maximum provably attained, which is how condition (2) above
       replaces the internal hypothesis.
     - [lt_neg_flip] must be corrected regardless: as it stands the class
       proves [False] (see [etc/floating_ord_unsound.fst]), so any proof
       against it -- including everything in this repo -- is vacuous.

   Condition (1) is WEAKER than the [all_finite] in the tree today, which
   demands [Finite] outright.  Allowing [-inf] matters: an additive mask of
   [-inf] is the standard PyTorch masking idiom and the kernel already
   handles it by select-to-zero (CUDA l.181).  So the new precondition is
   both auditable and more permissive than the current one.

   Check this file with:  ./fstar.sh etc/Kuiops.Floating.Axioms.fsti        *)

open Kuiper.Floating

(* ================================================================== *)
(* SITE 3 (CUDA l.173) -- obsolete if [floating] gains [is_finite].    *)
(* ================================================================== *)

(* Comparisons are false at NaN.  Every ordering law in [floating] is
   guarded by [~(NaN? (kind x))] and nothing points the other way, so no
   proof can conclude "x is a number" from "x compares". *)
val cmp_nan (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires NaN? (kind x) \/ NaN? (kind y))
          (ensures ~(lt x y) /\ ~(lte x y) /\ ~(eq x y))

(* A difference of numbers is not NaN.  Stated as the single implication the
   proof consumes -- NaN is ruled OUT.  (Your suggested wording, "inf - inf
   is NaN" and "NaN - NaN is NaN", is the converse; it says when a NaN
   APPEARS, which is never what is needed.)  [kind] records no sign, so
   "equal-signed infinities" is expressed as [x == y]; [+inf - -inf] is
   [+inf] and is correctly excluded. *)
val sub_not_nan (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(Infinite? (kind x) /\ Infinite? (kind y) /\ x == y))
          (ensures ~(NaN? (kind (x `sub` y))))

(* [x <= y ==> x - y <= 0].  Sound under round-to-nearest: rounding is
   monotone and [0] is exact.  [floating] has no monotonicity law at all. *)
val sub_nonpos (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ lte x y /\
                    ~(NaN? (kind (x `sub` y))))
          (ensures lte (x `sub` y) zero)

(* [exp] maps [[-inf, 0]] into [[0, 1]]: on a non-positive argument it
   cannot overflow, cannot be NaN, and cannot be negative.  This is the law
   that would replace the [isfinite] clamp, since [m_old <= m_new] always.
   [exp_approx] does not help: it relates [fexp] to the reals through the
   abstract [v_approximates], which says nothing about [kind]. *)
val exp_nonpos (#t : Type) {| floating t |} (x : t)
  : Lemma (requires ~(NaN? (kind x)) /\ lte x zero)
          (ensures Finite? (kind (fexp x)) /\
                   lte zero (fexp x) /\ lte (fexp x) one)

(* ================================================================== *)
(* SITE 4 (CUDA l.241) -- needed only to discharge [gt gl zero] rather  *)
(* than carry it.  Verified to be used for nothing else.               *)
(* ================================================================== *)

(* The order is transitive.  [floating] relates [lt], [lte] and [eq] to one
   another pointwise but never chains two comparisons.  Needed to compare a
   running maximum across warps in the block combine (CUDA l.218). *)
val lte_trans (#t : Type) {| floating t |} (x y z : t)
  : Lemma (requires lte x y /\ lte y z) (ensures lte x z)

(* [x - x == 0] for finite [x], exactly.  [sub_is_add_neg] rewrites [x - y]
   to [x + (-y)], but no law then gives [x + (-x) == 0]: the class has no
   additive inverse.  Used for the argmax warp, where [Msh[ww][i] == gm]. *)
val sub_self (#t : Type) {| floating t |} (x : t)
  : Lemma (requires Finite? (kind x)) (ensures x `sub` x == zero)

(* [exp 0 == 1], exactly.  This cannot be weakened to "[fexp zero] is
   positive": [gsum_pos] rewrites [1 * l] to [l] via [mul_one], and with
   positivity alone it would instead need [0 < x /\ 0 < y ==> 0 < x * y],
   which is FALSE for floats (the product can underflow).  Nor can it be
   generalised to finite [x], since [fexp] underflows to zero for very
   negative arguments. *)
val exp_zero (#t : Type) {| floating t |} (_ : unit)
  : Lemma (fexp (zero <: t) == one)

(* [0 < 1].  [floating] pins [kind zero] and [kind one] but never orders
   them, or even says they are distinct. *)
val one_pos (#t : Type) {| floating t |} (_ : unit)
  : Lemma (lt (zero <: t) one)

(* Non-negatives are closed under [+] and [*], and a sum with a positive
   summand is positive -- the folds behind [rowsum] (l.183) and [gl]
   (l.225).  Sound under round-to-nearest for the same reason as
   [sub_nonpos].  Stated in this specialised form rather than as general
   monotonicity because these hypotheses already exclude the [inf + -inf]
   and [0 * inf] NaN cases, so no side conditions are needed. *)
val add_nonneg (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (add x y))

val add_pos (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ (lt zero x \/ lt zero y))
          (ensures lt zero (add x y))

val mul_nonneg (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (mul x y))
