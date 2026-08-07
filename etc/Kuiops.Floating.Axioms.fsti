module Kuiops.Floating.Axioms

(* THE PROPOSED TRUSTED BASE, AND NOTHING ELSE.

   Every declaration below is a [val] with no definition: each one is an
   axiom that would be added to Kuiper's trusted computing base.  There are
   eleven.  Nothing else in the SDPA flash-attention correctness development
   is trusted -- everything else is derived from these plus the existing
   [floating] laws.  See [etc/floating_laws_proposal.fst] for the
   machine-checked argument that these are sufficient.

   They are stated as free-standing lemmas over the EXISTING
   [Kuiper.Floating.Base.floating] class, so adopting them requires no change
   to any typeclass and no re-instantiation of any backend.

   NO CHANGE IS NEEDED to [real_like] or [floating_real_like].  The
   float-to-real approximation argument already goes through with
   [exp_approx], [sub_approx], [div_approx] and [fmax_approx] as they stand;
   what is missing is purely about [kind] and the float ordering, which is
   what these eleven supply.

   THE ONE EXCEPTION -- a change rather than an addition -- is the correction
   to [lt_neg_flip] in [Kuiper.Floating.Base] and its four backing [val]s in
   [Kuiper.{Float16,BFloat16,Float32,Float64}.Base]:

     ensures lt x y <==> lte (zero `sub` y) (zero `sub` x)   (* current *)
     ensures lt x y <==> lt  (zero `sub` y) (zero `sub` x)   (* correct *)

   As it stands the class proves [False]; see [etc/floating_ord_unsound.fst].

   Check this file with:  ./fstar.sh etc/Kuiops.Floating.Axioms.fsti          *)

open Kuiper.Floating

(* ------------------------------------------------------------------ *)
(* A. The float order.  Independent of flash attention.                *)
(* ------------------------------------------------------------------ *)

(* Comparisons are false at NaN.

   Every ordering law in [floating] is guarded by [~(NaN? (kind x))], and
   nothing points the other way, so no proof can conclude "x is a number"
   from "x compares".  Without this, an ordering hypothesis carries no
   information about [kind] at all. *)
val cmp_nan (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires NaN? (kind x) \/ NaN? (kind y))
          (ensures ~(lt x y) /\ ~(lte x y) /\ ~(eq x y))

(* The order is transitive.

   [floating] relates [lt], [lte] and [eq] to one another pointwise but never
   chains two comparisons, so [lte] is not currently known to be an order.
   Needed as soon as a running maximum is compared across iterations. *)
val lte_trans (#t : Type) {| floating t |} (x y z : t)
  : Lemma (requires lte x y /\ lte y z) (ensures lte x z)

(* ------------------------------------------------------------------ *)
(* B. Subtraction.                                                     *)
(* ------------------------------------------------------------------ *)

(* When a difference is NaN: only for a NaN operand, or [inf - inf] with
   equal signs.  [floating] says nothing about [kind (sub x y)] beyond
   [neg_kind]. *)
val sub_nan_spec (#t : Type) {| floating t |} (x y : t)
  : Lemma (ensures NaN? (kind (x `sub` y)) <==>
                   (NaN? (kind x) \/ NaN? (kind y) \/
                    (Infinite? (kind x) /\ Infinite? (kind y) /\ x == y)))

(* [x - x == 0] for finite [x], exactly.  [sub_is_add_neg] rewrites [x - y]
   to [x + (-y)], but no law then gives [x + (-x) == 0]: the class has no
   additive inverse. *)
val sub_self (#t : Type) {| floating t |} (x : t)
  : Lemma (requires Finite? (kind x)) (ensures x `sub` x == zero)

(* Subtraction reverses the order in its second argument.  Sound under
   round-to-nearest: rounding is monotone and [0] is exact.  [floating] has
   no monotonicity law of any kind. *)
val sub_nonpos (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ lte x y /\
                    ~(NaN? (kind (x `sub` y))))
          (ensures lte (x `sub` y) zero)

(* ------------------------------------------------------------------ *)
(* C. The exponential.  [fexp] is currently a bare field with no laws.  *)
(* ------------------------------------------------------------------ *)

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

(* [exp 0 == 1], exactly.  This is what gives the softmax denominator its one
   strictly positive summand: the key that attains the row maximum. *)
val exp_zero (#t : Type) {| floating t |} (_ : unit)
  : Lemma (fexp (zero <: t) == one)

(* ------------------------------------------------------------------ *)
(* D. Signs.                                                           *)
(* ------------------------------------------------------------------ *)

(* [0 < 1].  [floating] pins [kind zero] and [kind one] but never orders
   them, or even says they are distinct. *)
val one_pos (#t : Type) {| floating t |} (_ : unit)
  : Lemma (lt (zero <: t) one)

(* Non-negatives are closed under [+] and [*], and a sum with a positive
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
