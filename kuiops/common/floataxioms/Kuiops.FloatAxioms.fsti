module Kuiops.FloatAxioms

(* Laws of IEEE 754 floating point, under round-to-nearest, that
   [Kuiper.Floating.Base.floating] does not currently provide.  These are
   ADMITTED: this module has no implementation.  Together with [floating]
   itself they are the only float-level facts the SDPA proof trusts.

   Candidates for upstreaming into [floating]. *)

open Kuiper.Floating

unfold let ninf (#t : Type) {| floating t |} : t = zero `sub` infinity

(* Comparison against a NaN is false. *)
val cmp_nan (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires NaN? (kind x) \/ NaN? (kind y))
          (ensures ~(lt x y) /\ ~(lte x y) /\ ~(eq x y))

val lte_trans (#t : Type) {| floating t |} (x y z : t)
  : Lemma (requires lte x y /\ lte y z) (ensures lte x z)

(* A difference of numbers is NaN only for two equal infinities. *)
val sub_not_nan (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\
                    ~(Infinite? (kind x) /\ Infinite? (kind y) /\ x == y))
          (ensures ~(NaN? (kind (x `sub` y))))

val sub_self (#t : Type) {| floating t |} (x : t)
  : Lemma (requires Finite? (kind x)) (ensures x `sub` x == zero)

val sub_nonpos (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ lte x y /\
                    ~(NaN? (kind (x `sub` y))))
          (ensures lte (x `sub` y) zero)

(* [exp] maps [[-inf, 0]] into [[0, 1]]: it cannot overflow there. *)
val exp_nonpos (#t : Type) {| floating t |} (x : t)
  : Lemma (requires ~(NaN? (kind x)) /\ lte x zero)
          (ensures Finite? (kind (fexp x)) /\
                   lte zero (fexp x) /\ lte (fexp x) one)

val exp_zero (#t : Type) {| floating t |} (_ : unit)
  : Lemma (fexp (zero <: t) == one)

val one_pos (#t : Type) {| floating t |} (_ : unit)
  : Lemma (lt (zero <: t) one)

val add_nonneg (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (x `add` y))

val add_pos (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ (lt zero x \/ lt zero y))
          (ensures lt zero (x `add` y))

val mul_nonneg (#t : Type) {| floating t |} (x y : t)
  : Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (x `mul` y))
