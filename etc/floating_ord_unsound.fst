module Floating_ord_unsound

(* A standalone demonstration that the ordering axioms of Kuiper's [floating]
   typeclass are jointly inconsistent, and a check that the one-token fix
   removes the contradiction.

   This file is NOT part of the build: it lives in [etc/] because it derives
   [False], which would poison anything that imported it.  Check it by hand:

     ./fstar.sh etc/floating_ord_unsound.fst

   THE DEFECT.  [Kuiper.Floating.Base.floating] declares

     lt_neg_flip : (x : t) -> (y : t) ->
       Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
             (ensures lt x y <==> lte (zero `sub` y) (zero `sub` x))
             [SMTPat (lt x y)];

   At [x == y] the right-hand side is [lte (-x) (-x)], which is true because
   [lte_is_lt_or_eq] makes [lte] reflexive.  So [lt x x] holds for every
   non-NaN [x].  But [negate_lt_is_lte] says [lt x y <==> not (lte y x)],
   which at [x == y] makes [lt x x] false.

   THE FIX.  The right-hand side should be [lt], not [lte]:

     (ensures lt x y <==> lt (zero `sub` y) (zero `sub` x))

   This is the correct IEEE fact (x < y iff -y < -x, sound because Kuiper
   identifies +0 and -0) and it is what the comment above the axiom states.
   The same [val] is repeated in the four concrete backends, so the fix must
   be applied in five places upstream:

     src/lib/kuiper/types/Kuiper.Floating.Base.fsti
     src/lib/kuiper/types/Kuiper.Float16.Base.fsti
     src/lib/kuiper/types/Kuiper.BFloat16.Base.fsti
     src/lib/kuiper/types/Kuiper.Float32.Base.fsti
     src/lib/kuiper/types/Kuiper.Float64.Base.fsti

   Every Kuiper proof needs re-checking afterwards. *)

open Kuiper.Floating
module F32 = Kuiper.Float32

(* ------------------------------------------------------------------ *)
(* Part 1: the contradiction, one step at a time.                      *)
(* ------------------------------------------------------------------ *)

(* [lte] is reflexive on non-NaN values. *)
let step1 (#t:Type) {| floating t |} (x : t)
  : Lemma (requires ~(NaN? (kind x))) (ensures lte x x)
  = lte_is_lt_or_eq x x

(* Negation preserves non-NaN-ness. *)
let step2 (#t:Type) {| floating t |} (x : t)
  : Lemma (requires ~(NaN? (kind x))) (ensures ~(NaN? (kind (zero `sub` x))))
  = neg_kind x

(* [lt_neg_flip] at [x == y] reduces to [lte (-x) (-x)], so [lt x x] holds. *)
let step3 (#t:Type) {| floating t |} (x : t)
  : Lemma (requires ~(NaN? (kind x))) (ensures lt x x)
  = step2 x;
    step1 (zero `sub` x);
    lt_neg_flip x x

(* [negate_lt_is_lte] at [x == y] reduces to [not (lte x x)], so it does not. *)
let step4 (#t:Type) {| floating t |} (x : t)
  : Lemma (requires ~(NaN? (kind x))) (ensures ~(lt x x))
  = step1 x;
    negate_lt_is_lte x x

(* [zero] is non-NaN by [kind_zero], so the two steps collide. *)
let unsound (#t:Type) {| floating t |} (_ : unit) : Lemma False
  = step3 (zero <: t); step4 (zero <: t)

(* Not an artifact of reasoning under an abstract instance: it holds at the
   concrete type the SDPA kernels are instantiated at. *)
let unsound_f32 (_ : unit) : Lemma False = unsound #F32.t ()

(* And therefore every proposition is provable. *)
let anything (p : prop) : squash p = unsound #F32.t ()

(* ------------------------------------------------------------------ *)
(* Part 2: the fix removes the contradiction.                          *)
(* ------------------------------------------------------------------ *)

(* A mock of exactly the axioms used above, with [lt_neg_flip] corrected. *)

type fk = | Fin | Inf | Nan

class ford (t : Type) = {
  zero : t;
  sub : t -> t -> t;
  lt : t -> t -> bool;
  lte : t -> t -> bool;
  kind : t -> fk;

  kind_zero : squash (kind zero == Fin);

  lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires ~(Nan? (kind x)) /\ ~(Nan? (kind y)))
          (ensures lte x y <==> lt x y \/ x == y) [SMTPat (lte x y)];

  neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x) [SMTPat (zero `sub` x)];

  neg_neg : (x : t) ->
    Lemma (requires ~(Nan? (kind x)))
          (ensures zero `sub` (zero `sub` x) == x)
          [SMTPat (zero `sub` (zero `sub` x))];

  (* CORRECTED: [lt] on the right-hand side, not [lte]. *)
  lt_neg_flip : (x : t) -> (y : t) ->
    Lemma (requires ~(Nan? (kind x)) /\ ~(Nan? (kind y)))
          (ensures lt x y <==> lt (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)];

  negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(Nan? (kind x)) /\ ~(Nan? (kind y)))
          (ensures lt x y <==> not (lte y x)) [SMTPat (lt x y)];
}

(* [lte] stays reflexive ... *)
let refl (#t:Type) {| ford t |} (x : t)
  : Lemma (requires ~(Nan? (kind x))) (ensures lte x x)
  = lte_is_lt_or_eq x x

(* ... and [lt] is now irreflexive, rather than both at once. *)
let irrefl (#t:Type) {| ford t |} (x : t)
  : Lemma (requires ~(Nan? (kind x))) (ensures ~(lt x x))
  = refl x; negate_lt_is_lte x x

(* The derivation of Part 1 no longer goes through. *)
[@@expect_failure]
let still_unsound (#t:Type) {| ford t |} (_ : unit) : Lemma False
  = irrefl (zero <: t);
    lt_neg_flip (zero <: t) zero;
    refl (zero `sub` (zero <: t));
    assert (lt (zero <: t) zero)
