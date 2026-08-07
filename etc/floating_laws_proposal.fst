module Floating_laws_proposal

(* PROPOSED ADDITIONAL LAWS FOR [Kuiper.Floating.Base.floating].

   These are the laws the SDPA flash-attention correctness proof needs in
   order to DISCHARGE two of the three conjuncts of
   [Kuiops.Sdpa.Flash.Spec.Top.row_no_overflow], which are currently carried
   as hypotheses:

     (2) [SS.cw_upto ...]                    -- online-softmax correction
                                                weights are finite
     (3) [acc1 (SV.flash_egl_at ...) i `gt` zero]
                                             -- the epilogue denominator is
                                                positive

   Neither is a real precondition on the kernel's inputs; both follow from
   conjunct (1), [all_finite].  They are stated as hypotheses only because
   [fexp] is an unaxiomatized primitive and the class has no monotonicity
   laws for [add] / [mul].

   This file is NOT part of the build (it lives in [etc/]).  Check it with

     ./fstar.sh etc/floating_laws_proposal.fst

   HOW TO READ IT.

   Part 1 declares a class [flt] that mirrors the subset of
   [Kuiper.Floating.Base.floating] the argument uses, plus the proposed new
   laws, flagged [NEW].  Every field is written in the exact shape it would
   take as a field of [floating], so the [NEW] ones can be pasted straight
   in.

   Part 2 proves, from those laws alone, the two facts above -- stated over
   faithful miniatures of the kernel's actual folds
   ([Kuiops.Sdpa.Flash.Spec.Float.row_sum], [.gmax], [.gsum]).  So this is
   not a wishlist: it is a machine-checked argument that the proposed laws
   are SUFFICIENT.  They are also close to minimal -- I derived them by
   walking the obligations backwards, and every one of them is used below.

   SUMMARY OF WHAT IS MISSING.  Two of the nine gaps are not about [fexp] at
   all, and are worth fixing on their own account:

     0. [cmp_nan]  -- nothing says a comparison is FALSE at NaN, so no proof
                      can conclude "[x] is a number" from "[x] compares".
     1. [lte_trans] -- the ordering axioms never chain two comparisons, so
                      [lte] is not currently known to be a transitive
                      relation at all.

   The rest are the [fexp] and sign/monotonicity laws the online-softmax
   argument needs:

     2. [sub_nan_spec]  when a difference is NaN
     3. [sub_self]      [x - x == 0] for finite [x]
     4. [sub_nonpos]    [x <= y ==> x - y <= 0]
     5. [exp_nonpos]    [exp] maps [[-inf, 0]] into [[0, 1]]   <- the key one
     6. [exp_zero]      [exp 0 == 1]
     7. [one_pos]       [0 < 1]
     8. [add_nonneg] / [add_pos] / [mul_nonneg]  sign closure

   WHY A MOCK CLASS RATHER THAN [floating] ITSELF.  [floating]'s ordering
   axioms are currently inconsistent (see [etc/floating_ord_unsound.fst]), so
   a proof conducted against the real class would prove nothing.  [flt]
   carries the CORRECTED [lt_neg_flip], so the results below are meaningful.

   RESIDUAL PRECONDITION.  Conjunct (3) does not follow from [all_finite]
   alone, and should not: if every key of a query row is masked out, the
   denominator really is zero.  What the laws buy is a reduction of (3) to

     "row [i] admits at least one key whose score is finite"

   which is a genuine, caller-checkable input condition, and which is
   automatically true when [has_mask = false] (under [causal], key [qpos] is
   always admitted).  See [gsum_pos] below.  *)

module T = FStar.Tactics.Typeclasses

(* ------------------------------------------------------------------ *)
(* Part 1: the class.                                                  *)
(* ------------------------------------------------------------------ *)

type fkind = | Finite | Infinite | NaN

class flt (t : Type) = {
  zero : t;
  one : t;
  add : t -> t -> t;
  mul : t -> t -> t;
  sub : t -> t -> t;
  eq : t -> t -> bool;
  lt : t -> t -> bool;
  lte : t -> t -> bool;
  kind : t -> fkind;
  infinity : t;
  fmax : t -> t -> t;
  fexp : t -> t;

  (* ---- existing laws of [floating], reproduced verbatim ---- *)

  kind_zero     : squash (kind zero == Finite);
  kind_one      : squash (kind one == Finite);
  kind_infinity : squash (kind infinity == Infinite);

  eq_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures eq x y <==> x == y) [SMTPat (eq x y)];

  lte_is_lt_or_eq : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lte x y <==> lt x y \/ x == y) [SMTPat (lte x y)];

  negate_lt_is_lte : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> not (lte y x)) [SMTPat (lt x y)];

  neg_kind : (x : t) ->
    Lemma (ensures kind (zero `sub` x) == kind x) [SMTPat (zero `sub` x)];

  neg_neg : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures zero `sub` (zero `sub` x) == x)
          [SMTPat (zero `sub` (zero `sub` x))];

  infinity_val_spec : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures lte x infinity) [SMTPat (lte x infinity)];

  (* NOTE: [lt] on the right, per etc/floating_ord_unsound.fst.  With the
     [lte] currently in [floating] this whole file is vacuous. *)
  lt_neg_flip : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lt x y <==> lt (zero `sub` y) (zero `sub` x))
          [SMTPat (lt x y)];

  fmax_spec : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures fmax x y == (if lt x y then y else x)) [SMTPat (fmax x y)];

  add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures add x y == add y x) [SMTPat (add x y)];

  mul_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures mul x y == mul y x) [SMTPat (mul x y)];

  add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures add x zero == x) [SMTPat (add x zero)];

  mul_one : (x : t) ->
    Lemma (requires ~(NaN? (kind x)))
          (ensures mul x one == x) [SMTPat (mul x one)];

  (* ---- [NEW] 0. comparisons are false at NaN ---------------------- *)

  (* IEEE: every ordered comparison involving a NaN is false.  The class
     currently constrains [eq], [lt] and [lte] ONLY under a non-NaN
     hypothesis, so nothing lets a proof conclude "[x] is not NaN" from
     "[x] compares".  Every ordering argument below needs that direction. *)
  cmp_nan : (x : t) -> (y : t) ->
    Lemma (requires NaN? (kind x) \/ NaN? (kind y))
          (ensures ~(lt x y) /\ ~(lte x y) /\ ~(eq x y));

  (* ---- [NEW] 1. the order is transitive --------------------------- *)

  (* The class relates [lt], [lte] and [eq] to each other pointwise, but
     never chains two comparisons, so [lte] is not currently known to BE an
     order.  Needed the moment a running maximum is compared across
     iterations. *)
  lte_trans : (x : t) -> (y : t) -> (z : t) ->
    Lemma (requires lte x y /\ lte y z) (ensures lte x z);

  (* ---- [NEW] 2. when subtraction is NaN --------------------------- *)

  (* IEEE: the only way a difference of numbers is NaN is a NaN operand or
     [inf - inf] with equal signs.  Needed to know that the online-softmax
     shift [m_t - m_{t+1}] is a real number and not NaN.  The class today
     says nothing at all about [kind (sub x y)] except through [neg_kind].
     Stated as an iff because both directions are used: the [<==] direction
     rules NaN out, the [==>] direction is what makes it checkable. *)
  sub_nan_spec : (x : t) -> (y : t) ->
    Lemma (ensures NaN? (kind (x `sub` y)) <==>
                   (NaN? (kind x) \/ NaN? (kind y) \/
                    (Infinite? (kind x) /\ Infinite? (kind y) /\ x == y)));

  (* ---- [NEW] 3. self-subtraction ---------------------------------- *)

  (* [x - x == 0] for finite [x].  Exact in IEEE (the subtraction is
     representable).  The class has no additive-inverse law at all:
     [sub_is_add_neg] rewrites [x - y] to [x + (-y)], but nothing then says
     [x + (-x) == 0].  This is what turns the argmax key's shifted score
     into [fexp zero]. *)
  sub_self : (x : t) ->
    Lemma (requires Finite? (kind x)) (ensures x `sub` x == zero);

  (* ---- [NEW] 4. subtraction is order-reversing in its second arg --- *)

  (* [x <= y ==> x - y <= 0].  Sound under round-to-nearest: rounding is
     monotone and [0] is exact.  The class has no monotonicity law
     whatsoever, for [sub], [add] or [mul]. *)
  sub_nonpos : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ lte x y /\
                    ~(NaN? (kind (x `sub` y))))
          (ensures lte (x `sub` y) zero);

  (* ---- [NEW] 5. exp of a non-positive argument -------------------- *)

  (* THE CENTRAL ONE.  [fexp] is currently a bare field with no laws, and
     [floating_real_like.exp_approx] only relates it to the reals through
     the abstract [v_approximates], which says nothing about [kind].

     [exp] maps [[-inf, 0]] into [[0, 1]]: it cannot overflow, cannot be
     NaN, and cannot be negative.  This is exactly why the online-softmax
     shift-by-the-max trick is safe, and it is the fact the flash kernel is
     built around.  Covers [x = -inf] (giving [0]), which is the case that
     arises when a whole key tile is masked out. *)
  exp_nonpos : (x : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ lte x zero)
          (ensures Finite? (kind (fexp x)) /\
                   lte zero (fexp x) /\ lte (fexp x) one);

  (* ---- [NEW] 6. exp at zero --------------------------------------- *)

  (* [exp 0 == 1], exactly.  Gives the denominator its one strictly
     positive summand: the key that attains the row maximum. *)
  exp_zero : squash (fexp zero == one);

  (* ---- [NEW] 7. zero is below one --------------------------------- *)

  (* The class pins [kind zero] and [kind one] but never says the two are
     ordered, or even distinct. *)
  one_pos : squash (lt zero one);

  (* ---- [NEW] 8. sign closure for add and mul ---------------------- *)

  (* Non-negatives are closed under [+] and [*], and adding a positive to a
     non-negative is positive.  Sound under round-to-nearest for the same
     reason as [sub_nonpos].  Stated in this specialised form rather than as
     general monotonicity because these hypotheses rule out the [inf + -inf]
     and [0 * inf] NaN cases with no side conditions. *)
  add_nonneg : (x : t) -> (y : t) ->
    Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (add x y));

  add_pos : (x : t) -> (y : t) ->
    Lemma (requires lte zero x /\ lte zero y /\ (lt zero x \/ lt zero y))
          (ensures lt zero (add x y));

  mul_nonneg : (x : t) -> (y : t) ->
    Lemma (requires lte zero x /\ lte zero y) (ensures lte zero (mul x y));
}

(* ------------------------------------------------------------------ *)
(* Part 2: the laws discharge the obligations.                         *)
(* ------------------------------------------------------------------ *)

(* Negative infinity: what the kernel folds its running maxima from, and
   what masking a key drives its score to. *)
unfold let ninf (#t : Type) {| flt t |} : t = zero `sub` infinity

let ninf_kind (#t : Type) {| flt t |} ()
  : Lemma (kind (ninf #t) == Infinite)
  = neg_kind (infinity <: t)

(* A score, as [all_finite] constrains it: either a real number, or the
   [-inf] that masking produces. *)
let ok (#t : Type) {| flt t |} (x : t) : prop =
  Finite? (kind x) \/ x == ninf

let ok_not_nan (#t : Type) {| flt t |} (x : t)
  : Lemma (requires ok x) (ensures ~(NaN? (kind x)))
  = ninf_kind #t ()

let lte_refl (#t : Type) {| flt t |} (x : t)
  : Lemma (requires ~(NaN? (kind x))) (ensures lte x x)
  = lte_is_lt_or_eq x x

(* [<] entails [<=].  Needs [cmp_nan] to know the operands are numbers. *)
let lt_imp_lte (#t : Type) {| flt t |} (x y : t)
  : Lemma (requires lt x y) (ensures lte x y)
  = FStar.Classical.move_requires (cmp_nan x) y;
    lte_is_lt_or_eq x y

(* A non-negative value is a number.  Same story: without [cmp_nan] an
   ordering hypothesis tells you nothing about [kind]. *)
let nonneg_not_nan (#t : Type) {| flt t |} (x : t)
  : Lemma (requires lte zero x) (ensures ~(NaN? (kind x)))
  = FStar.Classical.move_requires (cmp_nan (zero <: t)) x

(* [-inf] is the least non-NaN value.  Derivable from the existing laws --
   but only once [lt_neg_flip] is corrected, which is why the class above
   restates it. *)
let ninf_least (#t : Type) {| flt t |} (x : t)
  : Lemma (requires ~(NaN? (kind x))) (ensures lte (ninf #t) x)
  = ninf_kind #t ();
    neg_kind x;
    neg_neg (infinity <: t);
    neg_neg x;
    infinity_val_spec (zero `sub` x);
    lt_neg_flip x (ninf #t)

(* Hence a value that is [<= -inf] and not NaN IS [-inf]: masking is the
   only way a score reaches the bottom of the order. *)
let ok_lte_ninf (#t : Type) {| flt t |} (x : t)
  : Lemma (requires ~(NaN? (kind x)) /\ lte x (ninf #t)) (ensures x == ninf #t)
  = ninf_kind #t (); ninf_least x

(* -------------------------------------------------------------- *)
(* Obligation (2): correction weights are finite.                  *)
(* -------------------------------------------------------------- *)

(* One online-softmax step: the running maximum moves from [m] to
   [fmax m s], and the kernel rescales its accumulators by
   [fexp (m - fmax m s)].  This is [Spec.Step.cw_at]. *)

let fmax_lte_left (#t : Type) {| flt t |} (m s : t)
  : Lemma (requires ~(NaN? (kind m)) /\ ~(NaN? (kind s)))
          (ensures lte m (fmax m s))
  = fmax_spec m s; lte_refl m

(* THE OBLIGATION.  The only case the laws cannot cover is [m == s == -inf],
   where the shift is genuinely [inf - inf = NaN].  That case is exactly a
   key tile in which every key is masked out; the kernel's tile count
   [SF.key_tiles] is cut at the causal bound so that a warp never visits
   such a tile, which is how the hypothesis is met in context. *)
let cw_finite (#t : Type) {| flt t |} (m s : t)
  : Lemma (requires ok m /\ ok s /\ ~(m == ninf /\ s == ninf))
          (ensures Finite? (kind (fexp (m `sub` fmax m s))))
  = ok_not_nan m; ok_not_nan s; ninf_kind #t ();
    fmax_spec m s;
    fmax_lte_left m s;
    introduce lte s (ninf #t) ==> s == ninf #t
    with _. ok_lte_ninf s;
    sub_nan_spec m (fmax m s);
    sub_nonpos m (fmax m s);
    exp_nonpos (m `sub` fmax m s)

(* The rescaling factor also lies in [0, 1] -- the quantitative half of the
   same law, which is what keeps the accumulators from growing. *)
let cw_bounded (#t : Type) {| flt t |} (m s : t)
  : Lemma (requires ok m /\ ok s /\ ~(m == ninf /\ s == ninf))
          (ensures lte zero (fexp (m `sub` fmax m s)) /\
                   lte (fexp (m `sub` fmax m s)) one)
  = ok_not_nan m; ok_not_nan s; ninf_kind #t ();
    fmax_spec m s;
    fmax_lte_left m s;
    introduce lte s (ninf #t) ==> s == ninf #t
    with _. ok_lte_ninf s;
    sub_nan_spec m (fmax m s);
    sub_nonpos m (fmax m s);
    exp_nonpos (m `sub` fmax m s)

(* -------------------------------------------------------------- *)
(* Obligation (3): the epilogue denominator is positive.           *)
(* -------------------------------------------------------------- *)

(* A faithful miniature of [Spec.Float.sel_prob] and [.row_sum]: a warp's
   partial denominator for one query row.  Masked keys ([-inf]) contribute
   nothing; the rest contribute [exp (score - max)]. *)

let sel_prob (#t : Type) {| flt t |} (sv mnew : t) : t =
  if eq sv ninf then zero else fexp (sv `sub` mnew)

let rec row_sum (#t : Type) {| flt t |} (s : nat -> t) (mnew : t) (k : nat)
  : Tot t (decreases k)
  = if k = 0 then zero else add (row_sum s mnew (k - 1)) (sel_prob (s (k - 1)) mnew)

(* Every summand is non-negative: [exp] of a non-positive shift. *)
let sel_prob_nonneg (#t : Type) {| flt t |} (sv mnew : t)
  : Lemma (requires ok sv /\ ok mnew /\ lte sv mnew)
          (ensures lte zero (sel_prob sv mnew))
  = ok_not_nan sv; ok_not_nan mnew; ninf_kind #t ();
    lte_refl (zero <: t);
    if eq sv ninf then ()
    else begin
      eq_spec sv ninf;
      sub_nan_spec sv mnew;
      sub_nonpos sv mnew;
      exp_nonpos (sv `sub` mnew)
    end

let rec row_sum_nonneg (#t : Type) {| flt t |} (s : nat -> t) (mnew : t) (k : nat)
  : Lemma (requires ok mnew /\ (forall (j : nat). j < k ==> ok (s j) /\ lte (s j) mnew))
          (ensures lte zero (row_sum s mnew k))
          (decreases k)
  = if k = 0 then lte_refl (zero <: t)
    else begin
      row_sum_nonneg s mnew (k - 1);
      sel_prob_nonneg (s (k - 1)) mnew;
      add_nonneg (row_sum s mnew (k - 1)) (sel_prob (s (k - 1)) mnew)
    end

(* The key that attains the row maximum contributes exactly [exp 0 == 1].
   This is where the residual "row is non-empty" condition is discharged. *)
let sel_prob_argmax (#t : Type) {| flt t |} (mnew : t)
  : Lemma (requires Finite? (kind mnew))
          (ensures sel_prob mnew mnew == one)
  = ninf_kind #t ();
    eq_spec mnew ninf;
    sub_self mnew;
    exp_zero #t

let rec row_sum_pos (#t : Type) {| flt t |} (s : nat -> t) (mnew : t) (k : nat)
  : Lemma (requires Finite? (kind mnew) /\
                    (forall (j : nat). j < k ==> ok (s j) /\ lte (s j) mnew) /\
                    (exists (j : nat). j < k /\ s j == mnew))
          (ensures lt zero (row_sum s mnew k))
          (decreases k)
  = ninf_kind #t ();
    if k = 0 then ()
    else begin
      ok_not_nan (s (k - 1));
      eq_spec (s (k - 1)) mnew;
      if eq (s (k - 1)) mnew then begin
        row_sum_nonneg s mnew (k - 1);
        sel_prob_argmax mnew;
        one_pos #t;
        lt_imp_lte (zero <: t) one;
        add_pos (row_sum s mnew (k - 1)) (sel_prob (s (k - 1)) mnew)
      end else begin
        row_sum_pos s mnew (k - 1);
        sel_prob_nonneg (s (k - 1)) mnew;
        lt_imp_lte (zero <: t) (row_sum s mnew (k - 1));
        add_pos (row_sum s mnew (k - 1)) (sel_prob (s (k - 1)) mnew)
      end
    end

(* The cross-warp combine: [Spec.Float.gmax], [.gscale] and [.gsum].  Warp
   [w] published a partial maximum [em w] and denominator [el w]; warp 0
   folds them to a block-wide maximum and rescales. *)

let rec gmax (#t : Type) {| flt t |} (em : nat -> t) (n : nat)
  : Tot t (decreases n)
  = if n = 0 then ninf else fmax (gmax em (n - 1)) (em (n - 1))

let gscale (#t : Type) {| flt t |} (em : nat -> t) (gm : t) (w : nat) : t =
  fexp (em w `sub` gm)

let rec gsum (#t : Type) {| flt t |} (em el : nat -> t) (gm : t) (n : nat)
  : Tot t (decreases n)
  = if n = 0 then zero
    else add (gsum em el gm (n - 1)) (mul (gscale em gm (n - 1)) (el (n - 1)))

let rec gmax_ok (#t : Type) {| flt t |} (em : nat -> t) (n : nat)
  : Lemma (requires forall (w : nat). w < n ==> ok (em w))
          (ensures ok (gmax em n)) (decreases n)
  = ninf_kind #t ();
    if n = 0 then ()
    else begin
      gmax_ok em (n - 1);
      ok_not_nan (gmax em (n - 1));
      ok_not_nan (em (n - 1));
      fmax_spec (gmax em (n - 1)) (em (n - 1))
    end

(* The block maximum dominates every warp's maximum. *)
let rec gmax_ub (#t : Type) {| flt t |} (em : nat -> t) (n : nat) (w : nat)
  : Lemma (requires w < n /\ (forall (u : nat). u < n ==> ok (em u)))
          (ensures lte (em w) (gmax em n)) (decreases n)
  = ninf_kind #t ();
    gmax_ok em (n - 1);
    ok_not_nan (gmax em (n - 1));
    ok_not_nan (em (n - 1));
    fmax_spec (gmax em (n - 1)) (em (n - 1));
    if w = n - 1 then lte_refl (em (n - 1))
    else begin
      gmax_ub em (n - 1) w;
      ok_not_nan (em w);
      if lt (gmax em (n - 1)) (em (n - 1)) then begin
        lt_imp_lte (gmax em (n - 1)) (em (n - 1));
        lte_trans (em w) (gmax em (n - 1)) (em (n - 1))
      end
    end

let gscale_nonneg (#t : Type) {| flt t |} (em : nat -> t) (gm : t) (w : nat)
  : Lemma (requires ok (em w) /\ Finite? (kind gm) /\ lte (em w) gm)
          (ensures lte zero (gscale em gm w))
  = ok_not_nan (em w); ninf_kind #t ();
    sub_nan_spec (em w) gm;
    sub_nonpos (em w) gm;
    exp_nonpos (em w `sub` gm)

let rec gsum_nonneg (#t : Type) {| flt t |} (em el : nat -> t) (gm : t) (n : nat)
  : Lemma (requires Finite? (kind gm) /\
                    (forall (w : nat). w < n ==>
                       ok (em w) /\ lte (em w) gm /\ lte zero (el w)))
          (ensures lte zero (gsum em el gm n)) (decreases n)
  = if n = 0 then lte_refl (zero <: t)
    else begin
      gsum_nonneg em el gm (n - 1);
      gscale_nonneg em gm (n - 1);
      mul_nonneg (gscale em gm (n - 1)) (el (n - 1));
      add_nonneg (gsum em el gm (n - 1)) (mul (gscale em gm (n - 1)) (el (n - 1)))
    end

(* The warp that attains the block maximum contributes [exp 0 * l == l]. *)
let gscale_argmax (#t : Type) {| flt t |} (em : nat -> t) (gm : t) (w : nat)
  : Lemma (requires em w == gm /\ Finite? (kind gm))
          (ensures gscale em gm w == one)
  = sub_self gm; exp_zero #t

let rec gsum_pos (#t : Type) {| flt t |} (em el : nat -> t) (gm : t) (n : nat)
  : Lemma (requires Finite? (kind gm) /\
                    (forall (w : nat). w < n ==>
                       ok (em w) /\ lte (em w) gm /\ lte zero (el w)) /\
                    (exists (w : nat). w < n /\ em w == gm /\ lt zero (el w)))
          (ensures lt zero (gsum em el gm n)) (decreases n)
  = if n = 0 then ()
    else begin
      ok_not_nan (em (n - 1));
      eq_spec (em (n - 1)) gm;
      if eq (em (n - 1)) gm && lt zero (el (n - 1)) then begin
        gsum_nonneg em el gm (n - 1);
        gscale_argmax em gm (n - 1);
        nonneg_not_nan (el (n - 1));
        mul_comm (el (n - 1)) (one <: t);
        mul_one (el (n - 1));
        lt_imp_lte (zero <: t) (el (n - 1));
        add_pos (gsum em el gm (n - 1)) (mul (gscale em gm (n - 1)) (el (n - 1)))
      end else begin
        gsum_pos em el gm (n - 1);
        gscale_nonneg em gm (n - 1);
        mul_nonneg (gscale em gm (n - 1)) (el (n - 1));
        lt_imp_lte (zero <: t) (gsum em el gm (n - 1));
        add_pos (gsum em el gm (n - 1)) (mul (gscale em gm (n - 1)) (el (n - 1)))
      end
    end

(* -------------------------------------------------------------- *)
(* The two obligations, as the kernel proof would use them.        *)
(* -------------------------------------------------------------- *)

(* (3), assembled: the epilogue denominator [gsum] is positive as soon as
   ONE warp's partial denominator is, which [row_sum_pos] delivers from
   "row [i] admits at least one key with a finite score".  Note the only
   hypotheses left are consequences of [all_finite] plus that single
   non-emptiness condition -- no statement about correction weights, and
   nothing that mentions the kernel's internal state. *)
let denominator_pos
  (#t : Type) {| flt t |}
  (em el : nat -> t) (n : nat) (w0 : nat)
  (s : nat -> t) (k : nat)
  : Lemma
      (requires
        (forall (w : nat). w < n ==> ok (em w)) /\
        (forall (w : nat). w < n ==> lte zero (el w)) /\
        w0 < n /\ em w0 == gmax em n /\ Finite? (kind (gmax em n)) /\
        (* warp [w0]'s denominator is the [row_sum] of its own scores, and
           its own maximum is attained by one of them *)
        el w0 == row_sum s (em w0) k /\
        (forall (j : nat). j < k ==> ok (s j) /\ lte (s j) (em w0)) /\
        (exists (j : nat). j < k /\ s j == em w0))
      (ensures lt zero (gsum em el (gmax em n) n))
  = gmax_ok em n;
    row_sum_pos s (em w0) k;
    introduce forall (w : nat). w < n ==> lte (em w) (gmax em n)
    with introduce _ ==> _
    with _. gmax_ub em n w;
    gsum_pos em el (gmax em n) n
