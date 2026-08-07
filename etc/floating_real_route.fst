module Floating_real_route

(* CAN THE FLOAT-LEVEL AXIOMS BE REPLACED BY REAL-LEVEL REASONING?

   The suggestion under review: state the laws on reals and recover the float
   facts by approximation, since "any finite float should have a to_real", so
   only the genuinely NaN/finiteness-flavoured axioms should be needed.

   This file answers that question mechanically rather than by argument.  It
   is NOT part of the build; check it with

     ./fstar.sh etc/floating_real_route.fst

   SUMMARY OF THE ANSWER.

   Part 1 shows the approximation machinery cannot, on its own, transfer
   ANYTHING back from the reals: [v_approximates := fun _ _ -> True] is a
   model of [real_like] + [floating_real_like].  Every law of both classes
   holds trivially under it, so no float fact whatsoever is derivable from
   them.  This is not a quirk -- it is forced by the design.  [v_approximates]
   is a one-directional, error-tolerant relation ([a_add] holds with no
   rounding side condition, so it must absorb accumulated error), and it is
   deliberately abstract so instances may pick their own error model.

   The only law that rules the trivial model out is [precise_real_like]'s
   [v_approximates_inj].  In the installed tree that class is instantiated for
   exactly four types -- [u8], [u16], [u32], [u64] -- and for NO float type.
   That is not an oversight: it CANNOT hold for a float type, because
   [a_add]/[a_mul] force [v_approximates] to relate a rounded result to the
   exact real, so many distinct floats approximate a given real.

   Part 2 grants the route its two missing bridges anyway, as generously as
   possible:
     - [b_inj]  -- [precise_real_like], which no float type has;
     - [b_ord]  -- [lte x y <==> to_real x <=. to_real y], which exists
                   NOWHERE in Kuiper: there is no law of any kind relating
                   the float order to the real order.
   With both, three of the eight challenged axioms DO become derivable:
   [lte_trans], [sub_self] and [exp_zero].  Those are proved below.

   Part 3 shows the remaining five -- [sub_nonpos], [one_pos], [add_nonneg],
   [add_pos], [mul_nonneg] -- still do not follow, even with both bridges.
   The obstruction is exact and is spelled out there: they need to know where
   [to_real (add x y)] sits, and the ONLY link from [add] to the reals is the
   error-tolerant [v_approximates], which says nothing about [to_real] of the
   result.  Closing that gap needs a rounding-error law -- precisely what
   [v_approximates] exists to abstract away.

   CONCLUSION.  Of the eleven proposed axioms, three ([cmp_nan],
   [sub_nan_spec], [exp_nonpos]) were already agreed.  Three more
   ([lte_trans], [sub_self], [exp_zero]) could be dropped, but only in
   exchange for [b_inj] and [b_ord] -- and [b_inj] is unsound for floats, so
   in practice only [b_ord] is available and it buys just [lte_trans].  The
   five sign/monotonicity axioms cannot be obtained this way at all. *)

open FStar.Real
open FStar.Math.Exp

type fkind = | Finite | Infinite | NaN

(* ------------------------------------------------------------------ *)
(* The float class: the subset of [Kuiper.Floating.Base.floating] used *)
(* here, with the corrected [lt_neg_flip], plus the three axioms       *)
(* already agreed.                                                     *)
(* ------------------------------------------------------------------ *)

class flt (t : Type) = {
  zero : t; one : t; infinity : t;
  add : t -> t -> t;
  mul : t -> t -> t;
  sub : t -> t -> t;
  eq : t -> t -> bool;
  lt : t -> t -> bool;
  lte : t -> t -> bool;
  kind : t -> fkind;
  fmax : t -> t -> t;
  fexp : t -> t;
  fdiv : t -> t -> t;

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

  add_comm : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures add x y == add y x);

  add_zero : (x : t) ->
    Lemma (requires ~(NaN? (kind x))) (ensures add x zero == x);

  (* Already agreed: comparisons are false at NaN. *)
  cmp_nan : (x : t) -> (y : t) ->
    Lemma (requires NaN? (kind x) \/ NaN? (kind y))
          (ensures ~(lt x y) /\ ~(lte x y) /\ ~(eq x y));
}

(* ------------------------------------------------------------------ *)
(* Part 1: the approximation classes transfer nothing back.            *)
(* ------------------------------------------------------------------ *)

(* [real_like] and [floating_real_like] of [Kuiper.Approximates.Base],
   restated field for field as a record so it can be instantiated. *)
noeq type rl (t : Type) {| flt t |} = {
  tr   : t -> real;
  vapp : t -> real -> prop;

  r_to_real_ok : (x : t) -> Lemma (vapp x (tr x));
  r_a0 : squash (vapp zero 0.0R);
  r_a1 : squash (vapp one 1.0R);

  r_a_add : (x : t) -> (y : t) -> (r : real) -> (s : real) ->
    Lemma (requires vapp x r /\ vapp y s) (ensures vapp (add x y) (r +. s));
  r_a_mul : (x : t) -> (y : t) -> (r : real) -> (s : real) ->
    Lemma (requires vapp x r /\ vapp y s) (ensures vapp (mul x y) (r *. s));

  r_fmax : (x : t) -> (y : t) -> (r : real) -> (s : real) ->
    Lemma (requires vapp x r /\ vapp y s)
          (ensures vapp (fmax x y) (if r >. s then r else s));
  r_sub : (x : t) -> (y : t) -> (r : real) -> (s : real) ->
    Lemma (requires vapp x r /\ vapp y s) (ensures vapp (sub x y) (r -. s));
  r_exp : (x : t) -> (r : real) ->
    Lemma (requires vapp x r) (ensures vapp (fexp x) (exp r));
  r_div : (x : t) -> (y : t) -> (r : real) -> (s : real { s =!= 0.0R }) ->
    Lemma (requires vapp x r /\ vapp y s) (ensures vapp (fdiv x y) (r /. s));
}

(* THE KEY FACT.  The everywhere-true relation is a model of both classes,
   for ANY float type and ANY [to_real].  Since it carries no information,
   nothing about [add], [mul], [sub], [fexp] or the float order can be
   derived from [real_like] / [floating_real_like]. *)
let trivial_model (#t : Type) {| flt t |} : rl t = {
  tr   = (fun _ -> 0.0R);
  vapp = (fun _ _ -> True);
  r_to_real_ok = (fun _ -> ());
  r_a0 = (); r_a1 = ();
  r_a_add = (fun _ _ _ _ -> ());
  r_a_mul = (fun _ _ _ _ -> ());
  r_fmax  = (fun _ _ _ _ -> ());
  r_sub   = (fun _ _ _ _ -> ());
  r_exp   = (fun _ _ -> ());
  r_div   = (fun _ _ _ _ -> ());
}

(* ------------------------------------------------------------------ *)
(* Part 2: grant the two missing bridges.                              *)
(* ------------------------------------------------------------------ *)

noeq type bridge (#t : Type) {| flt t |} (m : rl t) = {
  (* [precise_real_like.v_approximates_inj].  Instantiated in Kuiper for
     [u8], [u16], [u32], [u64] only -- for no float type, and it cannot hold
     for one. *)
  b_inj : (x : t) -> (y : t) -> (r : real) ->
    Lemma (requires m.vapp x r /\ m.vapp y r) (ensures x == y);

  (* Float order reflects real order.  No such law exists anywhere in
     Kuiper; this is sound for floats and would be a reasonable addition. *)
  b_ord : (x : t) -> (y : t) ->
    Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)))
          (ensures lte x y <==> m.tr x <=. m.tr y);
}

(* --- 2a. [lte_trans] follows from [b_ord] alone. --- *)
let lte_trans (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (x y z : t)
  : Lemma (requires lte x y /\ lte y z) (ensures lte x z)
  = FStar.Classical.move_requires (cmp_nan x) y;
    FStar.Classical.move_requires (cmp_nan y) z;
    b.b_ord x y; b.b_ord y z; b.b_ord x z

(* --- 2b. [sub_self] follows from [b_inj]. --- *)
let sub_self (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (x : t)
  : Lemma (ensures x `sub` x == zero)
  = m.r_to_real_ok x;
    m.r_sub x x (m.tr x) (m.tr x);
    b.b_inj (x `sub` x) zero 0.0R

(* --- 2c. [exp_zero] follows from [b_inj]. --- *)
let exp_zero (#t : Type) {| flt t |} (m : rl t) (b : bridge m) ()
  : Lemma (ensures fexp (zero <: t) == one)
  = exp_base ();
    m.r_exp zero 0.0R;
    b.b_inj (fexp (zero <: t)) one 1.0R

(* ------------------------------------------------------------------ *)
(* Part 3: the sign / monotonicity axioms still do not follow.         *)
(* ------------------------------------------------------------------ *)

(* THE OBSTRUCTION, concretely.  To place [add x y] in the float order,
   [b_ord] demands a fact about [m.tr (add x y)].  What is available is
   [m.r_a_add], which yields [m.vapp (add x y) (m.tr x +. m.tr y)] -- a
   statement about [vapp], not about [tr].

   The only law pointing from [vapp] back to [tr] is [b_inj], and it needs
   TWO FLOATS approximating ONE real.  Here we have one float ([add x y])
   approximating two reals (the exact sum, and [tr (add x y)] via
   [r_to_real_ok]).  That is the other direction -- functionality of [vapp]
   in its real argument -- and it is exactly what must NOT hold for a float
   type, since [add] rounds: [add x y] approximates the exact sum while
   [tr (add x y)] is the rounded value, and the two differ.

   So no chain reaches [m.tr (add x y)], and each attempt below fails.  Each
   is given every relevant hypothesis, both bridges, and the fully expanded
   real-side reasoning. *)

[@@expect_failure]
let add_nonneg_via_reals (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ ~(NaN? (kind (add x y))))
          (ensures lte zero (add x y))
  = FStar.Classical.move_requires (cmp_nan (zero <: t)) x;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) y;
    b.b_ord zero x; b.b_ord zero y;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_a_add x y (m.tr x) (m.tr y);   (* vapp (add x y) (tr x +. tr y) *)
    m.r_to_real_ok (add x y);          (* vapp (add x y) (tr (add x y))  *)
    (* nothing connects the two reals above *)
    b.b_ord zero (add x y)

[@@expect_failure]
let add_pos_via_reals (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ (lt zero x \/ lt zero y) /\
                    ~(NaN? (kind (add x y))))
          (ensures lt zero (add x y))
  = FStar.Classical.move_requires (cmp_nan (zero <: t)) x;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) y;
    b.b_ord zero x; b.b_ord zero y;
    b.b_ord x zero; b.b_ord y zero;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_a_add x y (m.tr x) (m.tr y);
    m.r_to_real_ok (add x y);
    b.b_ord (add x y) zero

[@@expect_failure]
let mul_nonneg_via_reals (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ ~(NaN? (kind (mul x y))))
          (ensures lte zero (mul x y))
  = FStar.Classical.move_requires (cmp_nan (zero <: t)) x;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) y;
    b.b_ord zero x; b.b_ord zero y;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_a_mul x y (m.tr x) (m.tr y);
    m.r_to_real_ok (mul x y);
    b.b_ord zero (mul x y)

[@@expect_failure]
let sub_nonpos_via_reals (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ lte x y /\
                    ~(NaN? (kind (x `sub` y))))
          (ensures lte (x `sub` y) zero)
  = b.b_ord x y;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_sub x y (m.tr x) (m.tr y);
    m.r_to_real_ok (x `sub` y);
    b.b_ord (x `sub` y) zero

(* [one_pos] fails for a second, independent reason: nothing pins [tr zero]
   to [0.0R] or [tr one] to [1.0R].  [r_a0] / [r_a1] constrain [vapp], and
   [b_inj] cannot be used to pull a value of [tr] out of a [vapp] fact -- it
   again needs two floats and one real, not one float and two reals. *)
[@@expect_failure]
let one_pos_via_reals (#t : Type) {| flt t |} (m : rl t) (b : bridge m) ()
  : Lemma (ensures lt (zero <: t) one)
  = m.r_to_real_ok (zero <: t);
    m.r_to_real_ok (one <: t);
    b.b_ord (zero <: t) one;
    b.b_ord (one <: t) zero

(* ------------------------------------------------------------------ *)
(* Part 4: positive controls -- the failures above are attributable.   *)
(* ------------------------------------------------------------------ *)

(* Each attempt in Part 3 is now replayed with TWO extra hypotheses: that
   [vapp] is functional in its real argument -- the missing link named above
   -- and that the result is not NaN.  All five then go through, with no
   other change.  That pins the Part-3 failures on exactly these gaps and
   rules out a mistake in the attempts themselves.

   The second hypothesis is not free either: nothing in either class says
   arithmetic on non-NaN operands yields a non-NaN result, and [b_ord] is
   unusable at NaN.  So the real-level route would need a NaN-propagation
   law as well -- which is float-level content by construction. *)

let functional (#t : Type) {| flt t |} (m : rl t) =
  (x : t) -> (r : real) -> Lemma (requires m.vapp x r) (ensures r == m.tr x)

let one_pos_given_functionality
  (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (func : functional m) ()
  : Lemma (ensures lt (zero <: t) one)
  = func zero 0.0R;
    func one 1.0R;
    b.b_ord (one <: t) zero;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) one

let add_nonneg_given_functionality
  (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (func : functional m) (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ ~(NaN? (kind (add x y))))
          (ensures lte zero (add x y))
  = FStar.Classical.move_requires (cmp_nan (zero <: t)) x;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) y;
    b.b_ord zero x; b.b_ord zero y;
    func zero 0.0R;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_a_add x y (m.tr x) (m.tr y);
    func (add x y) (m.tr x +. m.tr y);
    b.b_ord zero (add x y)

let mul_nonneg_given_functionality
  (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (func : functional m) (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ ~(NaN? (kind (mul x y))))
          (ensures lte zero (mul x y))
  = FStar.Classical.move_requires (cmp_nan (zero <: t)) x;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) y;
    b.b_ord zero x; b.b_ord zero y;
    func zero 0.0R;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_a_mul x y (m.tr x) (m.tr y);
    func (mul x y) (m.tr x *. m.tr y);
    b.b_ord zero (mul x y)

let sub_nonpos_given_functionality
  (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (func : functional m) (x y : t)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ lte x y /\
                    ~(NaN? (kind (x `sub` y))))
          (ensures lte (x `sub` y) zero)
  = func zero 0.0R;
    b.b_ord x y;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_sub x y (m.tr x) (m.tr y);
    func (x `sub` y) (m.tr x -. m.tr y);
    b.b_ord (x `sub` y) zero

let add_pos_given_functionality
  (#t : Type) {| flt t |} (m : rl t) (b : bridge m) (func : functional m) (x y : t)
  : Lemma (requires lte zero x /\ lte zero y /\ (lt zero x \/ lt zero y) /\
                    ~(NaN? (kind (add x y))))
          (ensures lt zero (add x y))
  = func zero 0.0R;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) x;
    FStar.Classical.move_requires (cmp_nan (zero <: t)) y;
    b.b_ord zero x; b.b_ord zero y;
    b.b_ord x zero; b.b_ord y zero;
    m.r_to_real_ok x; m.r_to_real_ok y;
    m.r_a_add x y (m.tr x) (m.tr y);
    func (add x y) (m.tr x +. m.tr y);
    b.b_ord (add x y) zero

(* AND YET [functional] IS FALSE FOR EVERY FLOAT TYPE.  It says the real a
   float approximates is uniquely determined -- the exact opposite of what
   [r_a_add] forces.  Concretely, for [x] the unit in the last place of
   [one]:
     [r_a_add one x 1.0R (tr x)]  gives  [vapp (add one x) (1.0R +. tr x)],
   while [add one x == one] once [x] falls below half an ulp, so [functional]
   would give [1.0R +. tr x == tr one], i.e. [tr x == 0.0R] for every such
   nonzero [x].  With [b_ord] that forces [lte x zero] and [lte zero x], so
   [x == zero] -- false.

   This is why [precise_real_like] is instantiated in Kuiper for [u8], [u16],
   [u32] and [u64] and for no float type: exact arithmetic admits it, rounded
   arithmetic cannot.  Hence the five sign axioms have to be stated at the
   float level. *)
