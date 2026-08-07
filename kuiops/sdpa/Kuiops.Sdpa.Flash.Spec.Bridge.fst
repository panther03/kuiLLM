module Kuiops.Sdpa.Flash.Spec.Bridge

(* Bridge from the float-level description of the online-softmax update
   ([Kuiops.Sdpa.Flash.Spec.Float]) to the real-valued flash invariant
   ([Kuiops.Sdpa.Flash.Spec.Online]).

   The kernel encodes "key not admitted" as the [-inf] sentinel, so the running
   maximum is [-inf] exactly while no key has been absorbed.  That state
   approximates no real, so the invariant carried through the key loop is a
   case split on whether the absorbed key set is still empty: either nothing
   has been absorbed and the registers still hold the sentinel and zero, or
   some key has been absorbed and the registers approximate the real invariant
   at *some* running maximum.  Which real the maximum encodes never matters --
   see the header of [Kuiops.Sdpa.Flash.Spec.Online]. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Floating
open Kuiper.Approximates

module SF = Kuiops.Sdpa.Flash.Spec.Float
module SO = Kuiops.Sdpa.Flash.Spec.Online

(* ------------------------------------------------------------------ *)
(* Constants.                                                          *)
(* ------------------------------------------------------------------ *)

let zero_finite (#et : Type0) {| floating et |} ()
  : Lemma (kind (zero #et) == Finite)
  = kind_zero #et

let zero_approx (#et : Type0) {| floating et |} {| real_like et |} ()
  : Lemma ((zero #et) %~ 0.0R)
  = a0 #et

let ninf_not_nan (#et : Type0) {| floating et |} ()
  : Lemma (not_nan (neg #et infinity))
  = neg_kind (infinity #et)

(* ------------------------------------------------------------------ *)
(* The [-inf] sentinel.                                                *)
(* ------------------------------------------------------------------ *)

let lte_ninf (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires not_nan x) (ensures lte (neg #et infinity) x)
  = neg_kind x;
    infinity_val_spec (neg x);
    lte_is_lt_or_eq (neg x) infinity;
    lt_neg_flip (neg x) infinity;
    neg_neg x;
    lte_is_lt_or_eq (neg #et infinity) x

let fmax_ninf_l (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires not_nan x) (ensures fmax (neg #et infinity) x == x)
  = lte_ninf x;
    ninf_not_nan #et ();
    fmax_spec #et (neg #et infinity) x;
    negate_lt_is_lte #et x (neg #et infinity)

let fmax_ninf_r (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires not_nan x) (ensures fmax x (neg #et infinity) == x)
  = lte_ninf x;
    ninf_not_nan #et ();
    fmax_spec #et x (neg #et infinity);
    negate_lt_is_lte #et x (neg #et infinity)

let fmax_not_nan (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires not_nan x /\ not_nan y) (ensures not_nan (fmax x y))
  = fmax_spec x y

(* A masked entry compares equal to the sentinel, an admitted one does not. *)
let sel_prob_masked (#et : Type0) {| floating et |} (sv mnew : et)
  : Lemma (requires sv == neg #et infinity)
          (ensures SF.sel_prob sv mnew == zero)
  = ninf_not_nan #et ();
    eq_spec sv (neg #et infinity)

let sel_prob_admitted (#et : Type0) {| floating et |} (sv mnew : et)
  : Lemma (requires Finite? (kind sv))
          (ensures SF.sel_prob sv mnew == fexp (sv `sub` mnew))
  = ninf_not_nan #et ();
    eq_spec sv (neg #et infinity)

(* ------------------------------------------------------------------ *)
(* Emptiness of a key set, as a decidable ghost test.                  *)
(* ------------------------------------------------------------------ *)

let rec pnone (#n : nat) (p : SO.pred n) (k : nat { k <= n })
  : GTot bool (decreases k)
  = if k = 0 then true else pnone p (k - 1) && not (p (k - 1))

let rec pnone_spec (#n : nat) (p : SO.pred n) (k : nat { k <= n })
  : Lemma (ensures pnone p k <==> (forall (t : natlt n). t < k ==> ~(p t)))
          (decreases k)
  = if k = 0 then () else pnone_spec p (k - 1)

let pnone_ext (#n : nat) (p q : SO.pred n)
  : Lemma (requires forall (j : natlt n). p j == q j)
          (ensures pnone p n == pnone q n)
  = pnone_spec p n; pnone_spec q n

let pnone_witness (#n : nat) (p : SO.pred n) (j : natlt n)
  : Lemma (requires p j) (ensures ~(pnone p n))
  = pnone_spec p n

(* ------------------------------------------------------------------ *)
(* The row maximum, skipping masked entries.                           *)
(* ------------------------------------------------------------------ *)

let rec row_max_where
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (q : SO.pred bn) (k : nat { k <= bn })
  : GTot et (decreases k)
  = if k = 0 then neg infinity
    else if q (k - 1) then fmax (row_max_where s q (k - 1)) (acc1 s (k - 1))
    else row_max_where s q (k - 1)

let rec row_max_where_not_nan
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (q : SO.pred bn) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). not_nan (acc1 s t))
          (ensures not_nan (row_max_where s q k))
          (decreases k)
  = if k = 0 then ninf_not_nan #et ()
    else begin
      row_max_where_not_nan s q (k - 1);
      fmax_not_nan (row_max_where s q (k - 1)) (acc1 s (k - 1))
    end

let rec row_max_where_none
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (q : SO.pred bn) (k : nat { k <= bn })
  : Lemma (requires pnone q k)
          (ensures row_max_where s q k == neg #et infinity)
          (decreases k)
  = if k = 0 then () else row_max_where_none s q (k - 1)

(* With masked entries carrying the sentinel, the kernel's unconditional fold
   agrees with the fold over the admitted entries only. *)
let rec row_max_masked
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (q : SO.pred bn) (k : nat { k <= bn })
  : Lemma (requires (forall (t : natlt bn). not_nan (acc1 s t)) /\
                    (forall (t : natlt bn). ~(q t) ==> acc1 s t == neg infinity))
          (ensures SF.row_max s k == row_max_where s q k)
          (decreases k)
  = if k = 0 then ()
    else begin
      row_max_masked s q (k - 1);
      row_max_where_not_nan s q (k - 1);
      if q (k - 1) then () else fmax_ninf_r (row_max_where s q (k - 1))
    end

(* One step of [row_max_where_approx], with the witness made explicit so the
   existential can be eliminated without guessing. *)
let row_max_where_approx_step
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#bn : nat) (s : chest1 et bn) (q : SO.pred bn)
  (k : nat { 0 < k /\ k <= bn }) (r rj : real)
  : Lemma (requires q (k - 1) /\ row_max_where s q (k - 1) %~ r /\
                    acc1 s (k - 1) %~ rj)
          (ensures exists (rr : real). row_max_where s q k %~ rr)
  = fmax_approx (row_max_where s q (k - 1)) (acc1 s (k - 1)) r rj

(* The running maximum approximates *some* real as soon as one entry is
   admitted.  Which one is irrelevant: the flash invariant holds at an
   arbitrary maximum. *)
let rec row_max_where_approx
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#bn : nat) (s : chest1 et bn) (q : SO.pred bn) (xt : natlt bn -> GTot real)
  (k : nat { k <= bn })
  : Lemma (requires (forall (t : natlt bn). not_nan (acc1 s t)) /\
                    (forall (t : natlt bn). t < k /\ q t ==> acc1 s t %~ xt t))
          (ensures pnone q k \/ (exists (r : real). row_max_where s q k %~ r))
          (decreases k)
  = if k = 0 then ()
    else begin
      row_max_where_approx s q xt (k - 1);
      if q (k - 1)
      then
        (if pnone q (k - 1)
         then (row_max_where_none s q (k - 1);
               fmax_ninf_l (acc1 s (k - 1)))
         else
           FStar.Classical.exists_elim
             (exists (rr : real). row_max_where s q k %~ rr)
             #real #(fun r -> row_max_where s q (k - 1) %~ r)
             ()
             (fun r -> row_max_where_approx_step s q k r (xt (k - 1))))
      else ()
    end

(* ------------------------------------------------------------------ *)
(* The row denominator.                                                *)
(* ------------------------------------------------------------------ *)

(* An all-sentinel row contributes nothing. *)
let rec row_sum_all_masked
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (mnew : et) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). t < k ==> acc1 s t == neg infinity)
          (ensures SF.row_sum s mnew k == zero)
          (decreases k)
  = if k = 0 then ()
    else begin
      row_sum_all_masked s mnew (k - 1);
      sel_prob_masked (acc1 s (k - 1)) mnew;
      zero_finite #et ();
      add_zero (zero #et)
    end

(* The kernel's unconditional select-to-zero fold approximates the masked real
   sum of [exp (x j - m)]. *)
let rec row_sum_approx
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#bn : nat) (s : chest1 et bn) (q : SO.pred bn) (xt : natlt bn -> GTot real)
  (mnew : et) (mr : real) (k : nat { k <= bn })
  : Lemma (requires mnew %~ mr /\
                    (forall (t : natlt bn). ~(q t) ==> acc1 s t == neg infinity) /\
                    (forall (t : natlt bn). t < k /\ q t ==>
                       (Finite? (kind (acc1 s t)) /\ acc1 s t %~ xt t)))
          (ensures SF.row_sum s mnew k
                     %~ SO.sum_upto (fun j -> exp (xt j -. mr)) q k)
          (decreases k)
  = if k = 0 then zero_approx #et ()
    else begin
      row_sum_approx s q xt mnew mr (k - 1);
      let sv : et = acc1 s (k - 1) in
      if q (k - 1)
      then begin
        sel_prob_admitted sv mnew;
        sub_approx sv mnew (xt (k - 1)) mr;
        exp_approx (sv `sub` mnew) (xt (k - 1) -. mr);
        a_add (SF.row_sum s mnew (k - 1)) (SF.sel_prob sv mnew)
              (SO.sum_upto (fun j -> exp (xt j -. mr)) q (k - 1))
              (exp (xt (k - 1) -. mr))
      end
      else begin
        sel_prob_masked sv mnew;
        zero_approx #et ();
        a_add (SF.row_sum s mnew (k - 1)) (SF.sel_prob sv mnew)
              (SO.sum_upto (fun j -> exp (xt j -. mr)) q (k - 1))
              0.0R
      end
    end

(* ------------------------------------------------------------------ *)
(* The float-level flash state and its step.                           *)
(* ------------------------------------------------------------------ *)

(* [m] and [l] are one lane's running maximum and denominator after absorbing
   exactly the keys in [p].  While [p] is empty the registers hold the sentinel
   and zero -- a state that approximates no real, and that the kernel exits as
   soon as it absorbs a key. *)
let ml_state
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x : natlt n -> GTot real) (p : SO.pred n) (m l : et) : prop
  = not_nan m /\
    (if pnone p n
     then (m == neg infinity /\ l == zero)
     else (exists (mr : real). m %~ mr /\ l %~ (SO.dsum x p /. exp mr)))

(* [ml_state] only depends on the extension of the key set. *)
let ml_state_ext
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x : natlt n -> GTot real) (p q : SO.pred n) (m l : et)
  : Lemma (requires ml_state x p m l /\ (forall (j : natlt n). p j == q j))
          (ensures ml_state x q m l)
  = pnone_ext p q;
    SO.sum_where_ext (fun j -> exp (x j)) (fun j -> exp (x j)) p q

(* Hypotheses shared by every case of the step: the tile [q] of [bn] local keys
   starting at [k0] describes exactly the part of [t] the lane absorbs, the
   scores [es] carry the sentinel on rejected keys and approximate [x] on
   admitted ones, and the registers are updated by the kernel's formulas. *)
let step_pre
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) : prop
  = SO.disjoint p t /\
    (forall (j : natlt n). p' j == (p j || t j)) /\
    (forall (j : natlt n). t j ==> (k0 <= j /\ j < k0 + bn)) /\
    (forall (i : natlt bn). k0 + i < n ==> q i == t (k0 + i)) /\
    (forall (i : natlt bn). k0 + i >= n ==> ~(q i)) /\
    (forall (i : natlt bn). q i ==> (k0 + i < n /\ xt i == x (k0 + i))) /\
    (forall (i : natlt bn). not_nan (acc1 es i)) /\
    (forall (i : natlt bn). ~(q i) ==> acc1 es i == neg infinity) /\
    (forall (i : natlt bn). q i ==>
       (Finite? (kind (acc1 es i)) /\ acc1 es i %~ xt i)) /\
    not_nan vm /\ Finite? (kind cw') /\
    m' == fmax vm (SF.row_max es bn) /\
    cw' == fexp (vm `sub` m') /\
    l' == (vl `mul` cw') `add` (SF.row_sum es m' bn)

(* The tile predicate is empty exactly when the tile absorbs nothing. *)
let tile_empty
  (#n #bn : nat) (t : SO.pred n) (q : SO.pred bn) (k0 : natle n)
  : Lemma (requires (forall (j : natlt n). t j ==> (k0 <= j /\ j < k0 + bn)) /\
                    (forall (i : natlt bn). k0 + i < n ==> q i == t (k0 + i)) /\
                    pnone q bn)
          (ensures forall (j : natlt n). ~(t j))
  = pnone_spec q bn;
    introduce forall (j : natlt n). ~(t j)
    with (if t j then (let i : natlt bn = j - k0 in assert (q i)) else ())

(* Denominator step, in the case where the lane has already absorbed a key.
   This is [SO.dstep] transported across the approximation. *)
let step_absorbed
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) (mr mr' : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    vm %~ mr /\ m' %~ mr' /\ vl %~ (SO.dsum x p /. exp mr))
          (ensures l' %~ (SO.dsum x p' /. exp mr'))
  = sub_approx vm m' mr mr';
    exp_approx (vm `sub` m') (mr -. mr');
    a_mul vl cw' (SO.dsum x p /. exp mr) (exp (mr -. mr'));
    row_sum_approx es q xt m' mr' bn;
    SO.sum_where_tile (fun j -> exp (x j -. mr')) t
                      (fun j -> exp (xt j -. mr')) q k0;
    a_add (vl `mul` cw') (SF.row_sum es m' bn)
          (SO.dsum x p /. exp mr *. exp (mr -. mr'))
          (SO.tsum_d x t mr');
    SO.dstep x p t p' mr mr' (SO.dsum x p /. exp mr)

(* Denominator step, in the case where this tile absorbs the lane's first key.
   The old accumulator is the literal zero, so the (meaningless) correction
   factor is annihilated -- this is the only place the correction's finiteness
   is used, via [mul_zero]'s [Finite?] side condition.  That side condition is
   what the reference kernel's [if (!isfinite(corr)) corr = 0.0f;]
   (flash_attn_fa1.cu l.173) discharges by construction. *)
let step_fresh
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) (mr' : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    pnone p n /\ vl == zero /\ m' %~ mr')
          (ensures l' %~ (SO.dsum x p' /. exp mr'))
  = zero_finite #et ();
    mul_comm (zero #et) cw';
    mul_zero cw';
    zero_approx #et ();
    pnone_spec p n;
    SO.sum_upto_false (fun j -> exp (x j)) p n;
    row_sum_approx es q xt m' mr' bn;
    SO.sum_where_tile (fun j -> exp (x j -. mr')) t
                      (fun j -> exp (xt j -. mr')) q k0;
    a_add (vl `mul` cw') (SF.row_sum es m' bn) 0.0R (SO.tsum_d x t mr');
    SO.dstep x p t p' mr' mr' 0.0R

(* ------------------------------------------------------------------ *)
(* The four cases of the step.                                         *)
(* ------------------------------------------------------------------ *)

(* Nothing absorbed yet, and this tile admits nothing either. *)
let step_aa
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    pnone q bn /\ pnone p n /\
                    vm == neg infinity /\ vl == zero)
          (ensures ml_state x p' m' l')
  = tile_empty t q k0;
    pnone_ext p' p;
    row_max_where_none es q bn;
    row_max_masked es q bn;
    row_sum_all_masked es m' bn;
    ninf_not_nan #et ();
    fmax_ninf_l (neg #et infinity);
    zero_finite #et ();
    mul_comm (zero #et) cw';
    mul_zero cw';
    add_zero (zero #et)

(* Nothing absorbed yet, and this tile admits at least one key. *)
let step_ab
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) (i : natlt bn) (rR : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    pnone p n /\ vm == neg infinity /\ vl == zero /\
                    q i /\ row_max_where es q bn %~ rR)
          (ensures ml_state x p' m' l')
  = row_max_masked es q bn;
    row_max_where_not_nan es q bn;
    fmax_ninf_l (row_max_where es q bn);
    pnone_witness p' (k0 + i);
    step_fresh x p t p' q xt k0 es vm vl m' l' cw' rR

(* Already absorbing, and this tile admits nothing. *)
let step_ba
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) (mr : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    pnone q bn /\ ~(pnone p n) /\
                    vm %~ mr /\ vl %~ (SO.dsum x p /. exp mr))
          (ensures ml_state x p' m' l')
  = tile_empty t q k0;
    pnone_ext p' p;
    row_max_where_none es q bn;
    row_max_masked es q bn;
    fmax_ninf_r vm;
    step_absorbed x p t p' q xt k0 es vm vl m' l' cw' mr mr

(* Already absorbing, and this tile admits at least one key. *)
let step_bb
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) (i : natlt bn) (mr rR : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    ~(pnone p n) /\ vm %~ mr /\ vl %~ (SO.dsum x p /. exp mr) /\
                    q i /\ row_max_where es q bn %~ rR)
          (ensures ml_state x p' m' l')
  = row_max_masked es q bn;
    row_max_where_not_nan es q bn;
    fmax_not_nan vm (row_max_where es q bn);
    fmax_approx vm (row_max_where es q bn) mr rR;
    pnone_witness p' (k0 + i);
    step_absorbed x p t p' q xt k0 es vm vl m' l' cw' mr (rmax mr rR)

(* ------------------------------------------------------------------ *)
(* The step, packaged.                                                 *)
(* ------------------------------------------------------------------ *)

let step_b_ir
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) (i : natlt bn) (rR : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    ml_state x p vm vl /\
                    q i /\ row_max_where es q bn %~ rR)
          (ensures ml_state x p' m' l')
  = if pnone p n
    then step_ab x p t p' q xt k0 es vm vl m' l' cw' i rR
    else FStar.Classical.exists_elim
           (ml_state x p' m' l')
           #real #(fun mr -> vm %~ mr /\ vl %~ (SO.dsum x p /. exp mr))
           ()
           (fun mr -> step_bb x p t p' q xt k0 es vm vl m' l' cw' i mr rR)

let step_b_i
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et) (i : natlt bn)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    ml_state x p vm vl /\
                    q i /\ (exists (r : real). row_max_where es q bn %~ r))
          (ensures ml_state x p' m' l')
  = FStar.Classical.exists_elim
      (ml_state x p' m' l')
      #real #(fun r -> row_max_where es q bn %~ r)
      ()
      (fun r -> step_b_ir x p t p' q xt k0 es vm vl m' l' cw' i r)

(* One lane absorbs one tile of keys: the float-level state moves from the key
   set [p] to [p'], the union of [p] with the tile's admitted keys [t]. *)
let ml_step
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' : et)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    ml_state x p vm vl)
          (ensures ml_state x p' m' l')
  = row_max_where_approx es q xt bn;
    if pnone q bn
    then
      (if pnone p n
       then step_aa x p t p' q xt k0 es vm vl m' l' cw'
       else FStar.Classical.exists_elim
              (ml_state x p' m' l')
              #real #(fun mr -> vm %~ mr /\ vl %~ (SO.dsum x p /. exp mr))
              ()
              (fun mr -> step_ba x p t p' q xt k0 es vm vl m' l' cw' mr))
    else
      (pnone_spec q bn;
       FStar.Classical.exists_elim
         (ml_state x p' m' l')
         #(natlt bn) #(fun i -> b2t (q i))
         ()
         (fun i -> step_b_i x p t p' q xt k0 es vm vl m' l' cw' i))

(* ------------------------------------------------------------------ *)
(* The numerator register.                                             *)
(* ------------------------------------------------------------------ *)

(* Every float approximates [to_real] of itself, so a factor's real value
   exists even when the factor is the meaningless correction weight of an
   empty state.  That is what lets the numerator's zero be an approximation:
   unlike the denominator, the [P@V] accumulation is an opaque tensor-core
   chain and cannot be shown to return the literal zero on an all-zero
   probability tile. *)
let mul_zero_approx
  (#et : Type0) {| scalar et |} {| real_like et |} (x y : et)
  : Lemma (requires x %~ 0.0R) (ensures (x `mul` y) %~ 0.0R)
  = to_real_ok y;
    a_mul x y 0.0R (to_real y)

(* The same, annihilating on the right: the rescaling factor a warp with no
   keys publishes may well be NaN, so the factors cannot be commuted. *)
let mul_zero_approx_r
  (#et : Type0) {| scalar et |} {| real_like et |} (x y : et)
  : Lemma (requires y %~ 0.0R) (ensures (x `mul` y) %~ 0.0R)
  = to_real_ok x;
    a_mul x y (to_real x) 0.0R

(* [m], [l] and [o] are one lane's running maximum, denominator and output
   accumulator for one value column, after absorbing exactly the keys in [p].
   The single witness [mr] is shared between [l] and [o], so dividing them
   cancels it. *)
let mlo_state
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p : SO.pred n) (m l o : et) : prop
  = not_nan m /\
    (if pnone p n
     then (m == neg infinity /\ l == zero /\ o %~ 0.0R)
     else (exists (mr : real).
             m %~ mr /\ l %~ (SO.dsum x p /. exp mr) /\
             o %~ (SO.nsum x y p /. exp mr)))

let mlo_state_ml
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p : SO.pred n) (m l o : et)
  : Lemma (requires mlo_state x y p m l o) (ensures ml_state x p m l)
  = if pnone p n then ()
    else FStar.Classical.exists_elim
           (ml_state x p m l)
           #real
           #(fun mr -> m %~ mr /\ l %~ (SO.dsum x p /. exp mr) /\
                       o %~ (SO.nsum x y p /. exp mr))
           ()
           (fun mr -> ())

let mlo_state_ext
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p q : SO.pred n) (m l o : et)
  : Lemma (requires mlo_state x y p m l o /\ (forall (j : natlt n). p j == q j))
          (ensures mlo_state x y q m l o)
  = pnone_ext p q;
    SO.sum_where_ext (fun j -> exp (x j)) (fun j -> exp (x j)) p q;
    SO.sum_where_ext (fun j -> exp (x j) *. y j) (fun j -> exp (x j) *. y j) p q

(* The kernel's numerator update, alongside the softmax update of [step_pre]:
   the accumulator is rescaled by the same correction weight and the tile's
   [P@V] contribution [pv] is added. *)
let o_step_pre
  (#et : Type0) {| floating et |} {| real_like et |}
  (vo pv o' cw' : et) : prop
  = o' == (vo `mul` cw') `add` pv

(* What the tile's [P@V] contribution must satisfy: it approximates the shifted
   numerator sum over the tile's admitted keys, at whatever real the new
   maximum encodes.  A tile that admits nothing makes this say [pv %~ 0]. *)
let pv_ok
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (t : SO.pred n) (m' pv : et) : prop
  = forall (mr' : real). m' %~ mr' ==> pv %~ SO.tsum_n x y t mr'

(* An empty tile contributes an approximate zero at every shift. *)
let pv_ok_none
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (t : SO.pred n) (m' pv : et)
  : Lemma (requires pv_ok x y t m' pv /\ (forall (j : natlt n). ~(t j)))
          (ensures pv %~ 0.0R)
  = to_real_ok m';
    SO.sum_upto_false (fun j -> exp (x j) *. y j) t n;
    assert (SO.nsum x y t == 0.0R);
    SO.lem_tsum_n x y t (to_real m');
    assert (pv %~ SO.tsum_n x y t (to_real m'))

(* Numerator step, in the case where the lane has already absorbed a key. *)
let o_step_absorbed
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (vm vo pv o' m' cw' : et) (mr mr' : real)
  : Lemma (requires SO.disjoint p t /\
                    (forall (j : natlt n). p' j == (p j || t j)) /\
                    o_step_pre vo pv o' cw' /\
                    cw' == fexp (vm `sub` m') /\
                    vm %~ mr /\ m' %~ mr' /\
                    vo %~ (SO.nsum x y p /. exp mr) /\
                    pv %~ SO.tsum_n x y t mr')
          (ensures o' %~ (SO.nsum x y p' /. exp mr'))
  = sub_approx vm m' mr mr';
    exp_approx (vm `sub` m') (mr -. mr');
    a_mul vo cw' (SO.nsum x y p /. exp mr) (exp (mr -. mr'));
    a_add (vo `mul` cw') pv
          (SO.nsum x y p /. exp mr *. exp (mr -. mr'))
          (SO.tsum_n x y t mr');
    SO.nstep x y p t p' mr mr' (SO.nsum x y p /. exp mr)

(* Numerator step, in the case where this tile absorbs the lane's first key.
   The old accumulator approximates zero, which annihilates the meaningless
   correction factor. *)
let o_step_fresh
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (vo pv o' cw' : et) (mr' : real)
  : Lemma (requires SO.disjoint p t /\
                    (forall (j : natlt n). p' j == (p j || t j)) /\
                    o_step_pre vo pv o' cw' /\
                    pnone p n /\ vo %~ 0.0R /\
                    pv %~ SO.tsum_n x y t mr')
          (ensures o' %~ (SO.nsum x y p' /. exp mr'))
  = mul_zero_approx vo cw';
    pnone_spec p n;
    SO.sum_upto_false (fun j -> exp (x j) *. y j) p n;
    a_add (vo `mul` cw') pv 0.0R (SO.tsum_n x y t mr');
    SO.nstep x y p t p' mr' mr' 0.0R

(* Numerator step, in the case where nothing has been or is being absorbed. *)
let o_step_none
  (#et : Type0) {| floating et |} {| real_like et |}
  (vo pv o' cw' : et)
  : Lemma (requires o_step_pre vo pv o' cw' /\ vo %~ 0.0R /\ pv %~ 0.0R)
          (ensures o' %~ 0.0R)
  = mul_zero_approx vo cw';
    a_add (vo `mul` cw') pv 0.0R 0.0R

(* ------------------------------------------------------------------ *)
(* The numerator step, by the same four cases as [ml_step].            *)
(* ------------------------------------------------------------------ *)

(* Nothing absorbed yet, and this tile admits nothing either. *)
let mlo_aa
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n #bn : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' vo pv o' : et)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    o_step_pre vo pv o' cw' /\ pv_ok x y t m' pv /\
                    pnone q bn /\ pnone p n /\
                    vm == neg infinity /\ vl == zero /\ vo %~ 0.0R)
          (ensures mlo_state x y p' m' l' o')
  = step_aa x p t p' q xt k0 es vm vl m' l' cw';
    tile_empty t q k0;
    pnone_ext p' p;
    pv_ok_none x y t m' pv;
    o_step_none vo pv o' cw'

(* Already absorbing, and this tile admits nothing. *)
let mlo_ba
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' vo pv o' : et) (mr : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    o_step_pre vo pv o' cw' /\ pv_ok x y t m' pv /\
                    pnone q bn /\ ~(pnone p n) /\
                    vm %~ mr /\ vl %~ (SO.dsum x p /. exp mr) /\
                    vo %~ (SO.nsum x y p /. exp mr))
          (ensures mlo_state x y p' m' l' o')
  = step_ba x p t p' q xt k0 es vm vl m' l' cw' mr;
    tile_empty t q k0;
    pnone_ext p' p;
    row_max_where_none es q bn;
    row_max_masked es q bn;
    fmax_ninf_r vm;
    step_absorbed x p t p' q xt k0 es vm vl m' l' cw' mr mr;
    o_step_absorbed x y p t p' vm vo pv o' m' cw' mr mr

(* Nothing absorbed yet, and this tile admits at least one key. *)
let mlo_ab
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' vo pv o' : et)
  (i : natlt bn) (rR : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    o_step_pre vo pv o' cw' /\ pv_ok x y t m' pv /\
                    pnone p n /\ vm == neg infinity /\ vl == zero /\
                    vo %~ 0.0R /\ q i /\ row_max_where es q bn %~ rR)
          (ensures mlo_state x y p' m' l' o')
  = step_ab x p t p' q xt k0 es vm vl m' l' cw' i rR;
    row_max_masked es q bn;
    row_max_where_not_nan es q bn;
    fmax_ninf_l (row_max_where es q bn);
    pnone_witness p' (k0 + i);
    step_fresh x p t p' q xt k0 es vm vl m' l' cw' rR;
    o_step_fresh x y p t p' vo pv o' cw' rR

(* Already absorbing, and this tile admits at least one key. *)
let mlo_bb
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' vo pv o' : et)
  (i : natlt bn) (mr rR : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    o_step_pre vo pv o' cw' /\ pv_ok x y t m' pv /\
                    ~(pnone p n) /\ vm %~ mr /\
                    vl %~ (SO.dsum x p /. exp mr) /\
                    vo %~ (SO.nsum x y p /. exp mr) /\
                    q i /\ row_max_where es q bn %~ rR)
          (ensures mlo_state x y p' m' l' o')
  = step_bb x p t p' q xt k0 es vm vl m' l' cw' i mr rR;
    row_max_masked es q bn;
    row_max_where_not_nan es q bn;
    fmax_not_nan vm (row_max_where es q bn);
    fmax_approx vm (row_max_where es q bn) mr rR;
    pnone_witness p' (k0 + i);
    step_absorbed x p t p' q xt k0 es vm vl m' l' cw' mr (rmax mr rR);
    o_step_absorbed x y p t p' vm vo pv o' m' cw' mr (rmax mr rR)

(* ------------------------------------------------------------------ *)
(* The numerator step, packaged.                                       *)
(* ------------------------------------------------------------------ *)

let mlo_step_b_ir
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' vo pv o' : et)
  (i : natlt bn) (rR : real)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    o_step_pre vo pv o' cw' /\ pv_ok x y t m' pv /\
                    mlo_state x y p vm vl vo /\
                    q i /\ row_max_where es q bn %~ rR)
          (ensures mlo_state x y p' m' l' o')
  = if pnone p n
    then mlo_ab x y p t p' q xt k0 es vm vl m' l' cw' vo pv o' i rR
    else FStar.Classical.exists_elim
           (mlo_state x y p' m' l' o')
           #real
           #(fun mr -> vm %~ mr /\ vl %~ (SO.dsum x p /. exp mr) /\
                       vo %~ (SO.nsum x y p /. exp mr))
           ()
           (fun mr ->
              mlo_bb x y p t p' q xt k0 es vm vl m' l' cw' vo pv o' i mr rR)

let mlo_step_b_i
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' vo pv o' : et) (i : natlt bn)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    o_step_pre vo pv o' cw' /\ pv_ok x y t m' pv /\
                    mlo_state x y p vm vl vo /\
                    q i /\ (exists (r : real). row_max_where es q bn %~ r))
          (ensures mlo_state x y p' m' l' o')
  = FStar.Classical.exists_elim
      (mlo_state x y p' m' l' o')
      #real #(fun r -> row_max_where es q bn %~ r)
      ()
      (fun r ->
         mlo_step_b_ir x y p t p' q xt k0 es vm vl m' l' cw' vo pv o' i r)

(* One lane absorbs one tile of keys, denominator and numerator together: the
   float-level state moves from the key set [p] to [p'], the union of [p] with
   the tile's admitted keys [t]. *)
let mlo_step
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x y : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : natle n)
  (es : chest1 et bn) (vm vl m' l' cw' vo pv o' : et)
  : Lemma (requires step_pre x p t p' q xt k0 es vm vl m' l' cw' /\
                    o_step_pre vo pv o' cw' /\ pv_ok x y t m' pv /\
                    mlo_state x y p vm vl vo)
          (ensures mlo_state x y p' m' l' o')
  = row_max_where_approx es q xt bn;
    if pnone q bn
    then
      (if pnone p n
       then mlo_aa x y p t p' q xt k0 es vm vl m' l' cw' vo pv o'
       else FStar.Classical.exists_elim
              (mlo_state x y p' m' l' o')
              #real
              #(fun mr -> vm %~ mr /\ vl %~ (SO.dsum x p /. exp mr) /\
                          vo %~ (SO.nsum x y p /. exp mr))
              ()
              (fun mr ->
                 mlo_ba x y p t p' q xt k0 es vm vl m' l' cw' vo pv o' mr))
    else
      (pnone_spec q bn;
       FStar.Classical.exists_elim
         (mlo_state x y p' m' l' o')
         #(natlt bn) #(fun i -> b2t (q i))
         ()
         (fun i ->
            mlo_step_b_i x y p t p' q xt k0 es vm vl m' l' cw' vo pv o' i))

(* ------------------------------------------------------------------ *)
(* The cross-warp combine.                                             *)
(*                                                                     *)
(* Every warp published a running maximum, denominator and output       *)
(* accumulator for the key set it owns.  Warp 0 folds the maxima into a *)
(* block-wide one and re-weights the other two by [exp (m_w - gm)].     *)
(* A warp that absorbed nothing published the [-inf] sentinel, whose    *)
(* weight is meaningless; it is annihilated because that warp's         *)
(* accumulators are (approximately) zero.                              *)
(* ------------------------------------------------------------------ *)

(* The key set the first [k] warps cover between them. *)
let rec punion (#n #nw : nat) (pw : natlt nw -> SO.pred n) (k : nat { k <= nw })
  : SO.pred n
  = if k = 0 then SO.pfalse
    else (fun j -> punion pw (k - 1) j || pw (k - 1) j)

let rec punion_spec (#n #nw : nat) (pw : natlt nw -> SO.pred n)
                    (k : nat { k <= nw }) (j : natlt n)
  : Lemma (ensures punion pw k j <==> (exists (w : natlt nw). w < k /\ pw w j))
          (decreases k)
  = if k = 0 then () else punion_spec pw (k - 1) j

let punion_none (#n #nw : nat) (pw : natlt nw -> SO.pred n) (k : nat { k <= nw })
  : Lemma (pnone (punion pw k) n
           <==> (forall (w : natlt nw). w < k ==> pnone (pw w) n))
  = pnone_spec (punion pw k) n;
    introduce forall (w : natlt nw).
      (pnone (pw w) n <==> (forall (j : natlt n). ~(pw w j)))
    with pnone_spec (pw w) n;
    introduce forall (j : natlt n).
      (punion pw k j <==> (exists (w : natlt nw). w < k /\ pw w j))
    with punion_spec pw k j

(* The first [k] warps' keys are disjoint from warp [w1]'s whenever [w1] is
   not among them. *)
let rec punion_disjoint (#n #nw : nat) (pw : natlt nw -> SO.pred n)
                        (k : nat { k <= nw }) (w1 : natlt nw)
  : Lemma (requires k <= w1 /\
                    (forall (a b : natlt nw). ~(a == b) ==>
                       SO.disjoint (pw a) (pw b)))
          (ensures SO.disjoint (punion pw k) (pw w1))
          (decreases k)
  = if k = 0
    then assert (SO.disjoint (punion pw k) (pw w1))
    else begin
      let k1 : nat = k - 1 in
      let w0 : natlt nw = k1 in
      punion_disjoint pw k1 w1;
      assert (SO.disjoint (pw w0) (pw w1));
      introduce forall (j : natlt n). ~(punion pw k j /\ pw w1 j)
      with assert (punion pw k j == (punion pw k1 j || pw w0 j))
    end

(* The block-wide maximum after folding the first [k] warps. *)
let rec gmax_fold
  (#et : Type0) {| floating et |} (#nw : nat)
  (m : natlt nw -> GTot et) (k : nat { k <= nw }) : GTot et (decreases k)
  = if k = 0 then neg infinity else fmax (gmax_fold m (k - 1)) (m (k - 1))

(* The rescale-weighted sum of the first [k] warps' accumulators. *)
let rec gfold
  (#et : Type0) {| floating et |} (#nw : nat)
  (sc v : natlt nw -> GTot et) (k : nat { k <= nw }) : GTot et (decreases k)
  = if k = 0 then zero
    else add (gfold sc v (k - 1)) (mul (sc (k - 1)) (v (k - 1)))

(* What each warp published, as the combine sees it. *)
let warps_ok
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n #nw : nat) (x y : natlt n -> GTot real) (pw : natlt nw -> SO.pred n)
  (m l o : natlt nw -> GTot et) : prop
  = forall (w : natlt nw). mlo_state x y (pw w) (m w) (l w) (o w)

(* The block-wide maximum is not NaN, is the sentinel exactly while no warp
   has absorbed anything, and otherwise approximates some real. *)
let rec gmax_fold_ok
  (#et : Type0) {| floating et |} {| real_like et |} {| floating_real_like et |}
  (#n #nw : nat) (x y : natlt n -> GTot real) (pw : natlt nw -> SO.pred n)
  (m l o : natlt nw -> GTot et) (k : nat { k <= nw })
  : Lemma (requires warps_ok x y pw m l o)
          (ensures
            not_nan (gmax_fold m k) /\
            ((forall (w : natlt nw). w < k ==> pnone (pw w) n)
             ==> gmax_fold m k == neg infinity) /\
            ((forall (w : natlt nw). w < k ==> pnone (pw w) n) \/
             (exists (gmr : real). gmax_fold m k %~ gmr)))
          (decreases k)
  = if k = 0 then ninf_not_nan #et ()
    else begin
      let k1 : nat = k - 1 in
      let w1 : natlt nw = k1 in
      gmax_fold_ok x y pw m l o k1;
      let g1 = gmax_fold m k1 in
      let a = m w1 in
      if pnone (pw w1) n
      then fmax_ninf_r g1
      else begin
        fmax_not_nan g1 a;
        if (forall (w : natlt nw). w < k1 ==> pnone (pw w) n)
        then (fmax_ninf_l a;
              FStar.Classical.exists_elim
                (exists (gmr : real). fmax g1 a %~ gmr)
                #real
                #(fun mr -> a %~ mr /\ l w1 %~ (SO.dsum x (pw w1) /. exp mr) /\
                            o w1 %~ (SO.nsum x y (pw w1) /. exp mr))
                ()
                (fun mr -> ()))
        else
          FStar.Classical.exists_elim
            (exists (gmr : real). fmax g1 a %~ gmr)
            #real
            #(fun mr -> a %~ mr /\ l w1 %~ (SO.dsum x (pw w1) /. exp mr) /\
                        o w1 %~ (SO.nsum x y (pw w1) /. exp mr))
            ()
            (fun mr ->
               FStar.Classical.exists_elim
                 (exists (gmr : real). fmax g1 a %~ gmr)
                 #real
                 #(fun gmr1 -> g1 %~ gmr1)
                 ()
                 (fun gmr1 -> fmax_approx g1 a gmr1 mr))
      end
    end

(* One warp's rescaled accumulators are its share of the block sums, already
   shifted by the block-wide maximum.  A warp that absorbed nothing published
   zero, which annihilates its meaningless rescaling factor. *)
let combine_warp
  (#et : Type0) {| floating et |} {| real_like et |} {| floating_real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p : SO.pred n)
  (mw lw ow gm sc : et) (gmr : real)
  : Lemma (requires mlo_state x y p mw lw ow /\ gm %~ gmr /\
                    sc == fexp (mw `sub` gm))
          (ensures (sc `mul` lw) %~ SO.tsum_d x p gmr /\
                   (sc `mul` ow) %~ SO.tsum_n x y p gmr)
  = if pnone p n
    then begin
      pnone_spec p n;
      SO.tsum_d_none x p gmr;
      SO.tsum_n_none x y p gmr;
      zero_approx #et ();
      mul_zero_approx_r sc lw;
      mul_zero_approx_r sc ow
    end
    else
      FStar.Classical.exists_elim
        ((sc `mul` lw) %~ SO.tsum_d x p gmr /\ (sc `mul` ow) %~ SO.tsum_n x y p gmr)
        #real
        #(fun mr -> mw %~ mr /\ lw %~ (SO.dsum x p /. exp mr) /\
                    ow %~ (SO.nsum x y p /. exp mr))
        ()
        (fun mr ->
           sub_approx mw gm mr gmr;
           exp_approx (mw `sub` gm) (mr -. gmr);
           a_mul sc lw (exp (mr -. gmr)) (SO.dsum x p /. exp mr);
           a_mul sc ow (exp (mr -. gmr)) (SO.nsum x y p /. exp mr);
           SO.lem_rescale (SO.dsum x p /. exp mr) mr gmr (SO.dsum x p);
           SO.lem_rescale (SO.nsum x y p /. exp mr) mr gmr (SO.nsum x y p);
           SO.lem_tsum_d x p gmr;
           SO.lem_tsum_n x y p gmr)

(* The fold of warps that all published (approximate) zero is zero. *)
let rec gfold_none
  (#et : Type0) {| floating et |} {| real_like et |}
  (#nw : nat) (sc v : natlt nw -> GTot et) (k : nat { k <= nw })
  : Lemma (requires forall (w : natlt nw). w < k ==> v w %~ 0.0R)
          (ensures gfold sc v k %~ 0.0R)
          (decreases k)
  = if k = 0 then zero_approx #et ()
    else begin
      let k1 : nat = k - 1 in
      let w1 : natlt nw = k1 in
      gfold_none sc v k1;
      mul_zero_approx_r (sc w1) (v w1);
      a_add (gfold sc v k1) (mul (sc w1) (v w1)) 0.0R 0.0R
    end

(* The fold of the first [k] warps is the block sum over the keys they cover. *)
let rec gfold_ok
  (#et : Type0) {| floating et |} {| real_like et |} {| floating_real_like et |}
  (#n #nw : nat) (x y : natlt n -> GTot real) (pw : natlt nw -> SO.pred n)
  (m l o sc : natlt nw -> GTot et) (gm : et) (gmr : real) (k : nat { k <= nw })
  : Lemma (requires
             warps_ok x y pw m l o /\ gm %~ gmr /\
             (forall (w : natlt nw). sc w == fexp (m w `sub` gm)) /\
             (forall (w1 w2 : natlt nw). ~(w1 == w2) ==>
                SO.disjoint (pw w1) (pw w2)))
          (ensures gfold sc l k %~ SO.tsum_d x (punion pw k) gmr /\
                   gfold sc o k %~ SO.tsum_n x y (punion pw k) gmr)
          (decreases k)
  = if k = 0
    then (SO.tsum_d_none x (punion pw 0) gmr;
          SO.tsum_n_none x y (punion pw 0) gmr;
          zero_approx #et ())
    else begin
      let k1 : nat = k - 1 in
      let w1 : natlt nw = k1 in
      gfold_ok x y pw m l o sc gm gmr k1;
      combine_warp x y (pw w1) (m w1) (l w1) (o w1) gm (sc w1) gmr;
      punion_disjoint pw k1 w1;
      SO.tsum_d_split x (punion pw k1) (pw w1) gmr;
      SO.tsum_n_split x y (punion pw k1) (pw w1) gmr;
      SO.tsum_d_ext x (punion pw k) (SO.por (punion pw k1) (pw w1)) gmr;
      SO.tsum_n_ext x y (punion pw k) (SO.por (punion pw k1) (pw w1)) gmr;
      a_add (gfold sc l k1) (mul (sc w1) (l w1))
            (SO.tsum_d x (punion pw k1) gmr) (SO.tsum_d x (pw w1) gmr);
      a_add (gfold sc o k1) (mul (sc w1) (o w1))
            (SO.tsum_n x y (punion pw k1) gmr) (SO.tsum_n x y (pw w1) gmr)
    end

(* The block-wide state after the combine.  Unlike [mlo_state] the empty case
   only says the accumulators are approximately zero: an empty row's rescaling
   factors are meaningless floats and nothing pins the fold to a literal
   zero.  Nothing downstream needs more -- an empty row belongs to a query
   position the kernel never stores. *)
let gstate
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p : SO.pred n) (gm gl go : et) : prop
  = not_nan gm /\
    (if pnone p n
     then (gl %~ 0.0R /\ go %~ 0.0R)
     else (exists (gmr : real).
             gm %~ gmr /\ gl %~ (SO.dsum x p /. exp gmr) /\
             go %~ (SO.nsum x y p /. exp gmr)))

let mlo_combine
  (#et : Type0) {| floating et |} {| real_like et |} {| floating_real_like et |}
  (#n #nw : nat) (x y : natlt n -> GTot real) (pw : natlt nw -> SO.pred n)
  (m l o sc : natlt nw -> GTot et) (p' : SO.pred n)
  : Lemma (requires
             warps_ok x y pw m l o /\
             (forall (w : natlt nw). sc w == fexp (m w `sub` gmax_fold m nw)) /\
             (forall (w1 w2 : natlt nw). ~(w1 == w2) ==>
                SO.disjoint (pw w1) (pw w2)) /\
             (forall (j : natlt n). p' j == punion pw nw j))
          (ensures gstate x y p' (gmax_fold m nw)
                     (gfold sc l nw) (gfold sc o nw))
  = let gm = gmax_fold m nw in
    gmax_fold_ok x y pw m l o nw;
    punion_none pw nw;
    pnone_ext p' (punion pw nw);
    if pnone p' n
    then begin
      introduce forall (w : natlt nw). l w %~ 0.0R /\ o w %~ 0.0R
      with zero_approx #et ();
      gfold_none sc l nw;
      gfold_none sc o nw
    end
    else
      FStar.Classical.exists_elim
        (gstate x y p' gm (gfold sc l nw) (gfold sc o nw))
        #real
        #(fun gmr -> gm %~ gmr)
        ()
        (fun gmr ->
           gfold_ok x y pw m l o sc gm gmr nw;
           SO.lem_tsum_d x (punion pw nw) gmr;
           SO.lem_tsum_n x y (punion pw nw) gmr;
           SO.dsum_ext x p' (punion pw nw);
           SO.nsum_ext x y p' (punion pw nw))

(* ------------------------------------------------------------------ *)
(* The epilogue: normalising the block accumulators.                   *)
(* ------------------------------------------------------------------ *)

let one_approx (#et : Type0) {| floating et |} {| real_like et |} ()
  : Lemma ((one #et) %~ 1.0R)
  = a1 #et

(* Dividing the block numerator by the block denominator cancels the
   block-wide maximum and leaves the plain softmax-weighted average.  The
   denominator being a strictly positive float is a hypothesis: nothing in the
   [floating] laws relates the order on floats to the reals they approximate,
   so an underflowed denominator is exactly as far out of scope as an
   overflowed score. *)
let gstate_out
  (#et : Type0) {| floating et |} {| real_like et |} {| floating_real_like et |}
  (#n : nat) (x y : natlt n -> GTot real) (p : SO.pred n)
  (gm gl go : et) (j0 : natlt n)
  : Lemma (requires gstate x y p gm gl go /\ p j0)
          (ensures SO.dsum x p >. 0.0R /\
                   (go `mul` (one `div` gl))
                   %~ (SO.nsum x y p /. SO.dsum x p))
  = pnone_witness p j0;
    SO.dsum_pos x p j0;
    FStar.Classical.exists_elim
      ((go `mul` (one `div` gl)) %~ (SO.nsum x y p /. SO.dsum x p))
      #real
      #(fun gmr -> gm %~ gmr /\ gl %~ (SO.dsum x p /. exp gmr) /\
                   go %~ (SO.nsum x y p /. exp gmr))
      ()
      (fun gmr ->
         SO.dsum_pos x p j0;
         one_approx #et ();
         div_approx one gl 1.0R (SO.dsum x p /. exp gmr);
         a_mul go (one `div` gl) (SO.nsum x y p /. exp gmr)
           (1.0R /. (SO.dsum x p /. exp gmr));
         SO.div_cancel (SO.nsum x y p) (SO.dsum x p) (exp gmr))
