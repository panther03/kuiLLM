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

(* Hypotheses shared by every case of the step: the tile [q] of [bn] local keys
   starting at [k0] describes exactly the part of [t] the lane absorbs, the
   scores [es] carry the sentinel on rejected keys and approximate [x] on
   admitted ones, and the registers are updated by the kernel's formulas. *)
let step_pre
  (#et : Type0) {| floating et |} {| real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
  (es : chest1 et bn) (vm vl m' l' cw' : et) : prop
  = SO.disjoint p t /\
    (forall (j : natlt n). p' j == (p j || t j)) /\
    (forall (j : natlt n). t j ==> (k0 <= j /\ j < k0 + bn)) /\
    (forall (i : natlt bn). q i == t (k0 + i)) /\
    (forall (i : natlt bn). xt i == x (k0 + i)) /\
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
  (#n #bn : nat) (t : SO.pred n) (q : SO.pred bn) (k0 : nat { k0 + bn <= n })
  : Lemma (requires (forall (j : natlt n). t j ==> (k0 <= j /\ j < k0 + bn)) /\
                    (forall (i : natlt bn). q i == t (k0 + i)) /\
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
   is used. *)
let step_fresh
  (#et : Type0) {| floating et |} {| real_like et |}
  {| floating_real_like et |}
  (#n #bn : nat) (x : natlt n -> GTot real) (p t p' : SO.pred n)
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
  (q : SO.pred bn) (xt : natlt bn -> GTot real) (k0 : nat { k0 + bn <= n })
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
