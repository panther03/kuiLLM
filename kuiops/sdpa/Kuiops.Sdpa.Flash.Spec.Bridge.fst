module Kuiops.Sdpa.Flash.Spec.Bridge

(* Bridge from the float-level description of the online-softmax update
   ([Kuiops.Sdpa.Flash.Spec.Float]) to the real-valued flash invariant
   ([Kuiops.Sdpa.Flash.Spec.Online]).

   The kernel encodes "key not admitted" as the [-inf] sentinel, so the running
   maximum is [-inf] exactly while no key has been absorbed.  That state
   approximates no real, so the invariant carried through the key loop is a
   disjunction: either nothing has been absorbed and the registers still hold
   the sentinel and zero, or some key has been absorbed and the registers
   approximate the real invariant at *some* running maximum.  Which real the
   maximum encodes never matters -- see the header of
   [Kuiops.Sdpa.Flash.Spec.Online]. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Floating
open Kuiper.Approximates

module SEM = FStar.StrongExcludedMiddle
module SF = Kuiops.Sdpa.Flash.Spec.Float
module SO = Kuiops.Sdpa.Flash.Spec.Online

(* ------------------------------------------------------------------ *)
(* The [-inf] sentinel.                                                *)
(* ------------------------------------------------------------------ *)

let lte_ninf (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires not_nan x) (ensures lte (neg #et infinity) x)
  = let nx : et = neg x in
    neg_kind x;
    infinity_val_spec nx;
    lte_is_lt_or_eq nx infinity;
    lt_neg_flip nx infinity;
    neg_neg x;
    lte_is_lt_or_eq (neg #et infinity) x

let fmax_ninf_r (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires not_nan x) (ensures fmax x (neg #et infinity) == x)
  = lte_ninf x

let fmax_ninf_l (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires not_nan x) (ensures fmax (neg #et infinity) x == x)
  = lte_ninf x

let fmax_not_nan (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires not_nan x /\ not_nan y) (ensures not_nan (fmax x y))
  = ()

(* A finite score is never the sentinel, so [sel_prob] does not select it to
   zero. *)
let finite_not_ninf (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires Finite? (kind x)) (ensures ~(x == neg #et infinity))
  = ()

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
  = if k = 0 then () else row_max_where_not_nan s q (k - 1)

let rec row_max_where_none
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (q : SO.pred bn) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). t < k ==> ~(q t))
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
          (ensures (forall (t : natlt bn). t < k ==> ~(q t)) \/
                   (exists (r : real). row_max_where s q k %~ r))
          (decreases k)
  = if k = 0 then ()
    else begin
      row_max_where_approx s q xt (k - 1);
      if q (k - 1)
      then
        (if SEM.strong_excluded_middle
              (forall (t : natlt bn). t < k - 1 ==> ~(q t))
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
