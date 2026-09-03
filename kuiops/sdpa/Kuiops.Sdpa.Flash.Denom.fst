module Kuiops.Sdpa.Flash.Denom

(* The softmax denominator the epilogue divides by is positive.

   This is the one fact about the kernel's arithmetic that the approximation
   proof cannot get from the online-softmax invariant alone, because it is a
   statement about *floats*: a real-valued denominator is positive as soon as
   the key set is non-empty, but a float one could in principle be flushed to
   zero by an underflowing rescale.

   It is not.  The argument is entirely local: whenever the running maximum is
   finite, so is the running denominator, and it is bounded below by [1].  The
   two cases are

     - the new tile raises the maximum, so some key of the tile *equals* the
       new maximum and contributes [exp 0 == 1] to the tile's sum; or
     - it does not, so the correction weight is [exp 0 == 1] and the previous
       denominator survives multiplication unchanged.

   The same argument runs again one level up, across warps.  The only input
   hypothesis is that admitted scores are finite -- the non-emptiness the real
   argument needs is free, since every active query row admits key [0]. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Floating
open Kuiper.Shape

module FA = Kuiops.FloatAxioms
module SF = Kuiops.Sdpa.Flash.Spec.Float
module SB = Kuiops.Sdpa.Flash.Spec.Bridge
module SS = Kuiops.Sdpa.Flash.Spec.Step
module FC = Kuiper.Float.Casts
module SV = Kuiops.Sdpa.Flash.Vals
module SZ = Kuiper.SizeT

#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"

(* ------------------------------------------------------------------ *)
(* Ordering scaffolding.                                               *)
(* ------------------------------------------------------------------ *)

(* A value the folds may legitimately meet: a number, or the [-inf] the kernel
   uses to mark a key the row does not attend to. *)
let ok (#et : Type0) {| floating et |} (x : et) : prop =
  Finite? (kind x) \/ x == neg infinity

let ok_not_nan (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires ok x) (ensures not_nan x)
  = SB.ninf_not_nan #et ()

let lte_refl (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires not_nan x) (ensures lte x x)
  = lte_is_lt_or_eq x x

let lt_imp_lte (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires not_nan x /\ not_nan y /\ lt x y) (ensures lte x y)
  = lte_is_lt_or_eq x y

(* Nothing sits strictly below the sentinel. *)
let ok_lte_ninf (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires ok x /\ lte x (neg #et infinity))
          (ensures x == neg #et infinity)
  = SB.ninf_not_nan #et ();
    ok_not_nan x;
    SB.lte_ninf x;
    lte_is_lt_or_eq x (neg #et infinity);
    lte_is_lt_or_eq (neg #et infinity) x

let ok_fmax (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires ok x /\ ok y) (ensures ok (fmax x y))
  = ok_not_nan x; ok_not_nan y

(* [fmax] with a finite argument is finite. *)
let fmax_finite (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires ok x /\ ok y /\ (Finite? (kind x) \/ Finite? (kind y)))
          (ensures Finite? (kind (fmax x y)))
  = ok_not_nan x; ok_not_nan y

(* ------------------------------------------------------------------ *)
(* The row maximum.                                                    *)
(* ------------------------------------------------------------------ *)

let rec row_max_ok
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). ok (acc1 s t))
          (ensures ok (SF.row_max s k))
          (decreases k)
  = if k = 0 then ()
    else (row_max_ok s (k - 1); ok_fmax (SF.row_max s (k - 1)) (acc1 s (k - 1)))

let rec row_max_ub
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (k : nat { k <= bn }) (j : natlt bn)
  : Lemma (requires j < k /\ (forall (t : natlt bn). ok (acc1 s t)))
          (ensures lte (acc1 s j) (SF.row_max s k))
          (decreases k)
  = row_max_ok s (k - 1);
    ok_not_nan (SF.row_max s (k - 1));
    ok_not_nan (acc1 s (k - 1));
    if j = k - 1 then lte_refl (acc1 s (k - 1))
    else begin
      row_max_ub s (k - 1) j;
      ok_not_nan (acc1 s j);
      if lt (SF.row_max s (k - 1)) (acc1 s (k - 1))
      then begin
        lt_imp_lte (SF.row_max s (k - 1)) (acc1 s (k - 1));
        FA.lte_trans (acc1 s j) (SF.row_max s (k - 1)) (acc1 s (k - 1))
      end
    end

(* The fold returns the sentinel it started from or one of the entries. *)
let rec row_max_cases
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). ok (acc1 s t))
          (ensures SF.row_max s k == neg #et infinity \/
                   (exists (j : natlt bn). j < k /\ SF.row_max s k == acc1 s j))
          (decreases k)
  = if k = 0 then ()
    else begin
      row_max_cases s (k - 1);
      row_max_ok s (k - 1);
      ok_not_nan (SF.row_max s (k - 1));
      ok_not_nan (acc1 s (k - 1))
    end

(* Every entry is the sentinel exactly when the maximum is. *)
let row_max_ninf_all
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (k : nat { k <= bn }) (j : natlt bn)
  : Lemma (requires j < k /\ (forall (t : natlt bn). ok (acc1 s t)) /\
                    SF.row_max s k == neg #et infinity)
          (ensures acc1 s j == neg #et infinity)
  = row_max_ub s k j; ok_lte_ninf (acc1 s j)

(* ------------------------------------------------------------------ *)
(* The row sum.                                                        *)
(* ------------------------------------------------------------------ *)

let sel_prob_nonneg (#et : Type0) {| floating et |} (sv mnew : et)
  : Lemma (requires ok sv /\ not_nan mnew /\ lte sv mnew)
          (ensures Finite? (kind (SF.sel_prob sv mnew)) /\
                   lte zero (SF.sel_prob sv mnew))
  = kind_zero #et;
    lte_refl (zero #et);
    ok_not_nan sv;
    if eq sv (neg infinity) then ()
    else begin
      SB.ninf_not_nan #et ();
      FA.sub_not_nan sv mnew;
      FA.sub_nonpos sv mnew;
      FA.exp_nonpos (sv `sub` mnew)
    end

let rec row_sum_nonneg
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (mnew : et) (k : nat { k <= bn })
  : Lemma (requires (forall (t : natlt bn). ok (acc1 s t)) /\ not_nan mnew /\
                    (forall (t : natlt bn). t < k ==> lte (acc1 s t) mnew))
          (ensures lte zero (SF.row_sum s mnew k))
          (decreases k)
  = kind_zero #et;
    lte_refl (zero #et);
    if k = 0 then ()
    else begin
      row_sum_nonneg s mnew (k - 1);
      sel_prob_nonneg (acc1 s (k - 1)) mnew;
      FA.add_nonneg (SF.row_sum s mnew (k - 1)) (SF.sel_prob (acc1 s (k - 1)) mnew)
    end

(* The entry that attains a finite maximum contributes exactly [1]. *)
let sel_prob_argmax (#et : Type0) {| floating et |} (mnew : et)
  : Lemma (requires Finite? (kind mnew))
          (ensures SF.sel_prob mnew mnew == one)
  = SB.ninf_not_nan #et ();
    kind_infinity #et;
    eq_spec mnew (neg #et infinity);
    FA.sub_self mnew;
    FA.exp_zero #et ()

let rec row_sum_pos
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (mnew : et) (k : nat { k <= bn }) (j : natlt bn)
  : Lemma (requires (forall (t : natlt bn). ok (acc1 s t)) /\
                    Finite? (kind mnew) /\
                    (forall (t : natlt bn). t < k ==> lte (acc1 s t) mnew) /\
                    j < k /\ acc1 s j == mnew)
          (ensures lt zero (SF.row_sum s mnew k))
          (decreases k)
  = kind_zero #et;
    lte_refl (zero #et);
    row_sum_nonneg s mnew (k - 1);
    sel_prob_nonneg (acc1 s (k - 1)) mnew;
    if j = k - 1
    then begin
      sel_prob_argmax mnew;
      FA.one_pos #et ();
      FA.add_pos (SF.row_sum s mnew (k - 1)) (SF.sel_prob (acc1 s (k - 1)) mnew)
    end
    else begin
      row_sum_pos s mnew (k - 1) j;
      FA.add_pos (SF.row_sum s mnew (k - 1)) (SF.sel_prob (acc1 s (k - 1)) mnew)
    end

(* An all-sentinel row contributes nothing. *)
let rec row_sum_masked
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (mnew : et) (k : nat { k <= bn })
  : Lemma (requires forall (t : natlt bn). t < k ==> acc1 s t == neg #et infinity)
          (ensures SF.row_sum s mnew k == zero)
          (decreases k)
  = kind_zero #et;
    if k = 0 then ()
    else begin
      row_sum_masked s mnew (k - 1);
      SB.ninf_not_nan #et ();
      eq_spec (acc1 s (k - 1)) (neg #et infinity);
      add_zero (zero #et)
    end

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"

(* ------------------------------------------------------------------ *)
(* The cross-warp combine.                                             *)
(* ------------------------------------------------------------------ *)

let nonneg_not_nan (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires lte zero x) (ensures not_nan x)
  = introduce NaN? (kind x) ==> False
    with FA.cmp_nan (zero #et) x

let rec gmax_ok
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (i : natlt bm) (n : nat { n <= nw })
  : Lemma (requires forall (w : natlt nw). ok (acc2 eM w i))
          (ensures ok (SF.gmax eM i n))
          (decreases n)
  = if n = 0 then ()
    else (gmax_ok eM i (n - 1);
          ok_fmax (SF.gmax eM i (n - 1)) (acc2 eM (n - 1) i))

let rec gmax_ub
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (i : natlt bm) (n : nat { n <= nw }) (w : natlt nw)
  : Lemma (requires w < n /\ (forall (u : natlt nw). ok (acc2 eM u i)))
          (ensures lte (acc2 eM w i) (SF.gmax eM i n))
          (decreases n)
  = gmax_ok eM i (n - 1);
    ok_not_nan (SF.gmax eM i (n - 1));
    ok_not_nan (acc2 eM (n - 1) i);
    if w = n - 1 then lte_refl (acc2 eM (n - 1) i)
    else begin
      gmax_ub eM i (n - 1) w;
      ok_not_nan (acc2 eM w i);
      if lt (SF.gmax eM i (n - 1)) (acc2 eM (n - 1) i)
      then begin
        lt_imp_lte (SF.gmax eM i (n - 1)) (acc2 eM (n - 1) i);
        FA.lte_trans (acc2 eM w i) (SF.gmax eM i (n - 1)) (acc2 eM (n - 1) i)
      end
    end

let rec gmax_cases
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (i : natlt bm) (n : nat { n <= nw })
  : Lemma (requires forall (w : natlt nw). ok (acc2 eM w i))
          (ensures SF.gmax eM i n == neg #et infinity \/
                   (exists (w : natlt nw). w < n /\ SF.gmax eM i n == acc2 eM w i))
          (decreases n)
  = if n = 0 then ()
    else begin
      gmax_cases eM i (n - 1);
      gmax_ok eM i (n - 1);
      ok_not_nan (SF.gmax eM i (n - 1));
      ok_not_nan (acc2 eM (n - 1) i)
    end

let gscale_nonneg
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (gm : et) (w : natlt nw) (i : natlt bm)
  : Lemma (requires ok (acc2 eM w i) /\ Finite? (kind gm) /\
                    lte (acc2 eM w i) gm)
          (ensures Finite? (kind (SF.gscale eM gm w i)) /\
                   lte zero (SF.gscale eM gm w i))
  = ok_not_nan (acc2 eM w i);
    FA.sub_not_nan (acc2 eM w i) gm;
    FA.sub_nonpos (acc2 eM w i) gm;
    FA.exp_nonpos (acc2 eM w i `sub` gm)

let gscale_argmax
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM : chest2 et nw bm) (gm : et) (w : natlt nw) (i : natlt bm)
  : Lemma (requires Finite? (kind gm) /\ acc2 eM w i == gm)
          (ensures SF.gscale eM gm w i == one)
  = FA.sub_self gm; FA.exp_zero #et ()

let rec gsum_nonneg
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM eL : chest2 et nw bm) (gm : et) (i : natlt bm) (n : nat { n <= nw })
  : Lemma (requires Finite? (kind gm) /\
                    (forall (w : natlt nw). ok (acc2 eM w i)) /\
                    (forall (w : natlt nw). w < n ==> lte (acc2 eM w i) gm) /\
                    (forall (w : natlt nw). w < n ==> lte zero (acc2 eL w i)))
          (ensures lte zero (SF.gsum eM eL gm i n))
          (decreases n)
  = kind_zero #et;
    lte_refl (zero #et);
    if n = 0 then ()
    else begin
      gsum_nonneg eM eL gm i (n - 1);
      gscale_nonneg eM gm (n - 1) i;
      FA.mul_nonneg (SF.gscale eM gm (n - 1) i) (acc2 eL (n - 1) i);
      FA.add_nonneg (SF.gsum eM eL gm i (n - 1))
        (SF.gscale eM gm (n - 1) i `mul` acc2 eL (n - 1) i)
    end

let rec gsum_pos
  (#et : Type0) {| floating et |} (#nw #bm : nat)
  (eM eL : chest2 et nw bm) (gm : et) (i : natlt bm) (n : nat { n <= nw })
  (w0 : natlt nw)
  : Lemma (requires Finite? (kind gm) /\
                    (forall (w : natlt nw). ok (acc2 eM w i)) /\
                    (forall (w : natlt nw). w < n ==> lte (acc2 eM w i) gm) /\
                    (forall (w : natlt nw). w < n ==> lte zero (acc2 eL w i)) /\
                    w0 < n /\ acc2 eM w0 i == gm /\ lt zero (acc2 eL w0 i))
          (ensures lt zero (SF.gsum eM eL gm i n))
          (decreases n)
  = kind_zero #et;
    lte_refl (zero #et);
    gsum_nonneg eM eL gm i (n - 1);
    gscale_nonneg eM gm (n - 1) i;
    FA.mul_nonneg (SF.gscale eM gm (n - 1) i) (acc2 eL (n - 1) i);
    if w0 = n - 1
    then begin
      gscale_argmax eM gm (n - 1) i;
      nonneg_not_nan (acc2 eL (n - 1) i);
      mul_comm (one #et) (acc2 eL (n - 1) i);
      mul_one (acc2 eL (n - 1) i);
      FA.add_pos (SF.gsum eM eL gm i (n - 1))
        (SF.gscale eM gm (n - 1) i `mul` acc2 eL (n - 1) i)
    end
    else begin
      gsum_pos eM eL gm i (n - 1) w0;
      FA.add_pos (SF.gsum eM eL gm i (n - 1))
        (SF.gscale eM gm (n - 1) i `mul` acc2 eL (n - 1) i)
    end

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"

(* ------------------------------------------------------------------ *)
(* [fmax] as an upper bound.                                           *)
(* ------------------------------------------------------------------ *)

let fmax_ub_l (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires not_nan x /\ not_nan y) (ensures lte x (fmax x y))
  = lte_refl x; lte_is_lt_or_eq x y

let fmax_ub_r (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires not_nan x /\ not_nan y) (ensures lte y (fmax x y))
  = lte_refl y; negate_lt_is_lte x y

(* [fmax] picks the right operand only when it is strictly larger, so it is
   the left one whenever the right one is the sentinel. *)
let fmax_case_r (#et : Type0) {| floating et |} (x y : et)
  : Lemma (requires ok x /\ ok y /\ fmax x y == y /\ ~(fmax x y == x))
          (ensures Finite? (kind y))
  = ok_not_nan x; ok_not_nan y;
    introduce y == neg #et infinity ==> False
    with (SB.lte_ninf x; lte_is_lt_or_eq x y)

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"

(* ------------------------------------------------------------------ *)
(* The warp invariant, and its preservation by one online-softmax step. *)
(* ------------------------------------------------------------------ *)

let pos_not_nan (#et : Type0) {| floating et |} (x : et)
  : Lemma (requires lt zero x) (ensures not_nan x)
  = introduce NaN? (kind x) ==> False
    with FA.cmp_nan (zero #et) x

let ninf_not_finite (#et : Type0) {| floating et |} ()
  : Lemma (~(Finite? (kind (neg #et infinity))))
  = neg_kind (infinity #et); kind_infinity #et

(* The register pair of a warp is either still untouched, or it holds a finite
   maximum and a *strictly positive* denominator.  The second disjunct is the
   whole point: the running denominator never underflows to zero. *)
let warp_inv (#et : Type0) {| floating et |} (vm vl : et) : prop
  = ok vm /\ lte zero vl /\
    (vm == neg infinity ==> vl == zero) /\
    (Finite? (kind vm) ==> lt zero vl)

let step_inv
  (#et : Type0) {| floating et |}
  (es : chest1 et 16) (vm vl : et)
  : Lemma (requires warp_inv vm vl /\ (forall (t : natlt 16). ok (acc1 es t)))
          (ensures (let m' = fmax vm (SF.row_max es 16) in
                    warp_inv m'
                      ((vl `mul` SF.corr_weight vm m')
                       `add` SF.row_sum es m' 16)))
  = let rm = SF.row_max es 16 in
    let m' = fmax vm rm in
    let cw = SF.corr_weight vm m' in
    let rs = SF.row_sum es m' 16 in
    kind_zero #et;
    lte_refl (zero #et);
    SB.ninf_not_nan #et ();
    ninf_not_finite #et ();
    row_max_ok es 16;
    ok_not_nan vm; ok_not_nan rm;
    ok_fmax vm rm; ok_not_nan m';
    SF.corr_weight_finite vm m';
    fmax_ub_l vm rm; fmax_ub_r vm rm;
    introduce forall (t : natlt 16). lte (acc1 es t) m'
    with (row_max_ub es 16 t; ok_not_nan (acc1 es t);
          FA.lte_trans (acc1 es t) rm m');
    row_sum_nonneg es m' 16;
    (* the rescaled carry is nonnegative, whichever branch the state is in *)
    (if Finite? (kind vm)
     then begin
       SB.corr_weight_exp vm m';
       FA.sub_not_nan vm m';
       FA.sub_nonpos vm m';
       FA.exp_nonpos (vm `sub` m');
       FA.mul_nonneg vl cw
     end
     else (mul_comm vl cw; mul_zero cw));
    FA.add_nonneg (vl `mul` cw) rs;
    introduce m' == neg #et infinity ==> (vl `mul` cw) `add` rs == zero
    with begin
      ok_lte_ninf rm; ok_lte_ninf vm;
      introduce forall (t : natlt 16). acc1 es t == neg #et infinity
      with row_max_ninf_all es 16 t;
      row_sum_masked es m' 16;
      mul_comm vl cw; mul_zero cw;
      add_zero (zero #et)
    end;
    introduce Finite? (kind m') ==> lt zero ((vl `mul` cw) `add` rs)
    with begin
      if lt vm rm
      then begin
        row_max_cases es 16;
        eliminate exists (j : natlt 16). j < 16 /\ rm == acc1 es j
        with row_sum_pos es m' 16 j
      end
      else begin
        FA.sub_self vm;
        FA.exp_zero #et ();
        SB.corr_weight_exp vm m';
        nonneg_not_nan vl;
        mul_one vl
      end;
      FA.add_pos (vl `mul` cw) rs
    end

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 30"

(* ------------------------------------------------------------------ *)
(* The invariant, run over the warp's key tiles.                       *)
(* ------------------------------------------------------------------ *)

(* Every entry of a tile is a number or the sentinel: admitted keys are finite
   by hypothesis, the rest carry [-inf] by construction. *)
let tile_ok
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16) (k0 : nat)
  : Lemma
      (requires SS.all_finite emask has_mask row_active causal bi qh qpos
                  cbound scale eQ eKg i)
      (ensures
        (forall (t : natlt 16).
           ok (acc1 (SF.tile_scores #et_acc #et_ab #_f #_r #_s #_rb #_c1 emask has_mask row_active causal bi qh qpos
                       k0 cbound scale
                       (SS.tile_score_row #et_ab #et_acc #_s #_ #sk #d #sq16 eQ eKg k0 i)) t)))
  = introduce forall (t : natlt 16).
      ok (acc1 (SF.tile_scores #et_acc #et_ab #_f #_r #_s #_rb #_c1 emask has_mask row_active causal bi qh qpos
                  k0 cbound scale
                  (SS.tile_score_row #et_ab #et_acc #_s #_ #sk #d #sq16 eQ eKg k0 i)) t)
    with (if SF.key_ok row_active causal sk cbound (k0 + t)
          then assert (k0 <= sk)
          else ())

let rec run_inv
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw w t : nat)
  : Lemma
      (requires SS.all_finite emask has_mask row_active causal bi qh qpos
                  cbound scale eQ eKg i)
      (ensures
        (let ml = SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi qh qpos cbound
                    scale eQ eKg i nw w t in
         warp_inv (fst ml) (snd ml)))
      (decreases t)
  = if t = 0
    then (kind_zero #et_acc; lte_refl (zero #et_acc);
          SB.ninf_not_nan #et_acc (); ninf_not_finite #et_acc ())
    else begin
      run_inv emask has_mask row_active causal bi qh qpos cbound scale
        eQ eKg i nw w (t - 1);
      let k0 = (w + (t - 1) * nw) * 16 in
      tile_ok emask has_mask row_active causal bi qh qpos cbound scale
        eQ eKg i k0;
      let ml = SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi qh qpos cbound
                 scale eQ eKg i nw w (t - 1) in
      step_inv
        (SF.tile_scores #et_acc #et_ab #_f #_r #_s #_rb #_c1 emask has_mask row_active causal bi qh qpos k0 cbound
           scale (SS.tile_score_row #et_ab #et_acc #_s #_ #sk #d #sq16 eQ eKg k0 i))
        (fst ml) (snd ml)
    end

#pop-options

#push-options "--fuel 3 --ifuel 1 --z3rlimit 30"

(* ------------------------------------------------------------------ *)
(* Some key is admitted, so some maximum is finite.                    *)
(* ------------------------------------------------------------------ *)

let row_max_finite
  (#et : Type0) {| floating et |} (#bn : nat)
  (s : chest1 et bn) (k : nat { k <= bn }) (j : natlt bn)
  : Lemma (requires (forall (t : natlt bn). ok (acc1 s t)) /\ j < k /\
                    Finite? (kind (acc1 s j)))
          (ensures Finite? (kind (SF.row_max s k)))
  = row_max_ok s k;
    row_max_ub s k j;
    ninf_not_finite #et ();
    introduce SF.row_max s k == neg #et infinity ==> False
    with ok_lte_ninf (acc1 s j)

(* Once the running maximum is finite it stays finite: [fmax] never returns
   the sentinel over a number. *)
let rec run_max_mono
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw w t u : nat)
  : Lemma
      (requires SS.all_finite emask has_mask row_active causal bi qh qpos
                  cbound scale eQ eKg i /\ t <= u /\
                Finite? (kind (fst (SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb
                                      #_c1 #b #hq #sq #sk #d #sq16
                                      emask has_mask row_active causal bi qh
                                      qpos cbound scale eQ eKg i nw w t))))
      (ensures
        Finite? (kind (fst (SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                              #b #hq #sq #sk #d #sq16
                              emask has_mask row_active causal bi qh qpos
                              cbound scale eQ eKg i nw w u))))
      (decreases u - t)
  = if u = t then ()
    else begin
      run_max_mono emask has_mask row_active causal bi qh qpos cbound scale
        eQ eKg i nw w t (u - 1);
      let k0 = (w + (u - 1) * nw) * 16 in
      tile_ok emask has_mask row_active causal bi qh qpos cbound scale
        eQ eKg i k0;
      let es = SF.tile_scores #et_acc #et_ab #_f #_r #_s #_rb #_c1
                 emask has_mask row_active causal bi qh qpos k0 cbound scale
                 (SS.tile_score_row #et_ab #et_acc #_s #_ #sk #d #sq16
                    eQ eKg k0 i) in
      row_max_ok es 16;
      run_inv emask has_mask row_active causal bi qh qpos cbound scale
        eQ eKg i nw w (u - 1);
      fmax_finite
        (fst (SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi
                qh qpos cbound scale eQ eKg i nw w (u - 1)))
        (SF.row_max es 16)
    end

(* Warp [0] sees key [0] in its first tile, so its maximum is finite as soon
   as the row admits key [0] -- which every active row does. *)
let run_first_finite
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b #hq #sq : nat) (#sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask row_active causal : bool)
  (bi : natlt b) (qh : natlt hq) (qpos : natlt sq)
  (cbound : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d) (i : natlt 16)
  (nw : nat)
  : Lemma
      (requires SS.all_finite emask has_mask row_active causal bi qh qpos
                  cbound scale eQ eKg i /\
                SF.key_ok row_active causal sk cbound 0)
      (ensures
        Finite? (kind (fst (SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                              #b #hq #sq #sk #d #sq16
                              emask has_mask row_active causal bi qh qpos
                              cbound scale eQ eKg i nw 0 1))))
  = tile_ok emask has_mask row_active causal bi qh qpos cbound scale
      eQ eKg i 0;
    let es = SF.tile_scores #et_acc #et_ab #_f #_r #_s #_rb #_c1
               emask has_mask row_active causal bi qh qpos 0 cbound scale
               (SS.tile_score_row #et_ab #et_acc #_s #_ #sk #d #sq16
                  eQ eKg 0 i) in
    assert (Finite? (kind (acc1 es 0)));
    row_max_finite es 16 0;
    SB.ninf_not_nan #et_acc ();
    ninf_not_finite #et_acc ();
    row_max_ok es 16;
    fmax_finite (neg #et_acc infinity) (SF.row_max es 16);
    assert (fst (SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                   #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi
                   qh qpos cbound scale eQ eKg i nw 0 0) == neg infinity);
    assert (fst (SS.run_ml #et_ab #et_acc #_f #_r #_s #_rb #_c1
                   #b #hq #sq #sk #d #sq16 emask has_mask row_active causal bi
                   qh qpos cbound scale eQ eKg i nw 0 1)
            == fmax (neg #et_acc infinity) (SF.row_max es 16))

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"

(* ------------------------------------------------------------------ *)
(* The block-wide denominator.                                          *)
(* ------------------------------------------------------------------ *)

let block_denom_pos
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (#b : nat) (#hq #sq #sk : pos) (#d : pos) (#sq16 : squash (16 /?+ d))
  (emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (has_mask causal : bool)
  (bi : natlt b) (kvh group rows r0 : nat) (scale : et_acc)
  (eQ : chest2 et_ab 16 d) (eKg : chest2 et_ab sk d)
  (nw : pos) (nkt : nat) (i : natlt 16)
  : Lemma
      (requires
        SS.all_finite emask has_mask (SF.lane_active_row rows r0 i) causal bi
          (SF.lane_qh hq sq kvh group rows r0 i) (SF.lane_qpos sq rows r0 i)
          (SF.lane_cbound sq sk rows r0 i) scale eQ eKg i /\
        SF.key_ok (SF.lane_active_row rows r0 i) causal sk
          (SF.lane_cbound sq sk rows r0 i) 0 /\
        SF.warp_iters nw nkt 0 >= 1)
      (ensures
        (let eM = SS.block_m #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16 emask has_mask causal bi kvh group rows r0 scale
                    eQ eKg nw nkt in
         let eL = SS.block_l #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16 emask has_mask causal bi kvh group rows r0 scale
                    eQ eKg nw nkt in
         lt zero (SF.gsum eM eL (SF.gmax eM i nw) i nw)))
  = let ra = SF.lane_active_row rows r0 i in
    let qh = SF.lane_qh hq sq kvh group rows r0 i in
    let qpos = SF.lane_qpos sq rows r0 i in
    let cbound = SF.lane_cbound sq sk rows r0 i in
    let eM = SS.block_m #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16 emask has_mask causal bi kvh group rows r0 scale
               eQ eKg nw nkt in
    let eL = SS.block_l #et_ab #et_acc #_f #_r #_s #_rb #_c1 #b #hq #sq #sk #d #sq16 emask has_mask causal bi kvh group rows r0 scale
               eQ eKg nw nkt in
    introduce forall (w : natlt nw).
      ok (acc2 eM w i) /\ lte zero (acc2 eL w i) /\
      (Finite? (kind (acc2 eM w i)) ==> lt zero (acc2 eL w i))
    with begin
      SS.block_ml_acc emask has_mask causal bi kvh group rows r0 scale
        eQ eKg nw nkt w i;
      SS.run_mlv_t_acc emask has_mask causal bi kvh group rows r0 scale
        eQ eKg i nw w (SF.warp_iters nw nkt w);
      run_inv emask has_mask ra causal bi qh qpos cbound scale eQ eKg i
        nw w (SF.warp_iters nw nkt w)
    end;
    run_first_finite emask has_mask ra causal bi qh qpos cbound scale
      eQ eKg i nw;
    run_max_mono emask has_mask ra causal bi qh qpos cbound scale eQ eKg i
      nw 0 1 (SF.warp_iters nw nkt 0);
    SS.block_ml_acc emask has_mask causal bi kvh group rows r0 scale
      eQ eKg nw nkt 0 i;
    SS.run_mlv_t_acc emask has_mask causal bi kvh group rows r0 scale
      eQ eKg i nw 0 (SF.warp_iters nw nkt 0);
    assert (Finite? (kind (acc2 eM 0 i)));
    gmax_ok eM i nw;
    gmax_ub eM i nw 0;
    gmax_cases eM i nw;
    ninf_not_finite #et_acc ();
    introduce SF.gmax eM i nw == neg #et_acc infinity ==> False
    with ok_lte_ninf (acc2 eM 0 i);
    introduce forall (w : natlt nw). lte (acc2 eM w i) (SF.gmax eM i nw)
    with gmax_ub eM i nw w;
    eliminate exists (w0 : natlt nw).
      w0 < nw /\ SF.gmax eM i nw == acc2 eM w0 i
    with gsum_pos eM eL (SF.gmax eM i nw) i nw w0

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"

(* Warp [0] always runs at least once: an active row admits key [0], which
   lives in the first key tile. *)
let warp_iters_pos (nw : pos) (nkt : nat)
  : Lemma (requires nkt >= 1) (ensures SF.warp_iters nw nkt 0 >= 1)
  = FStar.Math.Lemmas.lemma_div_le nw (nkt + nw - 1) nw;
    FStar.Math.Lemmas.cancel_mul_div 1 nw

(* ------------------------------------------------------------------ *)
(* THE RESULT: the epilogue never divides by zero.                     *)
(* ------------------------------------------------------------------ *)

let flash_denom_pos
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#sqsk : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat) (i : natlt 16)
  : Lemma
      (requires
        (let hqv : pos = SZ.v hq in
         let sqv : pos = SZ.v sq in
         let skv : pos = SZ.v sk in
         let rowsv = SZ.v rows in
         SF.lane_active_row rowsv r0 i /\
         SS.all_finite emask has_mask true causal bi
           (SF.lane_qh hqv sqv kvh group rowsv r0 i)
           (SF.lane_qpos sqv rowsv r0 i)
           (SF.lane_cbound sqv skv rowsv r0 i) scale
           (SF.q_tile 16 rowsv group eQ bi kvh r0) eKg i))
      (ensures
        acc1 (SV.flash_egl_at #et_ab #et_acc #_f #_r #_s #_rb #_c1 nw d b hq sq rows sk #sqsk eQ eKg emask has_mask
                causal scale bi kvh group r0) i `gt` zero)
  = let hqv : pos = SZ.v hq in
    let sqv : pos = SZ.v sq in
    let skv : pos = SZ.v sk in
    let rowsv = SZ.v rows in
    let nwv : pos = SZ.v nw in
    let eQt = SF.q_tile 16 rowsv group eQ bi kvh r0 in
    let nkt = SF.key_tiles 16 16 sqv skv rowsv r0 causal in
    SS.key_ok_tile_lt sqv skv rowsv r0 causal i 0;
    warp_iters_pos nwv nkt;
    block_denom_pos emask has_mask causal bi kvh group rowsv r0 scale
      eQt eKg nwv nkt i;
    SV.flash_eM_at_def nw d b hq sq rows sk eQ eKg emask has_mask causal scale
      bi kvh group r0;
    SV.flash_eL_at_def nw d b hq sq rows sk eQ eKg emask has_mask causal scale
      bi kvh group r0;
    SV.flash_escale_egl_at_def nw d b hq sq rows sk eQ eKg emask has_mask causal
      scale bi kvh group r0 i

#pop-options
