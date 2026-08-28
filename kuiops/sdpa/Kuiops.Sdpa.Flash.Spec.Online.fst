module Kuiops.Sdpa.Flash.Spec.Online

(* Real-valued algebra of the online (flash) softmax used by
   [Kuiops.Sdpa.Flash], including masking and the cross-warp key split.

   The kernel never materializes the attention probabilities; it keeps, per
   query row, a running maximum [m], a running denominator [l] and a running
   output accumulator [o], and rescales them by [exp (m - m')] whenever the
   maximum moves.  The invariant tying those registers to the real spec is

     l == dsum x p /. exp m        o == nsum x y p /. exp m

   where [p] is the set of keys absorbed so far, [dsum] is the unnormalized
   softmax denominator over [p] and [nsum] the corresponding numerator against
   the value column [y].  Note the running maximum is unconstrained: it only
   exists to keep the floats in range, and every lemma here holds for an
   arbitrary [m].  That is what makes masked keys free -- absorbing a masked
   key leaves [p] alone -- and what makes the cross-warp combine a plain
   rescale-and-add over a partition of the key set.

   No extended reals appear anywhere: a masked key is simply absent from [p].
   The kernel's [-inf] sentinel is a float-level encoding of "not in [p]", and
   the states in which [p] is empty (where [exp m] would be [exp (-inf)]) are
   excluded by the invariant's users rather than modelled. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Seq.Common

module MS = Kuiper.Spec.GEMM

(* ------------------------------------------------------------------ *)
(* Masked sums over an index range.                                    *)
(* ------------------------------------------------------------------ *)

let pred (n : nat) = natlt n -> bool

let pfalse (#n : nat) : pred n = fun _ -> false
let ptrue (#n : nat) : pred n = fun _ -> true
let por (#n : nat) (p q : pred n) : pred n = fun j -> p j || q j
let pand (#n : nat) (p q : pred n) : pred n = fun j -> p j && q j
let disjoint (#n : nat) (p q : pred n) : prop = forall (j : natlt n). ~(p j /\ q j)

(* [sum_upto f p k] sums [f j] over the [j < k] admitted by [p], left to
   right, exactly like [chest1_rsum] over a select-to-zero sequence. *)
let rec sum_upto (#n : nat) (f : natlt n -> GTot real) (p : pred n) (k : natle n)
  : GTot real (decreases k)
  = if k = 0 then 0.0R
    else sum_upto f p (k - 1) +. (if p (k - 1) then f (k - 1) else 0.0R)

let sum_where (#n : nat) (f : natlt n -> GTot real) (p : pred n) : GTot real = sum_upto f p n

let rec sum_upto_ext (#n : nat) (f g : natlt n -> GTot real) (p q : pred n) (k : natle n)
  : Lemma (requires forall (j : natlt n). j < k ==> (p j == q j /\ (p j ==> f j == g j)))
          (ensures sum_upto f p k == sum_upto g q k)
          (decreases k)
  = if k = 0 then () else sum_upto_ext f g p q (k - 1)

let sum_where_ext (#n : nat) (f g : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires forall (j : natlt n). p j == q j /\ (p j ==> f j == g j))
          (ensures sum_where f p == sum_where g q)
  = sum_upto_ext f g p q n

let rec sum_upto_false (#n : nat) (f : natlt n -> GTot real) (p : pred n) (k : natle n)
  : Lemma (requires forall (j : natlt n). j < k ==> ~(p j))
          (ensures sum_upto f p k == 0.0R)
          (decreases k)
  = if k = 0 then () else sum_upto_false f p (k - 1)

let rec sum_upto_split (#n : nat) (f : natlt n -> GTot real) (p q : pred n) (k : natle n)
  : Lemma (requires disjoint p q)
          (ensures sum_upto f (por p q) k == sum_upto f p k +. sum_upto f q k)
          (decreases k)
  = if k = 0 then () else sum_upto_split f p q (k - 1)

let sum_where_split (#n : nat) (f : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires disjoint p q)
          (ensures sum_where f (por p q) == sum_where f p +. sum_where f q)
  = sum_upto_split f p q n

(* A constant divisor factors out of a masked sum. *)
let rec sum_upto_div (#n : nat) (f : natlt n -> GTot real) (p : pred n)
                     (c : real { c =!= 0.0R }) (k : natle n)
  : Lemma (ensures sum_upto (fun j -> f j /. c) p k == sum_upto f p k /. c)
          (decreases k)
  = if k = 0 then () else sum_upto_div f p c (k - 1)

let sum_where_div (#n : nat) (f : natlt n -> GTot real) (p : pred n) (c : real { c =!= 0.0R })
  : Lemma (sum_where (fun j -> f j /. c) p == sum_where f p /. c)
  = sum_upto_div f p c n

(* A masked sum of non-negative terms is non-negative, and strictly positive
   as soon as one admitted term is. *)
let rec sum_upto_nonneg (#n : nat) (f : natlt n -> GTot real) (p : pred n) (k : natle n)
  : Lemma (requires forall (j : natlt n). f j >=. 0.0R)
          (ensures sum_upto f p k >=. 0.0R)
          (decreases k)
  = if k = 0 then () else sum_upto_nonneg f p (k - 1)

let rec sum_upto_pos (#n : nat) (f : natlt n -> GTot real) (p : pred n) (k : natle n)
                     (i : natlt n)
  : Lemma (requires (forall (j : natlt n). f j >=. 0.0R) /\ i < k /\ p i /\ f i >. 0.0R)
          (ensures sum_upto f p k >. 0.0R)
          (decreases k)
  = if i = k - 1
    then sum_upto_nonneg f p (k - 1)
    else sum_upto_pos f p (k - 1) i

let sum_where_pos (#n : nat) (f : natlt n -> GTot real) (p : pred n) (i : natlt n)
  : Lemma (requires (forall (j : natlt n). f j >=. 0.0R) /\ p i /\ f i >. 0.0R)
          (ensures sum_where f p >. 0.0R)
  = sum_upto_pos f p n i

(* ------------------------------------------------------------------ *)
(* The running numerator / denominator.                                *)
(* ------------------------------------------------------------------ *)

(* Unnormalized softmax denominator over the absorbed keys [p]. *)
let dsum (#n : nat) (x : natlt n -> GTot real) (p : pred n) : GTot real
  = sum_where (fun j -> exp (x j)) p

(* Unnormalized numerator against one value column [y]. *)
let nsum (#n : nat) (x y : natlt n -> GTot real) (p : pred n) : GTot real
  = sum_where (fun j -> exp (x j) *. y j) p

let dsum_pos (#n : nat) (x : natlt n -> GTot real) (p : pred n) (i : natlt n)
  : Lemma (requires p i) (ensures dsum x p >. 0.0R)
  = sum_where_pos (fun j -> exp (x j)) p i

let dsum_split (#n : nat) (x : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires disjoint p q)
          (ensures dsum x (por p q) == dsum x p +. dsum x q)
  = sum_where_split (fun j -> exp (x j)) p q

let nsum_split (#n : nat) (x y : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires disjoint p q)
          (ensures nsum x y (por p q) == nsum x y p +. nsum x y q)
  = sum_where_split (fun j -> exp (x j) *. y j) p q

(* ------------------------------------------------------------------ *)
(* Rescaling: moving the running maximum from [m] to [m'].             *)
(* ------------------------------------------------------------------ *)

let rescale (a m m' : real) : real = a *. exp (m -. m')

let lem_rescale (a m m' d : real)
  : Lemma (requires a == d /. exp m)
          (ensures rescale a m m' == d /. exp m')
  = exp_sub m m'

(* The shifted tile sums the kernel actually computes. *)
let tsum_d (#n : nat) (x : natlt n -> GTot real) (t : pred n) (m' : real) : GTot real
  = sum_where (fun j -> exp (x j -. m')) t

let tsum_n (#n : nat) (x y : natlt n -> GTot real) (t : pred n) (m' : real) : GTot real
  = sum_where (fun j -> exp (x j -. m') *. y j) t

let lem_tsum_d (#n : nat) (x : natlt n -> GTot real) (t : pred n) (m' : real)
  : Lemma (tsum_d x t m' == dsum x t /. exp m')
  = let f : natlt n -> GTot real = fun j -> exp (x j) in
    sum_where_div f t (exp m');
    sum_where_ext (fun j -> exp (x j -. m')) (fun j -> f j /. exp m') t t

let lem_tsum_n (#n : nat) (x y : natlt n -> GTot real) (t : pred n) (m' : real)
  : Lemma (tsum_n x y t m' == nsum x y t /. exp m')
  = let f : natlt n -> GTot real = fun j -> exp (x j) *. y j in
    sum_where_div f t (exp m');
    sum_where_ext (fun j -> exp (x j -. m') *. y j) (fun j -> f j /. exp m') t t

let dsum_ext (#n : nat) (x : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires forall (j : natlt n). p j == q j) (ensures dsum x p == dsum x q)
  = sum_where_ext (fun j -> exp (x j)) (fun j -> exp (x j)) p q

let nsum_ext (#n : nat) (x y : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires forall (j : natlt n). p j == q j) (ensures nsum x y p == nsum x y q)
  = sum_where_ext (fun j -> exp (x j) *. y j) (fun j -> exp (x j) *. y j) p q

(* ------------------------------------------------------------------ *)
(* The flash accumulator invariant and its step.                       *)
(* ------------------------------------------------------------------ *)

(* [l] and [o] are the running denominator and output accumulator for the
   keys in [p], both scaled down by [exp m].  [m] is arbitrary. *)
let flash_inv (#n : nat) (x y : natlt n -> GTot real) (p : pred n) (m l o : real) : prop
  = l == dsum x p /. exp m /\ o == nsum x y p /. exp m

(* Absorb a whole tile [t] of fresh keys while moving the maximum to [m'].
   [p'] is any pointwise description of the enlarged key set. *)
let flash_step (#n : nat) (x y : natlt n -> GTot real) (p t p' : pred n) (m m' l o : real)
  : Lemma (requires disjoint p t /\
                    (forall (j : natlt n). p' j == (p j || t j)) /\
                    flash_inv x y p m l o)
          (ensures flash_inv x y p' m'
                     (rescale l m m' +. tsum_d x t m')
                     (rescale o m m' +. tsum_n x y t m'))
  = lem_rescale l m m' (dsum x p);
    lem_rescale o m m' (nsum x y p);
    lem_tsum_d x t m';
    lem_tsum_n x y t m';
    dsum_split x p t;
    nsum_split x y p t;
    dsum_ext x p' (por p t);
    nsum_ext x y p' (por p t)

let rec sum_upto_single (#n : nat) (f : natlt n -> GTot real) (i : natlt n) (t : pred n)
                        (k : natle n)
  : Lemma (requires forall (j : natlt n). t j == (j = i))
          (ensures sum_upto f t k == (if i < k then f i else 0.0R))
          (decreases k)
  = if k = 0 then () else sum_upto_single f i t (k - 1)

let sum_where_single (#n : nat) (f : natlt n -> GTot real) (i : natlt n) (t : pred n)
  : Lemma (requires forall (j : natlt n). t j == (j = i))
          (ensures sum_where f t == f i)
  = sum_upto_single f i t n

(* Absorb a single unmasked key [i]. *)
let flash_step_key (#n : nat) (x y : natlt n -> GTot real) (p p' : pred n) (i : natlt n)
                   (m m' l o : real)
  : Lemma (requires ~(p i) /\
                    (forall (j : natlt n). p' j == (p j || j = i)) /\
                    flash_inv x y p m l o)
          (ensures flash_inv x y p' m'
                     (rescale l m m' +. exp (x i -. m'))
                     (rescale o m m' +. exp (x i -. m') *. y i))
  = let t : pred n = fun j -> j = i in
    sum_where_single (fun j -> exp (x j -. m')) i t;
    sum_where_single (fun j -> exp (x j -. m') *. y j) i t;
    flash_step x y p t p' m m' l o

(* Absorb a masked key, or a fully masked tile: the key set is unchanged and
   only the rescale happens.  This is where the kernel would need [-inf] if
   the invariant mentioned the running maximum -- it does not. *)
let flash_step_masked (#n : nat) (x y : natlt n -> GTot real) (p p' : pred n) (m m' l o : real)
  : Lemma (requires (forall (j : natlt n). p' j == p j) /\ flash_inv x y p m l o)
          (ensures flash_inv x y p' m' (rescale l m m') (rescale o m m'))
  = let t : pred n = pfalse in
    sum_upto_false (fun j -> exp (x j -. m')) t n;
    sum_upto_false (fun j -> exp (x j -. m') *. y j) t n;
    flash_step x y p t p' m m' l o

(* ------------------------------------------------------------------ *)
(* Merging two partial states (the cross-warp key split).              *)
(* ------------------------------------------------------------------ *)

(* Each warp owns a disjoint set of key tiles and reaches its own
   [(m, l, o)].  Rescaling both to a common [gm] and adding merges them,
   because the invariant is linear in the key set once the [exp m] scaling is
   normalized away.  Iterating this absorbs any number of warps. *)
#push-options "--z3rlimit 60"
let flash_combine (#n : nat) (x y : natlt n -> GTot real) (p1 p2 p' : pred n)
                  (m1 m2 gm l1 l2 o1 o2 : real)
  : Lemma (requires disjoint p1 p2 /\
                    (forall (j : natlt n). p' j == (p1 j || p2 j)) /\
                    flash_inv x y p1 m1 l1 o1 /\
                    flash_inv x y p2 m2 l2 o2)
          (ensures flash_inv x y p' gm
                     (rescale l1 m1 gm +. rescale l2 m2 gm)
                     (rescale o1 m1 gm +. rescale o2 m2 gm))
  = lem_rescale l1 m1 gm (dsum x p1);
    lem_rescale l2 m2 gm (dsum x p2);
    lem_rescale o1 m1 gm (nsum x y p1);
    lem_rescale o2 m2 gm (nsum x y p2);
    dsum_split x p1 p2;
    nsum_split x y p1 p2;
    dsum_ext x p' (por p1 p2);
    nsum_ext x y p' (por p1 p2)
#pop-options

(* ------------------------------------------------------------------ *)
(* Restricting a sum to the support of its summand.                    *)
(* ------------------------------------------------------------------ *)

let rec sum_upto_restrict (#n : nat) (g : natlt n -> GTot real) (p : pred n) (k : natle n)
  : Lemma (requires forall (j : natlt n). ~(p j) ==> g j == 0.0R)
          (ensures sum_upto g ptrue k == sum_upto g p k)
          (decreases k)
  = if k = 0 then () else sum_upto_restrict g p (k - 1)

let sum_where_restrict (#n : nat) (g : natlt n -> GTot real) (p : pred n)
  : Lemma (requires forall (j : natlt n). ~(p j) ==> g j == 0.0R)
          (ensures sum_where g ptrue == sum_where g p)
  = sum_upto_restrict g p n

(* ------------------------------------------------------------------ *)
(* Bridge to [Kuiops.Sdpa.Flash.Spec] and to [Kuiper.Spec.GEMM].       *)
(* ------------------------------------------------------------------ *)

module FS = Kuiops.Sdpa.Flash.Spec

(* [chest1_rsum] folds left from [0], which is exactly [sum_upto]. *)
let rec sum_upto_rsum (#n : nat) (c : chest1 real n) (f : natlt n -> GTot real)
                      (p : pred n) (k : natle n)
  : Lemma (requires forall (j : natlt n). acc1 c j == (if p j then f j else 0.0R))
          (ensures seq_fold_left (+.) 0.0R (Seq.slice (chest1_to_seq c) 0 k)
                     == sum_upto f p k)
          (decreases k)
  = if k = 0
    then assert (Seq.length (Seq.slice (chest1_to_seq c) 0 0) == 0)
    else (sum_upto_rsum c f p (k - 1);
          lemma_seq_fold_left_slice 0.0R ( +. ) (chest1_to_seq c) 0 (k - 1))

let chest1_rsum_is_sum_where (#n : nat) (c : chest1 real n) (f : natlt n -> GTot real)
                             (p : pred n)
  : Lemma (requires forall (j : natlt n). acc1 c j == (if p j then f j else 0.0R))
          (ensures chest1_rsum c == sum_where f p)
  = sum_upto_rsum c f p n;
    assert (Seq.equal (Seq.slice (chest1_to_seq c) 0 n) (chest1_to_seq c))

let dsum_is_masked_denom (#n : pos) (valid : FS.valid_pred n) (r : chest1 real n)
  : Lemma (dsum (acc1 r) valid == FS.masked_denom valid r)
  = let x : natlt n -> GTot real = acc1 r in
    assert (FS.masked_denom valid r == chest1_rsum (FS.masked_exps valid r));
    assert (forall (j : natlt n).
      acc1 (FS.masked_exps valid r) j ==
        (if valid j then exp (x j) else 0.0R));
    chest1_rsum_is_sum_where
      (FS.masked_exps valid r) (fun j -> exp (x j)) valid;
    calc (==) {
      dsum (acc1 r) valid;
      == {} sum_where (fun j -> exp (x j)) valid;
      == {} chest1_rsum (FS.masked_exps valid r);
      == {} FS.masked_denom valid r;
    }

(* [matmul_single] is also a left fold from [zero]. *)
let rec matmul_single_is_sum_upto
  (#rows #shared #columns : nat)
  (m1 : chest2 real rows shared) (m2 : chest2 real shared columns)
  (row : natlt rows) (col : natlt columns) (k : natle shared)
  : Lemma (ensures MS.__matmul_single m1 m2 row col k
                     == sum_upto (fun j -> acc2 m1 row j *. acc2 m2 j col) ptrue k)
          (decreases k)
  = if k = 0 then () else matmul_single_is_sum_upto m1 m2 row col (k - 1)

(* Pointwise-hypothesis forms.  Stating these against a caller-supplied [g]
   rather than a literal lambda keeps callers from having to equate two
   syntactically distinct closures, which the SMT encoding cannot do. *)
let matmul_single_is_sum_where
  (#rows #shared #columns : nat)
  (m1 : chest2 real rows shared) (m2 : chest2 real shared columns)
  (row : natlt rows) (col : natlt columns)
  (g : natlt shared -> GTot real)
  : Lemma (requires forall (j : natlt shared). g j == acc2 m1 row j *. acc2 m2 j col)
          (ensures MS.matmul_single m1 m2 row col == sum_where g ptrue)
  = matmul_single_is_sum_upto m1 m2 row col shared;
    sum_where_ext g (fun j -> acc2 m1 row j *. acc2 m2 j col) ptrue ptrue

let sum_where_div_at (#n : nat) (f g : natlt n -> GTot real) (p : pred n)
                     (c : real { c =!= 0.0R })
  : Lemma (requires forall (j : natlt n). p j ==> g j == f j /. c)
          (ensures sum_where g p == sum_where f p /. c)
  = sum_where_div f p c;
    sum_where_ext g (fun j -> f j /. c) p p

let dsum_pointwise (#n : nat) (x g : natlt n -> GTot real) (p : pred n)
  : Lemma (requires forall (j : natlt n). p j ==> g j == exp (x j))
          (ensures sum_where g p == dsum x p)
  = sum_where_ext g (fun j -> exp (x j)) p p

let nsum_pointwise (#n : nat) (x y g : natlt n -> GTot real) (p : pred n)
  : Lemma (requires forall (j : natlt n). p j ==> g j == exp (x j) *. y j)
          (ensures sum_where g p == nsum x y p)
  = sum_where_ext g (fun j -> exp (x j) *. y j) p p

(* ------------------------------------------------------------------ *)
(* The epilogue: dividing the accumulator by the denominator.          *)
(* ------------------------------------------------------------------ *)

let div_cancel (a b e : real)
  : Lemma (requires e >. 0.0R /\ b >. 0.0R)
          (ensures (a /. e) /. (b /. e) == a /. b)
  = ()

let div_mul_left (a b z : real)
  : Lemma (requires z >. 0.0R) (ensures (a /. z) *. b == (a *. b) /. z)
  = ()

(* The spec's output cell, rewritten as the unnormalized numerator over the
   admitted keys divided by their denominator. *)
let masked_matmul_cell
  (#sk #dv #rows : pos)
  (valid : FS.valid_pred sk)
  (scores : chest1 real sk)
  (probs : chest2 real rows sk)
  (mV : chest2 real sk dv)
  (i : natlt rows) (c : natlt dv)
  : Lemma (requires forall (j : natlt sk).
                      acc2 probs i j == acc1 (FS.masked_softmax_real valid scores) j)
          (ensures MS.matmul_single probs mV i c
                     == nsum (acc1 scores) (fun (j : natlt sk) -> acc2 mV j c) valid
                        /. FS.masked_denom valid scores)
  = let x : natlt sk -> GTot real = acc1 scores in
    let y : natlt sk -> GTot real = fun (j : natlt sk) -> acc2 mV j c in
    let z : (w:real { w >. 0.0R }) = FS.masked_denom valid scores in
    let g : natlt sk -> GTot real = fun (j : natlt sk) -> acc2 probs i j *. acc2 mV j c in
    let f : natlt sk -> GTot real = fun (j : natlt sk) -> exp (x j) *. y j in
    assert (forall (j : natlt sk).
              acc2 probs i j == (if valid j then exp (x j) /. z else 0.0R));
    introduce forall (j : natlt sk). valid j ==> g j == f j /. z
    with introduce _ ==> _
    with div_mul_left (exp (x j)) (y j) z;
    matmul_single_is_sum_where probs mV i c g;
    sum_where_restrict g valid;
    sum_where_div_at f g valid z;
    nsum_pointwise x y f valid

(* [valid_pred] carries the non-emptiness proof; recover the witness. *)
let valid_witness (#n : pos) (valid : FS.valid_pred n) : GTot (k : natlt n { valid k })
  = FStar.IndefiniteDescription.indefinite_description_ghost
      (natlt n) (fun (j : natlt n) -> b2t (valid j))

(* Dividing the two registers cancels the [exp m] scaling. *)
let flash_quotient
  (#n : nat) (x y : natlt n -> GTot real) (p : pred n) (m l o : real) (i : natlt n)
  : Lemma (requires flash_inv x y p m l o /\ p i)
          (ensures l >. 0.0R /\
                   (l =!= 0.0R ==> o /. l == nsum x y p /. dsum x p))
  = dsum_pos x p i;
    div_cancel (nsum x y p) (dsum x p) (exp m)

(* Final correctness of one output cell.  [probs] is the spec's probability
   matrix, whose row [i] is the masked softmax of [scores]; [l] and [o] are the
   registers the kernel holds once every key has been absorbed.  Dividing them
   yields exactly the [(i, c)] cell of [probs @ mV]. *)
let flash_row_out
  (#sk #dv #rows : pos)
  (valid : FS.valid_pred sk)
  (scores : chest1 real sk)
  (probs : chest2 real rows sk)
  (mV : chest2 real sk dv)
  (i : natlt rows) (c : natlt dv)
  (m l o : real)
  : Lemma (requires flash_inv (acc1 scores) (fun (j : natlt sk) -> acc2 mV j c)
                              valid m l o /\
                    (forall (j : natlt sk).
                       acc2 probs i j == acc1 (FS.masked_softmax_real valid scores) j))
          (ensures l >. 0.0R /\
                   (l =!= 0.0R ==> o /. l == MS.matmul_single probs mV i c))
  = masked_matmul_cell valid scores probs mV i c;
    dsum_is_masked_denom valid scores;
    flash_quotient (acc1 scores) (fun (j : natlt sk) -> acc2 mV j c) valid m l o
                   (valid_witness valid)

(* ------------------------------------------------------------------ *)
(* Restricting a global masked sum to one contiguous key tile.         *)
(* ------------------------------------------------------------------ *)

(* Extending the range of a masked sum past the support of [p] is a no-op. *)
let rec sum_upto_stop (#n : nat) (f : natlt n -> GTot real) (p : pred n)
                      (k : natle n) (k' : natle n)
  : Lemma (requires k <= k' /\ (forall (j : natlt n). k <= j /\ j < k' ==> ~(p j)))
          (ensures sum_upto f p k' == sum_upto f p k)
          (decreases k')
  = if k' = k then () else sum_upto_stop f p k (k' - 1)

(* Agreement between a global mask [p] and a tile-local mask [q] anchored at
   [k0].  The tile may run past the end of the range -- the kernel's last key
   tile does -- and those local slots simply admit nothing. *)
let tile_agree
  (#n : nat) (f : natlt n -> GTot real) (p : pred n)
  (#bn : nat) (g : natlt bn -> GTot real) (q : pred bn) (k0 : nat)
  : prop
  = forall (i : natlt bn).
      (k0 + i < n ==> q i == p (k0 + i)) /\
      (k0 + i >= n ==> ~(q i)) /\
      (q i ==> (k0 + i < n /\ g i == f (k0 + i)))

(* A masked sum supported on the tile [[k0, k0 + bn)] is the tile-local fold. *)
let rec sum_upto_tile
  (#n : nat) (f : natlt n -> GTot real) (p : pred n)
  (#bn : nat) (g : natlt bn -> GTot real) (q : pred bn)
  (k0 : natle n) (k : natle bn)
  : Lemma (requires (forall (j : natlt n). j < k0 ==> ~(p j)) /\
                    tile_agree f p g q k0)
          (ensures (if k0 + k <= n
                    then sum_upto f p (k0 + k) == sum_upto g q k
                    else sum_upto f p n == sum_upto g q k))
          (decreases k)
  = if k = 0 then sum_upto_false f p k0 else sum_upto_tile f p g q k0 (k - 1)

let sum_where_tile
  (#n : nat) (f : natlt n -> GTot real) (p : pred n)
  (#bn : nat) (g : natlt bn -> GTot real) (q : pred bn)
  (k0 : natle n)
  : Lemma (requires (forall (j : natlt n).
                       p j ==> (k0 <= j /\ j < k0 + bn)) /\
                    tile_agree f p g q k0)
          (ensures sum_where f p == sum_upto g q bn)
  = sum_upto_tile f p g q k0 bn;
    if k0 + bn <= n then sum_upto_stop f p (k0 + bn) n else ()

(* ------------------------------------------------------------------ *)
(* The denominator half of the invariant, on its own.                  *)
(* ------------------------------------------------------------------ *)

(* The kernel updates [l] in the softmax pass and [o] in the P@V pass, so the
   two halves of [flash_inv] are established at different program points. *)
let dinv (#n : nat) (x : natlt n -> GTot real) (p : pred n) (m l : real) : prop
  = l == dsum x p /. exp m

let dstep (#n : nat) (x : natlt n -> GTot real) (p t p' : pred n) (m m' l : real)
  : Lemma (requires disjoint p t /\
                    (forall (j : natlt n). p' j == (p j || t j)) /\
                    dinv x p m l)
          (ensures dinv x p' m' (rescale l m m' +. tsum_d x t m'))
  = lem_rescale l m m' (dsum x p);
    lem_tsum_d x t m';
    dsum_split x p t;
    dsum_ext x p' (por p t)

let dcombine (#n : nat) (x : natlt n -> GTot real) (p1 p2 p' : pred n)
             (m1 m2 gm l1 l2 : real)
  : Lemma (requires disjoint p1 p2 /\
                    (forall (j : natlt n). p' j == (p1 j || p2 j)) /\
                    dinv x p1 m1 l1 /\ dinv x p2 m2 l2)
          (ensures dinv x p' gm (rescale l1 m1 gm +. rescale l2 m2 gm))
  = lem_rescale l1 m1 gm (dsum x p1);
    lem_rescale l2 m2 gm (dsum x p2);
    dsum_split x p1 p2;
    dsum_ext x p' (por p1 p2)

(* ------------------------------------------------------------------ *)
(* The numerator half of the invariant, on its own.                    *)
(* ------------------------------------------------------------------ *)

let ninv (#n : nat) (x y : natlt n -> GTot real) (p : pred n) (m o : real) : prop
  = o == nsum x y p /. exp m

let nstep (#n : nat) (x y : natlt n -> GTot real) (p t p' : pred n) (m m' o : real)
  : Lemma (requires disjoint p t /\
                    (forall (j : natlt n). p' j == (p j || t j)) /\
                    ninv x y p m o)
          (ensures ninv x y p' m' (rescale o m m' +. tsum_n x y t m'))
  = lem_rescale o m m' (nsum x y p);
    lem_tsum_n x y t m';
    nsum_split x y p t;
    nsum_ext x y p' (por p t)

let ncombine (#n : nat) (x y : natlt n -> GTot real) (p1 p2 p' : pred n)
             (m1 m2 gm o1 o2 : real)
  : Lemma (requires disjoint p1 p2 /\
                    (forall (j : natlt n). p' j == (p1 j || p2 j)) /\
                    ninv x y p1 m1 o1 /\ ninv x y p2 m2 o2)
          (ensures ninv x y p' gm (rescale o1 m1 gm +. rescale o2 m2 gm))
  = lem_rescale o1 m1 gm (nsum x y p1);
    lem_rescale o2 m2 gm (nsum x y p2);
    nsum_split x y p1 p2;
    nsum_ext x y p' (por p1 p2)

(* ------------------------------------------------------------------ *)
(* Dropping a mask that the summand already implements.                *)
(* ------------------------------------------------------------------ *)

(* The kernel's [P@V] matmul sums over every slot of the key tile; the slots it
   rejects carry a literal zero probability, so the unmasked fold agrees with
   the masked one. *)
let rec sum_upto_drop (#n : nat) (f : natlt n -> GTot real) (p : pred n) (k : natle n)
  : Lemma (requires forall (j : natlt n). ~(p j) ==> f j == 0.0R)
          (ensures sum_upto f ptrue k == sum_upto f p k)
          (decreases k)
  = if k = 0 then () else sum_upto_drop f p (k - 1)

let sum_where_drop (#n : nat) (f : natlt n -> GTot real) (p : pred n)
  : Lemma (requires forall (j : natlt n). ~(p j) ==> f j == 0.0R)
          (ensures sum_where f ptrue == sum_where f p)
  = sum_upto_drop f p n

(* ------------------------------------------------------------------ *)
(* Splitting the shifted sums.                                         *)
(*                                                                     *)
(* The cross-warp combine adds partial sums that are already shifted by *)
(* the block-wide maximum, so it needs the additive structure of        *)
(* [tsum_d]/[tsum_n] rather than that of [dsum]/[nsum].                 *)
(* ------------------------------------------------------------------ *)

let tsum_d_split (#n : nat) (x : natlt n -> GTot real) (p q : pred n) (m' : real)
  : Lemma (requires disjoint p q)
          (ensures tsum_d x (por p q) m' == tsum_d x p m' +. tsum_d x q m')
  = sum_where_split (fun j -> exp (x j -. m')) p q

let tsum_n_split (#n : nat) (x y : natlt n -> GTot real) (p q : pred n) (m' : real)
  : Lemma (requires disjoint p q)
          (ensures tsum_n x y (por p q) m' == tsum_n x y p m' +. tsum_n x y q m')
  = sum_where_split (fun j -> exp (x j -. m') *. y j) p q

let tsum_d_none (#n : nat) (x : natlt n -> GTot real) (p : pred n) (m' : real)
  : Lemma (requires forall (j : natlt n). ~(p j))
          (ensures tsum_d x p m' == 0.0R)
  = sum_upto_false (fun j -> exp (x j -. m')) p n

let tsum_n_none (#n : nat) (x y : natlt n -> GTot real) (p : pred n) (m' : real)
  : Lemma (requires forall (j : natlt n). ~(p j))
          (ensures tsum_n x y p m' == 0.0R)
  = sum_upto_false (fun j -> exp (x j -. m') *. y j) p n

let tsum_d_ext (#n : nat) (x : natlt n -> GTot real) (p q : pred n) (m' : real)
  : Lemma (requires forall (j : natlt n). p j == q j)
          (ensures tsum_d x p m' == tsum_d x q m')
  = sum_where_ext (fun j -> exp (x j -. m')) (fun j -> exp (x j -. m')) p q

let tsum_n_ext (#n : nat) (x y : natlt n -> GTot real) (p q : pred n) (m' : real)
  : Lemma (requires forall (j : natlt n). p j == q j)
          (ensures tsum_n x y p m' == tsum_n x y q m')
  = sum_where_ext (fun j -> exp (x j -. m') *. y j)
                  (fun j -> exp (x j -. m') *. y j) p q

(* The spec's output cell as the ratio of the two flash accumulators, with the
   denominator in the form the kernel's registers approximate. *)
let masked_out_cell
  (#sk #dv #rows : pos)
  (valid : FS.valid_pred sk)
  (scores : chest1 real sk)
  (probs : chest2 real rows sk)
  (mV : chest2 real sk dv)
  (i : natlt rows) (c : natlt dv)
  : Lemma (requires forall (j : natlt sk).
                      acc2 probs i j == acc1 (FS.masked_softmax_real valid scores) j)
          (ensures dsum (acc1 scores) valid >. 0.0R /\
                   MS.matmul_single probs mV i c
                   == nsum (acc1 scores) (fun (j : natlt sk) -> acc2 mV j c) valid
                      /. dsum (acc1 scores) valid)
  = dsum_pos (acc1 scores) valid (valid_witness valid);
    masked_matmul_cell valid scores probs mV i c;
    dsum_is_masked_denom valid scores

(* Both sums also transport along a pointwise-equal score / value function. *)
let dsum_ext2 (#n : nat) (x1 x2 : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires forall (j : natlt n). p j == q j /\ x1 j == x2 j)
          (ensures dsum x1 p == dsum x2 q)
  = sum_where_ext (fun j -> exp (x1 j)) (fun j -> exp (x2 j)) p q

let nsum_ext2 (#n : nat) (x1 x2 y1 y2 : natlt n -> GTot real) (p q : pred n)
  : Lemma (requires forall (j : natlt n).
                      p j == q j /\ x1 j == x2 j /\ y1 j == y2 j)
          (ensures nsum x1 y1 p == nsum x2 y2 q)
  = sum_where_ext (fun j -> exp (x1 j) *. y1 j) (fun j -> exp (x2 j) *. y2 j) p q
