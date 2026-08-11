module Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas

(* The mathematical core of split-K.

   Split [z] of [splits] covers the contiguous k-range [z*ks, (z+1)*ks) and
   produces the partial product [split_partial ks ra rb z].  Summing the
   partials left to right reproduces the full k reduction exactly, so the
   split is invisible in the top-level specification. *)

open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling

module MS = Kuiper.Spec.GEMM
module BM = Kuiops.Common.BlockMatmul
module ML = FStar.Math.Lemmas

#push-options "--fuel 1 --ifuel 0 --z3rlimit 10"

(* The partial product contributed by split [z]: the full width of [ra] rows
   against the full height of [rb] columns, restricted to k in [z*ks,(z+1)*ks). *)
let split_partial
  (#rows #columns #shared : pos)
  (ks : pos { ks /? shared })
  (ra : chest2 real rows shared) (rb : chest2 real shared columns)
  (z : natlt (shared / ks))
  : chest2 real rows columns
= MS.matmul (ematrix_subtile ra rows ks 0 z) (ematrix_subtile rb ks columns z 0)

(* The left-associated sum of the first [t] partials. *)
let split_sum
  (#rows #columns #shared : pos)
  (ks : pos { ks /? shared })
  (ra : chest2 real rows shared) (rb : chest2 real shared columns)
  (t : nat { t <= shared / ks })
  : GTot (chest2 real rows columns)
= MS.__matmul_single_tile rows columns ks ra rb 0 0 t

let split_sum_zero
  (#rows #columns #shared : pos)
  (ks : pos { ks /? shared })
  (ra : chest2 real rows shared) (rb : chest2 real shared columns)
  : Lemma (split_sum ks ra rb 0 == const (rows @| columns @| INil) 0.0R)
= MS.matmul_single_tile_zero_lemma rows columns ks ra rb 0 0

let split_sum_step
  (#rows #columns #shared : pos)
  (ks : pos { ks /? shared })
  (ra : chest2 real rows shared) (rb : chest2 real shared columns)
  (t : pos { t <= shared / ks })
  : Lemma (split_sum ks ra rb t
           == MS.matplus (split_sum ks ra rb (t - 1)) (split_partial ks ra rb (t - 1)))
= ML.lemma_mult_le_right (shared / ks) 1 ks;
  MS.matmul_single_tile_lemma rows columns ks ra rb 0 0 t

(* Summing every split reproduces the full k reduction. *)
let split_sum_full
  (#rows #columns #shared : pos)
  (ks : pos { ks /? shared })
  (ra : chest2 real rows shared) (rb : chest2 real shared columns)
  : Lemma (split_sum ks ra rb (shared / ks) == MS.matmul ra rb)
= BM.matmul_tile_chain_full ks ra rb

(* The same statement in the form the launcher sees it: [splits] splits of
   [ks] k-elements each. *)
let splitk_decomposition
  (#rows #columns : pos) (splits ks : pos)
  (ra : chest2 real rows (splits * ks)) (rb : chest2 real (splits * ks) columns)
  : Lemma
      (requires ks /? (splits * ks))
      (ensures (ML.cancel_mul_div splits ks;
                split_sum #rows #columns #(splits * ks) ks ra rb splits
                == MS.matmul ra rb))
= ML.cancel_mul_div splits ks;
  split_sum_full #rows #columns #(splits * ks) ks ra rb

#pop-options

(* ---------------------------------------------------------------------- *)
(* Cellwise view: the reduce kernel adds the [splits] partials of one cell
   left to right in the accumulator type.  These lemmas relate that fold to
   [split_sum]. *)

#push-options "--fuel 2 --ifuel 0 --z3rlimit 10"

let rec sum_upto (#et : Type) {| scalar et |} (f : nat -> et) (t : nat)
  : Tot et (decreases t)
= if t = 0 then zero else add (sum_upto f (t - 1)) (f (t - 1))

let rec rsum_upto (f : nat -> real) (t : nat) : Tot real (decreases t)
= if t = 0 then 0.0R else rsum_upto f (t - 1) +. f (t - 1)

let rec sum_upto_approx
  (#et : Type) {| scalar et |} {| real_like et |}
  (f : nat -> et) (g : nat -> real) (t : nat)
  : Lemma (requires forall (i : nat). i < t ==> f i %~ g i)
          (ensures sum_upto f t %~ rsum_upto g t)
          (decreases t)
= if t = 0 then (a0 #et)
  else begin
    sum_upto_approx f g (t - 1);
    a_add (sum_upto f (t - 1)) (f (t - 1)) (rsum_upto g (t - 1)) (g (t - 1))
  end

(* One cell of [split_sum] is the real fold of that cell of the partials. *)
let rec split_sum_cell
  (#rows #columns #shared : pos)
  (ks : pos { ks /? shared })
  (ra : chest2 real rows shared) (rb : chest2 real shared columns)
  (t : nat { t <= shared / ks })
  (i : natlt rows) (j : natlt columns)
  : Lemma
      (ensures acc2 (split_sum ks ra rb t) i j
               == rsum_upto (fun z -> if z < shared / ks
                                   then acc2 (split_partial ks ra rb z) i j
                                   else 0.0R) t)
      (decreases t)
= if t = 0 then split_sum_zero ks ra rb
  else begin
    split_sum_cell ks ra rb (t - 1) i j;
    split_sum_step ks ra rb t
  end

#pop-options
