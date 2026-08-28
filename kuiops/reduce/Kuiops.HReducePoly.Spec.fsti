module Kuiops.HReducePoly.Spec

(* Shared machinery for the innermost-dimension reductions: the pure
   [rfold1] specification, the balanced contiguous partition of a row over
   the threads of a block, and the shared-memory tree reduction, which is
   generic in the per-range predicate [p] relating a shared-memory cell to
   the range of per-thread partials it summarizes. [Exact] instantiates [p]
   with equality; [Approx] with approximation over the reals. *)

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Functions
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg
open Kuiper.Bijection { ( =~ ) }

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier

(* Boolean reflection of equality, replacing the removed compatibility name
   from older F★ releases. *)
let bool_eq (#a : eqtype) (x y : a) : GTot bool = x = y

unfold
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

(* ------------------------------------------------------------------ *)
(* Appending a trailing (reduced) dimension to a shape and its         *)
(* indices.                                                            *)
(* ------------------------------------------------------------------ *)

let rec snoc_shape (#r : nat) (d : shape r) (n : nat) : GTot (shape (r + 1)) =
  match d with
  | INil -> n @| INil
  | ICons h t -> h @| snoc_shape t n

let rec abs_snoc (#r : nat) (#d : shape r) (#n : nat)
  (i : abs d) (j : natlt n) : GTot (abs (snoc_shape d n)) =
  match d with
  | INil -> (j, ())
  | ICons h t ->
    let ih, it = i <: natlt h & abs t in
    (ih, abs_snoc it j)

(* The batch (non-reduced) dimensions, one CUDA block each. [csizeof] turns a
   [cshape] satisfying this into the block count, so the kernel entry points
   take no separate [rows] argument. *)
let batches_ok (#r : nat) (d : shape r) : prop =
  sizeof d > 0 /\ SZ.fits (sizeof d) /\ sizeof d <= SZ.v max_blocks

inline_for_extraction noextract
val conc_snoc (#r : erased nat) (#d : shape r) (#n : erased nat)
  (cd : cshape d) (i : conc d) (j : szlt n) : conc (snoc_shape d n)

val up_conc_snoc (#r : nat) (#d : shape r) (#n : nat)
  (cd : cshape d) (i : conc d) (j : szlt n)
  : Lemma (up (conc_snoc cd i j) == abs_snoc (up i) (SZ.v j))

val inner_seq (#et_i : Type0) (#r : nat) (#d : shape r) (#n : nat)
  (v : chest (snoc_shape d n) et_i) (i : abs d) : GTot (lseq et_i n)

val inner_seq_index (#et_i : Type0) (#r : nat) (#d : shape r) (#n : nat)
  (v : chest (snoc_shape d n) et_i) (i : abs d) (j : natlt n)
  : Lemma (inner_seq v i @! j == acc v (abs_snoc i j))
          [SMTPat (inner_seq v i @! j)]

(* ------------------------------------------------------------------ *)
(* Pure specification: non-empty left-to-right reduction ([foldl1]).   *)
(* ------------------------------------------------------------------ *)

let rfold1 (#et : Type0) (f : et -> et -> et)
  (s : seq et { Seq.length s > 0 }) : GTot et =
  seq_fold_left f (s @! 0) (Seq.slice s 1 (Seq.length s))

let reduced
  (#et_i #et #et_o : Type0)
  (#r : nat) (#d : shape r) (#n : pos)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (v : chest (snoc_shape d n) et_i)
  (i : abs d)
  : GTot et_o =
  post_map (rfold1 f (lseq_map pre_map (inner_seq v i)))

val seq_fold_left_append (#a #b:Type) (f: b -> a -> b) (acc:b) (s1 s2 : seq a)
  : Lemma (ensures seq_fold_left f acc (s1 @+ s2)
                   == seq_fold_left f (seq_fold_left f acc s1) s2)

val fold_left_reduce (#et:Type0) (f: et -> et -> et) (acc:et)
  (s:seq et{Seq.length s > 0})
  : Lemma (requires is_associative f)
          (ensures seq_fold_left f acc s == f acc (rfold1 f s))

val rfold1_singleton (#et:Type0) (f: et -> et -> et) (x:et)
  : Lemma (rfold1 f (Seq.create 1 x) == x)

(* The key fact: for an associative [f], the reduction of a concatenation is
   the reduction of the parts, combined. This is what lets adjacent contiguous
   ranges be merged in the tree reduction. *)
val rfold1_append (#et:Type0) (f: et -> et -> et)
  (s1 s2 : seq et { Seq.length s1 > 0 /\ Seq.length s2 > 0 })
  : Lemma (requires is_associative f)
          (ensures rfold1 f (s1 @+ s2) == f (rfold1 f s1) (rfold1 f s2))

(* Extending a left fold by one element needs no algebraic assumption. *)
val rfold1_snoc (#et:Type0) (f: et -> et -> et)
  (s : seq et { Seq.length s > 0 }) (x : et)
  : Lemma (rfold1 f (Seq.snoc s x) == f (rfold1 f s) x)

(* ------------------------------------------------------------------ *)
(* Balanced contiguous partition of [0, lena) into [nth] non-empty     *)
(* blocks. Block [tid] is [[bnd tid, bnd (tid+1))]. The first          *)
(* [lena % nth] blocks get one extra element; every block has at least *)
(* [lena / nth >= 1] elements (because [nth <= lena]).                 *)
(* ------------------------------------------------------------------ *)

let bnd (lena : nat) (nth : pos) (tid : nat) : nat =
  tid * (lena / nth) + (if tid <= lena % nth then tid else lena % nth)

val bnd_zero (lena : nat) (nth : pos)
  : Lemma (bnd lena nth 0 == 0)

val bnd_full (lena nth : pos)
  : Lemma (requires nth <= lena) (ensures bnd lena nth nth == lena)

val bnd_mono (lena nth : pos) (tid : nat { tid < nth })
  : Lemma (requires nth <= lena)
          (ensures bnd lena nth tid < bnd lena nth (tid + 1))

val bnd_le (lena nth : pos) (tid : nat { tid <= nth })
  : Lemma (requires nth <= lena) (ensures bnd lena nth tid <= lena)

val bnd_pos (lena nth : pos) (tid : nat { 0 < tid /\ tid <= nth })
  : Lemma (requires nth <= lena) (ensures bnd lena nth tid > 0)

(* The (non-empty) contiguous block owned by thread [tid]. *)
let block (#et:Type0) (lena nth : pos { nth <= lena })
  (input : seq et { Seq.length input == lena }) (tid : nat { tid < nth })
  : GTot (s:seq et { Seq.length s > 0 })
  = bnd_mono lena nth tid;
    bnd_le lena nth (tid + 1);
    Seq.slice input (bnd lena nth tid) (bnd lena nth (tid + 1))

(* The (non-empty) prefix [[0, bnd k)] covered by threads [0, k). *)
let iprefix (#et:Type0) (lena nth : pos { nth <= lena })
  (input : seq et { Seq.length input == lena }) (k : pos { k <= nth })
  : GTot (s:seq et { Seq.length s > 0 })
  = bnd_pos lena nth k;
    bnd_le lena nth k;
    Seq.slice input 0 (bnd lena nth k)

(* [init_ghost (k+1) g] is [init_ghost k g] with [g k] appended. *)
val init_ghost_snoc (#a:Type) (k:nat) (g : (i:nat{i < k+1} -> GTot a))
  : Lemma (Seq.init_ghost (k+1) g
           == Seq.snoc (Seq.init_ghost k (fun (i:nat{i<k}) -> g i)) (g k))

val blocks_fold
  (#et:Type0) (f : et -> et -> et)
  (lena nth : pos { nth <= lena }) (input : seq et { Seq.length input == lena })
  : Lemma (requires is_associative f)
          (ensures
            rfold1 f (Seq.init_ghost nth
                       (fun (tid:nat{tid<nth}) -> rfold1 f (block lena nth input tid)))
            == rfold1 f input)

(* The per-thread partial results, as reduced by [fold_block]. *)
let partials
  (#et_i #et : Type0)
  (#rank : nat) (#d : shape rank)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (cols nth : pos { nth <= cols })
  (va : chest (snoc_shape d cols) et_i)
  (batch : abs d)
  : GTot (lseq et nth)
  = Seq.init_ghost nth (fun (tid:nat{tid<nth}) ->
      rfold1 f (block cols nth (lseq_map pre_map (inner_seq va batch)) tid))

val partials_reduces
  (#et_i #et : Type0)
  (#rank : nat) (#d : shape rank)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (cols nth : pos { nth <= cols })
  (va : chest (snoc_shape d cols) et_i)
  (batch : abs d)
  : Lemma (requires is_associative f)
          (ensures rfold1 f (partials f pre_map cols nth va batch)
                   == rfold1 f (lseq_map pre_map (inner_seq va batch)))

(* Number of barrier calls in the reduction loop (identical to HReduce). *)
let hreduce_barrier_count (nth : pos) : GTot nat = log2 (2 * nth - 1)

val log2_hreduce (nth:pos) (it:nat)
  : Lemma (requires pow2 it >= nth /\ (it == 0 \/ pow2 (it - 1) < nth))
          (ensures it == log2 (2 * nth - 1))

inline_for_extraction noextract
let smin (a b : sz) : sz = if SZ.(a <=^ b) then a else b

(* ------------------------------------------------------------------ *)
(* Per-thread reduction of a contiguous block, seeded with its first   *)
(* element (no identity needed since the block is non-empty).          *)
(* ------------------------------------------------------------------ *)

inline_for_extraction noextract
fn fold_block
  (#et_i #et:Type0)
  (#rank : erased nat)
  (#d : shape rank)
  (cd : cshape d)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (cols : szp)
  (nth : szp { SZ.v nth <= SZ.v cols /\ SZ.fits (SZ.v cols + SZ.v nth) })
  (index : conc d -> szlt cols -> conc (snoc_shape d cols))
  (index_up : (i:conc d -> j:szlt cols ->
    Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
  (#l : tlayout (snoc_shape d cols)) {| ctlayout l |}
  (a : tensor et_i l)
  (batch : conc d)
  (tid : szlt nth)
  (#va : chest (snoc_shape d cols) et_i)
  (#fr : perm)
  preserves
    gpu ** a |-> Frac fr va
  returns
    res : et
  ensures
    pure (res == rfold1 f (block (SZ.v cols) (SZ.v nth)
      (lseq_map pre_map (inner_seq va (up batch))) (SZ.v tid)))

(* ------------------------------------------------------------------ *)
(* Shared-memory slice ownership (semantics-agnostic; identical to the *)
(* helpers in [Kuiper.Kernel.HReduce]).                                *)
(* ------------------------------------------------------------------ *)

(* Plain ownership of a slice of a rank-1 tensor. *)
let array1_pts_to_slice
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey]i
   [@@@mkey]j : nat{i <= j /\ j <= sz})
  (s : lseq et (j - i))
  : slprop
  = forall+ (k : nat{i <= k /\ k < j}).
      tensor_pts_to_cell r ((k <: natlt sz), ()) (s @! (k - i))

ghost
fn array1_slice_concat
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j k : nat{i <= j /\ j <= k /\ k <= sz})
  (#s1 : lseq et (j - i))
  (#s2 : lseq et (k - j))
  requires
    array1_pts_to_slice r i j s1 **
    array1_pts_to_slice r j k s2
  ensures
    array1_pts_to_slice r i k (s1 @+ s2)

inline_for_extraction noextract
fn array1_read_from_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : layout1 len) {| ctlayout l |}
  (r : array1 et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (lseq et (j - i)))
  preserves
    array1_pts_to_slice r i j s
  returns
    v : et
  ensures
    pure (v == s @! (idx - i))

inline_for_extraction noextract
fn array1_write_to_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : layout1 len) {| ctlayout l |}
  (r : array1 et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (lseq et (j - i)))
  (v : et)
  requires
    array1_pts_to_slice r i j s
  ensures
    array1_pts_to_slice r i j (Seq.upd s (idx - i) v)

(* ------------------------------------------------------------------ *)
(* Reduction invariant carried through the tree reduction: the first   *)
(* cell of shmem slice [i,j) summarizes the partials [i,j), as         *)
(* witnessed by the caller-chosen predicate [p].                       *)
(* ------------------------------------------------------------------ *)

(* [p i j x] must be preserved by combining adjacent ranges with [f]. *)
let red_comb_ty (#et : Type0) (f : et -> et -> et) (p : nat -> nat -> et -> prop) =
  i:nat -> j:nat -> k:nat -> x:et -> y:et ->
    Lemma (requires i < j /\ j < k /\ p i j x /\ p j k y)
          (ensures p i k (f x y))

unfold
let array1_pts_to_slice_red_inner
  (#et:Type0) (p : nat -> nat -> et -> prop)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j : nat{i < j /\ j <= sz})
  (s : lseq et (j - i))
  : slprop
  = array1_pts_to_slice r i j s ** pure (p i j (s @! 0))

let array1_pts_to_slice_red
  (#et:Type0) (p : nat -> nat -> et -> prop)
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  : slprop
  = exists* s. array1_pts_to_slice_red_inner p r i j s

let barrier_matrix
  (#et:Type0) (p : nat -> nat -> et -> prop)
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
  (it : nat)
  (from to : natlt nth)
: slprop
=
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (array1_pts_to_slice_red p r from (min (from + pow2 it) nth)))

(* One level of the shared-memory tree reduction. *)
inline_for_extraction noextract
fn iteration
  (#et:Type0) (f : et -> et -> et) (p : nat -> nat -> et -> prop)
  (p_comb : red_comb_ty f p)
  (nth : szp { SZ.v nth <= max_threads })
  (#l : layout1 nth) {| Kuiper.Tensor.ctlayout l |}
  (r : array1 et l)
  (tid : szlt nth)
  (it: szlt 31)
  preserves gpu
  preserves thread_id nth tid
  preserves mbarrier_tok nth (barrier_matrix p nth r)
  requires B.barrier_state it
  requires if_ (div_pow2 it tid) (array1_pts_to_slice_red p r tid (min (tid + pow2 it) nth))
  ensures  B.barrier_state (it + 1)
  ensures  if_ (div_pow2 (it+1) tid) (array1_pts_to_slice_red p r tid (min (tid + pow2 (it + 1)) nth))
