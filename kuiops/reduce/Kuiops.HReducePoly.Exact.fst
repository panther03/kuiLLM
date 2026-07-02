module Kuiops.HReducePoly.Exact

(* Exact reduction that is polymorphic in the (associative) reduction operator.

   This is an exact analogue of [Kuiper.Kernel.HReduce.reduce]: rather than
   summing with the scalar [add] up to a real-valued approximation, it folds an
   arbitrary [reduce : et -> et -> et] that is required only to be associative.
   The result is the *exact* left-to-right reduction of the (pre-mapped) input,
   so it can implement operators such as [any], [all], [max], ...

   Because a general associative operator has no inverse (unlike real [+.]), the
   input is partitioned into *contiguous* per-thread blocks rather than the
   strided blocks used by [HReduce]: combining adjacent contiguous ranges in
   thread order preserves the sequence order and hence needs associativity only.
   To avoid needing an identity element, the input must be non-empty and there
   must be no more threads than elements ([nth <= lena]), so every block is
   non-empty. *)

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.Functions
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier
module Array1 = Kuiper.Array1

(* ------------------------------------------------------------------ *)
(* Pure specification: non-empty left-to-right reduction ([foldl1]).   *)
(* ------------------------------------------------------------------ *)

(* Plain fold-left over an append (no algebraic assumption). *)
let rec seq_fold_left_append (#a #b:Type) (f: b -> a -> b) (acc:b) (s1 s2 : seq a)
  : Lemma (ensures seq_fold_left f acc (s1 @+ s2)
                   == seq_fold_left f (seq_fold_left f acc s1) s2)
          (decreases Seq.length s1)
  = match view_seq s1 with
    | SNil ->
      assert (Seq.equal (s1 @+ s2) s2)
    | SCons hd tl ->
      assert (Seq.equal (s1 @+ s2) (Seq.cons hd (tl @+ s2)));
      seq_fold_left_append f (f acc hd) tl s2

(* For an associative [f], seeding a fold with [acc] is the same as reducing the
   sequence on its own and combining once with [acc] on the left. *)
let rec fold_left_reduce (#et:Type0) (f: et -> et -> et) (acc:et)
  (s:seq et{Seq.length s > 0})
  : Lemma (requires is_associative f)
          (ensures seq_fold_left f acc s == f acc (rfold1 f s))
          (decreases Seq.length s)
  = match view_seq s with
    | SCons hd tl ->
      if Seq.length tl = 0 then begin
        assert (Seq.equal (Seq.slice s 1 (Seq.length s)) tl);
        assert (rfold1 f s == hd)
      end else begin
        assert (Seq.equal (Seq.slice s 1 (Seq.length s)) tl);
        assert (rfold1 f s == seq_fold_left f hd tl);
        fold_left_reduce f (f acc hd) tl;
        fold_left_reduce f hd tl;
        assert (seq_fold_left f acc s == seq_fold_left f (f acc hd) tl)
      end

let rfold1_singleton (#et:Type0) (f: et -> et -> et) (x:et)
  : Lemma (rfold1 f (Seq.create 1 x) == x)
  = let s = Seq.create 1 x in
    assert (Seq.equal (Seq.slice s 1 (Seq.length s)) (Seq.empty #et))

(* The key fact: for an associative [f], the reduction of a concatenation is the
   reduction of the parts, combined. This is what lets adjacent contiguous
   ranges be merged in the tree reduction. *)
let rfold1_append (#et:Type0) (f: et -> et -> et)
  (s1 s2 : seq et { Seq.length s1 > 0 /\ Seq.length s2 > 0 })
  : Lemma (requires is_associative f)
          (ensures rfold1 f (s1 @+ s2) == f (rfold1 f s1) (rfold1 f s2))
  = let s = s1 @+ s2 in
    assert (s @! 0 == s1 @! 0);
    assert (Seq.equal (Seq.slice s 1 (Seq.length s))
                      (Seq.slice s1 1 (Seq.length s1) @+ s2));
    seq_fold_left_append f (s1 @! 0) (Seq.slice s1 1 (Seq.length s1)) s2;
    assert (seq_fold_left f (s1 @! 0) (Seq.slice s1 1 (Seq.length s1)) == rfold1 f s1);
    fold_left_reduce f (rfold1 f s1) s2

let rfold1_snoc (#et:Type0) (f: et -> et -> et)
  (s : seq et { Seq.length s > 0 }) (x : et)
  : Lemma (requires is_associative f)
          (ensures rfold1 f (Seq.snoc s x) == f (rfold1 f s) x)
  = rfold1_singleton f x;
    assert (Seq.equal (Seq.snoc s x) (s @+ Seq.create 1 x));
    rfold1_append f s (Seq.create 1 x)

(* ------------------------------------------------------------------ *)
(* Balanced contiguous partition of [0, lena) into [nth] non-empty     *)
(* blocks. Block [tid] is [[bnd tid, bnd (tid+1))]. The first          *)
(* [lena % nth] blocks get one extra element; every block has at least *)
(* [lena / nth >= 1] elements (because [nth <= lena]).                 *)
(* ------------------------------------------------------------------ *)

module ML = FStar.Math.Lemmas

let bnd (lena : nat) (nth : pos) (tid : nat) : nat =
  tid * (lena / nth) + (if tid <= lena % nth then tid else lena % nth)

let bnd_zero (lena : nat) (nth : pos)
  : Lemma (bnd lena nth 0 == 0) = ()

let div_ge_one (lena nth : pos)
  : Lemma (requires nth <= lena) (ensures lena / nth >= 1)
  = ML.lemma_div_le nth lena nth

let bnd_full (lena nth : pos)
  : Lemma (requires nth <= lena) (ensures bnd lena nth nth == lena)
  = ML.euclidean_division_definition lena nth;
    ML.lemma_mod_lt lena nth

let bnd_mono (lena nth : pos) (tid : nat { tid < nth })
  : Lemma (requires nth <= lena)
          (ensures bnd lena nth tid < bnd lena nth (tid + 1))
  = div_ge_one lena nth;
    ML.lemma_mod_lt lena nth;
    ML.distributivity_add_left tid 1 (lena / nth)

let bnd_le (lena nth : pos) (tid : nat { tid <= nth })
  : Lemma (requires nth <= lena) (ensures bnd lena nth tid <= lena)
  = bnd_full lena nth;
    ML.lemma_mult_le_right (lena / nth) tid nth;
    ML.lemma_mod_lt lena nth

let bnd_pos (lena nth : pos) (tid : nat { 0 < tid /\ tid <= nth })
  : Lemma (requires nth <= lena) (ensures bnd lena nth tid > 0)
  = bnd_zero lena nth;
    bnd_mono lena nth 0

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
let init_ghost_snoc (#a:Type) (k:nat) (g : (i:nat{i < k+1} -> GTot a))
  : Lemma (Seq.init_ghost (k+1) g
           == Seq.snoc (Seq.init_ghost k (fun (i:nat{i<k}) -> g i)) (g k))
  = let lhs = Seq.init_ghost (k+1) g in
    let rhs = Seq.snoc (Seq.init_ghost k (fun (i:nat{i<k}) -> g i)) (g k) in
    Seq.init_ghost_index (k+1) g;
    Seq.init_ghost_index k (fun (i:nat{i<k}) -> g i);
    assert (Seq.equal lhs rhs)

(* Prefix version: reducing the first [k] block-partials equals reducing the
   corresponding prefix [[0, bnd k)] of the input. *)
let rec blocks_fold_prefix
  (#et:Type0) (f : et -> et -> et)
  (lena nth : pos { nth <= lena }) (input : seq et { Seq.length input == lena })
  (k : pos { k <= nth })
  : Lemma (requires is_associative f)
          (ensures
            rfold1 f (Seq.init_ghost k
                       (fun (tid:nat{tid<k}) -> rfold1 f (block lena nth input tid)))
            == rfold1 f (iprefix lena nth input k))
          (decreases k)
  = if k = 1 then begin
      bnd_zero lena nth;
      let g0 (tid:nat{tid<1}) : GTot et = rfold1 f (block lena nth input tid) in
      assert (Seq.equal (Seq.init_ghost 1 g0) (Seq.create 1 (g0 0)));
      rfold1_singleton f (g0 0)
    end else begin
      let g  (tid:nat{tid < k})   : GTot et = rfold1 f (block lena nth input tid) in
      let g' (tid:nat{tid < k-1}) : GTot et = rfold1 f (block lena nth input tid) in
      init_ghost_snoc (k-1) g;
      let pre  = Seq.init_ghost (k-1) g' in
      let last = Seq.create 1 (g (k-1)) in
      assert (Seq.equal (Seq.init_ghost k g) (pre @+ last));
      blocks_fold_prefix f lena nth input (k-1);
      rfold1_singleton f (g (k-1));
      rfold1_append f pre last;
      (* [iprefix (k-1)] and [block (k-1)] are adjacent slices whose
         concatenation is [iprefix k]. *)
      bnd_mono lena nth (k-1);
      bnd_le lena nth k;
      lem_append_slice input 0 (bnd lena nth (k-1)) (bnd lena nth k);
      rfold1_append f (iprefix lena nth input (k-1)) (block lena nth input (k-1))
    end

let blocks_fold
  (#et:Type0) (f : et -> et -> et)
  (lena nth : pos { nth <= lena }) (input : seq et { Seq.length input == lena })
  : Lemma (requires is_associative f)
          (ensures
            rfold1 f (Seq.init_ghost nth
                       (fun (tid:nat{tid<nth}) -> rfold1 f (block lena nth input tid)))
            == rfold1 f input)
  = blocks_fold_prefix f lena nth input nth;
    bnd_full lena nth;
    assert (Seq.equal (Seq.slice input 0 lena) input)

(* ------------------------------------------------------------------ *)
(* Per-thread reduction of a contiguous block, seeded with its first   *)
(* element (no identity needed since the block is non-empty).          *)
(* ------------------------------------------------------------------ *)

#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
inline_for_extraction noextract
let smin (a b : sz) : sz = if SZ.(a <=^ b) then a else b

inline_for_extraction noextract
fn fold_block
  (#et:Type0) {| sized et |}
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et -> et)
  (lena : sz)
  (nth : szp { SZ.v nth <= SZ.v lena /\ SZ.fits (SZ.v lena + SZ.v nth) })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (tid : szlt nth)
  (#va : erased (lseq et lena))
  (#fr : perm)
  preserves
    gpu ** a |-> Frac fr va
  returns
    res : et
  ensures
    pure (res == rfold1 f (block (SZ.v lena) (SZ.v nth) (lseq_map pre_map va) (SZ.v tid)))
{
  let q : sz = SZ.(lena /^ nth);
  let r : sz = SZ.(lena %^ nth);
  (**)assert pure (SZ.v q == SZ.v lena / SZ.v nth);
  (**)assert pure (SZ.v r == SZ.v lena % SZ.v nth);
  (**)ML.euclidean_division_definition (SZ.v lena) (SZ.v nth);
  (**)ML.lemma_mult_le_right (SZ.v q) (SZ.v tid) (SZ.v nth);
  (**)ML.lemma_mult_le_right (SZ.v q) (SZ.v tid + 1) (SZ.v nth);

  let tid1 : sz = SZ.(tid +^ 1sz);
  let mt  : sz = smin tid  r;
  let mt1 : sz = smin tid1 r;

  let lo : sz = SZ.(tid  *^ q +^ mt);
  let hi : sz = SZ.(tid1 *^ q +^ mt1);

  (**)bnd_mono  (SZ.v lena) (SZ.v nth) (SZ.v tid);
  (**)bnd_le    (SZ.v lena) (SZ.v nth) (SZ.v tid + 1);
  (**)assert pure (SZ.v lo == bnd (SZ.v lena) (SZ.v nth) (SZ.v tid));
  (**)assert pure (SZ.v hi == bnd (SZ.v lena) (SZ.v nth) (SZ.v tid + 1));
  (**)assert pure (SZ.v lo < SZ.v hi /\ SZ.v hi <= SZ.v lena);

  let x0 = Array1.read a lo;
  let mut acc : et = pre_map x0;
  let mut idx : sz = SZ.(lo +^ 1sz);

  (**)assert pure (Seq.equal (Seq.slice (lseq_map pre_map va) (SZ.v lo) (SZ.v lo + 1))
  (**)                       (Seq.create 1 (Seq.index (lseq_map pre_map va) (SZ.v lo))));
  (**)rfold1_singleton f (Seq.index (lseq_map pre_map va) (SZ.v lo));

  while (SZ.(!idx <^ hi))
    invariant
      gpu ** a |-> Frac fr va **
      live acc ** live idx **
      pure (SZ.v lo < SZ.v !idx /\ SZ.v !idx <= SZ.v hi /\
            !acc == rfold1 f (Seq.slice (lseq_map pre_map va) (SZ.v lo) (SZ.v !idx)))
    decreases (SZ.v hi - SZ.v !idx)
  {
    let xv = Array1.read a !idx;
    let v = pre_map xv;
    (**)assert pure (v == Seq.index (lseq_map pre_map va) (SZ.v !idx));
    (**)assert pure (Seq.equal (Seq.slice (lseq_map pre_map va) (SZ.v lo) (SZ.v !idx + 1))
    (**)                       (Seq.snoc (Seq.slice (lseq_map pre_map va) (SZ.v lo) (SZ.v !idx)) v));
    (**)rfold1_snoc f (Seq.slice (lseq_map pre_map va) (SZ.v lo) (SZ.v !idx)) v;
    acc := f !acc v;
    idx := SZ.(!idx +^ 1sz);
  };

  (**)assert pure (SZ.v !idx == SZ.v hi);
  (**)assert pure (Seq.equal (Seq.slice (lseq_map pre_map va) (SZ.v lo) (SZ.v hi))
  (**)                       (block (SZ.v lena) (SZ.v nth) (lseq_map pre_map va) (SZ.v tid)));
  !acc
}
#pop-options

(* ------------------------------------------------------------------ *)
(* Shared-memory slice ownership (semantics-agnostic; identical to the *)
(* helpers in [Kuiper.Kernel.HReduce]).                                *)
(* ------------------------------------------------------------------ *)

(* Plain ownership of a slice of an Array1. *)
let array1_pts_to_slice
  (#et : Type0)
  (#sz : nat)
  (#l : Array1.layout sz)
  ([@@@mkey] r : Array1.t et l)
  ([@@@mkey]i
   [@@@mkey]j : nat{i <= j /\ j <= sz})
  (s : lseq et (j - i))
  : slprop
  = forall+ (k : nat{i <= k /\ k < j}).
      Cell r (k <: natlt sz) |-> (s @! (k - i))

#push-options "--z3rlimit 80"
ghost
fn array1_slice_concat
  (#et : Type0)
  (#sz : nat)
  (#l : Array1.layout sz)
  (r : Array1.t et l)
  (i j k : nat{i <= j /\ j <= k /\ k <= sz})
  (#s1 : lseq et (j - i))
  (#s2 : lseq et (k - j))
  requires
    array1_pts_to_slice r i j s1 **
    array1_pts_to_slice r j k s2
  ensures
    array1_pts_to_slice r i k (s1 @+ s2)
{
  unfold array1_pts_to_slice r i j s1;
  unfold array1_pts_to_slice r j k s2;

  let s = s1 @+ s2;

  forevery_ext
    (fun (x:nat{i <= x /\ x < j}) -> Cell r (x <: natlt sz) |-> (s1 @! (x - i)))
    (fun (x:nat{i <= x /\ x < j}) -> Cell r (x <: natlt sz) |-> (s @! (x - i)));
  forevery_ext
    (fun (x:nat{j <= x /\ x < k}) -> Cell r (x <: natlt sz) |-> (s2 @! (x - j)))
    (fun (x:nat{j <= x /\ x < k}) -> Cell r (x <: natlt sz) |-> (s @! (x - i)));

  forevery_refine_join' #nat
    (fun (x:nat) -> i <= x /\ x < j)
    (fun (x:nat) -> j <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      Cell r (x <: natlt sz) |-> (s @! (x - i)));

  forevery_refine_ext' #nat
    #(fun (x:nat) -> (i <= x /\ x < j) \/ (j <= x /\ x < k))
    (fun (x:nat) -> i <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      Cell r (x <: natlt sz) |-> (s @! (x - i)));

  fold array1_pts_to_slice r i k s;
}
#pop-options

inline_for_extraction noextract
fn array1_read_from_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : Array1.layout len) {| ctlayout l |}
  (r : Array1.t et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (lseq et (j - i)))
  preserves
    array1_pts_to_slice r i j s
  returns
    v : et
  ensures
    pure (v == s @! (idx - i))
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  let v = Array1.read_cell r idx;
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s;
  v
}

inline_for_extraction noextract
fn array1_write_to_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : Array1.layout len) {| ctlayout l |}
  (r : Array1.t et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : erased (lseq et (j - i)))
  (v : et)
  requires
    array1_pts_to_slice r i j s
  ensures
    array1_pts_to_slice r i j (Seq.upd s (idx - i) v)
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract' #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  Array1.write_cell r idx v;
  let s' : erased (lseq et (j - i)) = Seq.upd s (idx - i) v;
  Pulse.Lib.Forall.elim_forall
    (fun (x:nat{i <= x /\ x < j}) ->
      Cell r (x <: natlt len) |-> (s' @! (x - i)));
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s';
  rewrite each s' as Seq.upd s (idx - i) v;
  ()
}

(* ------------------------------------------------------------------ *)
(* Reduction invariant carried through the tree reduction: the first   *)
(* cell of shmem slice [i,j) holds the reduction of partials [i,j).    *)
(* ------------------------------------------------------------------ *)

unfold
let array1_pts_to_slice_red_inner
  (#et:Type0) (f : et -> et -> et)
  (#sz : nat)
  (#l : Array1.layout sz)
  (r : Array1.t et l)
  (i j : nat{i < j /\ j <= sz})
  (parts : lseq et sz)
  (s : lseq et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((s @! 0) == rfold1 f (Seq.slice parts i j))

let array1_pts_to_slice_red
  (#et:Type0) (f : et -> et -> et)
  (#sz : nat)
  (#l : Array1.layout sz)
  ([@@@mkey] r : Array1.t et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  (parts : lseq et sz)
  : slprop
  = exists* s. array1_pts_to_slice_red_inner f r i j parts s

let barrier_matrix
  (#et:Type0) (f : et -> et -> et)
  (nth : szp)
  (#l : Array1.layout nth)
  (r : Array1.t et l)
  (parts : lseq et nth)
  (it : nat)
  (from to : natlt nth)
: slprop
=
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (array1_pts_to_slice_red f r from (min (from + pow2 it) nth) parts))

ghost
fn mk_barrier_pre
  (#et:Type0) (f : et -> et -> et)
  (nth : szp)
  (#l : Array1.layout nth)
  (r : Array1.t et l)
  (parts : lseq et nth)
  (tid : natlt nth)
  (it: natlt 31)
  requires
    if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (array1_pts_to_slice_red f r tid (min (tid + pow2 it) nth) parts)
  ensures
    forall+ (i:natlt nth). barrier_matrix f nth r parts it tid i
{
  open FStar.SizeT;
  if (tid >= pow2 it) {
    forevery_if_intro #(natlt nth) (tid - pow2 it) (fun i ->
      if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
        (array1_pts_to_slice_red f r tid (min (tid + pow2 it) nth) parts));
    forevery_ext
      (fun (i:natlt nth) ->
        if_ (op_Equality #(natlt nth) i (tid - pow2 it))
          (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
            (array1_pts_to_slice_red f r tid (min (tid + pow2 it) nth) parts)))
      (fun (i:natlt nth) -> barrier_matrix f nth r parts it tid i);
  } else {
    assert pure (pow2 it > tid);
    assert pure (tid % pow2 it == tid);
    if_elim_false _;
    forevery_emp_intro (natlt nth);
    forevery_ext
      (fun (i:natlt nth) -> emp)
      (fun (i:natlt nth) -> barrier_matrix f nth r parts it tid i);
  }
}

inline_for_extraction noextract
fn iteration
  (#et:Type0) (f : (et -> et -> et){ is_associative f })
  (nth : szp { SZ.v nth <= max_threads })
  (#l : Array1.layout nth) {| Kuiper.Tensor.ctlayout l |}
  (r : Array1.t et l)
  (parts : erased (lseq et nth))
  (tid : szlt nth)
  (it: szlt 31)
  preserves gpu
  preserves thread_id nth tid
  preserves mbarrier_tok nth (barrier_matrix f nth r parts)
  requires B.barrier_state it
  requires if_ (div_pow2 it tid) (array1_pts_to_slice_red f r tid (min (tid + pow2 it) nth) parts)
  ensures  B.barrier_state (it + 1)
  ensures  if_ (div_pow2 (it+1) tid) (array1_pts_to_slice_red f r tid (min (tid + pow2 (it + 1)) nth) parts)
{
  case_split (div_pow2 (it + 1) tid)
    (if_ (div_pow2 it tid) (array1_pts_to_slice_red f r tid (min (tid + pow2 it) nth) parts));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (array1_pts_to_slice_red f r tid (min (tid + pow2 it) nth) parts))
      as (if_ (div_pow2 (it + 1) tid)
            (array1_pts_to_slice_red f r tid (min (tid + pow2 it) nth) parts));

  mk_barrier_pre f nth r parts tid it;
  fold RPM.row (barrier_matrix f nth r parts) it tid;
  mbarrier_wait ();
  unfold RPM.col (barrier_matrix f nth r parts) it tid;

  let nextid = FStar.SizeT.(tid +^ spow2 it);

  let end_ : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (nextid <^ nth) {
    forevery_ext
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_red f r from (min (from + pow2 it) nth) parts)))
      (fun (from: natlt nth) ->
        if_ (op_Equality #(natlt nth) from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (array1_pts_to_slice_red f r from (min (from + pow2 it) nth) parts)));
    forevery_if_elim #(natlt nth)
      (tid + pow2 it)
      (fun (from: natlt nth) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
         (array1_pts_to_slice_red f r from (min (from + pow2 it) nth) parts));

    let b = sdiv_pow2 (it +^ 1sz) tid;

    rewrite each (div_pow2 (it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (array1_pts_to_slice_red f r nextid (min (tid + pow2 it + pow2 it) nth) parts)
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      (**)unfold (array1_pts_to_slice_red f r nextid end_ parts);
      (**)unfold (array1_pts_to_slice_red f r tid nextid parts);
      (**)array1_slice_concat #et #nth r tid nextid end_;

      let s1 = array1_read_from_slice r tid;
      (**)assert (pure (s1 == rfold1 f (Seq.slice parts tid nextid)));

      let s2 = array1_read_from_slice r nextid;
      (**)assert (pure (s2 == rfold1 f (Seq.slice parts nextid end_)));

      let s = f s1 s2;
      (**)lem_append_slice parts tid nextid end_;
      (**)rfold1_append f (Seq.slice parts tid nextid) (Seq.slice parts nextid end_);
      (**)assert (pure (s == rfold1 f (Seq.slice parts tid end_)));

      array1_write_to_slice r tid s;

      (**)with seq. assert (array1_pts_to_slice r tid end_ seq);
      (**)fold (array1_pts_to_slice_red f r tid end_ parts);
      (**)if_intro_true (array1_pts_to_slice_red f r tid end_ parts);
      (**)rewrite
      (**)  if_ true
      (**)      (array1_pts_to_slice_red f r (SZ.v tid) (reveal end_) parts)
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)      (array1_pts_to_slice_red f r (SZ.v tid) (reveal end_) parts);
    } else {
      if_elim_false _;
      if_intro_false (array1_pts_to_slice_red f r tid end_ parts);
    }
  } else {
    forevery_map
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_red f r from (min (from + pow2 it) nth) parts)))
      (fun from -> emp)
      fn from {
        if_rewrite_bool (from = tid + pow2 it) false _;
        if_elim_false _;
      };
    forevery_emp_elim _;
  }
}

(* ------------------------------------------------------------------ *)
(* Kernel spec plumbing.                                               *)
(* ------------------------------------------------------------------ *)

(* Per-thread partial results: thread [tid]'s reduction of its block. *)
let partials
  (#et:Type0) (f : et -> et -> et) (pre_map : et -> et)
  (lena nth : pos { nth <= lena })
  (va : lseq et lena)
  : GTot (lseq et nth)
  = Seq.init_ghost nth (fun (tid:nat{tid<nth}) ->
      rfold1 f (block lena nth (lseq_map pre_map va) tid))

(* Reducing the per-thread partials equals reducing the whole (mapped) input. *)
let partials_reduces
  (#et:Type0) (f : et -> et -> et) (pre_map : et -> et)
  (lena nth : pos { nth <= lena }) (va : lseq et lena)
  : Lemma (requires is_associative f)
          (ensures rfold1 f (partials f pre_map lena nth va)
                   == rfold1 f (lseq_map pre_map va))
  = blocks_fold f lena nth (lseq_map pre_map va)

(* Number of barrier calls in the reduction loop (identical to HReduce). *)
let hreduce_barrier_count (nth : pos) : GTot nat = log2 (2 * nth - 1)

private let rec log2_range (n:pos) (k:nat)
  : Lemma (requires pow2 k <= n /\ n < pow2 (k+1))
          (ensures log2 n == k)
          (decreases k)
= if k = 0 then ()
  else begin
    FStar.Math.Lemmas.lemma_div_le (pow2 k) n 2;
    log2_range (n/2) (k-1)
  end

private let log2_hreduce (nth:pos) (it:nat)
  : Lemma (requires pow2 it >= nth /\ (it == 0 \/ pow2 (it - 1) < nth))
          (ensures it == log2 (2 * nth - 1))
= if it = 0 then ()
  else log2_range (2 * nth - 1) it

unfold
let kpre
  (#et:Type0) {| sized et |}
  (f : et -> et -> et)
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (va : lseq et lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equality #nat tid 0) (live out) **
    exists* (v : et). Cell (Array1.from_array (l1_forward nth) shmem._1) tid |-> v

unfold
let kpost
  (#et:Type0) {| sized et |}
  (f : et -> et -> et)
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (va : lseq et lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))
    )

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn kf
  (#et:Type0) {| sized et |}
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (va : erased (lseq et lena))
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt 1sz)
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre f pre_map nth lena a va out shmem bid tid **
    thread_id nth tid **
    block_id 1 bid **
    mbarrier_tok nth (barrier_matrix f nth (Array1.from_array (l1_forward nth) shmem._1) (partials f pre_map (SZ.v lena) (SZ.v nth) va)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost f pre_map nth lena a va out shmem bid tid **
    thread_id nth tid **
    block_id 1 bid **
    mbarrier_tok nth (barrier_matrix f nth (Array1.from_array (l1_forward nth) shmem._1) (partials f pre_map (SZ.v lena) (SZ.v nth) va)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  let (gsa, _) = shmem;

  let sa = Array1.from_array (l1_forward nth) gsa;
  rewrite each Array1.from_array (l1_forward nth) gsa as sa;

  let parts : erased (lseq et nth) = partials f pre_map (SZ.v lena) (SZ.v nth) va;

  (* Compute partial reduction and write to shmem *)
  let psum : et = fold_block f pre_map lena nth a tid;
  Array1.write_cell sa tid psum;

  (* Now do tree reduction on shmem *)
  let mut n : szlt 32 = 0sz;

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> Cell sa (x <: natlt nth) |-> (seq![psum] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![psum];

  (**)Seq.init_ghost_index (SZ.v nth)
  (**)  (fun (i:nat{i < SZ.v nth}) -> rfold1 f (block (SZ.v lena) (SZ.v nth) (lseq_map pre_map va) i));
  (**)rfold1_singleton f (Seq.index parts (SZ.v tid));
  (**)assert pure (Seq.equal (Seq.slice parts (SZ.v tid) (SZ.v tid + 1))
  (**)                       (Seq.create 1 (Seq.index parts (SZ.v tid))));

  (**)fold (array1_pts_to_slice_red f sa tid (tid + 1) parts);
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_red f sa tid (min (tid + pow2 !n) nth) parts);

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_red f sa tid (min (tid + pow2 !n) nth) parts) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration f nth sa parts tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_red f sa tid (min (tid + pow2 it) nth) parts))
  as
    (if_ (op_Equality #nat tid 0) (array1_pts_to_slice_red f sa 0 nth parts));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  (* Thread zero owns the result at the end, and writes it out. *)
  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_red f sa 0 nth parts);
    if_elim_true' (op_Equality #nat tid 0) (live out);
    unfold array1_pts_to_slice_red f sa 0 nth parts;
    unfold array1_pts_to_slice_red_inner f sa 0 nth parts;
    (**)partials_reduces f pre_map (SZ.v lena) (SZ.v nth) va;
    (**)assert pure (Seq.equal (Seq.slice parts 0 (SZ.v nth)) parts);
    gpu_write out (array1_read_from_slice sa 0sz);
    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let bij : Kuiper.Bijection.bijection (k:nat{0 <= k /\ k < nth}) (Array1.ait nth) =
      Kuiper.Bijection.Mkbijection
        #(k:nat{0 <= k /\ k < nth})
        #(Array1.ait nth)
        (fun k -> k)
        (fun k -> k);
    forevery_iso bij _;
    forevery_ext _ (fun (k : natlt nth) -> Cell sa k |-> (ss @! k));
    Array1.implode sa;
    rewrite each sa as Array1.from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))
    )
  } else {
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_red f sa 0 nth parts);
    if_elim_false' (op_Equality #nat tid 0) (live out);
    if_intro_false' (op_Equality #nat tid 0) (
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))
    );
    ();
  };
}
#pop-options

ghost
fn block_setup
  (#et:Type0) {| sized et |}
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#va : lseq et lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (a |-> va ** live out)
  ensures
    (forall+ (i : natlt nth). kpre f pre_map nth lena a va out shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  Array1.share_n a nth;

  forevery_if_intro #(natlt nth) 0 (fun _ -> live out);
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0) (live out))
    (fun tid -> if_ (op_Equality #nat tid 0) (live out));

  forevery_zip (fun _ -> a |-> Frac (1 /. nth) va) _;

  Array1.raise' (l1_forward nth) gsa;
  Array1.explode (Array1.from_array (l1_forward nth) gsa);

  forevery_zip #(natlt nth)
    (fun tid -> a |-> Frac (1 /. nth) va ** if_ (op_Equality #nat tid 0) (live out))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (a |-> Frac (1 /. nth) va **
       if_ (op_Equality #nat tid 0) (live out)) **
      Cell (Array1.from_array (l1_forward nth) gsa) tid |-> (Array1.from_seq (l1_forward nth) vgsa @! tid)
    )
    (fun (tid : natlt nth) -> kpre f pre_map nth lena a va out shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}


ghost
fn block_teardown
  (#et:Type0) {| sized et |}
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#va : lseq et lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost f pre_map nth lena a va out shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (a |-> va ** (exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))))
{
  forevery_unzip _ _;

  Array1.gather_n a nth;

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (Array1.from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (Array1.from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))
  );

  Array1.lower (Array1.from_array (l1_forward nth) shmem._1);
  rewrite each Array1.core (Array1.from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

ghost
fn setup
  (#et:Type0) {| sized et |}
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  ()
  norewrite
  requires
    a |-> va ** live out
  ensures
    (forall+ (bid : natlt 1). a |-> va ** live out) **
    emp
{
  forevery_singleton_intro #(natlt 1) (fun _bid -> a |-> va ** live out);
}

ghost
fn teardown
  (#et:Type0) {| sized et |}
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). a |-> va ** exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va))) **
    emp
  ensures
    a |-> va ** (exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va)))
{
  forevery_singleton_elim #(natlt 1) _;
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| sized et |}
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  (#_ : squash (Seq.length va == SZ.v lena))
  (out : gpu_ref et)
  : kernel_desc
      (a |-> va ** live out)
      (a |-> va ** exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va)))
  = {
    nblk = 1sz;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun _bid shmem ->
      mbarrier_contract (barrier_matrix #et f nth (Array1.from_array _ shmem._1) (partials f pre_map (SZ.v lena) (SZ.v nth) va)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun _bid shmem ->
      mbarrier_transform (barrier_matrix f nth #(l1_forward nth) (Array1.from_array _ shmem._1) (partials f pre_map (SZ.v lena) (SZ.v nth) va)));

    f = kf f pre_map nth lena a va out;

    block_pre  = (fun bid -> a |-> va ** live out);
    block_post = (fun bid -> a |-> va ** exists* (v : et). out |-> v ** pure (v == rfold1 f (lseq_map pre_map va)));
    setup      = setup    nth lena a #va out;
    teardown   = teardown f pre_map nth lena a #va out;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup    f pre_map nth lena a #va out;
    block_teardown = block_teardown f pre_map nth lena a #va out;

    kpre =  kpre  f pre_map nth lena a va out;
    kpost = kpost f pre_map nth lena a va out;
    frame = emp;

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

inline_for_extraction noextract
fn reduce
  (#et:Type0) {| sized et |}
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et -> et)
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#va : erased (lseq et lena))
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    emp
  returns
    res : et
  ensures
    pure (res == rfold1 f (lseq_map pre_map va))
{
  let out = Kuiper.Ref.gpu_alloc0 #et ();
  launch_sync (kernel f pre_map nth lena a out);

  let mut hout : et = default #et;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  !hout;
}
