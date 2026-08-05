module Kuiops.HReducePoly.Spec

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
module ML = FStar.Math.Lemmas

let rec conc_snoc (#r : erased nat) (#d : shape r) (#n : erased nat)
  (cd : cshape d) (i : conc d) (j : szlt n) : conc (snoc_shape d n) =
  match cd with
  | CNil -> (j, ())
  | CCons ch ct ->
    let ih, it = i <: szlt (SZ.v ch) & conc _ in
    (ih, conc_snoc ct it j)

let rec up_conc_snoc (#r : nat) (#d : shape r) (#n : nat)
  (cd : cshape d) (i : conc d) (j : szlt n)
  : Lemma (up (conc_snoc cd i j) == abs_snoc (up i) (SZ.v j))
  = match cd with
    | CNil -> ()
    | CCons ch ct ->
      let _, it = i <: szlt (SZ.v ch) & conc _ in
      up_conc_snoc ct it j

let inner_seq (#et_i : Type0) (#r : nat) (#d : shape r) (#n : nat)
  (v : chest (snoc_shape d n) et_i) (i : abs d) : GTot (lseq et_i n) =
  Seq.init_ghost n (fun j -> acc v (abs_snoc i j))

let inner_seq_index (#et_i : Type0) (#r : nat) (#d : shape r) (#n : nat)
  (v : chest (snoc_shape d n) et_i) (i : abs d) (j : natlt n)
  : Lemma (inner_seq v i @! j == acc v (abs_snoc i j))
  = Seq.init_ghost_index n (fun (j:nat{j < n}) -> acc v (abs_snoc i j))

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
  : Lemma (rfold1 f (Seq.snoc s x) == f (rfold1 f s) x)
  = let s' = Seq.snoc s x in
    let t = Seq.slice s 1 (Seq.length s) in
    assert (s' @! 0 == s @! 0);
    assert (Seq.equal (Seq.slice s' 1 (Seq.length s')) (t @+ Seq.create 1 x));
    seq_fold_left_append f (s @! 0) t (Seq.create 1 x);
    assert (Seq.equal (Seq.slice (Seq.create 1 x) 1 1) (Seq.empty #et))

(* ------------------------------------------------------------------ *)
(* Balanced contiguous partition of [0, lena).                         *)
(* ------------------------------------------------------------------ *)

let bnd_zero (lena : nat) (nth : pos)
  : Lemma (bnd lena nth 0 == 0) = ()

private let div_ge_one (lena nth : pos)
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
private let rec blocks_fold_prefix
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

let partials_reduces
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
  = blocks_fold f cols nth (lseq_map pre_map (inner_seq va batch))

private let rec log2_range (n:pos) (k:nat)
  : Lemma (requires pow2 k <= n /\ n < pow2 (k+1))
          (ensures log2 n == k)
          (decreases k)
= if k = 0 then ()
  else begin
    FStar.Math.Lemmas.lemma_div_le (pow2 k) n 2;
    log2_range (n/2) (k-1)
  end

let log2_hreduce (nth:pos) (it:nat)
  : Lemma (requires pow2 it >= nth /\ (it == 0 \/ pow2 (it - 1) < nth))
          (ensures it == log2 (2 * nth - 1))
= if it = 0 then ()
  else log2_range (2 * nth - 1) it

(* ------------------------------------------------------------------ *)
(* Per-thread reduction of a contiguous block, seeded with its first   *)
(* element (no identity needed since the block is non-empty).          *)
(* ------------------------------------------------------------------ *)

#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
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
{
  let q : sz = SZ.(cols /^ nth);
  let r : sz = SZ.(cols %^ nth);
  (**)assert pure (SZ.v q == SZ.v cols / SZ.v nth);
  (**)assert pure (SZ.v r == SZ.v cols % SZ.v nth);
  (**)ML.euclidean_division_definition (SZ.v cols) (SZ.v nth);
  (**)ML.lemma_mult_le_right (SZ.v q) (SZ.v tid) (SZ.v nth);
  (**)ML.lemma_mult_le_right (SZ.v q) (SZ.v tid + 1) (SZ.v nth);

  let tid1 : sz = SZ.(tid +^ 1sz);
  let mt  : sz = smin tid  r;
  let mt1 : sz = smin tid1 r;

  let lo : sz = SZ.(tid  *^ q +^ mt);
  let hi : sz = SZ.(tid1 *^ q +^ mt1);

  (**)bnd_mono  (SZ.v cols) (SZ.v nth) (SZ.v tid);
  (**)bnd_le    (SZ.v cols) (SZ.v nth) (SZ.v tid + 1);
  (**)assert pure (SZ.v lo == bnd (SZ.v cols) (SZ.v nth) (SZ.v tid));
  (**)assert pure (SZ.v hi == bnd (SZ.v cols) (SZ.v nth) (SZ.v tid + 1));
  (**)assert pure (SZ.v lo < SZ.v hi /\ SZ.v hi <= SZ.v cols);

  let x0 = tensor_read a (index batch (lo <: szlt cols));
  (**)index_up batch (lo <: szlt cols);
  (**)Seq.init_ghost_index (SZ.v cols)
  (**)  (fun (j:nat{j < SZ.v cols}) -> Kuiper.Chest.acc va (abs_snoc (up batch) j));
  let mut acc : et = pre_map x0;
  let mut idx : sz = SZ.(lo +^ 1sz);

  (**)assert pure (Seq.equal
  (**)  (Seq.slice (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo) (SZ.v lo + 1))
  (**)  (Seq.create 1 (Seq.index (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo))));
  (**)rfold1_singleton f (Seq.index (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo));

  while (SZ.(!idx <^ hi))
    invariant
      gpu ** a |-> Frac fr va **
      live acc ** live idx **
      pure (SZ.v lo < SZ.v !idx /\ SZ.v !idx <= SZ.v hi /\
            !acc == rfold1 f
              (Seq.slice (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo) (SZ.v !idx)))
    decreases (SZ.v hi - SZ.v !idx)
  {
    assert pure (SZ.v !idx < SZ.v cols);
    let vidx = !idx;
    let xv = tensor_read a (index batch (vidx <: szlt cols));
    (**)index_up batch (vidx <: szlt cols);
    (**)Seq.init_ghost_index (SZ.v cols)
    (**)  (fun (j:nat{j < SZ.v cols}) -> Kuiper.Chest.acc va (abs_snoc (up batch) j));
    let v = pre_map xv;
    (**)assert pure (v == Seq.index (lseq_map pre_map (inner_seq va (up batch))) (SZ.v !idx));
    (**)assert pure (Seq.equal
    (**)  (Seq.slice (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo) (SZ.v !idx + 1))
    (**)  (Seq.snoc (Seq.slice (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo) (SZ.v !idx)) v));
    (**)rfold1_snoc f
    (**)  (Seq.slice (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo) (SZ.v !idx)) v;
    acc := f !acc v;
    idx := SZ.(!idx +^ 1sz);
  };

  (**)assert pure (SZ.v !idx == SZ.v hi);
  (**)assert pure (Seq.equal
  (**)  (Seq.slice (lseq_map pre_map (inner_seq va (up batch))) (SZ.v lo) (SZ.v hi))
  (**)  (block (SZ.v cols) (SZ.v nth)
  (**)    (lseq_map pre_map (inner_seq va (up batch))) (SZ.v tid)));
  !acc
}
#pop-options

(* ------------------------------------------------------------------ *)
(* Shared-memory slice ownership.                                      *)
(* ------------------------------------------------------------------ *)

#push-options "--z3rlimit 80"
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
{
  unfold array1_pts_to_slice r i j s1;
  unfold array1_pts_to_slice r j k s2;

  let s = s1 @+ s2;

  forevery_ext
    (fun (x:nat{i <= x /\ x < j}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s1 @! (x - i)))
    (fun (x:nat{i <= x /\ x < j}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));
  forevery_ext
    (fun (x:nat{j <= x /\ x < k}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s2 @! (x - j)))
    (fun (x:nat{j <= x /\ x < k}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

  forevery_refine_join' #nat
    (fun (x:nat) -> i <= x /\ x < j)
    (fun (x:nat) -> j <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

  forevery_refine_ext' #nat
    #(fun (x:nat) -> (i <= x /\ x < j) \/ (j <= x /\ x < k))
    (fun (x:nat) -> i <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

  forevery_ext
    _
    (fun (x : nat{i <= x /\ x < k}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (s @! (x - i)));

  fold array1_pts_to_slice r i k s;
}
#pop-options

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
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  let v = tensor_read_cell r (cidx1 idx);
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s;
  v
}

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
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract' #(x:nat{i <= x /\ x < j}) (SZ.v idx) _;
  tensor_write_cell r (cidx1 idx) v;
  let s' : erased (lseq et (j - i)) = Seq.upd s (idx - i) v;
  Pulse.Lib.Forall.elim_forall
    (fun (x:nat{i <= x /\ x < j}) ->
      tensor_pts_to_cell r ((x <: natlt len), ()) (s' @! (x - i)));
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s';
  rewrite each s' as Seq.upd s (idx - i) v;
  ()
}

(* ------------------------------------------------------------------ *)
(* The shared-memory tree reduction.                                   *)
(* ------------------------------------------------------------------ *)

ghost
fn mk_barrier_pre
  (#et:Type0) (p : nat -> nat -> et -> prop)
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
  (tid : natlt nth)
  (it: natlt 31)
  requires
    if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (array1_pts_to_slice_red p r tid (min (tid + pow2 it) nth))
  ensures
    forall+ (i:natlt nth). barrier_matrix p nth r it tid i
{
  open FStar.SizeT;
  if (tid >= pow2 it) {
    forevery_if_intro #(natlt nth) (tid - pow2 it) (fun i ->
      if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
        (array1_pts_to_slice_red p r tid (min (tid + pow2 it) nth)));
    forevery_ext
      (fun (i:natlt nth) ->
        if_ (op_Equality #(natlt nth) i (tid - pow2 it))
          (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
            (array1_pts_to_slice_red p r tid (min (tid + pow2 it) nth))))
      (fun (i:natlt nth) -> barrier_matrix p nth r it tid i);
  } else {
    assert pure (pow2 it > tid);
    assert pure (tid % pow2 it == tid);
    if_elim_false _;
    forevery_emp_intro (natlt nth);
    forevery_ext
      (fun (i:natlt nth) -> emp)
      (fun (i:natlt nth) -> barrier_matrix p nth r it tid i);
  }
}

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
{
  case_split (div_pow2 (it + 1) tid)
    (if_ (div_pow2 it tid) (array1_pts_to_slice_red p r tid (min (tid + pow2 it) nth)));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (array1_pts_to_slice_red p r tid (min (tid + pow2 it) nth)))
      as (if_ (div_pow2 (it + 1) tid)
            (array1_pts_to_slice_red p r tid (min (tid + pow2 it) nth)));

  mk_barrier_pre p nth r tid it;
  fold RPM.row (barrier_matrix p nth r) it tid;
  mbarrier_wait ();
  unfold RPM.col (barrier_matrix p nth r) it tid;

  let nextid = FStar.SizeT.(tid +^ spow2 it);

  let end_ : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (nextid <^ nth) {
    forevery_ext
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_red p r from (min (from + pow2 it) nth))))
      (fun (from: natlt nth) ->
        if_ (op_Equality #(natlt nth) from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (array1_pts_to_slice_red p r from (min (from + pow2 it) nth))));
    forevery_if_elim #(natlt nth)
      (tid + pow2 it)
      (fun (from: natlt nth) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
         (array1_pts_to_slice_red p r from (min (from + pow2 it) nth)));

    let b = sdiv_pow2 (it +^ 1sz) tid;

    rewrite each (div_pow2 (it + 1) (SZ.v tid)) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (array1_pts_to_slice_red p r nextid (min (tid + pow2 it + pow2 it) nth))
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) (SZ.v tid)));
      if_elim_true _;

      (**)unfold (array1_pts_to_slice_red p r nextid end_);
      (**)unfold (array1_pts_to_slice_red p r tid nextid);
      (**)array1_slice_concat #et #nth r tid nextid end_;

      let s1 = array1_read_from_slice r tid;
      (**)assert (pure (p tid nextid s1));

      let s2 = array1_read_from_slice r nextid;
      (**)assert (pure (p nextid end_ s2));

      let s = f s1 s2;
      (**)p_comb tid nextid end_ s1 s2;
      (**)assert (pure (p tid end_ s));

      array1_write_to_slice r tid s;

      (**)with seq. assert (array1_pts_to_slice r tid end_ seq);
      (**)fold (array1_pts_to_slice_red p r tid end_);
      (**)if_intro_true (array1_pts_to_slice_red p r tid end_);
      (**)rewrite
      (**)  if_ true
      (**)      (array1_pts_to_slice_red p r (SZ.v tid) (reveal end_))
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) (SZ.v tid))
      (**)      (array1_pts_to_slice_red p r (SZ.v tid) (reveal end_));
    } else {
      if_elim_false _;
      if_intro_false (array1_pts_to_slice_red p r tid end_);
    }
  } else {
    forevery_map
      (fun (from: natlt nth) ->
        if_ (op_Equality #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_red p r from (min (from + pow2 it) nth))))
      (fun from -> emp)
      fn from {
        if_rewrite_bool (from = tid + pow2 it) false _;
        if_elim_false _;
      };
    forevery_emp_elim _;
  }
}
