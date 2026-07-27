module Kuiops.HReducePoly.Exact

(* Exact reduction over the innermost dimension of a tensor. Each outer
   (batch) index is reduced by one CUDA block. *)

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Functions
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Bijection { ( =~ ) }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier

unfold
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

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

inline_for_extraction noextract
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
  (#et_i #et:Type0)
  (#rank : erased nat)
  (#d : shape rank)
  (cd : cshape d)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (cols : szp)
  (nth : szp { SZ.v nth <= SZ.v cols /\ SZ.fits (SZ.v cols + SZ.v nth) })
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

  let x0 = tensor_read a (conc_snoc cd batch (lo <: szlt cols));
  (**)up_conc_snoc cd batch (lo <: szlt cols);
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
    let xv = tensor_read a (conc_snoc cd batch (vidx <: szlt cols));
    (**)up_conc_snoc cd batch (vidx <: szlt cols);
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
(* Reduction invariant carried through the tree reduction: the first   *)
(* cell of shmem slice [i,j) holds the reduction of partials [i,j).    *)
(* ------------------------------------------------------------------ *)

unfold
let array1_pts_to_slice_red_inner
  (#et:Type0) (f : et -> et -> et)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j : nat{i < j /\ j <= sz})
  (parts : lseq et sz)
  (s : lseq et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((s @! 0) == rfold1 f (Seq.slice parts i j))

let array1_pts_to_slice_red
  (#et:Type0) (f : et -> et -> et)
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  (parts : lseq et sz)
  : slprop
  = exists* s. array1_pts_to_slice_red_inner f r i j parts s

let barrier_matrix
  (#et:Type0) (f : et -> et -> et)
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
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
  (#l : layout1 nth)
  (r : array1 et l)
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
  (#l : layout1 nth) {| Kuiper.Tensor.ctlayout l |}
  (r : array1 et l)
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
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (output : tensor et_o lout)
  (va : chest (snoc_shape d cols) et_i)
  (vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = a |-> Frac ((1 /. rows) /. nth) va **
    if_ (op_Equality #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)) **
    exists* (v : et).
      tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

unfold
let kpost
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (output : tensor et_o lout)
  (va : chest (snoc_shape d cols) et_i)
  (vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = a |-> Frac ((1 /. rows) /. nth) va **
    if_ (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid)
    )

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn kf
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin)
  (output : tensor et_o lout)
  (va : chest (snoc_shape d cols) et_i)
  (vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre f pre_map post_map rows cols nth a output va vout shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix f nth
      (from_array (l1_forward nth) shmem._1)
      (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid)))) **
    B.barrier_state 0
  ensures
    gpu **
    kpost f pre_map post_map rows cols nth a output va vout shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix f nth
      (from_array (l1_forward nth) shmem._1)
      (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid)))) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre f pre_map post_map rows cols nth a output va vout shmem bid tid;
  let (gsa, _) = shmem;

  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  let batch = cunflatten cd bid;
  let parts : erased (lseq et nth) =
    partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid));

  let psum : et = fold_block cd f pre_map cols nth a batch tid;
  assert pure (up batch == unflatten d (SZ.v bid));
  tensor_write_cell sa (cidx1 tid) psum;

  let mut n : szlt 32 = 0sz;

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> tensor_pts_to_cell sa ((x <: natlt nth), ()) (seq![psum] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![psum];

  (**)Seq.init_ghost_index (SZ.v nth)
  (**)  (fun (i:nat{i < SZ.v nth}) -> rfold1 f
  (**)    (block (SZ.v cols) (SZ.v nth)
  (**)      (lseq_map pre_map (inner_seq va (unflatten d (SZ.v bid)))) i));
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

  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0) (array1_pts_to_slice_red f sa 0 nth parts);
    if_elim_true' (op_Equality #nat tid 0)
      (Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid)));
    unfold array1_pts_to_slice_red f sa 0 nth parts;
    unfold array1_pts_to_slice_red_inner f sa 0 nth parts;
    (**)partials_reduces f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid));
    (**)assert pure (Seq.equal (Seq.slice parts 0 (SZ.v nth)) parts);
    let red = array1_read_from_slice sa 0sz;
    rewrite
      Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid))
    as
      Cell output (up batch) |-> acc vout (unflatten d (SZ.v bid));
    tensor_write_cell output batch (post_map red);
    assert pure (post_map red == reduced f pre_map post_map va (unflatten d (SZ.v bid)));
    rewrite
      Cell output (up batch) |-> post_map red
    as
      Cell output (unflatten d (SZ.v bid)) |->
        reduced f pre_map post_map va (unflatten d (SZ.v bid));
    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let css : erased (chest1 et nth) = hide (seq_to_chest1 (reveal ss));
    forevery_refine_ext'
      #nat
      #(fun (k:nat) -> 0 <= k /\ k < nth)
      (fun (k:nat) -> k < nth)
      _;
    forevery_ext
      (fun (k:natlt nth) ->
        tensor_pts_to_cell sa ((k <: natlt nth), ()) (ss @! (k - 0)))
      (fun (k:natlt nth) ->
        tensor_pts_to_cell sa (abs_bij.gg k) (acc (reveal css) (abs_bij.gg k)));
    forevery_iso_back (abs_bij #nth)
      (fun (i : abs (nth @| INil)) -> tensor_pts_to_cell sa i (acc (reveal css) i));
    tensor_implode sa;
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d (SZ.v bid)) |->
        reduced f pre_map post_map va (unflatten d (SZ.v bid))
    )
  } else {
    if_elim_false' (op_Equality #nat tid 0) (array1_pts_to_slice_red f sa 0 nth parts);
    if_elim_false' (op_Equality #nat tid 0)
      (Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid)));
    if_intro_false' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d (SZ.v bid)) |->
        reduced f pre_map post_map va (unflatten d (SZ.v bid))
    );
    ();
  };
}
#pop-options

ghost
fn block_setup
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (a |-> Frac (1 /. rows) va **
     Cell output (unflatten d bid) |-> acc vout (unflatten d bid))
  ensures
    (forall+ (i : natlt nth).
      kpre f pre_map post_map rows cols nth a output va vout shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  tensor_share_n a (SZ.v nth);

  forevery_if_intro #(natlt nth) 0 (fun _ ->
    Cell output (unflatten d bid) |-> acc vout (unflatten d bid));
  forevery_ext
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)))
    (fun tid -> if_ (op_Equality #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)));

  forevery_zip (fun _ -> a |-> Frac ((1 /. rows) /. nth) va) _;

  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
  (fun tid ->
    a |-> Frac ((1 /. rows) /. nth) va **
    if_ (op_Equality #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)))
  _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (a |-> Frac ((1 /. rows) /. nth) va **
       if_ (op_Equality #nat tid 0)
         (Cell output (unflatten d bid) |-> acc vout (unflatten d bid))) **
      tensor_pts_to_cell (from_array (l1_forward nth) gsa)
        (abs_bij.gg (tid <: natlt nth))
        (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) ->
      kpre f pre_map post_map rows cols nth a output va vout shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}


ghost
fn block_teardown
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth).
      kpost f pre_map post_map rows cols nth a output va vout shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (a |-> Frac (1 /. rows) va **
     Cell output (unflatten d bid) |->
       reduced f pre_map post_map va (unflatten d bid))
{
  forevery_unzip _ _;

  tensor_gather_n a (SZ.v nth);

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        Cell output (unflatten d bid) |->
          reduced f pre_map post_map va (unflatten d bid)))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        Cell output (unflatten d bid) |->
          reduced f pre_map post_map va (unflatten d bid)));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid)
  );

  tensor_concr (from_array (l1_forward nth) shmem._1);
  rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

ghost
fn setup
  (#et_i #et_o : Type0)
  (#rank : nat) (#d : shape rank)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vout : chest d et_o)
  ()
  norewrite
  requires
    a |-> va ** output |-> vout
  ensures
    (forall+ (bid : natlt rows).
      a |-> Frac (1 /. rows) va **
      Cell output (unflatten d bid) |-> acc vout (unflatten d bid)) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n a (SZ.v rows);
  tensor_explode output;
  forevery_iso (flatten_bij d)
    (fun i -> Cell output i |-> acc vout i);
  forevery_rw_size (sizeof d) (SZ.v rows);
  forevery_zip
    (fun (_ : natlt rows) -> a |-> Frac (1 /. rows) va)
    (fun (bid : natlt rows) ->
      Cell output (unflatten d bid) |-> acc vout (unflatten d bid));
}

ghost
fn teardown
  (#et_i #et #et_o : Type0)
  (#rank : nat) (#d : shape rank)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  ()
  norewrite
  requires
    (forall+ (bid : natlt rows).
      a |-> Frac (1 /. rows) va **
      Cell output (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid)) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    a |-> va **
    output |-> mk d (fun i -> reduced f pre_map post_map va i)
{
  forevery_unzip _ _;
  tensor_gather_n a (SZ.v rows);
  forevery_rw_size (SZ.v rows) (sizeof d);
  forevery_iso_back (flatten_bij d)
    (fun i -> Cell output i |-> reduced f pre_map post_map va i);
  let result = mk d (fun i -> reduced f pre_map post_map va i);
  forevery_map
    (fun i -> Cell output i |-> reduced f pre_map post_map va i)
    (fun i -> Cell output i |-> acc result i)
    fn _ { () };
  tensor_implode output;
}

inline_for_extraction noextract
let kernel
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin { is_global a })
  (output : tensor et_o lout { is_global output })
  (#va : chest (snoc_shape d cols) et_i)
  (#vout : chest d et_o)
  : kernel_desc
      (a |-> va ** output |-> vout)
      (a |-> va ** output |-> mk d (fun i -> reduced f pre_map post_map va i))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et f nth
        (from_array _ shmem._1)
        (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d bid))));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix f nth #(l1_forward nth)
        (from_array _ shmem._1)
        (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d bid))));

    f = kf cd f pre_map post_map rows cols nth a output va vout;

    block_pre  = (fun bid ->
      a |-> Frac (1 /. rows) va **
      Cell (output <: tensor et_o lout) (unflatten d bid) |->
        acc vout (unflatten d bid));
    block_post = (fun bid ->
      a |-> Frac (1 /. rows) va **
      Cell (output <: tensor et_o lout) (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid));
    setup      = setup rows cols a #va output #vout;
    teardown   = teardown f pre_map post_map rows cols a #va output;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup
      f pre_map post_map rows cols nth a #va output #vout;
    block_teardown = block_teardown
      f pre_map post_map rows cols nth a #va output #vout;

    kpre = kpre f pre_map post_map rows cols nth a output va vout;
    kpost = kpost f pre_map post_map rows cols nth a output va vout;
    frame = pure (SZ.fits (tlayout_ulen lout));

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

inline_for_extraction noextract
fn reduce
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin { is_global a })
  (output : tensor et_o lout { is_global output })
  (#va : chest (snoc_shape d cols) et_i)
  (#vout : chest d et_o)
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    on gpu_loc (output |-> vout)
  ensures
    on gpu_loc (output |-> mk d (fun i -> reduced f pre_map post_map va i))
{
  launch_sync (kernel cd f pre_map post_map rows cols nth a output);
  ()
}
