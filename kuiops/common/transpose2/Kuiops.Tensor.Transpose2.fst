module Kuiops.Tensor.Transpose2

(* ==================================================================== *)
(* Rank-2 index-swap (transpose) layer for Kuiper layouts, chests and    *)
(* array2 views, plus a column-major variant of the Copy.Vec2 strided-   *)
(* chunk ownership partition, obtained by DELEGATION to the row-major    *)
(* partition applied to the transposed view.                             *)
(*                                                                       *)
(* !!! THIS MODULE IS DELIBERATELY INTERFACE-FREE. DO NOT ADD AN .fsti.   *)
(*                                                                       *)
(* Every client of this module depends on DEFINITIONAL TRANSPARENCY:      *)
(*                                                                       *)
(*   * [own_strided_chunks_cm] / [live_strided_chunks_cm] are usable only  *)
(*     because they unfold definitionally to the corresponding            *)
(*     [Kuiper.Kernel.GEMM.Copy.Vec2] predicate at the transposed view.   *)
(*     Behind a [val] the delegation stops typechecking entirely.        *)
(*                                                                       *)
(*   * [srm_of_scm] must expose its [offset] and [stride] fields, or      *)
(*     [lemma_aligned_srm_of_scm] becomes unprovable downstream.  This is *)
(*     the exact trap that forced [Kuiops.Array2.Strided.ColMajor] to     *)
(*     restate upstream's abstract [instance val strided_col_major_-      *)
(*     l2_col_major] with transparent fields.                             *)
(*                                                                       *)
(* Adding an interface here to "tidy up" before upstreaming WILL break    *)
(* the GEMM-TN staging path.  If you want abstraction, export explicit    *)
(* [val] lemmas instead -- do not hide the definitions.                   *)
(* ==================================================================== *)

#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Bijection
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Array2.Strided
open Pulse.Lib.Trade { (@==>) }

module SZ = Kuiper.SizeT
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.TensorRO { vtlayout_of_tlayout }

(* ------------------------------------------------------------------ *)
(* The rank-2 index-swap bijection.                                    *)
(* ------------------------------------------------------------------ *)

let transpose2 (rows cols : nat)
  : (abs (rows @| cols @| INil) =~ abs (cols @| rows @| INil))
= {
    ff = (fun (i, (j, ())) -> (j, (i, ())));
    gg = (fun (j, (i, ())) -> (i, (j, ())));
    ff_gg = (fun x ->
      let (j, (i, ())) = x in
      assert ((fun (i, (j, ())) -> (j, (i, ()))) ((fun (j, (i, ())) -> (i, (j, ()))) x))
             == (j, (i, ())));
    gg_ff = (fun x ->
      let (i, (j, ())) = x in
      assert ((fun (j, (i, ())) -> (i, (j, ()))) ((fun (i, (j, ())) -> (j, (i, ()))) x))
             == (i, (j, ())));
  }

(* ------------------------------------------------------------------ *)
(* Transposed layout.                                                  *)
(* ------------------------------------------------------------------ *)

let ltranspose (#rows #cols : erased nat) (l : layout2 rows cols)
  : layout2 cols rows
= tlayout_bij (transpose2 rows cols) l

let ltranspose_cell (#rows #cols : erased nat) (l : layout2 rows cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (cell_of_pos (ltranspose l) j i == cell_of_pos l i j)
= ()

(* ------------------------------------------------------------------ *)
(* Bridge: a column-major layout is row-major when transposed.         *)
(* ------------------------------------------------------------------ *)

inline_for_extraction noextract
let srm_of_scm (#rows #cols : erased nat) (#l : layout2 rows cols)
  (scm : strided_col_major (vtlayout_of_tlayout l))
  : strided_row_major (vtlayout_of_tlayout (ltranspose l))
= {
    offset = scm.offset;
    stride = scm.stride;
    pf = (fun (j : natlt cols) (i : natlt rows) ->
            ltranspose_cell l i j; scm.pf i j);
  }

(* Alignment carries over verbatim, since [srm_of_scm] copies [offset] and
   [stride] unchanged. *)
let lemma_aligned_srm_of_scm (#rows #cols : erased nat) (#l : layout2 rows cols)
  (scm : strided_col_major (vtlayout_of_tlayout l)) (n : pos)
  : Lemma (requires aligned_strided_col_major n scm)
          (ensures aligned_strided_row_major n (srm_of_scm scm))
= ()

(* Dual bridge: a row-major layout is column-major when transposed.  This is
   what lets a shared B tile staged as [bn x bk] row-major be handed to
   [mma_loadB_map_cm] as a [bk x bn] column-major operand, which is the whole
   point of storing B transposed: the tensor core sees it transposed for free,
   by the choice of fragment layout tag rather than by moving any data. *)
inline_for_extraction noextract
let scm_of_srm (#rows #cols : erased nat) (#l : layout2 rows cols)
  (srm : strided_row_major (vtlayout_of_tlayout l))
  : strided_col_major (vtlayout_of_tlayout (ltranspose l))
= {
    offset = srm.offset;
    stride = srm.stride;
    pf = (fun (j : natlt cols) (i : natlt rows) ->
            ltranspose_cell l i j; srm.pf i j);
  }

let lemma_aligned_scm_of_srm (#rows #cols : erased nat) (#l : layout2 rows cols)
  (srm : strided_row_major (vtlayout_of_tlayout l)) (n : pos)
  : Lemma (requires aligned_strided_row_major n srm)
          (ensures aligned_strided_col_major n (scm_of_srm srm))
= ()

(* ------------------------------------------------------------------ *)
(* Concrete index map for the transposed layout.                       *)
(* ------------------------------------------------------------------ *)

inline_for_extraction noextract
let transpose2_conc (#rows #cols : erased nat)
  (x : conc (cols @| rows @| INil))
  : Tot (conc (rows @| cols @| INil))
= [@@inline_let] let (j, (i, ())) = x in (i, (j, ()))

let transpose2_conc_correct (#rows #cols : erased nat)
  (x : conc (cols @| rows @| INil))
  : (up (transpose2_conc #rows #cols x) == (transpose2 rows cols).gg (up x))
= ()

inline_for_extraction noextract
let ctlayout_ltranspose (#rows #cols : erased nat)
  (l : layout2 rows cols) {| c : ctlayout l |}
  (#_ : squash (SZ.fits rows /\ SZ.fits cols))
  : ctlayout (ltranspose l)
= ctlayout_bij
    (transpose2 rows cols)
    (transpose2_conc #rows #cols)
    (transpose2_conc_correct #rows #cols)
    l

(* Instance forms.  Both are keyed on the head symbol [ltranspose], so they
   can only fire on a transposed layout and cannot shadow any upstream
   instance.  They exist so that generic library functions such as
   [cp_array2_vec], which take their layout witnesses as typeclass
   arguments, resolve on transposed views without the caller having to
   spell out every implicit positionally. *)
inline_for_extraction noextract
instance ctlayout_ltranspose_inst (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| c : ctlayout l |}
  (#_ : squash (SZ.fits rows /\ SZ.fits cols))
  : ctlayout (ltranspose l)
= ctlayout_ltranspose l #c

inline_for_extraction noextract
instance srm_of_scm_inst (#rows #cols : erased nat) (#l : layout2 rows cols)
  {| scm : strided_col_major (vtlayout_of_tlayout l) |}
  : strided_row_major (vtlayout_of_tlayout (ltranspose l))
= srm_of_scm scm

(* ------------------------------------------------------------------ *)
(* Transposed chest and transposed array2 view.                        *)
(* ------------------------------------------------------------------ *)

(* Transposed chest.

   The definition is phrased as [mk] over [(transpose2 cols rows).ff] rather
   than the more readable [mk2 (fun j i -> acc2 em i j)] on purpose: this is
   *syntactically* the shape that [tensor_apply_bij_st]'s trade produces, so
   the Pulse matcher discharges the [forall*] round-trip without needing
   functional extensionality.  [lemma_ctranspose_acc] below is the
   characterisation downstream code should actually reason with. *)
let ctranspose (#et : Type0) (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  : chest2 et cols rows
= mk (cols @| rows @| INil) (fun i -> acc em ((transpose2 cols rows).ff i))

let lemma_ctranspose_acc (#et : Type0) (#rows #cols : erased nat)
  (em : chest2 et rows cols) (j : natlt cols) (i : natlt rows)
  : Lemma (acc2 (ctranspose em) j i == acc2 em i j)
        [SMTPat (acc2 (ctranspose em) j i)]
= ()

let lemma_ctranspose_involutive (#et : Type0) (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  : Lemma (ctranspose (ctranspose em) == em)
= assert (equal (ctranspose (ctranspose em)) em)

(* [inline_for_extraction] matters: [atranspose] is a runtime no-op (it only
   changes the ghost layout index), so without it KaRaMeL emits an identity
   function that shows up in the generated CUDA. *)
inline_for_extraction noextract
let atranspose (#et : Type0) (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  : array2 et (ltranspose l)
= from_array (ltranspose l) (core m)

(* [tensor_apply_bij] hands back the chest in [mk]-of-bijection form; this is
   the same chest as [ctranspose], by extensionality. *)
let lemma_ctranspose_mk (#et : Type0) (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  : Lemma (mk (cols @| rows @| INil)
               (fun i -> acc em (i <~| transpose2 rows cols))
           == ctranspose em)
= assert (equal (mk (cols @| rows @| INil)
                    (fun i -> acc em (i <~| transpose2 rows cols)))
                (ctranspose em))

ghost
fn tensor_transpose2
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| is_full l |}
  (m : array2 et l)
  (#fp : perm) (#em : chest2 et rows cols)
  requires m |-> Frac fp em
  ensures atranspose m |-> Frac fp (ctranspose em)
{
  tensor_apply_bij (transpose2 rows cols) m;
  lemma_ctranspose_mk em;
  ()
}

(* The chest handed back by the [tensor_apply_bij_st] trade, in [ff] form. *)
let lemma_ctranspose_mk_ff (#et : Type0) (#rows #cols : erased nat)
  (em' : chest2 et cols rows)
  : Lemma (mk (rows @| cols @| INil)
               (fun i -> acc em' ((transpose2 rows cols).ff i))
           == ctranspose em')
        [SMTPat (mk (rows @| cols @| INil)
                    (fun i -> acc em' ((transpose2 rows cols).ff i)))]
= assert (equal (mk (rows @| cols @| INil)
                    (fun i -> acc em' ((transpose2 rows cols).ff i)))
                (ctranspose em'))

(* Read/write transposed view.  Note we deliberately do NOT try to prove
   [ltranspose (ltranspose l) == l] -- that would need functional
   extensionality inside the [injection] record's [f] field.  The trade
   returned here closes the round trip without ever needing it. *)
inline_for_extraction noextract
fn array2_transpose2_st
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| is_full l |}
  (m : array2 et l)
  (#fp : perm) (#em : chest2 et rows cols)
  requires m |-> Frac fp em
  returns tm : array2 et (ltranspose l)
  ensures
    rewrites_to tm (atranspose m) **
    tm |-> Frac fp (ctranspose em) **
    (forall* (em' : chest2 et cols rows).
      tm |-> Frac fp em' @==> m |-> Frac fp (ctranspose em'))
{
  lemma_ctranspose_mk em;
  let tm = tensor_apply_bij_st (transpose2 rows cols) m;
  tm
}

(* ------------------------------------------------------------------ *)
(* Column-major chunk ownership, by DELEGATION to the row-major        *)
(* partition applied to the transposed view.                          *)
(*                                                                     *)
(* [CV2.in_chunk] flattens row-major, [flat = i*cols + j].  For a       *)
(* column-major matrix that is not the physical address order, so the  *)
(* row-major partition of a column-major matrix is uncoalesced.        *)
(* Partitioning the TRANSPOSED view at dims (cols, rows) flattens as   *)
(* [j*rows + i], which IS the physical address.  The two partitions    *)
(* are genuinely different per-thread cell sets -- this is not an      *)
(* slprop equality and no such theorem is attempted.                   *)
(* ------------------------------------------------------------------ *)

let own_strided_chunks_cm
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (em : chest2 et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
= CV2.own_strided_chunks (atranspose m) (ctranspose em) nthr tid

let live_strided_chunks_cm
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (nthr : nat)
  (tid : natlt nthr)
  : slprop
= CV2.live_strided_chunks (atranspose m) nthr tid

ghost
fn split_array2_into_strided_chunks_cm
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| is_full l |}
  (m : array2 et l)
  (#em : chest2 et rows cols)
  (nthr : pos)
  requires
    m |-> em
  ensures
    pure (SZ.fits (l.ulen))
  ensures
    forall+ (tid : natlt nthr). own_strided_chunks_cm m em nthr tid
{
  tensor_transpose2 m;
  CV2.split_array2_into_strided_chunks (atranspose m) nthr;
  ()
}

(* ------------------------------------------------------------------ *)
(* The INVERSE view change, without layout-level involutivity.          *)
(*                                                                      *)
(* [ltranspose (ltranspose l) == l] is NOT provable (it needs            *)
(* functional extensionality inside the [injection] record's [f] field), *)
(* so the round trip cannot be closed at the layout level.  But it can   *)
(* be closed one CELL at a time: [atranspose m] and [m] share a [core],  *)
(* and [ltranspose_cell] says their index maps agree after swapping i    *)
(* and j, so the two cell slprops are literally the same slprop.         *)
(* ------------------------------------------------------------------ *)

let lemma_cell_atranspose
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l) (i : natlt rows) (j : natlt cols)
  (f : perm) (v : et)
  : Lemma (tensor_pts_to_cell (atranspose m) #f (idx2 j i) v
           == tensor_pts_to_cell m #f (idx2 i j) v)
= tensor_pts_to_cell_eq (atranspose m) (idx2 j i) f v;
  tensor_pts_to_cell_eq m (idx2 i j) f v;
  ltranspose_cell l i j

ghost
fn atranspose_back
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l)
  (#f : perm)
  (#e : chest2 et cols rows)
  requires
    atranspose m |-> Frac f e
  ensures
    m |-> Frac f (ctranspose e)
{
  tensor_ilower2 (atranspose m);
  forevery_commute
    (fun (r : natlt cols) (c : natlt rows) ->
       tensor_pts_to_cell (atranspose m) #f (idx2 r c) (acc e (idx2 r c)));
  ghost
  fn aux (c : natlt rows) (r : natlt cols)
    requires
      tensor_pts_to_cell (atranspose m) #f (idx2 r c) (acc e (idx2 r c))
    ensures
      tensor_pts_to_cell m #f (idx2 c r) (acc (ctranspose e) (idx2 c r))
  {
    lemma_cell_atranspose m c r f (acc e (idx2 r c));
    rewrite
      (tensor_pts_to_cell (atranspose m) #f (idx2 r c) (acc e (idx2 r c)))
      as (tensor_pts_to_cell m #f (idx2 c r) (acc (ctranspose e) (idx2 c r)));
  };
  forevery_map_2 _ _ aux;
  tensor_iraise2 m;
}

(* Forward direction of [atranspose_back], proved the same way: cell by cell.
   Unlike [tensor_transpose2] this does NOT require [is_full l], which matters
   for the global operand, whose layout is an arbitrary strided one. *)
ghost
fn atranspose_fwd
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l)
  (#f : perm)
  (#e : chest2 et rows cols)
  requires
    m |-> Frac f e
  ensures
    atranspose m |-> Frac f (ctranspose e)
{
  tensor_ilower2 m;
  forevery_commute
    (fun (c : natlt rows) (r : natlt cols) ->
       tensor_pts_to_cell m #f (idx2 c r) (acc e (idx2 c r)));
  ghost
  fn aux (r : natlt cols) (c : natlt rows)
    requires
      tensor_pts_to_cell m #f (idx2 c r) (acc e (idx2 c r))
    ensures
      tensor_pts_to_cell (atranspose m) #f (idx2 r c) (acc (ctranspose e) (idx2 r c))
  {
    lemma_cell_atranspose m c r f (acc e (idx2 c r));
    rewrite
      (tensor_pts_to_cell m #f (idx2 c r) (acc e (idx2 c r)))
      as (tensor_pts_to_cell (atranspose m) #f (idx2 r c) (acc (ctranspose e) (idx2 r c)));
  };
  forevery_map_2 _ _ aux;
  tensor_iraise2 (atranspose m);
}

(* Both joins are trade-free: the inverse view change is [atranspose_back]
   (defined below), which closes the round trip cell-by-cell instead of at
   the layout level.  This matters because chunk ownership arriving through
   a barrier carries no lexically-scoped borrow to trade against. *)
ghost
fn join_array2_from_strided_chunks_cm
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#em : chest2 et rows cols)
  (nthr : pos)
  requires
    pure (SZ.fits (l.ulen)) **
    (forall+ (tid : natlt nthr). own_strided_chunks_cm m em nthr tid)
  ensures
    m |-> em
{
  CV2.join_array2_from_strided_chunks (atranspose m) nthr;
  atranspose_back m;
  lemma_ctranspose_involutive em;
  rewrite each (ctranspose (ctranspose em)) as em;
  ()
}

ghost
fn join_array2_from_strided_chunks_underspec_cm
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (nthr : pos)
  requires
    pure (SZ.fits (l.ulen)) **
    (forall+ (tid : natlt nthr). live_strided_chunks_cm m nthr tid)
  ensures
    live m
{
  CV2.join_array2_from_strided_chunks_underspec (atranspose m) nthr;
  unfold live (atranspose m);
  with e. assert (atranspose m |-> e);
  atranspose_back m;
  fold live m;
  ()
}


(* Transposing commutes with subtiling, with the tile dimensions and the
   tile indices both swapped.  This is what lets the column-major staging
   path relate the [(bn, bk)] tile it actually copies to the [(bk, bn)]
   tile named in the k-loop's specification. *)
let lemma_ctranspose_subtile
  (#et : Type0) (#rows #cols : nat)
  (em : chest2 et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (ctranspose (Kuiper.EMatrix.Tiling.ematrix_subtile em trows tcols tr tc)
           == Kuiper.EMatrix.Tiling.ematrix_subtile (ctranspose em) tcols trows tc tr)
= assert (equal
    (ctranspose (Kuiper.EMatrix.Tiling.ematrix_subtile em trows tcols tr tc))
    (Kuiper.EMatrix.Tiling.ematrix_subtile (ctranspose em) tcols trows tc tr))
