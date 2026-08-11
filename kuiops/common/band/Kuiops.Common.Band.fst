module Kuiops.Common.Band
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Injection
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Array2.Strided
open Kuiper.TradeHelpers
open Kuiper.ForEvery
open Kuiper.Bijection
open Pulse.Lib.Trade { (@==>) }
module SZ = Kuiper.SizeT

#push-options "--split_queries always"
inline_for_extraction noextract
instance c_band_layout
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| cc : ctlayout l |}
  (coff : erased nat)
  {| c_coff : concrete_sz coff |}
  (clen : erased nat { coff + clen <= cols })
  : ctlayout (band_layout l coff clen)
  = {
      ulen_fits = ();
      all_fit = ();
      cimap = (fun (x : conc (rows @| clen @| INil)) ->
                match x with | (i, (j, ())) ->
                [@@inline_let] let x' = (i, (concr' c_coff +^ j, ())) in
                cc.cimap x');
  }
#pop-options

inline_for_extraction noextract
let array2_band
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : erased nat)
  (clen : erased nat { coff + clen <= cols })
  : Tot (array2 et (band_layout l coff clen))
  = from_array (band_layout l coff clen) (core gm)

let array2_band_core gm coff clen = ()

let array2_band_is_global gm coff clen = ()

#push-options "--fuel 0 --ifuel 0"
inline_for_extraction noextract
let strided_row_major_band_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (coff : sz { coff < cols /\ rows > 0 })
  : res : sz { SZ.v res == sub.offset + coff }
  = sub.pf 0 coff;
    assert (cell_of_pos l 0 coff == sub.offset + coff);
    sub.offset +^ coff
#pop-options

#push-options "--fuel 0 --ifuel 0"
let strided_row_major_band_proof
  (#rows #cols : nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (coff : nat)
  (clen : nat { 0 < clen /\ coff + clen <= cols })
  (i : natlt rows)
  (j : natlt clen)
  : Lemma (
      cell_of_pos (band_layout l coff clen) i j ==
      sub.offset + coff + sub.stride * i + j
    )
=
  calc (==) {
    cell_of_pos (band_layout l coff clen) i j <: int;
    == {}
    cell_of_pos l i (coff + j);
    == { sub.pf i (coff + j) }
    sub.offset + sub.stride * i + (coff + j);
    == {}
    sub.offset + coff + sub.stride * i + j;
  };
  ()
#pop-options

inline_for_extraction noextract
instance strided_row_major_band
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen /\ rows > 0))
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (coff : erased nat)
  {| c_coff : concrete_sz coff |}
  (clen : erased nat { 0 < clen /\ coff + clen <= cols })
  : strided_row_major (vtlayout_of_tlayout (band_layout l coff clen)) =
{
  offset = strided_row_major_band_offset l (concr' c_coff);
  stride = sub.stride;
  pf = (fun i j -> strided_row_major_band_proof #rows #cols l coff clen i j);
}

let lemma_band_strided_row_major_offset l coff clen = ()

let lemma_band_strided_row_major_stride l coff clen = ()

let lemma_aligned_strided_row_major_band #rows #cols l #sub coff #c_coff clen n =
  Kuiper.Divides.lemma_nat_divides_pos_divides n sub.offset;
  Kuiper.Divides.lemma_nat_divides_pos_divides n coff;
  Kuiper.Divides.lemma_divides_sum n sub.offset coff;
  Kuiper.Divides.lemma_nat_divides_pos_divides n (sub.offset + coff)

unfold
let in_band (rows cols : nat) (coff clen : nat) (ij : natlt rows & natlt cols) : prop =
  coff <= snd ij /\ snd ij < coff + clen

#push-options "--fuel 0 --ifuel 2"
let band_bij
  (rows cols : nat)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  : ( (ij : (natlt rows & natlt cols) { in_band rows cols coff clen ij })
      =~ (natlt rows & natlt clen) ) =
  mk_bijection
    (fun (ij : (natlt rows & natlt cols) { in_band rows cols coff clen ij }) ->
       let (i, j) = ij in
       ((i, (j - coff <: natlt clen)) <: (natlt rows & natlt clen)))
    (fun (y : natlt rows & natlt clen) ->
       let (i, j) = y in
       ((i, (coff + j <: natlt cols))
          <: (ij : (natlt rows & natlt cols) { in_band rows cols coff clen ij })))
    (fun (y : natlt rows & natlt clen) -> let (i, j) = y in ())
    (fun (ij : (natlt rows & natlt cols) { in_band rows cols coff clen ij }) ->
       let (i, j) = ij in ())
#pop-options

#push-options "--fuel 0 --ifuel 1"
let lemma_band_cell_eq
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  (em : chest2 et rows cols)
  (f : perm)
  (i : natlt rows)
  (j : natlt clen)
  : Lemma (
      (Cell (array2_band gm coff clen) (idx2 i j)
         |-> Frac f (acc (ematrix_band em coff clen) (idx2 i j)))
      ==
      (Cell gm (idx2 i (coff + j <: natlt cols)) |-> Frac f (acc em (idx2 i (coff + j <: natlt cols))))
    )
=
  let gm' = array2_band gm coff clen in
  let l' = band_layout l coff clen in
  assert (l'.imap.f (idx2 i j) == l.imap.f (idx2 i (coff + j <: natlt cols)));
  tensor_pts_to_cell_eq gm' (idx2 i j) f (acc (ematrix_band em coff clen) (idx2 i j));
  tensor_pts_to_cell_eq gm (idx2 i (coff + j <: natlt cols)) f (acc em (idx2 i (coff + j <: natlt cols)))
#pop-options

let lemma_band_cell_eq_all
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  (em : chest2 et rows cols)
  (f : perm)
  : Lemma (
      forall (i : natlt rows) (j : natlt clen).
        (Cell (array2_band gm coff clen) (idx2 i j)
           |-> Frac f (acc (ematrix_band em coff clen) (idx2 i j)))
        ==
        (Cell gm (idx2 i (coff + j <: natlt cols))
           |-> Frac f (acc em (idx2 i (coff + j <: natlt cols))))
    )
=
  introduce forall (i : natlt rows) (j : natlt clen).
    (Cell (array2_band gm coff clen) (idx2 i j)
       |-> Frac f (acc (ematrix_band em coff clen) (idx2 i j)))
    ==
    (Cell gm (idx2 i (coff + j <: natlt cols))
       |-> Frac f (acc em (idx2 i (coff + j <: natlt cols))))
  with lemma_band_cell_eq gm coff clen em f i j

ghost
fn array2_extract_band_ro
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em ** pure (SZ.fits l.ulen)
  ensures
    factored
      (array2_band gm coff clen |-> Frac f (ematrix_band em coff clen))
      (gm |-> Frac f em)
{
  tensor_explode2 gm;
  forevery_refine_split
    (fun (ij : natlt rows & natlt cols) ->
       Cell gm (idx2 #rows #cols (fst ij) (snd ij)) |-> Frac f (acc em (idx2 #rows #cols (fst ij) (snd ij))))
    (fun (ij : natlt rows & natlt cols) -> in_band rows cols coff clen ij);
  forevery_iso
    (band_bij rows cols coff clen)
    (fun (ij : (natlt rows & natlt cols) { in_band rows cols coff clen ij }) ->
       Cell gm (idx2 #rows #cols (fst ij) (snd ij)) |-> Frac f (acc em (idx2 #rows #cols (fst ij) (snd ij))));
  lemma_band_cell_eq_all gm coff clen em f;
  forevery_ext
    (fun (y : natlt rows & natlt clen) ->
       Cell gm (idx2 (fst ((band_bij rows cols coff clen).gg y))
                     (snd ((band_bij rows cols coff clen).gg y)))
         |-> Frac f (acc em (idx2 (fst ((band_bij rows cols coff clen).gg y))
                                  (snd ((band_bij rows cols coff clen).gg y)))))
    (fun (y : natlt rows & natlt clen) ->
       Cell (array2_band gm coff clen) (idx2 (fst y) (snd y))
         |-> Frac f (acc (ematrix_band em coff clen) (idx2 (fst y) (snd y))));
  tensor_implode2 (array2_band gm coff clen) #f #(ematrix_band em coff clen);
  ghost
  fn aux ()
    requires
      (forall+ (ij : (natlt rows & natlt cols)
                     { ~(in_band rows cols coff clen ij) }).
         Cell gm (idx2 #rows #cols (fst ij) (snd ij)) |-> Frac f (acc em (idx2 #rows #cols (fst ij) (snd ij))))
    requires
      array2_band gm coff clen |-> Frac f (ematrix_band em coff clen)
    ensures
      gm |-> Frac f em
  {
    tensor_explode2 (array2_band gm coff clen);
    lemma_band_cell_eq_all gm coff clen em f;
    forevery_ext
      (fun (y : natlt rows & natlt clen) ->
         Cell (array2_band gm coff clen) (idx2 (fst y) (snd y))
           |-> Frac f (acc (ematrix_band em coff clen) (idx2 (fst y) (snd y))))
      (fun (y : natlt rows & natlt clen) ->
         Cell gm (idx2 (fst ((band_bij rows cols coff clen).gg y))
                       (snd ((band_bij rows cols coff clen).gg y)))
           |-> Frac f (acc em (idx2 (fst ((band_bij rows cols coff clen).gg y))
                                    (snd ((band_bij rows cols coff clen).gg y)))));
    forevery_iso_back
      (band_bij rows cols coff clen)
      (fun (ij : (natlt rows & natlt cols) { in_band rows cols coff clen ij }) ->
         Cell gm (idx2 #rows #cols (fst ij) (snd ij)) |-> Frac f (acc em (idx2 #rows #cols (fst ij) (snd ij))));
    forevery_refine_join'
      (fun (ij : natlt rows & natlt cols) -> in_band rows cols coff clen ij)
      (fun (ij : natlt rows & natlt cols) -> ~(in_band rows cols coff clen ij))
      (fun (ij : (natlt rows & natlt cols)
                 { in_band rows cols coff clen ij
                   \/ ~(in_band rows cols coff clen ij) }) ->
         Cell gm (idx2 #rows #cols (fst ij) (snd ij)) |-> Frac f (acc em (idx2 #rows #cols (fst ij) (snd ij))));
    forevery_unrefine _;
    tensor_implode2 gm #f #em;
    ()
  };
  Pulse.Lib.Trade.intro_trade _ _ _ aux;
}

inline_for_extraction noextract
fn array2_extract_band_ro'
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : erased nat)
  (clen : erased nat { coff + clen <= cols })
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em ** pure (SZ.fits l.ulen)
  returns gm' : array2 et (band_layout l coff clen)
  ensures
    rewrites_to gm' (array2_band gm coff clen) **
    factored
      (gm' |-> Frac f (ematrix_band em coff clen))
      (gm |-> Frac f em)
{
  array2_extract_band_ro gm coff clen;
  array2_band gm coff clen;
}
