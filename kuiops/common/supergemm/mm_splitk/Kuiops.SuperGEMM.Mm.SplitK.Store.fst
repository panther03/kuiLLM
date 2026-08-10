module Kuiops.SuperGEMM.Mm.SplitK.Store

(* Drain a warp's accumulator fragments DIRECTLY to global memory.

   The split-K partial product is fp32 and so is the workspace, so there is no
   conversion to stage through shared memory: every [frag x frag] accumulator
   fragment is written with one [store_matrix_sync] against the warp's
   [wm x wn] tile of the workspace.  This is the whole reason the split-K
   kernel has no epilogue scratch (and hence a smaller shared allocation than
   the non-split sibling).

   Layout-generic: the destination is any [layout2 wm wn] carrying a
   [strided_row_major] witness, which is what [mma_store]'s [ldm] argument
   needs. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor { array2, layout2 }
open Kuiper.Tensor.Tiling { array2_extract_tile_st, subtile_layout }
open Kuiper.Array2.Strided { strided_row_major, strided_row_major_subtile }
open Kuiper.TensorCore { FragAcc, FragLAcc, value_for, fragment, array_fragment_pts_to,
                         array_fragment_pts_to_ref, array_fragment_extract_ro, mma_store }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { chest2 }
open Pulse.Lib.Trade
open Kuiops.SuperGEMM.Mm.Params { frag, frag_sz, mfrag, nfrag }
open Pulse.Lib.Array { op_Array_Access }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module ML = FStar.Math.Lemmas

inline_for_extraction noextract
fn store_frag_row
  (#et_acc : Type0) {| scalar et_acc |}
  (wm wn : szp)
  (#_ : squash (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (#lT : layout2 (SZ.v wm) (SZ.v wn)) {| T.ctlayout lT |}
       {| strT : strided_row_major (vtlayout_of_tlayout lT) |}
  (tile : array2 et_acc lT)
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#em0 : erased (seq (value_for et_acc FragAcc frag frag frag)))
  (i_sz : szlt (mfrag wm))
  (nfrag_sz : szp { SZ.v nfrag_sz == nfrag wn })
  (#_ : squash (Pulse.Lib.Array.length accFrags == mfrag wm * nfrag wn))
  (#_ : squash (SZ.fits lT.ulen))
  (#_ : squash (SZ.fits (mfrag wm * nfrag wn)))
  ()
  preserves gpu
  preserves array_fragment_pts_to accFrags #1.0R em0
  requires exists* (e : chest2 et_acc (SZ.v wm) (SZ.v wn)).
    tile |-> Frac (1.0R /. warp_size) e
  ensures exists* (e : chest2 et_acc (SZ.v wm) (SZ.v wn)).
    tile |-> Frac (1.0R /. warp_size) e
{
  let mut j = 0sz;
  while (!j <^ nfrag_sz)
    invariant live j
    invariant exists* (e : chest2 et_acc (SZ.v wm) (SZ.v wn)).
      tile |-> Frac (1.0R /. warp_size) e **
      array_fragment_pts_to accFrags #1.0R em0 **
      pure (SZ.v !j <= nfrag wn)
    decreases (nfrag wn - SZ.v !j)
  {
    let jv = !j;
    let sTile = array2_extract_tile_st tile frag frag (SZ.v i_sz) (SZ.v jv);
    ML.lemma_mult_le_right (nfrag wn) (SZ.v i_sz + 1) (mfrag wm);
    let idx : szlt (mfrag wm * nfrag wn) = i_sz *^ nfrag_sz +^ jv;
    array_fragment_pts_to_ref accFrags;
    array_fragment_extract_ro accFrags idx;
    mma_store accFrags.(idx) #_
      #(strided_row_major_subtile lT frag frag (SZ.v i_sz) (SZ.v jv))
      sTile;
    with vf. assert (sTile |-> Frac (1.0R /. warp_size) vf);
    Pulse.Lib.Forall.elim_forall #_ vf;
    ambig_trade_elim ();
    ambig_trade_elim ();
    j := !j +^ 1sz;
  };
}

inline_for_extraction noextract
fn store_warp_tile
  (#et_acc : Type0) {| scalar et_acc |}
  (wm wn : szp)
  (#_ : squash (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (#lT : layout2 (SZ.v wm) (SZ.v wn)) {| T.ctlayout lT |}
       {| strT : strided_row_major (vtlayout_of_tlayout lT) |}
  (tile : array2 et_acc lT)
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#em0 : erased (seq (value_for et_acc FragAcc frag frag frag)))
  (#_ : squash (Pulse.Lib.Array.length accFrags == mfrag wm * nfrag wn))
  (#_ : squash (SZ.fits lT.ulen))
  (#_ : squash (SZ.fits (mfrag wm * nfrag wn)))
  ()
  preserves gpu
  preserves array_fragment_pts_to accFrags #1.0R em0
  requires exists* (e : chest2 et_acc (SZ.v wm) (SZ.v wn)).
    tile |-> Frac (1.0R /. warp_size) e
  ensures exists* (e : chest2 et_acc (SZ.v wm) (SZ.v wn)).
    tile |-> Frac (1.0R /. warp_size) e
{
  let mfrag_sz : szp = wm /^ frag_sz;
  let nfrag_sz : szp = wn /^ frag_sz;
  let mut i = 0sz;
  while (!i <^ mfrag_sz)
    invariant live i
    invariant exists* (e : chest2 et_acc (SZ.v wm) (SZ.v wn)).
      tile |-> Frac (1.0R /. warp_size) e **
      array_fragment_pts_to accFrags #1.0R em0 **
      pure (SZ.v !i <= mfrag wm)
    decreases (mfrag wm - SZ.v !i)
  {
    let iv = !i;
    store_frag_row wm wn tile accFrags iv nfrag_sz ();
    i := !i +^ 1sz;
  };
}
