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
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.SplitK.StoreLemmas
  { frag_div, frags_approx, subtiles_approx, subtiles_approx_row_done,
    tile_approx_from_subtiles }
open Pulse.Lib.Trade
open Kuiops.SuperGEMM.Mm.Params { frag, frag_sz, mfrag, nfrag }
open Pulse.Lib.Array { op_Array_Access }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module ML = FStar.Math.Lemmas

inline_for_extraction noextract
fn store_frag_row
  (#et_acc : Type0) {| scalar et_acc |} {| real_like et_acc |}
  (#a #b : erased nat)
  (asz : szp { SZ.v asz == reveal a })
  (bsz : szp { SZ.v bsz == reveal b })
  (#lT : layout2 (a * frag) (b * frag)) {| T.ctlayout lT |}
       {| strT : strided_row_major (vtlayout_of_tlayout lT) |}
  (tile : array2 et_acc lT)
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#em0 : erased (seq (value_for et_acc FragAcc frag frag frag)))
  (rAcc : chest2 real (reveal a * frag) (reveal b * frag))
  (i_sz : szlt asz)
  (#e0 : chest2 et_acc (reveal a * frag) (reveal b * frag))
  (#_ : squash (Pulse.Lib.Array.length accFrags == reveal a * reveal b))
  (#_ : squash (SZ.fits lT.ulen))
  (#_ : squash (SZ.fits (reveal a * reveal b)))
  ()
  preserves gpu
  preserves array_fragment_pts_to accFrags #1.0R em0
  requires
    tile |-> Frac (1.0R /. warp_size) e0 **
    pure (frags_approx (reveal a) (reveal b) em0 rAcc /\
          subtiles_approx (reveal a) (reveal b) e0 rAcc (SZ.v i_sz) 0)
  ensures exists* (e : chest2 et_acc (reveal a * frag) (reveal b * frag)).
    tile |-> Frac (1.0R /. warp_size) e **
    pure (subtiles_approx (reveal a) (reveal b) e rAcc (SZ.v i_sz + 1) 0)
{
  frag_div (reveal a);
  frag_div (reveal b);
  let mut j = 0sz;
  while (!j <^ bsz)
    invariant live j
    invariant exists* (e : chest2 et_acc (reveal a * frag) (reveal b * frag)).
      tile |-> Frac (1.0R /. warp_size) e **
      array_fragment_pts_to accFrags #1.0R em0 **
      pure (SZ.v !j <= reveal b /\
            subtiles_approx (reveal a) (reveal b) e rAcc (SZ.v i_sz) (SZ.v !j))
    decreases (reveal b - SZ.v !j)
  {
    let jv = !j;
    let sTile = array2_extract_tile_st tile frag frag (SZ.v i_sz) (SZ.v jv);
    ML.lemma_mult_le_right (reveal b) (SZ.v i_sz + 1) (reveal a);
    let idx : szlt (reveal a * reveal b) = i_sz *^ bsz +^ jv;
    array_fragment_pts_to_ref accFrags;
    array_fragment_extract_ro accFrags idx;
    mma_store accFrags.(idx) #_
      #(strided_row_major_subtile lT frag frag (SZ.v i_sz) (SZ.v jv))
      sTile;
    with vf. assert (sTile |-> Frac (1.0R /. warp_size) vf);
    assert pure (vf == Seq.index em0 (SZ.v idx));
    assert pure (Seq.index em0 (SZ.v idx)
      %~ ematrix_subtile rAcc frag frag (SZ.v i_sz) (SZ.v jv));
    Pulse.Lib.Forall.elim_forall #_ vf;
    ambig_trade_elim ();
    ambig_trade_elim ();
    j := !j +^ 1sz;
  };
  with e_fin. assert (tile |-> Frac (1.0R /. warp_size) e_fin);
  subtiles_approx_row_done (reveal a) (reveal b) e_fin rAcc (SZ.v i_sz);
}

inline_for_extraction noextract
fn store_warp_tile
  (#et_acc : Type0) {| scalar et_acc |} {| real_like et_acc |}
  (#a #b : erased nat)
  (asz : szp { SZ.v asz == reveal a })
  (bsz : szp { SZ.v bsz == reveal b })
  (#lT : layout2 (a * frag) (b * frag)) {| T.ctlayout lT |}
       {| strT : strided_row_major (vtlayout_of_tlayout lT) |}
  (tile : array2 et_acc lT)
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#em0 : erased (seq (value_for et_acc FragAcc frag frag frag)))
  (rAcc : chest2 real (reveal a * frag) (reveal b * frag))
  (#_ : squash (Pulse.Lib.Array.length accFrags == reveal a * reveal b))
  (#_ : squash (SZ.fits lT.ulen))
  (#_ : squash (SZ.fits (reveal a * reveal b)))
  ()
  preserves gpu
  preserves array_fragment_pts_to accFrags #1.0R em0
  requires
    (exists* (e : chest2 et_acc (reveal a * frag) (reveal b * frag)).
      tile |-> Frac (1.0R /. warp_size) e) **
    pure (frags_approx (reveal a) (reveal b) em0 rAcc)
  ensures exists* (e : chest2 et_acc (reveal a * frag) (reveal b * frag)).
    tile |-> Frac (1.0R /. warp_size) e **
    pure (e %~ rAcc)
{
  frag_div (reveal a);
  frag_div (reveal b);
  let mut i = 0sz;
  while (!i <^ asz)
    invariant live i
    invariant exists* (e : chest2 et_acc (reveal a * frag) (reveal b * frag)).
      tile |-> Frac (1.0R /. warp_size) e **
      array_fragment_pts_to accFrags #1.0R em0 **
      pure (SZ.v !i <= reveal a /\
            subtiles_approx (reveal a) (reveal b) e rAcc (SZ.v !i) 0)
    decreases (reveal a - SZ.v !i)
  {
    let iv = !i;
    store_frag_row asz bsz tile accFrags rAcc iv ();
    i := !i +^ 1sz;
  };
  with e_fin. assert (tile |-> Frac (1.0R /. warp_size) e_fin);
  let ifin = !i;
  assert pure (SZ.v ifin == reveal a);
  assert pure (subtiles_approx (reveal a) (reveal b) e_fin rAcc (reveal a) 0);
  tile_approx_from_subtiles (reveal a) (reveal b) e_fin rAcc ();
  assert pure (e_fin %~ rAcc);
}
