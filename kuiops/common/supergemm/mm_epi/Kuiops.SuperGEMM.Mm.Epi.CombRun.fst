module Kuiops.SuperGEMM.Mm.Epi.CombRun

(* Read a [chunk et_d]-wide run of cells from a source matrix, combine it
   elementwise with the matching run of a read-only C VIEW, and 128-bit
   (vectorized) store the result into a destination matrix.

   This is [Kuiops.Vec.MapRun.vec_map_run_write] with the unary [post_map]
   replaced by a binary [comb : et_c -> et_s -> et_d] whose first argument comes
   from [gC].  [gC] is a [rotensor] over an arbitrary [vtlayout]: its index map
   is not assumed injective, contiguous or aligned, so C is read SCALARLY, one
   [tensor_read] per element.  That is the definition of the C read (see the
   "THE C VIEW" section of gemm_tc_flat_nosplitk_epi.cu); only the D store is
   vectorized.

   Reusable and layout-generic; belongs upstream in Kuiper.Array2.Vectorized. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiops.Array2.Vectorized { row_cells, array2_vec_write_cells }
open Kuiper.Tensor { array2, layout2, tensor_read }
open Kuiops.Array2.Strided { cell_of_pos, strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { acc2 }

module RO = Kuiper.TensorRO
module T = Kuiper.Tensor
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn vec_comb_run_write
  (#et_c : Type0) {| sized et_c |}
  (#et_s : Type0) {| sized et_s |}
  (#et_d : Type0) {| sized et_d, has_vec_cpy et_d |}
  (#crows #ccols : erased nat)
  (#cl : RO.vlayout2 crows ccols) {| RO.cvtlayout cl |}
  (gC : RO.roarray2 et_c cl)
  (#srows #scols : erased nat)
  (#sl : layout2 srows scols) {| T.ctlayout sl |}
  (src : array2 et_s sl)
  (#drows #dcols : erased nat)
  (#dl : layout2 drows dcols) {| T.ctlayout dl, strided_row_major (vtlayout_of_tlayout dl) |}
  (dst : array2 et_d dl)
  (comb : et_c -> et_s -> et_d)
  (obuf : array et_d)
  (cr : szlt crows)
  (cc : SZ.t { SZ.v cc + SZ.v (chunk et_d) <= ccols })
  (sr : szlt srows)
  (sc : SZ.t { SZ.v sc + SZ.v (chunk et_d) <= scols })
  (di : szlt drows)
  (dj : szlt (dcols - chunk et_d + 1))
  (#fc #fs : perm)
  (#ec : T.chest2 et_c crows ccols)
  (#es : T.chest2 et_s srows scols)
  (dold : erased (seq et_d))
  (#bufv : erased (seq et_d))
  (#_ : squash (Seq.length dold == SZ.v (chunk et_d) /\ Seq.length bufv == SZ.v (chunk et_d)))
  ()
  preserves gpu
  preserves src |-> Frac fs es
  preserves gC |-> Frac fc ec
  requires  obuf |-> bufv
  requires  row_cells dst 1.0R di dj (chunk et_d) dold
  requires  pure (A.length obuf == SZ.v (chunk et_d))
  requires  pure (aligned' 16 (T.core dst) (cell_of_pos dl di dj))
  requires  pure (aligned 16 obuf)
  ensures   exists* (nv : seq et_d { Seq.length nv == SZ.v (chunk et_d) }).
              row_cells dst 1.0R di dj (chunk et_d) nv **
              obuf |-> nv **
              pure (nv ==
                Seq.init_ghost (SZ.v (chunk et_d))
                  (fun (x : natlt (SZ.v (chunk et_d))) ->
                    comb (acc2 ec (SZ.v cr) (SZ.v cc + x))
                         (acc2 es (SZ.v sr) (SZ.v sc + x))))
{
  let mut e = 0sz;
  while (!e <^ (chunk et_d))
    invariant exists* (ev : SZ.t) (so : seq et_d).
      e |-> ev **
      obuf |-> so **
      src |-> Frac fs es **
      gC |-> Frac fc ec **
      pure (SZ.v ev <= SZ.v (chunk et_d) /\ Seq.length so == SZ.v (chunk et_d) /\
            (forall (x : nat). x < SZ.v ev ==>
              Seq.index so x
              == comb (acc2 ec (SZ.v cr) (SZ.v cc + x))
                      (acc2 es (SZ.v sr) (SZ.v sc + x))))
    decreases (SZ.v (chunk et_d) - SZ.v !e)
  {
    let ve = !e;
    let scc : szlt scols = sc +^ ve;
    let ccc : szlt ccols = cc +^ ve;
    let sv = tensor_read src (sr, (scc, ()));
    let cv = RO.tensor_read gC (cr, (ccc, ()));
    with so. assert obuf |-> so;
    obuf.(ve) <- comb cv sv;
    e := !e +^ 1sz;
  };
  with so. assert obuf |-> so;
  assert pure (Seq.equal so
    (Seq.init_ghost (SZ.v (chunk et_d))
      (fun (x : natlt (SZ.v (chunk et_d))) ->
        comb (acc2 ec (SZ.v cr) (SZ.v cc + x))
             (acc2 es (SZ.v sr) (SZ.v sc + x)))));
  array2_vec_write_cells dst di dj obuf dold so ();
}
