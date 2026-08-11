module Kuiops.Vec.MapRun

(* Read a [chunk et_d]-wide run of cells from a source matrix, apply a pure
   [post_map] elementwise, and 128-bit (vectorized) store the mapped run into a
   destination matrix.  This is the "retype + narrow" store the GEMM epilogue
   needs: the tensor-core accumulator can only be drained through fp32 shared
   memory (there is no bf16 accumulator fragment in CUDA), so the fp32 scratch
   values must be read scalarly and cast to the output element type before the
   coalesced global store.

   The written run is specified exactly ([post_map] of the source run) so a
   caller draining a whole fragment can reassemble its lane's [own_lane_cells]
   with the [lane_fade] chest machinery; the epilogue's own postcondition still
   leaves the global output values unspecified (step 1: memory safety).

   Reusable and layout-generic; belongs upstream in Kuiper.Array2.Vectorized. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Array2.Vectorized { row_cells, array2_vec_write_cells }
open Kuiper.Tensor { array2, layout2, tensor_read }
open Kuiper.Array2.Strided { cell_of_pos, strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { acc2 }

module T = Kuiper.Tensor
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn vec_map_run_write
  (#et_s : Type0) {| sized et_s |}
  (#et_d : Type0) {| sized et_d, has_vec_cpy et_d |}
  (#srows #scols : erased nat)
  (#sl : layout2 srows scols) {| T.ctlayout sl |}
  (src : array2 et_s sl)
  (#drows #dcols : erased nat)
  (#dl : layout2 drows dcols) {| T.ctlayout dl, strided_row_major (vtlayout_of_tlayout dl) |}
  (dst : array2 et_d dl)
  (post_map : et_s -> et_d)
  (obuf : array et_d)
  (sr : szlt srows)
  (sc : SZ.t { SZ.v sc + SZ.v (chunk et_d) <= scols })
  (di : szlt drows)
  (dj : szlt (dcols - chunk et_d + 1))
  (#fs : perm)
  (#es : T.chest2 et_s srows scols)
  (dold : erased (seq et_d))
  (#bufv : erased (seq et_d))
  (#_ : squash (Seq.length dold == SZ.v (chunk et_d) /\ Seq.length bufv == SZ.v (chunk et_d)))
  ()
  preserves gpu
  preserves src |-> Frac fs es
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
                    post_map (acc2 es (SZ.v sr) (SZ.v sc + x))))
{
  let mut e = 0sz;
  while (!e <^ (chunk et_d))
    invariant exists* (ev : SZ.t) (so : seq et_d).
      e |-> ev **
      obuf |-> so **
      src |-> Frac fs es **
      pure (SZ.v ev <= SZ.v (chunk et_d) /\ Seq.length so == SZ.v (chunk et_d) /\
            (forall (x : nat). x < SZ.v ev ==>
              Seq.index so x == post_map (acc2 es (SZ.v sr) (SZ.v sc + x))))
    decreases (SZ.v (chunk et_d) - SZ.v !e)
  {
    let ve = !e;
    let scc : szlt scols = sc +^ ve;
    let sv = tensor_read src (sr, (scc, ()));
    with so. assert obuf |-> so;
    A.op_Array_Assignment obuf ve (post_map sv) #so;
    e := !e +^ 1sz;
  };
  with so. assert obuf |-> so;
  assert pure (Seq.equal so
    (Seq.init_ghost (SZ.v (chunk et_d))
      (fun (x : natlt (SZ.v (chunk et_d))) ->
        post_map (acc2 es (SZ.v sr) (SZ.v sc + x)))));
  array2_vec_write_cells dst di dj obuf dold so ();
}
