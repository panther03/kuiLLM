module Kuiops.Array2.Vectorized.Pipelined

(* Asynchronous (cp.async) vectorized copy from one [array2] to another.

   [Kuiper.Array2.Vectorized.array2_vec_read_pipelined] stages a global tile
   into a *local* array, which is the wrong shape for a GEMM: the destination
   has to be the shared-memory tile itself, viewed through its own layout, so
   the fragment loaders can read it back.  This is the array2-to-array2
   counterpart, and it is what a software-pipelined GEMM issues [A_ITERS +
   B_ITERS] times per k-step.

   Only the [chunk et] cells actually touched need to be owned on either side,
   so a thread can hold a disjoint share of a tile and stage into it without
   any whole-matrix permission.

   TODO(upstream): belongs in [Kuiper.Array2.Vectorized]. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Chest
open Kuiops.PipelineCopy
open Pulse.Lib.Pledge

open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiops.Array2.Strided
open Kuiops.Array2.Vectorized { row_cells }
open Kuiper.TensorRO { vtlayout_of_tlayout }

module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn array2_vec_cpy_pipelined
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#drows #dcols : erased nat)
  (#dl : layout2 drows dcols)
  {| T.ctlayout dl, dstrided : strided_row_major (vtlayout_of_tlayout dl) |}
  (dm : array2 et dl)
  (di : szlt drows)
  (dj : szlt (dcols - chunk et + 1))
  (#srows #scols : erased nat)
  (#sl : layout2 srows scols)
  {| T.ctlayout sl, sstrided : strided_row_major (vtlayout_of_tlayout sl) |}
  (sm : array2 et sl)
  (si : szlt srows)
  (sj : szlt (scols - chunk et + 1))
  (#f : perm)
  (#dold #sv : erased (seq et))
  (#_ : squash (Seq.length dold == chunk et /\ Seq.length sv == chunk et))
  (#b : pipeline_batch_t)
  ()
  preserves gpu
  preserves batch_live b
  requires  row_cells dm 1.0R (SZ.v di) (SZ.v dj) (chunk et) dold
  requires  row_cells sm f (SZ.v si) (SZ.v sj) (chunk et) sv
  requires  pure (aligned' 16 (T.core dm) (cell_of_pos dl (SZ.v di) (SZ.v dj)))
  requires  pure (aligned' 16 (T.core sm) (cell_of_pos sl (SZ.v si) (SZ.v sj)))
  ensures   pledge0 (batch_done b)
              (row_cells dm 1.0R (SZ.v di) (SZ.v dj) (chunk et) sv **
               row_cells sm f (SZ.v si) (SZ.v sj) (chunk et) sv)
