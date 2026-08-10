module Kuiops.SuperGEMM.Mm.Output

(* Layout-generic output tiling for the software-pipelined tensor-core GEMM.

   [Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc]'s [output_fragment] and
   [output_lane_live] hardcode the output operand to row-major
   ([gD : array2 et (rm m n)]).  SuperGEMM requires A, B *and* D to carry a
   [strided_row_major] typeclass witness rather than being pinned to row-major,
   so this module generalises the two definitions to an arbitrary
   [lD : layout2 m n].  Everything underneath ([block_tile], [warp_tile],
   [array2_subtile], [live_lane_cells]) is already layout-generic, so the change
   is purely mechanical.

   TODO(upstream): fold these back into [...To.KernelDesc] as the primary
   definitions, with the row-major versions recovered at [lD := rm m n]. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile, warp_tile }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc { output_fragment, output_lane_live,
                                                     live_lane_cells, own_lane_cells }
open Kuiper.EMatrix.Tiling { ematrix_subtile }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

inline_for_extraction noextract
let output_fragment'
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (#lD : layout2 m n)
  (gD : array2 et lD)
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (mi : natlt wm)
  (nj : natlt wn)
= array2_subtile
    (warp_tile (block_tile gD bm bn bid) (wm * tm) (wn * tn) wid)
    tm tn mi nj

let output_lane_live'
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : nat)
  (#lD : layout2 m n)
  (gD : array2 et lD)
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  : slprop
= forall+ (mi : natlt wm) (nj : natlt wn).
    live_lane_cells
      (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
      (tid % warp_size)

(* Functional (approximation) counterpart of [output_lane_live']: instead of
   merely re-establishing that the lane owns its output cells, it states that
   for every [tm x tn] output fragment the lane holds cells [eD] that
   approximate the matching subtile of the real target [rD].  Layout-generic
   version of upstream
   [Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.output_lane_approximates].

   Hosted here beside [output_lane_live'] because both [Epilogue] (which commits
   to it in its [ensures]) and [Shared] (which restates the [kf] postcondition)
   need it, and [Shared] cannot import [Epilogue] (that module imports [Shared]).
   [Mm.Output] is imported by both, so it is the single, cycle-free home. *)
let output_lane_approximates'
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (#m #n : nat)
  (#lD : layout2 m n)
  (gD : array2 et lD)
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  (rD : chest2 real (wm * tm) (wn * tn))
  : slprop
= forall+ (mi : natlt wm) (nj : natlt wn).
    exists* (eD : chest2 et tm tn).
      own_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
        eD (tid % warp_size) **
      pure (eD %~ ematrix_subtile rD tm tn mi nj)

(* ---- layout-generic live split of the output tile among warp lanes ----
   The generalisation of upstream
   [Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.split_output_to_lanes] to an
   arbitrary [lD : layout2 m n]. *)
ghost
fn split_output_to_lanes'
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
  (gD : array2 et lD)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires live gD
  ensures
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_live' gD bm bn tm tn wm wn bid tid

(* ---- layout-generic live inverse of [split_output_to_lanes'] ---- *)
ghost
fn gather_output_live'
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
  (gD : array2 et lD)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (#_ : squash (SZ.fits (m * n)))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_live' gD bm bn tm tn wm wn bid tid
  ensures exists* (eD : chest2 et (SZ.v m) (SZ.v n)). gD |-> eD

(* ---- layout-generic FUNCTIONAL inverse of [split_output_to_lanes'] ----
   The approximation counterpart of [gather_output_live']: given, for every
   [(bid, tid)], that the thread's lane holds output cells approximating the
   matching doubly-nested subtile of the real target [rD], it gathers the whole
   output array back and certifies [eD %~ rD].  Layout-generic version of
   upstream [Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Teardown.gather_output]
   (which is pinned to row-major [gD : array2 et (rm m n)] and folds in the
   accumulate-into-C combine; SuperGEMM has no C operand, so [rD] is the whole
   post-mapped product). *)
ghost
fn gather_output_approximates'
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
  (gD : array2 et lD)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (#_ : squash (SZ.fits (m * n)))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  (rD : chest2 real (SZ.v m) (SZ.v n))
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_approximates' gD bm bn tm tn wm wn bid tid
        (ematrix_subtile
          (ematrix_subtile rD bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
  ensures
    exists* (eD : chest2 et (SZ.v m) (SZ.v n)).
      gD |-> eD ** pure (eD %~ rD)
