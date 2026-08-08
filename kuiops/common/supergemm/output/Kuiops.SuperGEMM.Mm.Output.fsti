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
                                                     live_lane_cells }

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
