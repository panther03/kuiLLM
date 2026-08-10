module Kuiops.SuperGEMM.Mm.SplitK.Output

(* Per-warp ownership of the split-K fp32 workspace.

   The split-K GEMM has no epilogue: the accumulator is already fp32 and so is
   the workspace, so each warp drains its fragments with [mma_store] straight to
   global memory.  [mma_store] consumes a [Frac (1/warp_size)] share of the
   destination tile held cooperatively by all 32 lanes, so the natural per-thread
   output capability here is a warp-tile share -- NOT the per-lane cell partition
   ([Mm.Output.output_lane_live']) the shared-scratch epilogue needs.

   [warp_tile_pts_to] from [Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc] is
   already exactly that predicate, layout-generic; this module only adds the
   [live] split/gather around it. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { warp_tile_pts_to }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

(* One thread's share of its warp's output tile, contents unspecified. *)
let ws_warp_live
  (#et : Type0) {| scalar et |}
  (#mm #nn : nat)
  (#lW : layout2 mm nn)
  (gW : array2 et lW)
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ mm /\ bn /?+ nn /\
                wm * tm /?+ bm /\ wn * tn /?+ bn /\
                tm /?+ bm /\ tn /?+ bn))
  (bid : natlt (mm / bm * (nn / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  : slprop
= exists* (em : chest2 et (wm * tm) (wn * tn)).
    warp_tile_pts_to gW bm bn tm tn wm wn bid (tid / warp_size) em

ghost
fn split_ws_to_warps
  (#et : Type0) {| scalar et |}
  (#mm #nn : szp)
  (#lW : layout2 (SZ.v mm) (SZ.v nn)) {| T.ctlayout lW |}
  (gW : array2 et lW)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ mm /\ bn /?+ nn /\
                wm * tm /?+ bm /\ wn * tn /?+ bn /\
                tm /?+ bm /\ tn /?+ bn))
  (nblk : szp{SZ.v nblk == SZ.v mm / SZ.v bm * (SZ.v nn / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires live gW
  ensures
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      ws_warp_live gW bm bn tm tn wm wn bid tid

ghost
fn gather_ws_warps
  (#et : Type0) {| scalar et |}
  (#mm #nn : szp)
  (#lW : layout2 (SZ.v mm) (SZ.v nn)) {| T.ctlayout lW |}
  (gW : array2 et lW)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ mm /\ bn /?+ nn /\
                wm * tm /?+ bm /\ wn * tn /?+ bn /\
                tm /?+ bm /\ tn /?+ bn))
  (#_ : squash (SZ.fits lW.ulen))
  (nblk : szp{SZ.v nblk == SZ.v mm / SZ.v bm * (SZ.v nn / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      ws_warp_live gW bm bn tm tn wm wn bid tid
  ensures live gW
