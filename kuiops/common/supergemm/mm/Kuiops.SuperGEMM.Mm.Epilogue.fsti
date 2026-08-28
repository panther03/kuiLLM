module Kuiops.SuperGEMM.Mm.Epilogue

(* Module 6: the epilogue of the software-pipelined tensor-core GEMM.

   One lane's part of draining its warp's [mfrag x nfrag] fp32 accumulator
   fragments to the global output D, through the per-warp fp32 shared scratch,
   applying a caller-supplied [post_map : et_acc -> et_d] elementwise.

   The fp32 scratch round-trip is FORCED by CUDA: there is no bf16 accumulator
   fragment, so [store_matrix_sync] / [mma_store] cannot write the accumulator
   at [et_d] directly.  The accumulator is stored to fp32 shared memory, then
   read back scalarly, cast by [post_map], and 128-bit (coalesced) stored to D.

   Structure (faithful to gemm_tc_flat_nosplitk_noepi.cu):
     for i in 0 .. mfrag-1:                    -- one 16-row band at a time
       warp_barrier_wait                       -- __syncwarp()
       for j in 0 .. nfrag-1:                  -- store band i, fragment j
         mma_store  acc[i*nfrag+j] -> scratch 16x16 subtile at column j*16
       warp_barrier_wait                       -- __syncwarp()
       drain band i: for each vec-group of the (16 x wn) band owned by this
         lane, scalar-read the fp32 scratch run, [post_map] it, 128-bit store
         into D at (row_base + i*16 + r, col_base + c).

   Step 1: memory safety only.  All data values are existentially quantified;
   the post merely re-establishes ownership of D, the scratch and the fragments.

   Step 6: functional.  On entry [fragarrayAcc_approximates] states the
   accumulator fragments approximate the warp's real tile [rAcc]; on exit
   [output_lane_approximates'] states the lane's D cells approximate
   [post_map_r] of the matching cells of [rAcc].  The chain is: [mma_store]
   copies each [frag x frag] accumulator fragment into a disjoint 16-column
   slice of the fp32 band, so the band content approximates the [frag x wn]
   row-band of [rAcc]; the vectorized drain writes [post_map] of that band into
   D, and [post_map %~ post_map_r] carries the approximation through the cast.
   The accumulator's [frag x frag] tiling and the drain's [frag x wn] band
   tiling differ, hence the [coerce_chest2_cols] on the post's real target.

   The lane's slice of D is [output_lane_live'] (the layout-generic version from
   [Kuiops.SuperGEMM.Mm.Output], identical to [Shared.kpre]/[kpost]) under the
   BAND tiling [tm = frag, tn = wn, wm = mfrag, wn = 1], so the drain's vec-group
   lane partition coincides exactly with D's lane ownership.  The Kernel must
   instantiate [Shared.kpre]'s [(tm, tn, wmf, wnf)] with [(frag, wn, mfrag, 1)]
   (reconciliation [mfrag * frag == wm], [1 * wn == wn]).

   D is layout-generic: it carries [ctlayout] and [strided_row_major] witnesses
   (like A and B) rather than being pinned to row-major.  The vectorized store's
   alignment obligation for a general strided layout cannot be discharged from
   [chunk et_d /?+ n] alone, so the caller supplies it as the precondition
   [aligned_strided_row_major (chunk et_d) strD] (the same pattern A/B use in
   [Kuiops.Kernel.GEMM.TensorCore2D.To.KernelBody.kf]). *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor { array2, layout2 }
open Kuiops.Array2.Strided
  { strided_row_major, cell_of_pos, aligned_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.TensorCore { FragAcc, FragLAcc, fragment }
open Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }
open Kuiper.Chest { chest2, chest_map }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', output_fragment', output_lane_approximates' }

open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Shared { scratch_tile_live, shmems_desc }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params

(* Value-preserving coercion between two [chest2] column extents that are
   provably equal.  The accumulator fragments are tiled [frag x frag]
   ([nfrag wn] of them across a band), whereas the output drain tiles the same
   warp tile into [frag x wn] bands.  [rAcc] therefore reaches the epilogue with
   columns [nfrag wn * frag], but [output_lane_approximates'] indexes it with
   [1 * wn]; both equal [wn]. *)
inline_for_extraction noextract
let coerce_chest2_cols (#et : Type) (#r #c1 #c2 : nat)
  (_ : squash (c1 == c2)) (x : chest2 et r c1) : chest2 et r c2
= coerce_eq () x

inline_for_extraction noextract
fn epilogue
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc,
     scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (#_ : squash (mfrag wm * frag == SZ.v wm))
  (#_ : squash (nfrag wn * frag == SZ.v wn))
  (nblk : szp { SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn) })
  (nthr : szp { SZ.v nthr == P.nthr bm bn wm wn })
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (rAcc : chest2 real (mfrag wm * frag) (nfrag wn * frag))
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (#_ : squash (Pulse.Lib.Array.length accFrags == mfrag wm * nfrag wn))
  ()
  preserves gpu
  preserves thread_id nthr (SZ.v tid)
  preserves fragarrayAcc_approximates (mfrag wm) (nfrag wn) accFrags rAcc
  preserves scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid)
  preserves pure (aligned 16 (T.core gD))
  preserves pure (aligned_strided_row_major (SZ.v (chunk et_d)) strD)
  requires output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid)
  ensures output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid)
    (coerce_chest2_cols #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
      (chest_map post_map_r rAcc))
