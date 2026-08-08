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
   [Kuiper.Kernel.GEMM.TensorCore2D.To.KernelBody.kf]). *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor { array2, layout2 }
open Kuiper.Array2.Strided
  { strided_row_major, cell_of_pos, aligned_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.TensorCore { FragAcc, FragLAcc, value_for, array_fragment_pts_to, fragment }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live' }

open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Shared { scratch_tile_live, shmems_desc }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params

inline_for_extraction noextract
fn epilogue
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab,
     scalar et_acc, has_vec_cpy et_acc,
     scalar et_d, has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (post_map : et_acc -> et_d)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (#_ : squash (mfrag wm * frag == SZ.v wm))
  (nblk : szp { SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn) })
  (nthr : szp { SZ.v nthr == P.nthr bm bn wm wn })
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#fAcc : perm)
  (#ems : erased (seq (value_for et_acc FragAcc frag frag frag)))
  (bid : natlt (SZ.v nblk))
  (tid : natlt (SZ.v nthr))
  (#_ : squash (Pulse.Lib.Array.length accFrags == mfrag wm * nfrag wn))
  ()
  preserves gpu
  preserves thread_id nthr tid
  preserves array_fragment_pts_to accFrags #fAcc ems
  preserves output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
  preserves scratch_tile_live bm bn bk wm wn skew sh nthr tid
  preserves pure (aligned 16 (T.core gD))
  preserves pure (aligned_strided_row_major (SZ.v (chunk et_d)) strD)
