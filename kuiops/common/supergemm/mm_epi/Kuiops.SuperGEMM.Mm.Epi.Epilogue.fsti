module Kuiops.SuperGEMM.Mm.Epi.Epilogue

(* Epilogue of the C-combining tensor-core GEMM: [D = comb C (A @ B^T)].

   Identical to [Kuiops.SuperGEMM.Mm.Epilogue] except for the drain: instead of
   casting the fp32 scratch band by a unary [post_map], each 128-bit output run
   is combined with the matching run of C by [comb : et_c -> et_acc -> et_d].

   C is a READ-ONLY VIEW over an arbitrary [vlayout2]: the index map is not
   assumed injective (broadcast is a non-injective map), contiguous or aligned,
   so C is read SCALARLY, one [tensor_read] per element, while the D store stays
   128-bit.  See the "THE C VIEW -- NO LAYOUT IS ASSUMED" section of
   gemm_tc_flat_nosplitk_epi.cu.

   The pure lemmas and ghost helpers live in
   [Kuiops.SuperGEMM.Mm.Epi.EpilogueLemmas]. *)

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiper.TensorCore { FragAcc, FragLAcc, value_for, array_fragment_pts_to, fragment,
                         array_fragment_pts_to_ref, array_fragment_extract_ro, mma_store }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc
  { own_lane_cells, live_lane_cells, in_lane }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', output_fragment', output_lane_approximates' }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep
  { own_lane_cells_rw, lane_fade, lane_fade_start, lane_fade_done }

open Kuiops.Array2.Vectorized { row_cells }
open Kuiper.Tensor.Tiling { array2_subtile, array2_extract_tile_st, subtile_layout }
open Kuiops.Array2.Strided
  { strided_row_major, strided_row_major_subtile, cell_of_pos, aligned_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { mk2, acc2, chest2, chest_comb }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }

open Pulse.Lib.Trade

open Kuiops.SuperGEMM.Mm.Epi.EpilogueLemmas
open Kuiops.Array.LocalAligned { local_aligned16 }
open Kuiper.Barrier.Warp { warp_barrier_wait }
open Kuiops.SuperGEMM.Mm.Shared
  { scratch_tile_live, scratch_tile, sar_scratch, shmems_desc }
open Kuiops.Array2.Layout.Skewed
  { l2_skewed_row_major, srm_l2_skewed_row_major }

open Kuiops.SuperGEMM.Mm.Params

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module RO = Kuiper.TensorRO
module A = Pulse.Lib.Array
module P = Kuiops.SuperGEMM.Mm.Params
module VG = Kuiops.Array2.Vectorized.Group
module ML = FStar.Math.Lemmas
inline_for_extraction noextract
fn epilogue
  (#et_ab #et_c #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab,
     scalar et_c, real_like et_c,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc,
     scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n)) {| RO.cvtlayout lC |}
  (gC : RO.roarray2 et_c lC)
  (#fC : perm)
  (#eC : chest2 et_c (SZ.v m) (SZ.v n))
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (comb : et_c -> et_acc -> et_d)
  (comb_r : real -> real -> real { approx2 comb comb_r })
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
  preserves gC |-> Frac fC eC
  preserves pure (aligned 16 (T.core gD))
  preserves pure (aligned_strided_row_major (SZ.v (chunk et_d)) strD)
  preserves pure (reveal eC %~ rC)
  requires output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid)
  ensures output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid)
    (coerce_chest2_cols #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
      (chest_comb comb_r
        (lane_c_target rC bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid)) rAcc))
