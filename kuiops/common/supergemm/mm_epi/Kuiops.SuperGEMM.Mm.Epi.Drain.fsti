module Kuiops.SuperGEMM.Mm.Epi.Drain

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
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
  { own_lane_cells, live_lane_cells, in_lane }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', output_fragment', output_lane_approximates' }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep
  { own_lane_cells_rw, lane_fade, lane_fade_start, lane_fade_done }

open Kuiper.Array2.Vectorized { row_cells }
open Kuiper.Tensor.Tiling { array2_subtile, array2_extract_tile_st, subtile_layout }
open Kuiper.Array2.Strided
  { strided_row_major, strided_row_major_subtile, cell_of_pos, aligned_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { mk2, acc2, chest2, chest_comb }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }

open Pulse.Lib.Trade

open Kuiops.SuperGEMM.Mm.Epi.CombRun { vec_comb_run_write }
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
open Pulse.Lib.Array { op_Array_Access }
module P = Kuiops.SuperGEMM.Mm.Params
module VG = Kuiper.Array2.Vectorized.Group
module ML = FStar.Math.Lemmas
(* ---------------------------------------------------------------------------
   drain_band: drain band [idx] of the warp's output fragment (a rows x cols
   tile) to the global output, combining with the C window at global origin
   [(crb, ccb)].  Consumes and restores [live_lane_cells], draining each
   vec-group via [drain_group].
   --------------------------------------------------------------------------- *)
inline_for_extraction noextract
fn drain_band
  (#et_c #et_acc #et_d : Type0)
  {| scalar et_c, real_like et_c, scalar et_acc, real_like et_acc,
     scalar et_d, real_like et_d, hvc : has_vec_cpy et_d |}
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
  (obuf : array et_d)
  (bm bn rows cols wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm * SZ.v rows /?+ SZ.v bm /\
                SZ.v wn * SZ.v cols /?+ SZ.v bn))
  (#_ : squash (chunk et_d /?+ SZ.v cols))
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (warpRow : szlt (bm / (wm * rows)))
  (warpCol : szlt (bn / (wn * cols)))
  (bid : szlt (m / bm * (n / bn)))
  (wid : szlt (bm / (wm * rows) * (bn / (wn * cols))))
  (#_ : squash (
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) /\
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) /\
    SZ.v warpRow == SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v cols)) /\
    SZ.v warpCol == SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v cols))))
  (#lAcc : layout2 (SZ.v rows) (SZ.v cols)) {| T.ctlayout lAcc |}
  (acc : array2 et_acc lAcc)
  (#fs : perm)
  (#eAcc : chest2 et_acc (SZ.v rows) (SZ.v cols))
  (rBand : chest2 real (SZ.v rows) (SZ.v cols))
  (idx : szlt (wm * wn))
  (crb ccb : erased nat)
  (#_ : squash (
    reveal crb + SZ.v rows <= SZ.v m /\ reveal ccb + SZ.v cols <= SZ.v n /\
    reveal crb == SZ.v mrow * SZ.v bm + SZ.v warpRow * (SZ.v wm * SZ.v rows)
                  + (SZ.v idx / SZ.v wn) * SZ.v rows /\
    reveal ccb == SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
                  + (SZ.v idx % SZ.v wn) * SZ.v cols))
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (SZ.v rows * SZ.v cols + Kuiper.Barrier.Warp.warp_size)))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  ()
  preserves
    gpu **
    (acc |-> Frac fs eAcc) **
    (gC |-> Frac fC eC)
  requires
    (exists* (bufv : seq et_d). obuf |-> bufv)
  requires
    pure (aligned 16 (T.core gD) /\
          aligned_strided_row_major (chunk et_d) strD /\
          A.length obuf == SZ.v (chunk et_d) /\ aligned 16 obuf /\
          reveal eAcc %~ rBand /\ reveal eC %~ rC)
  requires
    live_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      (SZ.v lane)
  ensures
    (exists* (bufv : seq et_d). obuf |-> bufv)
  ensures
    (exists* (eD : chest2 et_d (SZ.v rows) (SZ.v cols)).
      own_lane_cells
        (output_fragment' gD bm bn rows cols wm wn
          (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
        eD (SZ.v lane) **
      pure (eD %~ chest_comb comb_r
                    (cband rC (SZ.v rows) (SZ.v cols) crb ccb ()) rBand))

inline_for_extraction noextract
fn store_band
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc, real_like et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#em0 : erased (seq (value_for et_acc FragAcc frag frag frag)))
  (rAcc : chest2 real (mfrag wm * frag) (nfrag wn * frag))
  (wid_sz : SZ.t { SZ.v wid_sz < warps bm bn wm wn })
  (i_sz : szlt (mfrag wm))
  (band : array2 et_acc
    (subtile_layout
      (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
      frag (SZ.v wn) (SZ.v wid_sz) 0))
  (#e0 : chest2 et_acc frag (SZ.v wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == mfrag wm * nfrag wn))
  ()
  preserves array_fragment_pts_to accFrags #1.0R em0
  requires
    band |-> Frac (1.0R /. warp_size) e0 **
    pure (Seq.length em0 == mfrag wm * nfrag wn /\
          (forall (i : natlt (mfrag wm)) (j : natlt (nfrag wn)).
            Seq.index em0 (i * nfrag wn + j) %~ ematrix_subtile rAcc frag frag i j))
  ensures
    exists* (e : chest2 et_acc frag (SZ.v wn)).
      band |-> Frac (1.0R /. warp_size) e **
      pure (e %~ ematrix_subtile rAcc frag (SZ.v wn) (SZ.v i_sz) 0)
