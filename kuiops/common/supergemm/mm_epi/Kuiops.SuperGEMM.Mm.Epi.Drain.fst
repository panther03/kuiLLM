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
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc
  { own_lane_cells, live_lane_cells, in_lane }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell { tiled_cell }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', output_fragment', output_lane_approximates' }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep
  { own_lane_cells_rw, lane_fade, lane_fade_start, lane_fade_done }

open Kuiops.Array2.Vectorized { row_cells }
open Kuiper.Tensor.Tiling { array2_subtile, array2_extract_tile_st, subtile_layout }
open Kuiops.Array2.Strided
  { strided_row_major, strided_row_major_subtile, cell_of_pos,
    aligned_strided_row_major, to_kuiper_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { mk2, acc2, chest2, chest_comb }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }

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
module P = Kuiops.SuperGEMM.Mm.Params
module VG = Kuiops.Array2.Vectorized.Group
module ML = FStar.Math.Lemmas

[@@"opaque_to_smt"]
let fragment_index
  (rows cols : pos) (row : natlt rows) (col : natlt cols)
  : i : natlt (rows * cols) { i == row * cols + col }
= ML.lemma_mult_le_right cols (row + 1) rows;
  ML.distributivity_add_left row 1 cols;
  row * cols + col

(* Flattening three nested tiles is just the corresponding row-major sum.
   State it once in a small pure context so resource conversions do not have
   to normalize the arithmetic while also reasoning about the heap. *)
let tiled_cell3_flat
  (e0 e1 e2 e3 : pos)
  (#_ : squash (e1 /?+ e0 /\ e2 /?+ e1 /\ e3 /?+ e2))
  (i0 : natlt (e0 / e1))
  (i1 : natlt (e1 / e2))
  (i2 : natlt (e2 / e3))
  (i3 : natlt e3)
  : Lemma
      (tiled_cell e0 e1 i0
        (tiled_cell e1 e2 i1 (tiled_cell e2 e3 i2 i3))
       == i0 * e1 + i1 * e2 + i2 * e3 + i3)
= ()

(* ---------------------------------------------------------------------------
   drain_group: drain one vec-group [vg] of band [idx]'s (rows x cols) D
   fragment to the global output: read the fp32 scratch band [acc], read the
   matching run of C scalarly, [comb] them and 128-bit store to D.
   [crb]/[ccb] are the global coordinates of the band's top-left corner, so the
   C window the drain target is written against is [cband eC rows cols crb ccb].
   --------------------------------------------------------------------------- *)
#push-options "--z3rlimit 15"
inline_for_extraction noextract
fn drain_group
  (#et_c #et_acc #et_d : Type0)
  {| scalar et_c, scalar et_acc, scalar et_d, hvc : has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n)) {| RO.cvtlayout lC |}
  (gC : RO.roarray2 et_c lC)
  (#fC : perm)
  (#eC : chest2 et_c (SZ.v m) (SZ.v n))
  (comb : et_c -> et_acc -> et_d)
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
  (idx : szlt (wm * wn))
  (idxRow : szlt wm)
  (idxCol : szlt wn)
  (#_ : squash (SZ.v idx == SZ.v idxRow * SZ.v wn + SZ.v idxCol))
  (crb ccb : erased nat)
  (#_ : squash (
    reveal crb + SZ.v rows <= SZ.v m /\ reveal ccb + SZ.v cols <= SZ.v n /\
    reveal crb == SZ.v mrow * SZ.v bm + SZ.v warpRow * (SZ.v wm * SZ.v rows)
                  + (SZ.v idxRow) * SZ.v rows /\
    reveal ccb == SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
                  + (SZ.v idxCol) * SZ.v cols))
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (SZ.v rows * SZ.v cols + Kuiper.Barrier.Warp.warp_size)))
  (eD0 eTarget : chest2 et_d (SZ.v rows) (SZ.v cols))
  (#_ : squash (
    eTarget == mk2 #et_d #(SZ.v rows) #(SZ.v cols)
      (fun (a : natlt (SZ.v rows)) (b : natlt (SZ.v cols)) ->
        comb (acc2 (cband eC (SZ.v rows) (SZ.v cols) crb ccb ()) a b)
             (acc2 eAcc a b))))
  (vg : sz{
    SZ.v vg < VG.ngroups (chunk et_d) (SZ.v rows) (SZ.v cols) /\
    SZ.v vg % Kuiper.Barrier.Warp.warp_size == SZ.v lane})
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
          A.length obuf == SZ.v (chunk et_d) /\ aligned 16 obuf)
  requires
    own_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      (lane_fade eD0 eTarget (SZ.v lane) (SZ.v vg))
      (SZ.v lane)
  ensures
    (exists* (bufv : seq et_d). obuf |-> bufv)
  ensures
    own_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      (lane_fade eD0 eTarget (SZ.v lane)
        (SZ.v vg + Kuiper.Barrier.Warp.warp_size))
      (SZ.v lane)
{
  VG.group_bounds (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg);
  assert pure (SZ.fits (SZ.v vg * chunk et_d));
  let flat = vg *^ chunk et_d;
  let row : szlt rows = flat /^ cols;
  let col : szlt cols = flat %^ cols;
  assert pure (SZ.v row == VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg));
  assert pure (SZ.v col == VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg));

  let em = lane_fade eD0 eTarget (SZ.v lane) (SZ.v vg);
  let em' = lane_fade eD0 eTarget (SZ.v lane)
    (SZ.v vg + Kuiper.Barrier.Warp.warp_size);

  unfold own_lane_cells
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
    em (SZ.v lane);
  VG.cells_extract_group
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
    (chunk et_d) em
    (fun ij -> in_lane (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v lane) ij)
    (SZ.v vg) ();

  lane_fade_others eD0 eTarget (SZ.v lane) (SZ.v vg);
  forevery_ext
    #(ij : (natlt (SZ.v rows) & natlt (SZ.v cols)){
      in_lane (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v lane) ij /\
      VG.group_of (chunk et_d) (SZ.v cols) ij._1 ij._2 =!= SZ.v vg})
    (fun ij -> T.tensor_pts_to_cell
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      (idx2 ij._1 ij._2)
      (acc2 em ij._1 ij._2))
    (fun ij -> T.tensor_pts_to_cell
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      (idx2 ij._1 ij._2)
      (acc2 em' ij._1 ij._2));

  rewrite
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      1.0R
      (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (chunk et_d)
      (VG.group_seq (chunk et_d) em (SZ.v vg))
  as
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      1.0R (SZ.v row) (SZ.v col) (chunk et_d)
      (VG.group_seq (chunk et_d) em (SZ.v vg));

  // ---- Compute the global coordinates of the run, mirror epilogue_chunk_update.
  assert pure (SZ.v idxRow < SZ.v wm);
  assert pure (SZ.v idxCol < SZ.v wn);
  assert pure (
    (SZ.v idxCol) * SZ.v cols + SZ.v col + (chunk et_d)
    <= SZ.v wn * SZ.v cols);
  assert pure (SZ.v warpCol + 1 <= SZ.v bn / (SZ.v wn * SZ.v cols));
  ML.div_exact_r (SZ.v bn) (SZ.v wn * SZ.v cols);
  assert pure (
    (SZ.v bn / (SZ.v wn * SZ.v cols)) * (SZ.v wn * SZ.v cols) == SZ.v bn);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idxCol) * SZ.v cols + SZ.v col + (chunk et_d)
    <= (SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols));
  assert pure ((SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols) <= SZ.v bn);
  nested_row_bound (SZ.v bm) (SZ.v wm) (SZ.v rows)
    (SZ.v warpRow) (SZ.v idxRow) (SZ.v row);
  assert pure (
    SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idxRow) * SZ.v rows + SZ.v row
    < SZ.v bm);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idxCol) * SZ.v cols + SZ.v col + (chunk et_d)
    <= SZ.v bn);
  assert pure (
    SZ.v mrow * SZ.v bm + SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idxRow) * SZ.v rows + SZ.v row
    < SZ.v m);
  assert pure (
    SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idxCol) * SZ.v cols + SZ.v col + (chunk et_d)
    <= SZ.v n);
  let globalRow : szlt m =
    mrow *^ bm +^ warpRow *^ (wm *^ rows) +^ idxRow *^ rows +^ row;
  let globalCol : szlt (n -^ (chunk et_d) +^ 1sz) =
    mcol *^ bn +^ warpCol *^ (wn *^ cols) +^ idxCol *^ cols +^ col;

  global_col_divides (chunk et_d) (SZ.v bn) (SZ.v rows) (SZ.v cols)
    (SZ.v wm) (SZ.v wn)
    (SZ.v mcol) (SZ.v warpCol) (SZ.v idxCol) (SZ.v col);
  assert pure (SZ.v globalCol ==
    SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idxCol) * SZ.v cols + SZ.v col);
  let strD_off = strD.offset;
  let strD_str = strD.stride;
  strD.pf globalRow globalCol;
  assert pure (chunk et_d /? strD_off);
  assert pure (chunk et_d /? strD_str);
  assert pure (chunk et_d /? SZ.v globalCol);
  divides_helper
    (chunk et_d) strD_off strD_str (SZ.v globalRow) (SZ.v globalCol);
  assert pure ((chunk et_d) /? cell_of_pos lD
    (SZ.v globalRow) (SZ.v globalCol));
  assert pure ((chunk et_d) * size #et_d == 16);
  scale_align (chunk et_d)
    (cell_of_pos lD (SZ.v globalRow) (SZ.v globalCol))
    (size #et_d);
  assert pure (
    16 /?+ (cell_of_pos lD
              (SZ.v globalRow) (SZ.v globalCol) * size #et_d));
  Kuiper.Divides.lemma_divides_sum 16 (base_address (T.core gD))
    (cell_of_pos lD
      (SZ.v globalRow) (SZ.v globalCol) * size #et_d);
  assert pure (aligned' 16 (T.core gD)
    (cell_of_pos lD (SZ.v globalRow) (SZ.v globalCol)));

  // ---- Convert the run to global coordinates, store, convert back.
  tiled_cell3_flat (SZ.v m) (SZ.v bm) (SZ.v wm * SZ.v rows) (SZ.v rows)
    (SZ.v bid / (SZ.v n / SZ.v bn))
    (SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v cols)))
    (SZ.v idxRow) (SZ.v row);
  assert pure (SZ.v globalRow ==
    tiled_cell (SZ.v m) (SZ.v bm) (SZ.v bid / (SZ.v n / SZ.v bn))
      (tiled_cell (SZ.v bm) (SZ.v wm * SZ.v rows)
        (SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v cols)))
        (tiled_cell (SZ.v wm * SZ.v rows) (SZ.v rows)
          (SZ.v idxRow) (SZ.v row))));
  row_cells_frag_to_global gD
    (SZ.v bm) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
    (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol)
    (SZ.v row) (SZ.v col) (chunk et_d)
    (SZ.v globalRow) (SZ.v globalCol) 1.0R
    (VG.group_seq (chunk et_d) em (SZ.v vg));

  with bufv0. assert obuf |-> bufv0;
  A.pts_to_len obuf;
  vec_comb_run_write gC acc gD comb obuf globalRow globalCol row col
    globalRow globalCol (VG.group_seq (chunk et_d) em (SZ.v vg)) ();

  // The stored run is exactly group [vg] of [em']: for the cells of group
  // [vg] (which is [lane]'s group), [em'] takes the [eTarget] value, which is
  // [comb] of the C window and the scratch [eAcc].  [vec_comb_run_write]
  // characterizes its result at the GLOBAL C coordinates; the window's
  // origin [(crb, ccb)] is exactly [(globalRow - row, globalCol - col)], so the
  // two agree cell by cell.
  assert pure (reveal crb + SZ.v row == SZ.v globalRow);
  assert pure (reveal ccb + SZ.v col == SZ.v globalCol);
  cband_run_global eC (SZ.v rows) (SZ.v cols) (reveal crb) (reveal ccb) ()
    (SZ.v row) (SZ.v col) (SZ.v (chunk et_d))
    (SZ.v globalRow) (SZ.v globalCol) ();
  with nv. assert
    (row_cells gD 1.0R (SZ.v globalRow) (SZ.v globalCol) (chunk et_d) nv
       ** obuf |-> nv);
  assert pure (forall (x : natlt (chunk et_d)).
    acc2 (cband eC (SZ.v rows) (SZ.v cols) crb ccb ()) (SZ.v row) (SZ.v col + x)
    == acc2 eC (SZ.v globalRow) (SZ.v globalCol + x));
  assert pure (forall (x : natlt (chunk et_d)).
    acc2 em' (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
             (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg) + x)
    == comb (acc2 eC (SZ.v globalRow) (SZ.v globalCol + x))
            (acc2 eAcc
               (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
               (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg) + x)));
  // [group_seq em' vg] delta-unfolds to this [init_ghost]; prove [nv] equal to
  // the *unfolded* form so downstream frame-matches (which unfold [group_seq])
  // discharge [nv == group_seq em' vg] at fuel 0.
  Seq.lemma_eq_elim nv
    (Seq.init_ghost (SZ.v (chunk et_d))
      (fun (x : natlt (SZ.v (chunk et_d))) ->
        acc2 em'
          (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
          (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg) + x)));

  row_cells_global_to_frag gD
    (SZ.v bm) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
    (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol)
    (SZ.v row) (SZ.v col) (chunk et_d)
    (SZ.v globalRow) (SZ.v globalCol) 1.0R
    (VG.group_seq (chunk et_d) em' (SZ.v vg));

  rewrite
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      1.0R (SZ.v row) (SZ.v col) (chunk et_d)
      (VG.group_seq (chunk et_d) em' (SZ.v vg))
  as
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      1.0R
      (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (chunk et_d)
      (VG.group_seq (chunk et_d) em' (SZ.v vg));

  VG.cells_restore_group
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
    (chunk et_d) em'
    (fun ij -> in_lane (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v lane) ij)
    (SZ.v vg) ();
  fold own_lane_cells
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
    em' (SZ.v lane);
}
#pop-options

(* ---------------------------------------------------------------------------
   drain_band: drain band [idx] of the warp's output fragment (a rows x cols
   tile) to the global output, combining with the C window at global origin
   [(crb, ccb)].  Consumes and restores [live_lane_cells], draining each
   vec-group via [drain_group].
   --------------------------------------------------------------------------- *)
#push-options "--z3rlimit 15"
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
  (idxRow : szlt wm)
  (idxCol : szlt wn)
  (#_ : squash (SZ.v idx == SZ.v idxRow * SZ.v wn + SZ.v idxCol))
  (crb ccb : erased nat)
  (#_ : squash (
    reveal crb + SZ.v rows <= SZ.v m /\ reveal ccb + SZ.v cols <= SZ.v n /\
    reveal crb == SZ.v mrow * SZ.v bm + SZ.v warpRow * (SZ.v wm * SZ.v rows)
                  + (SZ.v idxRow) * SZ.v rows /\
    reveal ccb == SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
                  + (SZ.v idxCol) * SZ.v cols))
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
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      (SZ.v lane)
  ensures
    (exists* (bufv : seq et_d). obuf |-> bufv)
  ensures
    (exists* (eD : chest2 et_d (SZ.v rows) (SZ.v cols)).
      own_lane_cells
        (output_fragment' gD bm bn rows cols wm wn
          (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
        eD (SZ.v lane) **
      pure (eD %~ chest_comb comb_r
                    (cband rC (SZ.v rows) (SZ.v cols) crb ccb ()) rBand))
{
  let eTarget : chest2 et_d (SZ.v rows) (SZ.v cols) =
    mk2 #et_d #(SZ.v rows) #(SZ.v cols)
      (fun (a : natlt (SZ.v rows)) (b : natlt (SZ.v cols)) ->
        comb (acc2 (cband eC (SZ.v rows) (SZ.v cols) crb ccb ()) a b)
             (acc2 eAcc a b));
  unfold live_lane_cells
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
    (SZ.v lane);
  with (eD0 : chest2 _ _ _).
    assert own_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
      eD0 (SZ.v lane);
  lane_fade_start eD0 eTarget (SZ.v lane);
  own_lane_cells_rw
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
    (SZ.v lane) eD0 (lane_fade eD0 eTarget (SZ.v lane) (SZ.v lane));

  let area = rows *^ cols /^ chunk et_d;
  FStar.Math.Lib.slash_decr_axiom (SZ.v rows * SZ.v cols) (SZ.v (chunk et_d));
  assert pure (SZ.v area == VG.ngroups (chunk et_d) (SZ.v rows) (SZ.v cols));
  let mut vg : sz = lane;
  while (!vg <^ area)
    invariant live vg
    invariant (exists* (bufv : seq et_d). obuf |-> bufv)
    invariant
      own_lane_cells
        (output_fragment' gD bm bn rows cols wm wn
          (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
        (lane_fade eD0 eTarget (SZ.v lane) (SZ.v !vg))
        (SZ.v lane)
    invariant pure (SZ.v !vg % Kuiper.Barrier.Warp.warp_size == SZ.v lane)
    invariant pure (SZ.v !vg <= SZ.v area + Kuiper.Barrier.Warp.warp_size)
    decreases (SZ.v area + Kuiper.Barrier.Warp.warp_size - SZ.v !vg)
  {
    let vvg = !vg;
    drain_group gD gC comb obuf bm bn rows cols wm wn
      mrow mcol warpRow warpCol bid wid acc idx idxRow idxCol crb ccb
      lane eD0 eTarget vvg ();
    FStar.Math.Lemmas.add_div_mod_1 (SZ.v vvg) Kuiper.Barrier.Warp.warp_size;
    assert pure (SZ.v vvg < SZ.v vvg + Kuiper.Barrier.Warp.warp_size);
    assert pure (SZ.v (!vg +^ Kuiper.warp_size)
      == SZ.v vvg + Kuiper.Barrier.Warp.warp_size);
    vg := !vg +^ Kuiper.warp_size;
  };

  lane_fade_done eD0 eTarget (SZ.v lane) (SZ.v !vg);
  own_lane_cells_rw
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol))
    (SZ.v lane)
    (lane_fade eD0 eTarget (SZ.v lane) (SZ.v !vg))
    eTarget;
  (* [eTarget = comb (C window) eAcc]; with [eAcc %~ rBand],
     [eC %~ rC] and [comb %~ comb_r], [eTarget %~ comb_r (rC window) rBand]. *)
  cband_approx eC rC (SZ.v rows) (SZ.v cols) crb ccb () ();
  comb_approx #et_c #et_acc #et_d (SZ.v rows) (SZ.v cols) comb comb_r
    (cband eC (SZ.v rows) (SZ.v cols) crb ccb ())
    (cband rC (SZ.v rows) (SZ.v cols) crb ccb ())
    eAcc rBand () ();
  target_eq_chest_comb #et_c #et_acc #et_d comb (SZ.v rows) (SZ.v cols)
    (cband eC (SZ.v rows) (SZ.v cols) crb ccb ()) eAcc;
}
#pop-options
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
  (#_ : squash (nfrag wn * frag == SZ.v wn))
  ()
  preserves array_fragment_pts_to accFrags #1.0R em0
  requires
    band |-> Frac (1.0R /. warp_size) e0 **
    pure (Seq.length em0 == mfrag wm * nfrag wn /\
          (forall (i : natlt (mfrag wm)) (j : natlt (nfrag wn)).
            Seq.index em0 (fragment_index (mfrag wm) (nfrag wn) i j)
              %~ ematrix_subtile rAcc frag frag i j))
  ensures
    exists* (e : chest2 et_acc frag (SZ.v wn)).
      band |-> Frac (1.0R /. warp_size) e **
      pure (e %~ ematrix_subtile rAcc frag (SZ.v wn) (SZ.v i_sz) 0)
{
  let ld_sz : SZ.t = wn `SZ.add` chunk et_acc;
  assert pure (SZ.v ld_sz == lde et_acc wn);
  let nfrag_sz : SZ.t = wn /^ frag_sz;
  assert pure (SZ.v nfrag_sz == nfrag wn);
  assert pure (frag /? SZ.v wn);
  assert pure (nfrag wn * frag == SZ.v wn);
  assert pure (frag /? frag);
  ML.lemma_mult_le_right frag (SZ.v i_sz + 1) (mfrag wm);
  assert pure (SZ.v i_sz * frag + frag <= mfrag wm * frag);
  assert pure (SZ.v wn /? (nfrag wn * frag));
  let mut j = 0sz;
  while (!j <^ nfrag_sz)
    invariant live j
    invariant exists* (e : chest2 et_acc frag (SZ.v wn)).
      band |-> Frac (1.0R /. warp_size) e **
      array_fragment_pts_to accFrags #1.0R em0 **
      pure (SZ.v !j <= nfrag wn /\
            (forall (j' : nat). j' < SZ.v !j ==>
              ematrix_subtile e frag frag 0 j'
                %~ ematrix_subtile (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v i_sz) 0)
                                   frag frag 0 j'))
    decreases (nfrag wn - SZ.v !j)
  {
    let jv = !j;
    with e_cur. assert (band |-> Frac (1.0R /. warp_size) e_cur);
    let sTile = array2_extract_tile_st band frag frag 0 (SZ.v jv);
    ML.lemma_mult_le_right (nfrag wn) (SZ.v i_sz + 1) (mfrag wm);
    assert pure (SZ.v i_sz * nfrag wn + SZ.v jv < mfrag wm * nfrag wn);
    let idx : szlt (mfrag wm * nfrag wn) = i_sz *^ nfrag_sz +^ jv;
    assert pure (SZ.v idx == SZ.v i_sz * nfrag wn + SZ.v jv);
    array_fragment_pts_to_ref accFrags;
    array_fragment_extract_ro accFrags idx;
    // The extraction plugin appends a [__syncwarp()] to every overwrite-combiner
    // [mma_store_comb] (overwrite branch of the [Kuiper.TensorCore.Base.mma_store_comb]
    // case in [$KUIPER_HOME/extraction/dune/generated/ExtractKuiper.ml], ~L1308-1318
    // and ~L1364-1374), so the generated code carries one extra warp barrier per
    // fragment store vs. the reference. It is redundant here: the [j] stores hit
    // disjoint 16-column slices of a warp-private band and [store_matrix_sync] is
    // warp-collective. Removing it requires dropping the [__syncwarp] element from
    // that overwrite branch upstream, out of scope for this module.
    mma_store accFrags.(idx) #_
      #(Kuiper.Array2.Strided.strided_row_major_subtile
          (subtile_layout
            (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
            frag (SZ.v wn) (SZ.v wid_sz) 0)
          #_ #(Kuiper.Array2.Strided.strided_row_major_subtile
            (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
            #_ #(to_kuiper_strided_row_major
                   (l2_skewed_row_major (warps bm bn wm wn * frag)
                     (SZ.v wn) (eskew et_acc))
                   (srm_l2_skewed_row_major
                     #(warps bm bn wm wn * frag) #(SZ.v wn) #(eskew et_acc) ld_sz))
            frag (SZ.v wn) (SZ.v wid_sz) 0)
          frag frag 0 (SZ.v jv))
      sTile;
    with vf. assert (sTile |-> Frac (1.0R /. warp_size) vf);
    assert pure (vf == Seq.index em0 (SZ.v idx));
    assert pure (Seq.index em0 (SZ.v idx)
      %~ ematrix_subtile rAcc frag frag (SZ.v i_sz) (SZ.v jv));
    Pulse.Lib.Forall.elim_forall #_ vf;
    ambig_trade_elim ();
    ambig_trade_elim ();
    (* Reslice the just-written [jv]-th fragment into the band-subtile form the
       loop invariant carries: [rAcc]'s [(i_sz, jv)] subtile is column [jv] of
       [i_sz]'s [frag x wn] row-band. *)
    ML.lemma_mult_le_right frag (SZ.v i_sz + 1) (mfrag wm);
    ML.lemma_mult_le_right frag (SZ.v jv + 1) (nfrag wn);
    subtile_band_frag #real frag (SZ.v wn)
      (mfrag wm * frag) (nfrag wn * frag) rAcc (SZ.v i_sz) (SZ.v jv) ();
    j := !j +^ 1sz;
  };
  with e_fin. assert (band |-> Frac (1.0R /. warp_size) e_fin);
  let jf = !j;
  assert pure (SZ.v jf == nfrag wn);
  assert pure (nfrag wn == SZ.v wn / frag);
  assert pure (forall (j' : natlt (SZ.v wn / frag)).
        ematrix_subtile e_fin frag frag 0 j'
          %~ ematrix_subtile (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v i_sz) 0)
                             frag frag 0 j');
  band_approx_from_subtiles #et_acc frag (SZ.v wn) e_fin
    (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v i_sz) 0) () ();
  assert pure (e_fin %~ ematrix_subtile rAcc frag (SZ.v wn) (SZ.v i_sz) 0);
}
