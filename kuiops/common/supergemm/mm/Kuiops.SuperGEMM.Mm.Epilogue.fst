module Kuiops.SuperGEMM.Mm.Epilogue

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore { FragAcc, FragLAcc, value_for, array_fragment_pts_to, fragment,
                         array_fragment_pts_to_ref, array_fragment_extract_ro, mma_store }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
  { own_lane_cells, live_lane_cells, in_lane }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', output_fragment' }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile, warp_tile }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell
  { tiled_cell }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep
  { lane_coincide, own_lane_cells_rw, lane_fade, lane_fade_start, lane_fade_done }

open Kuiper.Array2.Vectorized { row_cells }
open Kuiper.Tensor.Tiling { array2_subtile, array2_extract_tile_st, subtile_layout, c_subtile_layout, cell_convert_eq }
open Kuiper.Array2.Strided
  { strided_row_major, strided_row_major_l2_row_major, strided_row_major_subtile,
    cell_of_pos, aligned_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { mk2, acc2, chest2 }

open Kuiper.Concrete { concrete_sz, concrete_sz_sz }
open Pulse.Lib.Trade

open Kuiops.Vec.MapRun { vec_map_run_write }
open Kuiops.Array.LocalAligned { local_aligned16 }
open Kuiper.Barrier.Warp { warp_barrier_wait }
open Kuiops.SuperGEMM.Mm.Shared
  { scratch_tile_live, scratch_tile, scratch_matrix, sar_scratch, shmems_desc }
open Kuiops.Array2.Layout.Skewed
  { l2_skewed_row_major, srm_l2_skewed_row_major, c_l2_skewed_row_major }

open Kuiops.SuperGEMM.Mm.Params

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module A = Pulse.Lib.Array
open Pulse.Lib.Array { op_Array_Access }
module P = Kuiops.SuperGEMM.Mm.Params
module VG = Kuiper.Array2.Vectorized.Group
module ML = FStar.Math.Lemmas

(* ---------------------------------------------------------------------------
   Pure helper lemmas copied from
   Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueChunkUpdate
   (that module has no interface, so its helpers cannot be imported).
   --------------------------------------------------------------------------- *)

let divides_helper
  (d : pos)
  (a b r c : int)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
= Kuiper.Divides.lemma_divides_product_l d b r;
  Kuiper.Divides.lemma_divides_sum d a (b * r);
  Kuiper.Divides.lemma_divides_sum d (a + b * r) c

(* [chunk et] divides 16 (its byte width is exactly 16 bytes). *)
let chunk_div16 (et : Type0) {| sized et, hvc : has_vec_cpy et |}
  : Lemma (chunk et /? 16)
= assert (chunk et * size #et == 16);
  introduce exists (z : int). chunk et * z == 16
  with (size #et) and ()

(* If [d | c] and [d * s == 16], then [16 | (c * s)].  Isolated into its own
   query so the nonlinear step is robust. *)
let scale_align (d : pos) (c : nat) (s : pos)
  : Lemma (requires d /? c /\ d * s == 16)
          (ensures 16 /?+ (c * s))
= let z = Kuiper.Divides.get_factor d c in
  calc (==) {
    c * s;
    == { }
    (d * z) * s;
    == { FStar.Math.Lemmas.paren_mul_right d z s }
    d * (z * s);
    == { FStar.Math.Lemmas.swap_mul z s }
    d * (s * z);
    == { FStar.Math.Lemmas.paren_mul_right d s z }
    (d * s) * z;
  };
  FStar.Math.Lemmas.swap_mul (d * s) z;
  FStar.Math.Lemmas.multiple_modulo_lemma z (d * s)

let global_col_divides
  (w : pos)
  (bn rows cols wm wn : pos)
  (#_ : squash (wn * cols /?+ bn))
  (mcol warpCol mi : nat)
  (col : nat)
  : Lemma (requires w /? cols /\ w /? col)
          (ensures w /? (mcol * bn + warpCol * (wn * cols) + mi * cols + col))
= Kuiper.Divides.lemma_divides_product_r w wn cols;
  ML.div_exact_r bn (wn * cols);
  Kuiper.Divides.lemma_divides_product_r w (bn / (wn * cols)) (wn * cols);
  assert (bn == (bn / (wn * cols)) * (wn * cols));
  Kuiper.Divides.lemma_divides_product_r w mcol bn;
  Kuiper.Divides.lemma_divides_product_r w warpCol (wn * cols);
  Kuiper.Divides.lemma_divides_product_r w mi cols;
  Kuiper.Divides.lemma_divides_sum w (mcol * bn) (warpCol * (wn * cols));
  Kuiper.Divides.lemma_divides_sum w
    (mcol * bn + warpCol * (wn * cols)) (mi * cols);
  Kuiper.Divides.lemma_divides_sum w
    (mcol * bn + warpCol * (wn * cols) + mi * cols) col

(* Layout-generic counterpart of
   [Kuiper...EpilogueCell.output_fragment_cell_convert_eq], stated for the
   generic [output_fragment'] over an arbitrary [lD].  Same proof: three
   [cell_convert_eq] steps down the block/warp/subtile chain, all of which are
   already layout-generic.
   TODO(upstream): fold into [EpilogueCell] alongside [output_fragment']. *)
let output_fragment_cell_convert_eq'
  (#et : Type0) {| scalar et |}
  (#m #n : nat)
  (#lD : layout2 m n)
  (gD : array2 et lD)
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn /\
                tm /?+ (wm * tm) /\ tn /?+ (wn * tn) /\
                (wm * tm) / tm == wm /\ (wn * tn) / tn == wn))
  (bid : natlt (m / bm * (n / bn)))
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  (mi : natlt wm)
  (nj : natlt wn)
  (i : natlt tm)
  (j : natlt tn)
  (f : perm)
  (v : et)
  : Lemma (
      let blockRow = bid / (n / bn) in
      let blockCol = bid % (n / bn) in
      let warpRow = wid / (bn / (wn * tn)) in
      let warpCol = wid % (bn / (wn * tn)) in
      let fragRow = tiled_cell (wm * tm) tm mi i in
      let fragCol = tiled_cell (wn * tn) tn nj j in
      let blockCellRow = tiled_cell bm (wm * tm) warpRow fragRow in
      let blockCellCol = tiled_cell bn (wn * tn) warpCol fragCol in
      T.tensor_pts_to_cell
        (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
        #f (idx2 i j) v
      ==
      T.tensor_pts_to_cell gD #f
        (idx2
          (tiled_cell m bm blockRow blockCellRow)
          (tiled_cell n bn blockCol blockCellCol))
        v)
= let blockRow = bid / (n / bn) in
  let blockCol = bid % (n / bn) in
  let warpRow = wid / (bn / (wn * tn)) in
  let warpCol = wid % (bn / (wn * tn)) in
  let fragRow = tiled_cell (wm * tm) tm mi i in
  let fragCol = tiled_cell (wn * tn) tn nj j in
  let blockCellRow = tiled_cell bm (wm * tm) warpRow fragRow in
  let blockCellCol = tiled_cell bn (wn * tn) warpCol fragCol in
  let dBlock = block_tile gD bm bn bid in
  let dWarp = warp_tile dBlock (wm * tm) (wn * tn) wid in
  cell_convert_eq dWarp tm tn mi nj i j f v;
  cell_convert_eq dBlock (wm * tm) (wn * tn)
    warpRow warpCol fragRow fragCol f v;
  cell_convert_eq gD bm bn
    blockRow blockCol blockCellRow blockCellCol f v;
  ()

let frag_global_cell_eq
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
  (i : natlt tm)
  (j : nat)
  (w : nat { j + w <= tn })
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : nat { globalCol + w <= n /\
    globalCol == bid % (n / bn) * bn
      + wid % (bn / (wn * tn)) * (wn * tn) + nj * tn + j })
  (f : perm)
  (v : seq et { Seq.length v == w })
  (x : natlt w)
  : Lemma (
      T.tensor_pts_to_cell
        (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
        #f (idx2 i (j + x)) (Seq.index v x)
      == T.tensor_pts_to_cell gD #f
           (idx2 globalRow (globalCol + x)) (Seq.index v x))
= ML.cancel_mul_div wm tm;
  ML.cancel_mul_div wn tn;
  ML.cancel_mul_mod wm tm;
  ML.cancel_mul_mod wn tn;
  output_fragment_cell_convert_eq' gD bm bn tm tn wm wn
    bid wid mi nj i (j + x) f (Seq.index v x)

ghost
fn row_cells_frag_to_global
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
  (i : natlt tm)
  (j : nat)
  (w : nat { j + w <= tn })
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : nat { globalCol + w <= n /\
    globalCol == bid % (n / bn) * bn
      + wid % (bn / (wn * tn)) * (wn * tn) + nj * tn + j })
  (f : perm)
  (v : seq et { Seq.length v == w })
  requires
    row_cells (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
      f i j w v
  ensures
    row_cells gD f globalRow globalCol w v
{
  FStar.Classical.forall_intro
    (frag_global_cell_eq gD bm bn tm tn wm wn bid wid mi nj
      i j w globalRow globalCol f v);
  unfold row_cells (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
    f i j w v;
  forevery_ext #(natlt w)
    (fun x -> T.tensor_pts_to_cell
      (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
      #f (idx2 i (j + x)) (Seq.index v x))
    (fun x -> T.tensor_pts_to_cell gD #f
      (idx2 globalRow (globalCol + x)) (Seq.index v x));
  fold row_cells gD f globalRow globalCol w v;
}

ghost
fn row_cells_global_to_frag
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
  (i : natlt tm)
  (j : nat)
  (w : nat { j + w <= tn })
  (globalRow : natlt m {
    globalRow == tiled_cell m bm (bid / (n / bn))
      (tiled_cell bm (wm * tm) (wid / (bn / (wn * tn)))
        (tiled_cell (wm * tm) tm mi i)) })
  (globalCol : nat { globalCol + w <= n /\
    globalCol == bid % (n / bn) * bn
      + wid % (bn / (wn * tn)) * (wn * tn) + nj * tn + j })
  (f : perm)
  (v : seq et { Seq.length v == w })
  requires
    row_cells gD f globalRow globalCol w v
  ensures
    row_cells (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
      f i j w v
{
  FStar.Classical.forall_intro
    (frag_global_cell_eq gD bm bn tm tn wm wn bid wid mi nj
      i j w globalRow globalCol f v);
  unfold row_cells gD f globalRow globalCol w v;
  forevery_ext #(natlt w)
    (fun x -> T.tensor_pts_to_cell gD #f
      (idx2 globalRow (globalCol + x)) (Seq.index v x))
    (fun x -> T.tensor_pts_to_cell
      (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
      #f (idx2 i (j + x)) (Seq.index v x));
  fold row_cells (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
    f i j w v;
}

(* ---------------------------------------------------------------------------
   Pure helper lemmas copied from
   Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep.
   --------------------------------------------------------------------------- *)

let group_gap (a b : nat) (lane : natlt Kuiper.Barrier.Warp.warp_size)
  : Lemma (requires a % Kuiper.Barrier.Warp.warp_size == lane /\
                    b % Kuiper.Barrier.Warp.warp_size == lane /\ a < b)
          (ensures a + Kuiper.Barrier.Warp.warp_size <= b)
= FStar.Math.Lemmas.euclidean_division_definition a Kuiper.Barrier.Warp.warp_size;
  FStar.Math.Lemmas.euclidean_division_definition b Kuiper.Barrier.Warp.warp_size;
  if b / Kuiper.Barrier.Warp.warp_size <= a / Kuiper.Barrier.Warp.warp_size then
    FStar.Math.Lemmas.lemma_mult_le_right Kuiper.Barrier.Warp.warp_size
      (b / Kuiper.Barrier.Warp.warp_size) (a / Kuiper.Barrier.Warp.warp_size)

let lane_fade_others
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : pos)
  (em0 em1 : chest2 et rows cols)
  (lane : natlt Kuiper.Barrier.Warp.warp_size)
  (vg : nat { vg % Kuiper.Barrier.Warp.warp_size == lane })
  : Lemma (forall (i : natlt rows) (j : natlt cols).
      in_lane (chunk et #_ #hvc) rows cols lane (i, j) /\
      VG.group_of (chunk et #_ #hvc) cols i j <> vg ==>
        acc2 (lane_fade em0 em1 lane vg) i j
        == acc2 (lane_fade em0 em1 lane (vg + Kuiper.Barrier.Warp.warp_size)) i j)
= introduce forall (i : natlt rows) (j : natlt cols).
      in_lane (chunk et #_ #hvc) rows cols lane (i, j) /\
      VG.group_of (chunk et #_ #hvc) cols i j <> vg ==>
        acc2 (lane_fade em0 em1 lane vg) i j
        == acc2 (lane_fade em0 em1 lane (vg + Kuiper.Barrier.Warp.warp_size)) i j
  with introduce _ ==> _
  with _. (
    let g = VG.group_of (chunk et #_ #hvc) cols i j in
    if vg < g then group_gap vg g lane)

(* Trivial warp-barrier proof: all real resources are framed (p = q = emp). *)
unfold
let warp_emp_pred (_ : natlt Kuiper.Barrier.Warp.warp_size) : slprop = emp

ghost
fn warp_sync_noop (p : natlt Kuiper.Barrier.Warp.warp_size -> slprop)
  requires forall+ (i : natlt Kuiper.Barrier.Warp.warp_size). p i
  ensures  forall+ (i : natlt Kuiper.Barrier.Warp.warp_size). p i
{
  ()
}

let warp_emp_proof
  : stt_ghost unit emp_inames
      (requires forall+ (i : natlt Kuiper.Barrier.Warp.warp_size). warp_emp_pred i)
      (ensures  fun _ -> forall+ (i : natlt Kuiper.Barrier.Warp.warp_size). warp_emp_pred i)
  = warp_sync_noop warp_emp_pred

(* ---------------------------------------------------------------------------
   drain_group: drain one vec-group [vg] of band [idx]'s (rows x cols) D
   fragment to the global output, reading the fp32 scratch band [acc],
   [post_map]-casting and 128-bit storing to D.  Mirror of
   epilogue_fragment_step, with the C/comb machinery replaced by a
   [vec_map_run_write] against the global matrix.
   --------------------------------------------------------------------------- *)
#push-options "--z3rlimit 15"
inline_for_extraction noextract
fn drain_group
  (#et_acc #et_d : Type0)
  {| scalar et_acc, scalar et_d, hvc : has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (post_map : et_acc -> et_d)
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
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (SZ.v rows * SZ.v cols + Kuiper.Barrier.Warp.warp_size)))
  (eD0 eTarget : chest2 et_d (SZ.v rows) (SZ.v cols))
  (#_ : squash (
    eTarget == mk2 #et_d #(SZ.v rows) #(SZ.v cols)
      (fun (a : natlt (SZ.v rows)) (b : natlt (SZ.v cols)) ->
        post_map (acc2 eAcc a b))))
  (vg : sz{
    SZ.v vg < VG.ngroups (chunk et_d) (SZ.v rows) (SZ.v cols) /\
    SZ.v vg % Kuiper.Barrier.Warp.warp_size == SZ.v lane})
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  ()
  preserves
    gpu **
    acc |-> Frac fs eAcc
  requires
    (exists* (bufv : seq et_d). obuf |-> bufv)
  requires
    pure (aligned 16 (T.core gD) /\
          aligned_strided_row_major (chunk et_d) strD /\
          A.length obuf == SZ.v (chunk et_d) /\ aligned 16 obuf)
  requires
    own_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      (lane_fade eD0 eTarget (SZ.v lane) (SZ.v vg))
      (SZ.v lane)
  ensures
    (exists* (bufv : seq et_d). obuf |-> bufv)
  ensures
    own_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
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
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
    em (SZ.v lane);
  VG.cells_extract_group
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
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
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      (idx2 ij._1 ij._2)
      (acc2 em ij._1 ij._2))
    (fun ij -> T.tensor_pts_to_cell
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      (idx2 ij._1 ij._2)
      (acc2 em' ij._1 ij._2));

  rewrite
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      1.0R
      (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (chunk et_d)
      (VG.group_seq (chunk et_d) em (SZ.v vg))
  as
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      1.0R (SZ.v row) (SZ.v col) (chunk et_d)
      (VG.group_seq (chunk et_d) em (SZ.v vg));

  // ---- Compute the global coordinates of the run, mirror epilogue_chunk_update.
  assert pure (SZ.v idx / SZ.v wn < SZ.v wm);
  assert pure (SZ.v idx % SZ.v wn < SZ.v wn);
  assert pure (
    (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_d)
    <= SZ.v wn * SZ.v cols);
  assert pure (SZ.v warpCol + 1 <= SZ.v bn / (SZ.v wn * SZ.v cols));
  ML.div_exact_r (SZ.v bn) (SZ.v wn * SZ.v cols);
  assert pure (
    (SZ.v bn / (SZ.v wn * SZ.v cols)) * (SZ.v wn * SZ.v cols) == SZ.v bn);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_d)
    <= (SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols));
  assert pure ((SZ.v warpCol + 1) * (SZ.v wn * SZ.v cols) <= SZ.v bn);
  assert pure (
    SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idx / SZ.v wn) * SZ.v rows + SZ.v row
    < SZ.v bm);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_d)
    <= SZ.v bn);
  assert pure (
    SZ.v mrow * SZ.v bm + SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idx / SZ.v wn) * SZ.v rows + SZ.v row
    < SZ.v m);
  assert pure (
    SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col + (chunk et_d)
    <= SZ.v n);
  let globalRow : szlt m =
    mrow *^ bm +^ warpRow *^ (wm *^ rows) +^ (idx /^ wn) *^ rows +^ row;
  let globalCol : szlt (n -^ (chunk et_d) +^ 1sz) =
    mcol *^ bn +^ warpCol *^ (wn *^ cols) +^ (idx %^ wn) *^ cols +^ col;

  global_col_divides (chunk et_d) (SZ.v bn) (SZ.v rows) (SZ.v cols)
    (SZ.v wm) (SZ.v wn)
    (SZ.v mcol) (SZ.v warpCol) (SZ.v idx % SZ.v wn) (SZ.v col);
  assert pure (SZ.v globalCol ==
    SZ.v mcol * SZ.v bn + SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idx % SZ.v wn) * SZ.v cols + SZ.v col);
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
  row_cells_frag_to_global gD
    (SZ.v bm) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
    (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn)
    (SZ.v row) (SZ.v col) (chunk et_d)
    (SZ.v globalRow) (SZ.v globalCol) 1.0R
    (VG.group_seq (chunk et_d) em (SZ.v vg));

  with bufv0. assert obuf |-> bufv0;
  A.pts_to_len obuf;
  vec_map_run_write acc gD post_map obuf row col globalRow globalCol
    (VG.group_seq (chunk et_d) em (SZ.v vg)) ();

  // The stored run is exactly group [vg] of [em']: for the cells of group
  // [vg] (which is [lane]'s group), [em'] takes the [eTarget] value, which is
  // [post_map] of the scratch [eAcc]. [vec_map_run_write] characterizes its
  // result as [init_ghost (fun x -> post_map (acc2 eAcc row (col+x)))]; we show
  // that this [init_ghost] equals [group_seq em' vg], so any frame-match that
  // needs [<witness> == group_seq em' vg] is discharged via the [vec_map]
  // hypothesis [<witness> == init_ghost ...].
  with nv. assert
    (row_cells gD 1.0R (SZ.v globalRow) (SZ.v globalCol) (chunk et_d) nv
       ** obuf |-> nv);
  assert pure (forall (x : natlt (chunk et_d)).
    acc2 em' (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
             (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg) + x)
    == post_map (acc2 eAcc
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
    (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn)
    (SZ.v row) (SZ.v col) (chunk et_d)
    (SZ.v globalRow) (SZ.v globalCol) 1.0R
    (VG.group_seq (chunk et_d) em' (SZ.v vg));

  rewrite
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      1.0R (SZ.v row) (SZ.v col) (chunk et_d)
      (VG.group_seq (chunk et_d) em' (SZ.v vg))
  as
    row_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      1.0R
      (VG.grow (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (VG.gcol (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v vg))
      (chunk et_d)
      (VG.group_seq (chunk et_d) em' (SZ.v vg));

  VG.cells_restore_group
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
    (chunk et_d) em'
    (fun ij -> in_lane (chunk et_d) (SZ.v rows) (SZ.v cols) (SZ.v lane) ij)
    (SZ.v vg) ();
  fold own_lane_cells
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
    em' (SZ.v lane);
}
#pop-options

(* ---------------------------------------------------------------------------
   drain_band: drain band [idx] of the warp's output fragment (a rows x cols
   tile) to the global output.  Mirror of epilogue_fragment_from_warp, but
   memory-safety only (no C/approx machinery): consumes and restores
   [live_lane_cells], draining each vec-group via [drain_group].
   --------------------------------------------------------------------------- *)
#push-options "--z3rlimit 15"
inline_for_extraction noextract
fn drain_band
  (#et_acc #et_d : Type0)
  {| scalar et_acc, scalar et_d, hvc : has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (post_map : et_acc -> et_d)
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
  (lane : szlt warp_size)
  (#_ : squash (SZ.fits (SZ.v rows * SZ.v cols + Kuiper.Barrier.Warp.warp_size)))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  ()
  preserves
    gpu **
    acc |-> Frac fs eAcc
  requires
    (exists* (bufv : seq et_d). obuf |-> bufv)
  requires
    pure (aligned 16 (T.core gD) /\
          aligned_strided_row_major (chunk et_d) strD /\
          A.length obuf == SZ.v (chunk et_d) /\ aligned 16 obuf)
  requires
    live_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      (SZ.v lane)
  ensures
    (exists* (bufv : seq et_d). obuf |-> bufv)
  ensures
    live_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      (SZ.v lane)
{
  let eTarget : chest2 et_d (SZ.v rows) (SZ.v cols) =
    mk2 #et_d #(SZ.v rows) #(SZ.v cols)
      (fun (a : natlt (SZ.v rows)) (b : natlt (SZ.v cols)) ->
        post_map (acc2 eAcc a b));
  unfold live_lane_cells
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
    (SZ.v lane);
  with (eD0 : chest2 _ _ _).
    assert own_lane_cells
      (output_fragment' gD bm bn rows cols wm wn
        (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
      eD0 (SZ.v lane);
  lane_fade_start eD0 eTarget (SZ.v lane);
  own_lane_cells_rw
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
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
          (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
        (lane_fade eD0 eTarget (SZ.v lane) (SZ.v !vg))
        (SZ.v lane)
    invariant pure (SZ.v !vg % Kuiper.Barrier.Warp.warp_size == SZ.v lane)
    invariant pure (SZ.v !vg <= SZ.v area + Kuiper.Barrier.Warp.warp_size)
    decreases (SZ.v area + Kuiper.Barrier.Warp.warp_size - SZ.v !vg)
  {
    let vvg = !vg;
    drain_group gD post_map obuf bm bn rows cols wm wn
      mrow mcol warpRow warpCol bid wid acc idx lane eD0 eTarget vvg ();
    FStar.Math.Lemmas.add_div_mod_1 (SZ.v vvg) Kuiper.Barrier.Warp.warp_size;
    assert pure (SZ.v vvg < SZ.v vvg + Kuiper.Barrier.Warp.warp_size);
    assert pure (SZ.v (!vg +^ Kuiper.warp_size)
      == SZ.v vvg + Kuiper.Barrier.Warp.warp_size);
    vg := !vg +^ Kuiper.warp_size;
  };

  lane_fade_done eD0 eTarget (SZ.v lane) (SZ.v !vg);
  own_lane_cells_rw
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
    (SZ.v lane)
    (lane_fade eD0 eTarget (SZ.v lane) (SZ.v !vg))
    eTarget;
  fold live_lane_cells
    (output_fragment' gD bm bn rows cols wm wn
      (SZ.v bid) (SZ.v wid) (SZ.v idx / SZ.v wn) (SZ.v idx % SZ.v wn))
    (SZ.v lane);
}
#pop-options

inline_for_extraction noextract
fn store_band
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp { SZ.v nthr == P.nthr bm bn wm wn })
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#fAcc : perm)
  (#ems : erased (seq (value_for et_acc FragAcc frag frag frag)))
  (tid : natlt (SZ.v nthr))
  (wid_sz : SZ.t { SZ.v wid_sz == tid / SZ.v warp_size })
  (i_sz : SZ.t { SZ.v i_sz < mfrag wm })
  (#_ : squash (Pulse.Lib.Array.length accFrags == mfrag wm * nfrag wn))
  ()
  preserves array_fragment_pts_to accFrags #fAcc ems
  preserves scratch_tile_live bm bn bk wm wn skew sh nthr tid
{
  let ld_sz : SZ.t = wn `SZ.add` chunk et_acc;
  assert pure (SZ.v ld_sz == lde et_acc wn);
  (* Destructure [sh] concretely so the scratch band pointer is obtained by a
     tuple projection of the shared-memory tuple; KaRaMeL only emits the
     [(et * ) KPR_SHMEM_AT(..)] C++ cast on that path, not via the [sar_scratch]
     accessor (which leaves a [void *] initialiser nvcc rejects). *)
  let (sA0, (sA1, (sB0, (sB1, (sScratch, su))))) = sh;
  assert rewrites_to sScratch (sar_scratch bm bn bk wm wn skew sh);
  rewrite (scratch_tile_live bm bn bk wm wn skew
             (sA0, (sA1, (sB0, (sB1, (sScratch, su))))) nthr tid)
       as (scratch_tile_live bm bn bk wm wn skew sh nthr tid);
  with eIn. unfold scratch_tile_live bm bn bk wm wn skew sh nthr tid;
  let band = array2_subtile
    (T.from_array
      (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)) sScratch)
    frag (SZ.v wn) (SZ.v wid_sz) 0;
  rewrite each scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size)
    as band;

  let nfrag_sz : SZ.t = wn /^ frag_sz;
  assert pure (SZ.v nfrag_sz == nfrag wn);
  let mut j = 0sz;
  while (!j <^ nfrag_sz)
    invariant live j
    invariant exists* (e : chest2 et_acc frag (SZ.v wn)).
      band |-> Frac (1.0R /. warp_size) e **
      array_fragment_pts_to accFrags #fAcc ems
    invariant pure (SZ.v !j <= nfrag wn)
    decreases (nfrag wn - SZ.v !j)
  {
    let jv = !j;
    let sTile = array2_extract_tile_st band frag frag 0 (SZ.v jv);
    let idx : szlt (mfrag wm * nfrag wn) = i_sz *^ nfrag_sz +^ jv;
    array_fragment_pts_to_ref accFrags;
    array_fragment_extract_ro accFrags idx;
    mma_store accFrags.(idx) #_
      #(strided_row_major_subtile
          (subtile_layout
            (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
            frag (SZ.v wn) (SZ.v wid_sz) 0)
          #_ #(strided_row_major_subtile
            (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
            #_ #(srm_l2_skewed_row_major
                   #(warps bm bn wm wn * frag) #(SZ.v wn) #(eskew et_acc) ld_sz)
            frag (SZ.v wn) (SZ.v wid_sz) 0)
          frag frag 0 (SZ.v jv))
      sTile;
    with vf. assert (sTile |-> Frac (1.0R /. warp_size) vf);
    Pulse.Lib.Forall.elim_forall #_ vf;
    ambig_trade_elim ();
    ambig_trade_elim ();
    j := !j +^ 1sz;
  };
  rewrite each band
    as scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size);
  fold scratch_tile_live bm bn bk wm wn skew sh nthr tid;
}

let div_lt_mul (a q d : nat)
  : Lemma (requires d > 0 /\ a < q * d) (ensures a / d < q)
  = if q = 0 then ()
    else begin
      FStar.Math.Lemmas.lemma_div_le a (q * d - 1) d;
      FStar.Math.Lemmas.division_addition_lemma (d - 1) d (q - 1);
      FStar.Math.Lemmas.small_div (d - 1) d;
      assert ((q - 1) * d + (d - 1) == q * d - 1)
    end

let div_rem_one (x : nat) : Lemma (x / 1 == x /\ x % 1 == 0)
  = FStar.Math.Lemmas.lemma_div_mod x 1;
    FStar.Math.Lemmas.lemma_mod_lt x 1

let sz_v_one : squash (SZ.v 1sz == 1) = ()

inline_for_extraction noextract
let epi_scratch_base_ctlayout
  (et_acc : Type0) {| scalar et_acc |} {| has_vec_cpy et_acc |}
  (rows_wf cols : nat)
  (ld : SZ.t {
     SZ.v ld == cols + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_ /\
     SZ.fits rows_wf /\
     SZ.fits (rows_wf * (cols + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_)) })
  : T.ctlayout
      (l2_skewed_row_major rows_wf cols
        (eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_))
  = c_l2_skewed_row_major
      #rows_wf #cols
      #(eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_) ld

inline_for_extraction noextract
let epi_scratch_tile_ctlayout
  (et_acc : Type0) {| scalar et_acc |} {| has_vec_cpy et_acc |}
  (rows_wf : nat)
  (cols wid : SZ.t)
  (ld : SZ.t {
     SZ.v ld == SZ.v cols + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_ /\
     SZ.fits rows_wf /\
     SZ.fits (rows_wf * (SZ.v cols + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_)) /\
     frag /? rows_wf /\ 0 < SZ.v cols /\ SZ.v wid < rows_wf / frag })
  : T.ctlayout
      (subtile_layout
        (l2_skewed_row_major rows_wf (SZ.v cols)
          (eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_))
        frag (SZ.v cols) (SZ.v wid) 0)
  = c_subtile_layout
      (l2_skewed_row_major rows_wf (SZ.v cols)
        (eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_))
      #(c_l2_skewed_row_major
          #rows_wf #(SZ.v cols)
          #(eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_) ld)
      frag (SZ.v cols) (SZ.v wid) 0

#push-options "--split_queries always --z3rlimit 15"
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
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (#_ : squash (Pulse.Lib.Array.length accFrags == mfrag wm * nfrag wn))
  ()
  preserves gpu
  preserves thread_id nthr (SZ.v tid)
  preserves array_fragment_pts_to accFrags #fAcc ems
  preserves output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid)
  preserves scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid)
  preserves pure (aligned 16 (T.core gD))
  preserves pure (aligned_strided_row_major (SZ.v (chunk et_d)) strD)
{
  P.nthr_le_max_threads et_ab et_acc bm bn bk wm wn skew;
  let tid_sz : SZ.t = tid;
  let wid_sz : SZ.t = tid_sz /^ Kuiper.warp_size;
  let lane_sz : SZ.t = tid_sz %^ Kuiper.warp_size;
  let bid_sz : SZ.t = bid;

  // ---- divisibility of the output stride: chunk et_d | 16 | wn | bn | n
  chunk_div16 et_d #_ #_;
  assert pure (16 /? SZ.v wn);
  assert pure (SZ.v wn /? SZ.v bn);
  assert pure (SZ.v bn /? SZ.v n);
  assert pure (chunk et_d /? SZ.v wn);
  assert pure (chunk et_d /? SZ.v bn);
  assert pure (chunk et_d /? SZ.v n);
  assert pure (chunk et_d /?+ SZ.v n);
  assert pure (chunk et_d /?+ SZ.v wn);

  // ---- warp/block coordinates (loop-invariant)
  let blocksN : SZ.t = n /^ bn;
  let mrow : szlt (m /^ bm) = bid_sz /^ blocksN;
  let mcol : szlt blocksN = bid_sz %^ blocksN;
  let warpsN : SZ.t = bn /^ wn;
  assert pure (SZ.v wid_sz < warps bm bn wm wn);
  let warpRow : SZ.t = wid_sz /^ warpsN;
  let warpCol : SZ.t = wid_sz %^ warpsN;

  // ---- band count and the reconciliation [mfrag wm * frag == wm]
  let mfrag_wm_sz : szp = wm /^ frag_sz;
  assert pure (SZ.v mfrag_wm_sz == mfrag wm);

  // ---- skewed shared-tile leading dimension (row stride)
  let ld_sz : SZ.t = wn `SZ.add` chunk et_acc;
  assert pure (SZ.v ld_sz == lde et_acc wn);

  let zd : et_d = zero #et_d;
  // NOTE: the length must be written inline, not via a `let`-bound size: a
  // let-bound length extracts as a non-constant stack-array bound, which
  // KaRaMeL rejects.
  let mut obuf = [| zd; chunk et_d |];
  A.pts_to_len obuf;
  local_aligned16 #et_d obuf;
  {
    let mut i : sz = 0sz;
      while (!i <^ mfrag_wm_sz)
        invariant live i
        invariant
          gpu **
          thread_id nthr (SZ.v tid) **
          array_fragment_pts_to accFrags #fAcc ems **
          output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid) **
          scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid) **
          (exists* (bufv : seq et_d). obuf |-> bufv)
        invariant pure (aligned 16 (T.core gD) /\ SZ.v !i <= mfrag wm /\
                        aligned_strided_row_major (SZ.v (chunk et_d)) strD /\
                        A.length obuf == SZ.v (chunk et_d) /\ aligned 16 obuf)
        decreases (mfrag wm - SZ.v !i)
      {
        let iv = !i;
    
        // barrier 1: __syncwarp before overwriting the shared band
        warp_barrier_wait () warp_emp_pred warp_emp_pred warp_emp_proof
          #(SZ.v nthr) #(SZ.v tid);
    
        // store all nfrag accumulator fragments of band [iv] into the band
        store_band bm bn bk wm wn skew nthr sh accFrags (SZ.v tid) wid_sz iv ();
    
        // barrier 2: __syncwarp before reading the band back
        warp_barrier_wait () warp_emp_pred warp_emp_pred warp_emp_proof
          #(SZ.v nthr) #(SZ.v tid);
    
        // extract this band's [live_lane_cells] instance (mi = iv, nj = 0)
        unfold output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid);
        Kuiper.ForEvery.forevery_extract_2
          #(natlt (mfrag wm)) #(natlt 1)
          (SZ.v iv <: natlt (mfrag wm)) (0 <: natlt 1)
          (fun (mi : natlt (mfrag wm)) (nj : natlt 1) ->
            live_lane_cells
              (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
                (SZ.v bid) (SZ.v tid / Kuiper.Barrier.Warp.warp_size) mi nj)
              (SZ.v tid % Kuiper.Barrier.Warp.warp_size));
    
        // expose the shared band as an array2 [acc]; destructure [sh] concretely
        // so the band pointer is a tuple projection (emits the C++ cast).
        let (sA0, (sA1, (sB0, (sB1, (sScratch, su))))) = sh;
        assert rewrites_to sScratch (sar_scratch bm bn bk wm wn skew sh);
        rewrite (scratch_tile_live bm bn bk wm wn skew
                   (sA0, (sA1, (sB0, (sB1, (sScratch, su))))) nthr (SZ.v tid))
             as (scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid));
        with eAcc0. unfold scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid);
        let band = array2_subtile
          (T.from_array
            (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)) sScratch)
          frag (SZ.v wn) (SZ.v wid_sz) 0;
        rewrite each scratch_tile bm bn bk wm wn skew sh (SZ.v tid / SZ.v warp_size)
          as band;

        assert pure (SZ.sizet_to_nat frag_sz == frag);
        assert pure (SZ.sizet_to_nat mfrag_wm_sz == mfrag wm);
        assert pure (SZ.sizet_to_nat 1sz == 1);
        assert pure (SZ.v bid_sz == SZ.v bid);
        assert pure (SZ.v wid_sz == SZ.v tid / Kuiper.Barrier.Warp.warp_size);
        assert pure (SZ.v lane_sz == SZ.v tid % Kuiper.Barrier.Warp.warp_size);
        assert pure (SZ.v 1sz == 1);
        div_rem_one (SZ.v iv);
        assert pure (SZ.v iv / SZ.v 1sz == SZ.v iv);
        assert pure (SZ.v iv % SZ.v 1sz == 0);
        assert pure (SZ.v frag_sz == frag);
        assert pure (SZ.v mfrag_wm_sz == mfrag wm);
        assert pure (SZ.v iv < SZ.v mfrag_wm_sz);
        assert pure (SZ.v iv < mfrag wm);
        assert pure (SZ.v iv / SZ.v 1sz < SZ.v mfrag_wm_sz);

        rewrite
          live_lane_cells
            (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
              (SZ.v bid) (SZ.v tid / Kuiper.Barrier.Warp.warp_size) (SZ.v iv) 0)
            (SZ.v tid % Kuiper.Barrier.Warp.warp_size)
        as
          live_lane_cells
            (output_fragment' gD bm bn frag_sz wn mfrag_wm_sz 1sz
              (SZ.v bid_sz) (SZ.v wid_sz) (SZ.v iv / SZ.v 1sz) (SZ.v iv % SZ.v 1sz))
            (SZ.v lane_sz);
    
        assert pure (SZ.fits (warps bm bn wm wn * frag * lde et_acc wn));
        assert pure (SZ.fits (warps bm bn wm wn * frag));
        assert pure (SZ.fits ((warps bm bn wm wn * frag) * (SZ.v wn + eskew et_acc)));
        assert pure (SZ.v ld_sz == SZ.v wn + eskew et_acc);

        // Force [sized et_acc] to the scalar-derived witness so our explicitly
        // built layout matches the band's layout (which resolves it the same
        // way); otherwise [sized] is ambiguous (scalar vs. has_vec_cpy) and the
        // ctlayout implicit fails to unify.

        let ld_sz2 : (ld : SZ.t {
            SZ.v ld == SZ.v wn + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_ /\
            SZ.fits (warps bm bn wm wn * frag) /\
            SZ.fits ((warps bm bn wm wn * frag) * (SZ.v wn + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_)) }) = ld_sz;

        // Build the base skewed-matrix ctlayout via a top-level pure helper: the
        // [reveal (hide _)] that [c_l2_skewed_row_major]'s erased indices produce
        // does not reduce against a bare layout inside a Pulse tot-bind, but it
        // does at module level, so the helper hands back a clean bare-form type.
        assert pure (frag /? (warps bm bn wm wn * frag));
        assert pure (0 < SZ.v wn);
        FStar.Math.Lemmas.cancel_mul_div (warps bm bn wm wn) frag;
        assert pure (SZ.v wid_sz < (warps bm bn wm wn * frag) / frag);

        // Recompute the block/warp coordinates *inside* the closure: the outer
        // let-bound coordinate equations do not survive across the
        // block boundary, so we rebuild them here where
        // their defining equations are in scope for drain_band's coord squash.
        assert pure (SZ.v bid_sz < SZ.v nblk);
        assert pure (SZ.v nblk == (SZ.v m / SZ.v bm) * (SZ.v n / SZ.v bn));
        div_lt_mul (SZ.v bid_sz) (SZ.v m / SZ.v bm) (SZ.v n / SZ.v bn);
        let blocksN : SZ.t = n /^ bn;
        let mrow : szlt (m /^ bm) = bid_sz /^ blocksN;
        let mcol : szlt blocksN = bid_sz %^ blocksN;
        let warpsN : SZ.t = bn /^ wn;
        let warpRow : SZ.t = wid_sz /^ warpsN;
        let warpCol : SZ.t = wid_sz %^ warpsN;

        // [SZ.rem] gives the [mod_spec] form [a - (a/b)*b]; convert to [a % b]
        assert pure (SZ.v blocksN > 0);
        assert pure (SZ.v warpsN > 0);
        FStar.Math.Lemmas.lemma_div_mod (SZ.v bid_sz) (SZ.v blocksN);
        FStar.Math.Lemmas.lemma_div_mod (SZ.v wid_sz) (SZ.v warpsN);

        // reconcile drain_band's [cols = 1sz] coord squash: [wn * 1 == wn]
        assert pure (SZ.v wn * SZ.v 1sz == SZ.v wn);
        assert pure (SZ.v bn / (SZ.v wn * SZ.v 1sz) == SZ.v bn / SZ.v wn);
        assert pure (SZ.v warpsN == SZ.v bn / SZ.v wn);
        assert pure (SZ.v warpRow == SZ.v wid_sz / SZ.v warpsN);
        assert pure (SZ.v warpCol == SZ.v wid_sz % SZ.v warpsN);
        assert pure (SZ.v warpRow == SZ.v wid_sz / (SZ.v bn / (SZ.v wn * SZ.v 1sz)));
        assert pure (SZ.v warpCol == SZ.v wid_sz % (SZ.v bn / (SZ.v wn * SZ.v 1sz)));
        assert pure (SZ.v mrow == SZ.v bid_sz / (SZ.v n / SZ.v bn));
        assert pure (SZ.v blocksN > 0);
        assert pure (SZ.v blocksN == SZ.v n / SZ.v bn);
        assert pure (SZ.v mcol == SZ.v bid_sz % SZ.v blocksN);
        assert pure (SZ.v mcol == SZ.v bid_sz % (SZ.v n / SZ.v bn));

        // divisibility / warp-bound facts for drain_band's refined params
        assert pure (SZ.v mfrag_wm_sz * SZ.v frag_sz == SZ.v wm);
        assert pure (SZ.v wm /?+ SZ.v bm);
        assert pure (SZ.v wn /?+ SZ.v bn);
        assert pure (SZ.v bm / (SZ.v mfrag_wm_sz * SZ.v frag_sz) == SZ.v bm / SZ.v wm);
        assert pure (warps_n bn wn > 0);
        div_lt_mul (SZ.v wid_sz) (warps_m bm wm) (warps_n bn wn);
        FStar.Math.Lemmas.lemma_mod_lt (SZ.v wid_sz) (warps_n bn wn);
        assert pure (SZ.v warpRow < SZ.v bm / SZ.v wm);
        assert pure (SZ.v warpCol < SZ.v bn / SZ.v wn);
        assert pure (SZ.v iv < SZ.v mfrag_wm_sz * SZ.v 1sz);

        drain_band #et_acc #et_d
          gD post_map obuf
          bm bn frag_sz wn mfrag_wm_sz 1sz
          mrow mcol warpRow warpCol bid_sz wid_sz
          #_
          #(subtile_layout
              (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn)
                (eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_))
              frag (SZ.v wn) (SZ.v wid_sz) 0)
          #(epi_scratch_tile_ctlayout et_acc (warps bm bn wm wn * frag) wn wid_sz ld_sz2)
          band
          iv lane_sz ();
    
        rewrite
          live_lane_cells
            (output_fragment' gD bm bn frag_sz wn mfrag_wm_sz 1sz
              (SZ.v bid_sz) (SZ.v wid_sz) (SZ.v iv / SZ.v 1sz) (SZ.v iv % SZ.v 1sz))
            (SZ.v lane_sz)
        as
          live_lane_cells
            (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
              (SZ.v bid) (SZ.v tid / Kuiper.Barrier.Warp.warp_size) (SZ.v iv) 0)
            (SZ.v tid % Kuiper.Barrier.Warp.warp_size);
    
        // re-fold the shared band
        rewrite each band
          as scratch_tile bm bn bk wm wn skew sh (SZ.v tid / SZ.v warp_size);
        fold scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid);
    
        // re-insert this band's cells into the [forall+], re-fold output_lane_live
        Pulse.Lib.Trade.elim_trade _ _;
        fold output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid);
    
        i := !i +^ 1sz;
      }
  };
}

#pop-options

