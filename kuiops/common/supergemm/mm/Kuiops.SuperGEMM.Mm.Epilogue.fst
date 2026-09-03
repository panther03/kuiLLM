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
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc
  { own_lane_cells, live_lane_cells, in_lane }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', output_fragment', output_lane_approximates' }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile, warp_tile }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueCell
  { tiled_cell }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep
  { lane_coincide, own_lane_cells_rw, lane_fade, lane_fade_start, lane_fade_done }

open Kuiops.Array2.Vectorized { row_cells }
open Kuiper.Tensor.Tiling { array2_subtile, array2_extract_tile_st, subtile_layout, c_subtile_layout, cell_convert_eq }
open Kuiops.Array2.Strided
  { strided_row_major, strided_row_major_l2_row_major, strided_row_major_subtile,
    cell_of_pos, aligned_strided_row_major, to_kuiper_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { mk2, acc2, chest2, chest_map }
open Kuiper.EMatrix { lemma_approximates_intro }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }

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
module P = Kuiops.SuperGEMM.Mm.Params
module VG = Kuiops.Array2.Vectorized.Group
module ML = FStar.Math.Lemmas

(* ---------------------------------------------------------------------------
   Step-6 pure lemmas: relate the fp32 scratch band content to the real
   accumulator tile [rAcc] and carry [post_map %~ post_map_r] through the cast.
   --------------------------------------------------------------------------- *)

(* Cell reduction for [mk2]: needed because a [let]-bound [mk2 f] value hides
   its [mk] head from the [acc_pat] SMTPat, so [acc2 (mk2 f) a b] does not
   auto-reduce to [f a b] at use sites. *)
let acc2_mk2 (#et : Type) (#d0 #d1 : nat)
  (f : natlt d0 -> natlt d1 -> GTot et) (a : natlt d0) (b : natlt d1)
  : Lemma (acc2 (mk2 f) a b == f a b)
          [SMTPat (acc2 (mk2 f) a b)]
= ()

let tile_index_lt
  (extent : nat)
  (tile : pos { tile /?+ extent })
  (i : nat)
  (_ : squash (i * tile + tile <= extent))
  : Lemma (i < extent / tile)
= Kuiper.Divides.lemma_nat_divides_pos_divides tile extent;
  Kuiper.Divides.lemma_divides_exact tile extent;
  if extent / tile <= i then
    ML.lemma_mult_le_left tile (extent / tile) i

(* [ematrix_subtile] of a [frag x wn] row-band, sliced at column [j], is the
   [frag x frag] accumulator subtile [(iv, j)] of the whole warp tile. *)
let subtile_band_frag
  (#et : Type0)
  (frag wn : pos)
  (rows cols : nat)
  (rAcc : chest2 et rows cols)
  (iv j : nat)
  (_ : squash (frag /? rows /\ wn /? cols /\ frag /? wn /\
               iv * frag + frag <= rows /\ j * frag + frag <= wn /\
               (j * frag) < cols /\ iv < rows / frag /\
               j < wn / frag /\ j < cols / frag))
  : Lemma
      (ematrix_subtile (ematrix_subtile rAcc frag wn iv 0) frag frag 0 j
       == ematrix_subtile rAcc frag frag iv j)
= tile_index_lt rows frag iv ();
  tile_index_lt wn frag j ();
  let iv' : natlt (rows / frag) = iv in
  let jw : natlt (wn / frag) = j in
  assert (cols > 0);
  Kuiper.Divides.lemma_divides_trans frag wn cols;
  Kuiper.Divides.lemma_divides_le wn cols;
  assert (j * frag + frag <= cols);
  tile_index_lt cols frag j ();
  let jc : natlt (cols / frag) = j in
  introduce forall (a : natlt frag) (b : natlt frag).
    acc2 (ematrix_subtile (ematrix_subtile rAcc frag wn iv' 0)
                          frag frag 0 jw) a b
    == acc2 (ematrix_subtile rAcc frag frag iv' jc) a b
  with assert_norm (
    acc2 (ematrix_subtile (ematrix_subtile rAcc frag wn iv' 0)
                          frag frag 0 jw) a b
    == acc2 (ematrix_subtile rAcc frag frag iv' jc) a b);
  Kuiper.Chest.ext
    (ematrix_subtile (ematrix_subtile rAcc frag wn iv' 0) frag frag 0 jw)
    (ematrix_subtile rAcc frag frag iv' jc)

(* A [frag x wn] band approximates the real band [rBand] as soon as each of its
   [nfrag] [frag x frag] slices approximates the matching slice of [rBand]. *)
let band_approx_from_subtiles
  (#et : Type0) {| scalar et, real_like et |}
  (frag wn : pos)
  (e : chest2 et frag wn)
  (rBand : chest2 real frag wn)
  (_ : squash (frag /? wn))
  (_ : squash (forall (j : natlt (wn / frag)).
        ematrix_subtile e frag frag 0 j %~ ematrix_subtile rBand frag frag 0 j))
  : Lemma (e %~ rBand)
= introduce forall (a : natlt frag) (b : natlt wn). acc2 e a b %~ acc2 rBand a b
  with begin
    let j = b / frag in
    let b' = b % frag in
    ML.lemma_div_mod b frag;
    ML.div_exact_r wn frag;
    introduce (b / frag) >= (wn / frag) ==> False
    with ML.lemma_mult_le_right frag (wn / frag) (b / frag);
    assert (j < wn / frag);
    assert (b == j * frag + b');
    assert (acc2 (ematrix_subtile e frag frag 0 j) a b'
            == acc2 e a (j * frag + b'));
    assert (acc2 (ematrix_subtile rBand frag frag 0 j) a b'
            == acc2 rBand a (j * frag + b'))
  end;
  lemma_approximates_intro e rBand

(* [eTarget = post_map . e] approximates [post_map_r . rBand] whenever
   [e %~ rBand] and [post_map %~ post_map_r]. *)
let map_approx
  (#et1 #et2 : Type0) {| scalar et1, real_like et1, scalar et2, real_like et2 |}
  (frag wn : pos)
  (post_map : et1 -> et2)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (e : chest2 et1 frag wn)
  (rBand : chest2 real frag wn)
  (_ : squash (e %~ rBand))
  : Lemma (chest_map post_map e %~ chest_map post_map_r rBand)
= introduce forall (a : natlt frag) (b : natlt wn).
    acc2 (chest_map post_map e) a b %~ acc2 (chest_map post_map_r rBand) a b
  with begin
    assert (acc2 e a b %~ acc2 rBand a b);
    assert (acc2 (chest_map post_map e) a b == post_map (acc2 e a b));
    assert (acc2 (chest_map post_map_r rBand) a b == post_map_r (acc2 rBand a b))
  end;
  lemma_approximates_intro (chest_map post_map e) (chest_map post_map_r rBand)

(* The [mk2] cell-wise form of [post_map . eAcc] is propositionally the
   [chest_map] form; proven with literal [mk2] so the [acc2_mk2] SMTPat fires. *)
let target_eq_chest_map
  (#et_acc #et_d : Type0)
  (post_map : et_acc -> et_d)
  (rows cols : nat)
  (eAcc : chest2 et_acc rows cols)
  : Lemma
      (mk2 #et_d #rows #cols
        (fun (a : natlt rows) (b : natlt cols) -> post_map (acc2 eAcc a b))
       == chest_map post_map eAcc)
= let lhs = mk2 #et_d #rows #cols
      (fun (a : natlt rows) (b : natlt cols) -> post_map (acc2 eAcc a b)) in
  introduce forall (a : natlt rows) (b : natlt cols).
    acc2 lhs a b == acc2 (chest_map post_map eAcc) a b
  with begin
    assert (acc2 lhs a b == post_map (acc2 eAcc a b));
    assert (acc2 (chest_map post_map eAcc) a b == post_map (acc2 eAcc a b))
  end;
  Kuiper.EMatrix.lemma_equal_intro lhs (chest_map post_map eAcc);
  Kuiper.Chest.ext lhs (chest_map post_map eAcc)

(* [chest_map] commutes with the column coercion and subtiling on the drain
   target: [rAcc]'s [nfrag*frag]-column form and D's [1*wn] form both mean the
   [frag x wn] band [iv]. *)
let coerced_drain_target_eq_cnf
  (post_map_r : real -> real)
  (rows wn frag cnf : pos)
  (rAcc : chest2 real rows cnf)
  (iv : nat)
  (_ : squash (frag /? rows /\ cnf == wn /\ 1 * wn == wn /\ iv * frag + frag <= rows))
  : Lemma
      (chest_map post_map_r (ematrix_subtile rAcc frag wn iv 0)
       == ematrix_subtile
            (coerce_chest2_cols #real #rows #cnf #(1 * wn) () (chest_map post_map_r rAcc))
            frag wn iv 0)
= let lhs = chest_map post_map_r (ematrix_subtile rAcc frag wn iv 0) in
  let rhs = ematrix_subtile
              (coerce_chest2_cols #real #rows #cnf #(1 * wn) () (chest_map post_map_r rAcc))
              frag wn iv 0 in
  assert (forall (a : natlt frag) (b : natlt wn). acc2 lhs a b == acc2 rhs a b);
  Kuiper.Chest.ext lhs rhs

(* ---------------------------------------------------------------------------
   Pure helper lemmas copied from
   Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueChunkUpdate
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
   Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.EpilogueStep.
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
  with (
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
  (idxRow : szlt wm)
  (idxCol : szlt wn)
  (#_ : squash (SZ.v idx == SZ.v idxRow * SZ.v wn + SZ.v idxCol))
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
  assert pure (SZ.v idxRow + 1 <= SZ.v wm);
  ML.distributivity_add_left (SZ.v idxRow) 1 (SZ.v rows);
  ML.lemma_mult_le_right (SZ.v rows)
    (SZ.v idxRow + 1) (SZ.v wm);
  assert pure (
    (SZ.v idxRow) * SZ.v rows + SZ.v row
    < SZ.v wm * SZ.v rows);
  ML.div_exact_r (SZ.v bm) (SZ.v wm * SZ.v rows);
  assert pure (SZ.v warpRow + 1 <=
    SZ.v bm / (SZ.v wm * SZ.v rows));
  ML.distributivity_add_left (SZ.v warpRow) 1
    (SZ.v wm * SZ.v rows);
  ML.lemma_mult_le_right (SZ.v wm * SZ.v rows)
    (SZ.v warpRow + 1) (SZ.v bm / (SZ.v wm * SZ.v rows));
  assert pure (
    SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idxRow) * SZ.v rows + SZ.v row
    < SZ.v bm);
  assert pure (
    SZ.v warpCol * (SZ.v wn * SZ.v cols)
      + (SZ.v idxCol) * SZ.v cols + SZ.v col + (chunk et_d)
    <= SZ.v bn);
  ML.div_exact_r (SZ.v m) (SZ.v bm);
  assert pure (SZ.v mrow + 1 <= SZ.v m / SZ.v bm);
  ML.distributivity_add_left (SZ.v mrow) 1 (SZ.v bm);
  ML.lemma_mult_le_right (SZ.v bm) (SZ.v mrow + 1) (SZ.v m / SZ.v bm);
  assert pure (
    SZ.v mrow * SZ.v bm + SZ.v warpRow * (SZ.v wm * SZ.v rows)
      + (SZ.v idxRow) * SZ.v rows + SZ.v row
    < SZ.v m);
  ML.div_exact_r (SZ.v n) (SZ.v bn);
  assert pure (SZ.v mcol + 1 <= SZ.v n / SZ.v bn);
  ML.distributivity_add_left (SZ.v mcol) 1 (SZ.v bn);
  ML.lemma_mult_le_right (SZ.v bn) (SZ.v mcol + 1) (SZ.v n / SZ.v bn);
  assert pure (
    SZ.v mcol * SZ.v bn +
      (SZ.v warpCol * (SZ.v wn * SZ.v cols)
       + (SZ.v idxCol) * SZ.v cols + SZ.v col + (chunk et_d))
    <= (SZ.v mcol + 1) * SZ.v bn);
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
  row_cells_frag_to_global gD
    (SZ.v bm) (SZ.v bn) (SZ.v rows) (SZ.v cols) (SZ.v wm) (SZ.v wn)
    (SZ.v bid) (SZ.v wid) (SZ.v idxRow) (SZ.v idxCol)
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
   tile) to the global output.  Mirror of epilogue_fragment_from_warp, but
   memory-safety only (no C/approx machinery): consumes and restores
   [live_lane_cells], draining each vec-group via [drain_group].
   --------------------------------------------------------------------------- *)
#push-options "--z3rlimit 15"
inline_for_extraction noextract
fn drain_band
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, scalar et_d, real_like et_d, hvc : has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
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
          A.length obuf == SZ.v (chunk et_d) /\ aligned 16 obuf /\
          reveal eAcc %~ rBand)
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
      pure (eD %~ chest_map post_map_r rBand))
{
  let eTarget : chest2 et_d (SZ.v rows) (SZ.v cols) =
    mk2 #et_d #(SZ.v rows) #(SZ.v cols)
      (fun (a : natlt (SZ.v rows)) (b : natlt (SZ.v cols)) ->
        post_map (acc2 eAcc a b));
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
    drain_group gD post_map obuf bm bn rows cols wm wn
      mrow mcol warpRow warpCol bid wid acc idx idxRow idxCol #_
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
  (* [eTarget = post_map . eAcc]; with [eAcc %~ rBand] and [post_map %~
     post_map_r], [eTarget %~ post_map_r . rBand]. *)
  map_approx #et_acc #et_d (SZ.v rows) (SZ.v cols) post_map post_map_r
    eAcc rBand ();
  target_eq_chest_map #et_acc #et_d post_map (SZ.v rows) (SZ.v cols) eAcc;
}
#pop-options

(* Per-fragment approximation fact for the accumulator content [em0], packaged
   so its (nonlinear) flat-index bound [i*nfrag wn+j < mfrag wm*nfrag wn] is
   discharged once at this top-level definition.  Restating the raw [forall]
   inside the epilogue's [withlocal] block fails to re-elaborate that bound. *)
let em0_frag_approx
  (#et_acc : Type0) {| scalar et_acc, real_like et_acc |}
  (wm wn : szp)
  (em0 : seq (value_for et_acc FragAcc frag frag frag))
  (rAcc : chest2 real (mfrag wm * frag) (nfrag wn * frag))
  : prop
= Seq.length em0 == mfrag wm * nfrag wn /\
  (forall (i : natlt (mfrag wm)) (j : natlt (nfrag wn)).
     Seq.index em0 (i * nfrag wn + j) %~ ematrix_subtile rAcc frag frag i j)

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

let div_lt_mul (a q d : nat)
  : Lemma (requires d > 0 /\ a < q * d) (ensures a / d < q)
  = if q = 0 then ()
    else begin
      FStar.Math.Lemmas.lemma_div_le a (q * d - 1) d;
      FStar.Math.Lemmas.division_addition_lemma (d - 1) d (q - 1);
      FStar.Math.Lemmas.small_div (d - 1) d;
      assert ((q - 1) * d + (d - 1) == q * d - 1)
    end

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

#push-options " --z3rlimit 15"

(* Swap the (eD-independent) drain target [rA] under an [exists* eD] for an
   equal chest [rB].  Needed because a [rewrite]'s slprop-equivalence cannot see
   the [rA == rB] fact through the [exists*]/[pure] binder, but opening the
   witness and re-closing discharges it by SMT congruence. *)
ghost
fn swap_lane_target
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et_d l) (lane : natlt Kuiper.Barrier.Warp.warp_size)
  (rA rB : chest2 real rows cols)
  (_ : squash (rA == rB))
  requires (exists* (eD : chest2 et_d rows cols). own_lane_cells m eD lane ** pure (eD %~ rA))
  ensures  (exists* (eD : chest2 et_d rows cols). own_lane_cells m eD lane ** pure (eD %~ rB))
{
  ()
}
#pop-options

(* Per-band cell predicates, packaged at top level so their bodies (which carry
   the nonlinear [output_fragment']/[ematrix_subtile] index arithmetic and the
   [natlt]/[pos] bound checks) are elaborated once here, where nonlinear
   reasoning is available, rather than re-elaborated inside the epilogue's
   [withlocal] block (where it fails as "ill-typed").  The bounds are threaded as
   EXPLICIT [squash] witnesses so the call sites pass only simple [szp]/[szlt]
   values -- an explicit squash yields an SMT obligation, not a typing
   re-elaboration, sidestepping the block's ill-typedness.
   [epi_live_band] = the lane still owns band [mi]'s output cells;
   [epi_approx_band] = it has drained them and they approximate [rD]'s band. *)
let epi_bounds_ok (bm bn wm wn m n nblk nthr : szp) : prop =
  SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\ SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
  mfrag wm * frag == SZ.v wm /\ 1 * SZ.v wn == SZ.v wn /\
  SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn) /\
  SZ.v nthr == SZ.v bm / SZ.v wm * (SZ.v bn / SZ.v wn) * warp_size

let epi_live_band
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n : szp) (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn nblk nthr : szp)
  (sq : squash (epi_bounds_ok bm bn wm wn m n nblk nthr))
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (mi : natlt (mfrag wm))
  : slprop
= live_lane_cells
    (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
      (SZ.v bid) (SZ.v tid / warp_size) mi 0)
    (SZ.v tid % warp_size)

(* [own_lane_approx m lane rTile]: opaque wrapper for "the lane owns cells of [m]
   that approximate real tile [rTile]".  Keyed on [m] ([mkey]) so Pulse frame-
   matches by the [output_fragment'] pointer and discharges index-coordinate
   equalities by congruence, hiding the inner [exists*] the same way
   [live_lane_cells] does -- which is what makes the sizet<->nat reshape a plain
   [rewrite] rather than an [exists*] equality Pulse cannot prove directly. *)
let own_lane_approx
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  ([@@@mkey] m : array2 et l)
  (lane : natlt warp_size)
  (rTile : chest2 real rows cols)
  : slprop
= exists* (em : chest2 et rows cols).
    own_lane_cells m em lane ** pure (em %~ rTile)

let epi_approx_band
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n : szp) (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn nblk nthr : szp)
  (sq : squash (epi_bounds_ok bm bn wm wn m n nblk nthr))
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (rD : chest2 real (mfrag wm * frag) (1 * SZ.v wn))
  (mi : natlt (mfrag wm))
  : slprop
= own_lane_approx
    (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
      (SZ.v bid) (SZ.v tid / warp_size) mi 0)
    (SZ.v tid % warp_size)
    (ematrix_subtile rD frag (SZ.v wn) mi 0)

let natlt1_singleton () : Lemma (forall (y : natlt 1). (0 <: natlt 1) == y) = ()

(* [epi_mixed ... kk mi]: band [mi] is drained ([epi_approx_band]) for [mi < kk],
   still live ([epi_live_band]) otherwise.  Top-level (not a local lambda) so its
   applications stay opaque: an inlined local [mixed] would beta-reduce to a bare
   [if], breaking the frame-match against the [forevery_extract_if] leftover which
   keeps [mixed] applied. *)
let epi_mixed
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n : szp) (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn nblk nthr : szp)
  (sq : squash (epi_bounds_ok bm bn wm wn m n nblk nthr))
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (rD : chest2 real (mfrag wm * frag) (1 * SZ.v wn))
  (kk : nat)
  (mi : natlt (mfrag wm))
  : slprop
= if mi < kk
  then epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD mi
  else epi_live_band gD bm bn wm wn nblk nthr () bid tid mi

(* Off-diagonal invariance of the pivot: shifting [kk -> kk+1] leaves band [mi]
   unchanged when [mi <> kk] (it stays on the same side of the pivot).  Used to
   re-establish the drain loop invariant after draining band [kk]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
let nat_lt_or_gt (a b : nat)
  : Lemma (requires a <> b) (ensures a < b \/ a > b)
= ()

let epi_mixed_shift
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n : szp) (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn nblk nthr : szp)
  (sq : squash (epi_bounds_ok bm bn wm wn m n nblk nthr))
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (rD : chest2 real (mfrag wm * frag) (1 * SZ.v wn))
  (kk : nat)
  (mi : natlt (mfrag wm))
  : Lemma (requires mi <> kk)
          (ensures epi_mixed gD bm bn wm wn nblk nthr sq bid tid rD kk mi
                == epi_mixed gD bm bn wm wn nblk nthr sq bid tid rD (kk + 1) mi)
= nat_lt_or_gt mi kk;
  if mi < kk then begin
    assert (mi < kk + 1);
    ()
  end else begin
    assert (mi > kk);
    assert (~(mi < kk + 1));
    ()
  end
#pop-options

(* [forall]-lifted pivot-shift: discharges the [squash] argument of
   [forevery_extract_replace_eqtype] in one shot. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
let epi_mixed_shift_all
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n : szp) (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn nblk nthr : szp)
  (sq : squash (epi_bounds_ok bm bn wm wn m n nblk nthr))
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (rD : chest2 real (mfrag wm * frag) (1 * SZ.v wn))
  (kk : nat)
  : Lemma
      (forall (mi : natlt (mfrag wm)). mi =!= kk ==>
         epi_mixed gD bm bn wm wn nblk nthr sq bid tid rD kk mi
      == epi_mixed gD bm bn wm wn nblk nthr sq bid tid rD (kk + 1) mi)
= introduce forall (mi : natlt (mfrag wm)). mi =!= kk ==>
      epi_mixed gD bm bn wm wn nblk nthr sq bid tid rD kk mi
   == epi_mixed gD bm bn wm wn nblk nthr sq bid tid rD (kk + 1) mi
  with introduce _ ==> _
  with epi_mixed_shift gD bm bn wm wn nblk nthr sq bid tid rD kk mi
#pop-options

(* Extract element [z] and hand back a trade that, once [z] is re-provided at the
   replaced predicate [p2], re-establishes the whole forever at [p2].  [p1]/[p2]
   are abstract here so the internal [forall+] frame-matches succeed even when the
   concrete predicate (our [epi_mixed ...]) would otherwise be unfolded and
   mismatched by the matcher.
   TODO(upstream): the library already has this as a private
   [forevery_extract_replace_eqtype] in
   [Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueLoopStep]; expose it in that
   module's [.fsti] and drop this copy. *)
#push-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"
ghost
fn forevery_extract_replace_eqtype
  (#a : eqtype)
  (z : a)
  (p1 p2 : a -> slprop)
  (#_ : squash (forall x. x =!= z ==> p1 x == p2 x))
  requires forall+ (x : a). p1 x
  ensures p1 z ** (p2 z @==> forall+ (x : a). p2 x)
{
  forevery_extract_if_eqtype z p1;
  intro_trade #emp_inames
    (p2 z)
    (forall+ (x : a). p2 x)
    (forall+ (x : a). if x = z then emp else p1 x)
    fn _ {
      forevery_map #a
        (fun x -> if x = z then emp else p1 x)
        (fun x -> if x = z then emp else p2 x)
        fn x {
          let is_z = x = z;
          if is_z {
            rewrite (if x = z then emp else p1 x) as emp;
            rewrite emp as (if x = z then emp else p2 x)
          } else {
            rewrite (if x = z then emp else p1 x) as (p1 x);
            rewrite (p1 x) as (p2 x);
            rewrite (p2 x) as (if x = z then emp else p2 x)
          }
        };
      forevery_unextract_if_eqtype z p2
    };
}
#pop-options

(* Expand [forall+ mi. epi_approx_band .. mi] to [output_lane_approximates'].
   Top-level (not inline in [epilogue]) so its body -- which mentions the
   nonlinear [ematrix_subtile rD frag wn mi nj] -- is elaborated here rather than
   re-elaborated inside the epilogue's [withlocal] block, where the nonlinear
   subtile-dimension refinements fail to typecheck. *)
#push-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"
ghost
fn epi_out_gather
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n : szp) (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn nblk nthr : szp)
  (sq : squash (epi_bounds_ok bm bn wm wn m n nblk nthr))
  (bid : szlt (SZ.v nblk))
  (tid : szlt (SZ.v nthr))
  (rD : chest2 real (mfrag wm * frag) (1 * SZ.v wn))
  requires
    forall+ (mi : natlt (mfrag wm)).
      epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD mi
  ensures
    output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
      (SZ.v bid) (SZ.v tid) rD
{
  forevery_map
    #(natlt (mfrag wm))
    (fun (mi : natlt (mfrag wm)) ->
      epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD mi)
    (fun (mi : natlt (mfrag wm)) ->
      forall+ (nj : natlt 1).
        exists* (eD : chest2 et_d frag (SZ.v wn)).
          own_lane_cells
            (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
              (SZ.v bid) (SZ.v tid / warp_size) mi nj)
            eD (SZ.v tid % warp_size) **
          pure (eD %~ ematrix_subtile rD frag (SZ.v wn) mi nj))
    fn mi {
      unfold (epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD mi);
      unfold (own_lane_approx
                (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
                  (SZ.v bid) (SZ.v tid / warp_size) mi 0)
                (SZ.v tid % warp_size)
                (ematrix_subtile rD frag (SZ.v wn) mi 0));
      natlt1_singleton ();
      forevery_singleton_intro'
        (fun (nj : natlt 1) ->
          exists* (eD : chest2 et_d frag (SZ.v wn)).
            own_lane_cells
              (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
                (SZ.v bid) (SZ.v tid / warp_size) mi nj)
              eD (SZ.v tid % warp_size) **
            pure (eD %~ ematrix_subtile rD frag (SZ.v wn) mi nj))
        (0 <: natlt 1);
    };
  fold output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
    (SZ.v bid) (SZ.v tid) rD;
}
#pop-options

#push-options "--z3rlimit 40"
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
{
  P.nthr_le_max_threads et_ab et_acc bm bn bk wm wn skew;
  let tid_sz : SZ.t = tid;
  let wid_sz : SZ.t = tid_sz /^ Kuiper.warp_size;
  let lane_sz : SZ.t = tid_sz %^ Kuiper.warp_size;
  let bid_sz : SZ.t = bid;

  unfold (fragarrayAcc_approximates #et_acc #(solve) #(solve) #frag #frag #frag
            (mfrag wm) (nfrag wn) accFrags rAcc);
  with em0. assert (array_fragment_pts_to accFrags #1.0R em0);

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

  // Band dimension products, bound OUTSIDE the [withlocal] block so the
  // positivity refinement is discharged here (where the [constraints] context
  // is available) rather than re-elaborated inside the block, which fails.
  // They feed a lemma only, so they must be [erased] or they extract as
  // statement-position [Prims_op_Star]/[Prims_op_Division] applications.
  let rows_prod : (p:erased nat{p > 0}) = hide (mfrag wm * frag);
  let cnf_prod : (p:erased nat{p > 0}) = hide (nfrag wn * frag);

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
    // ---- real drain target: [rAcc] retiled from [frag x frag] fragments to
    // [frag x wn] bands, cast by [post_map_r], with the column coercion.
    let rD : chest2 real (mfrag wm * frag) (1 * SZ.v wn) =
      coerce_chest2_cols #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
        (chest_map post_map_r rAcc);

    // ---- per-band predicates: [live_cell] = the lane still owns band [mi]'s
    // cells; [approx_cell] = it has drained them and they approximate [rD]'s
    // band [mi]; [mixed kk] = drained for [mi < kk], live otherwise.
    let live_cell = epi_live_band gD bm bn wm wn nblk nthr () bid tid;
    let mixed = epi_mixed gD bm bn wm wn nblk nthr () bid tid rD;

    // ---- [array_fragment_pts_to accFrags #1.0R em0] is already in context,
    // exposed once at the top of the fn (the [unfold] typechecks there but not
    // inside this [withlocal] block).

    // ---- collapse output_lane_live' (nj : natlt 1) to [forall+ mi. mixed 0 mi]
    unfold output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 (SZ.v bid) (SZ.v tid);
    forevery_map
      #(natlt (mfrag wm))
      (fun (mi : natlt (mfrag wm)) ->
        forall+ (nj : natlt 1).
          live_lane_cells
            (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
              (SZ.v bid) (SZ.v tid / Kuiper.Barrier.Warp.warp_size) mi nj)
            (SZ.v tid % Kuiper.Barrier.Warp.warp_size))
      (fun (mi : natlt (mfrag wm)) -> mixed 0 mi)
      fn mi {
        natlt1_singleton ();
        forevery_singleton_elim'
          (fun (nj : natlt 1) ->
            live_lane_cells
              (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
                (SZ.v bid) (SZ.v tid / Kuiper.Barrier.Warp.warp_size) mi nj)
              (SZ.v tid % Kuiper.Barrier.Warp.warp_size))
          (0 <: natlt 1);
        rewrite
          (live_lane_cells
            (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
              (SZ.v bid) (SZ.v tid / Kuiper.Barrier.Warp.warp_size) mi 0)
            (SZ.v tid % Kuiper.Barrier.Warp.warp_size))
        as (live_cell mi);
        rewrite (live_cell mi) as (mixed 0 mi);
      };

    let mut i : sz = 0sz;
    while (!i <^ mfrag_wm_sz)
      invariant live i
      invariant
        gpu **
        thread_id nthr (SZ.v tid) **
        array_fragment_pts_to accFrags #1.0R em0 **
        scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid) **
        (exists* (bufv : seq et_d). obuf |-> bufv) **
        (forall+ (mi : natlt (mfrag wm)). mixed (SZ.v !i) mi)
      invariant pure (aligned 16 (T.core gD) /\ SZ.v !i <= mfrag wm /\
                      aligned_strided_row_major (SZ.v (chunk et_d)) strD /\
                      A.length obuf == SZ.v (chunk et_d) /\ aligned 16 obuf /\
                      em0_frag_approx wm wn em0 rAcc)
      decreases (mfrag wm - SZ.v !i)
    {
      let iv = !i;
      // Ghost: consumed only by rewrites/folds.  A concrete binding here
      // extracts as a statement-position [FStar_SizeT_v] call.
      let iv_nat : (x:erased nat{x < mfrag wm}) = hide (SZ.v iv);
      // [chest_map post_map_r (band iv of rAcc)] == [band iv of rD].  Proven
      // up-front (pure, slprop-independent) where the proof state is light; the
      // nonlinear lemma-argument elaboration is fragile deeper in the block.
      coerced_drain_target_eq_cnf post_map_r (reveal rows_prod) (SZ.v wn) frag
        (reveal cnf_prod) rAcc (SZ.v iv) ();
      assert pure (chest_map post_map_r (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v iv) 0)
                   == ematrix_subtile rD frag (SZ.v wn) (SZ.v iv) 0);

      // barrier 1: __syncwarp before overwriting the shared band
      warp_barrier_wait () warp_emp_pred warp_emp_pred warp_emp_proof
        #(SZ.v nthr) #(SZ.v tid);

      // expose the shared band as an array2 [band]; destructure [sh] concretely
      // so the band pointer is a tuple projection (emits the C++ cast).
      let (sA0, (sA1, (sB0, (sB1, (sScratch, su))))) = sh;
      assert rewrites_to sScratch (sar_scratch bm bn bk wm wn skew sh);
      rewrite (scratch_tile_live bm bn bk wm wn skew
                 (sA0, (sA1, (sB0, (sB1, (sScratch, su))))) nthr (SZ.v tid))
           as (scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid));
      unfold scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid);
      let band = array2_subtile
        (T.from_array
          (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)) sScratch)
        frag (SZ.v wn) (SZ.v wid_sz) 0;
      rewrite each scratch_tile bm bn bk wm wn skew sh (SZ.v tid / SZ.v warp_size)
        as band;

      // store all nfrag accumulator fragments of band [iv] into the band
      store_band #et_ab #et_acc bm bn bk wm wn skew accFrags rAcc wid_sz iv band ();

      // barrier 2: __syncwarp before reading the band back
      warp_barrier_wait () warp_emp_pred warp_emp_pred warp_emp_proof
        #(SZ.v nthr) #(SZ.v tid);

      // Extract band [iv]'s live cells and get a trade that re-establishes the
      // forever at pivot [iv+1] once band [iv] is drained.  Off-diagonal bands
      // are pivot-invariant ([epi_mixed_shift_all]).  The abstract combinator
      // performs the pivot-shift [forevery_map] on abstract predicates, where
      // the [forall+] frame-match succeeds (a concrete-predicate shift in the
      // drain body's rich context does not).
      epi_mixed_shift_all gD bm bn wm wn nblk nthr () bid tid rD (SZ.v iv);
      forevery_extract_replace_eqtype #(natlt (mfrag wm)) (reveal iv_nat)
        (mixed (SZ.v iv)) (mixed (SZ.v (iv +^ 1sz)));
      rewrite (mixed (SZ.v iv) (reveal iv_nat)) as (live_cell (reveal iv_nat));

      assert pure (SZ.sizet_to_nat frag_sz == frag);
      assert pure (SZ.sizet_to_nat mfrag_wm_sz == mfrag wm);
      assert pure (SZ.sizet_to_nat 1sz == 1);
      assert pure (SZ.v bid_sz == SZ.v bid);
      assert pure (SZ.v wid_sz == SZ.v tid / Kuiper.Barrier.Warp.warp_size);
      assert pure (SZ.v lane_sz == SZ.v tid % Kuiper.Barrier.Warp.warp_size);
      assert pure (SZ.v 1sz == 1);
      assert pure (SZ.v frag_sz == frag);
      assert pure (SZ.v mfrag_wm_sz == mfrag wm);
      assert pure (SZ.v iv < SZ.v mfrag_wm_sz);
      assert pure (SZ.v iv < mfrag wm);

      // [wid] warp-in-block bound, needed by [output_fragment']'s refinement
      // (this nonlinear div fact no longer falls out of the split-query budget
      // on its own here).
      assert pure (warps_n bn wn > 0);
      div_lt_mul (SZ.v wid_sz) (warps_m bm wm) (warps_n bn wn);
      assert pure (SZ.v mfrag_wm_sz * SZ.v frag_sz == SZ.v wm);
      assert pure (warps_m bm wm == SZ.v bm / SZ.v wm);
      assert pure (warps_n bn wn == SZ.v bn / SZ.v wn);

      // reshape [live_cell iv] to drain_band's drain-coordinate form
      rewrite (live_cell (reveal iv_nat))
      as
        (live_lane_cells
          (output_fragment' gD bm bn frag_sz wn mfrag_wm_sz 1sz
            (SZ.v bid_sz) (SZ.v wid_sz) (SZ.v iv) 0)
          (SZ.v lane_sz));

      assert pure (SZ.fits (warps bm bn wm wn * frag * lde et_acc wn));
      assert pure (SZ.fits (warps bm bn wm wn * frag));
      assert pure (SZ.fits ((warps bm bn wm wn * frag) * (SZ.v wn + eskew et_acc)));
      assert pure (SZ.v ld_sz == SZ.v wn + eskew et_acc);

      let ld_sz2 : (ld : SZ.t {
          SZ.v ld == SZ.v wn + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_ /\
          SZ.fits (warps bm bn wm wn * frag) /\
          SZ.fits ((warps bm bn wm wn * frag) * (SZ.v wn + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_)) }) = ld_sz;

      assert pure (frag /? (warps bm bn wm wn * frag));
      assert pure (0 < SZ.v wn);
      FStar.Math.Lemmas.cancel_mul_div (warps bm bn wm wn) frag;
      assert pure (SZ.v wid_sz < (warps bm bn wm wn * frag) / frag);

      assert pure (SZ.v bid_sz < SZ.v nblk);
      assert pure (SZ.v nblk == (SZ.v m / SZ.v bm) * (SZ.v n / SZ.v bn));
      div_lt_mul (SZ.v bid_sz) (SZ.v m / SZ.v bm) (SZ.v n / SZ.v bn);
      let blocksN : SZ.t = n /^ bn;
      let mrow : szlt (m /^ bm) = bid_sz /^ blocksN;
      let mcol : szlt blocksN = bid_sz %^ blocksN;
      let warpsN : SZ.t = bn /^ wn;
      let warpRow : SZ.t = wid_sz /^ warpsN;
      let warpCol : SZ.t = wid_sz %^ warpsN;

      assert pure (SZ.v blocksN > 0);
      assert pure (SZ.v warpsN > 0);
      FStar.Math.Lemmas.lemma_div_mod (SZ.v bid_sz) (SZ.v blocksN);
      FStar.Math.Lemmas.lemma_div_mod (SZ.v wid_sz) (SZ.v warpsN);

      assert pure (SZ.v wn * SZ.v 1sz == SZ.v wn);
      assert pure (SZ.v bn / (SZ.v wn * SZ.v 1sz) == SZ.v bn / SZ.v wn);
      assert pure (SZ.v warpsN == SZ.v bn / SZ.v wn);
      assert pure (SZ.v warpRow == SZ.v wid_sz / SZ.v warpsN);
      assert pure (SZ.v warpCol == SZ.v wid_sz % SZ.v warpsN);
      assert pure (SZ.v warpRow == SZ.v wid_sz / (SZ.v bn / (SZ.v wn * SZ.v 1sz)));
      assert pure (SZ.v warpCol == SZ.v wid_sz % (SZ.v bn / (SZ.v wn * SZ.v 1sz)));
      assert pure (SZ.v mrow == SZ.v bid_sz / (SZ.v n / SZ.v bn));
      assert pure (SZ.v blocksN == SZ.v n / SZ.v bn);
      assert pure (SZ.v mcol == SZ.v bid_sz % SZ.v blocksN);
      assert pure (SZ.v mcol == SZ.v bid_sz % (SZ.v n / SZ.v bn));

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

      // drain band [iv]: read fp32 scratch, cast by [post_map], store to D
      epilogue_band_fits et_ab et_acc bm bn bk wm wn skew;
      drain_band #et_acc #et_d
        gD post_map post_map_r obuf
        bm bn frag_sz wn mfrag_wm_sz 1sz
        mrow mcol warpRow warpCol bid_sz wid_sz
        #_
        #(subtile_layout
            (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn)
              (eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_))
            frag (SZ.v wn) (SZ.v wid_sz) 0)
        #(epi_scratch_tile_ctlayout et_acc (warps bm bn wm wn * frag) wn wid_sz ld_sz2)
        band
        (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v iv) 0)
        iv iv 0sz #_ lane_sz ();

      // reshape the drained cells back to [approx_cell iv], i.e. [mixed (iv+1) iv].
      // First swap the (eD-independent) drain target from the [chest_map . rAcc]
      // form to the [rD] form under the [exists*], then fold the sizet-coordinate
      // drained slprop into the opaque [approx_cell] (= [epi_approx_band]); the
      // sizet<->nat index equalities asserted above bridge the two, exactly as
      // the forward [live_cell] reshape at the top of the loop does.
      swap_lane_target #et_d
        (output_fragment' gD bm bn frag_sz wn mfrag_wm_sz 1sz
          (SZ.v bid_sz) (SZ.v wid_sz) (SZ.v iv) 0)
        (SZ.v lane_sz)
        (chest_map post_map_r (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v iv) 0))
        (ematrix_subtile rD frag (SZ.v wn) (SZ.v iv) 0)
        ();
      // Fold the drained cells into the opaque [own_lane_approx] at sizet
      // coordinates, rewrite it to the canonical nat form by [mkey] congruence
      // (the same asserted index equalities as the forward [live_cell] reshape),
      // then fold into [epi_approx_band].
      fold (own_lane_approx
              (output_fragment' gD bm bn frag_sz wn mfrag_wm_sz 1sz
                (SZ.v bid_sz) (SZ.v wid_sz) (SZ.v iv) 0)
              (SZ.v lane_sz)
              (ematrix_subtile rD frag (SZ.v wn) (SZ.v iv) 0));
      rewrite
        (own_lane_approx
          (output_fragment' gD bm bn frag_sz wn mfrag_wm_sz 1sz
            (SZ.v bid_sz) (SZ.v wid_sz) (SZ.v iv) 0)
          (SZ.v lane_sz)
          (ematrix_subtile rD frag (SZ.v wn) (SZ.v iv) 0))
      as
        (own_lane_approx
          (output_fragment' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
            (SZ.v bid) (SZ.v tid / Kuiper.Barrier.Warp.warp_size) (reveal iv_nat) 0)
          (SZ.v tid % Kuiper.Barrier.Warp.warp_size)
          (ematrix_subtile rD frag (SZ.v wn) (reveal iv_nat) 0));
      fold (epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD (reveal iv_nat));
      rewrite (epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD (reveal iv_nat))
        as (mixed (SZ.v (iv +^ 1sz)) (reveal iv_nat));

      // re-fold the shared band
      rewrite each band
        as scratch_tile bm bn bk wm wn skew sh (SZ.v tid / SZ.v warp_size);
      fold scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid);

      // re-insert band [iv]'s (now drained) cell: eliminate the trade built at
      // extract time. It consumes [mixed (iv+1) iv_nat] and re-establishes the
      // whole forever at pivot [iv+1].
      elim_trade
        (mixed (SZ.v (iv +^ 1sz)) (reveal iv_nat))
        (forall+ (x : natlt (mfrag wm)). mixed (SZ.v (iv +^ 1sz)) x);
      // Increment via [!i], not a body-local name: karamel turns this while
      // into a C [for] and moves the update into the increment slot, where a
      // body-scoped binding would be out of scope.  Everything above is stated
      // over [SZ.v (iv +^ 1sz)] so it matches the post-assignment invariant
      // syntactically.
      i := !i +^ 1sz;
    };

    // ---- re-fold the accumulator fragments' abstraction
    fold fragarrayAcc_approximates (mfrag wm) (nfrag wn) accFrags rAcc;

    // loop exited with [!i == mfrag_wm_sz]; rewrite the invariant's symbolic
    // exit deref to the concrete bound so the forever's index reads [mfrag wm].
    rewrite each !i as mfrag_wm_sz;
    rewrite each (SZ.v mfrag_wm_sz) as (mfrag wm);

    // ---- convert [forall+ mi. mixed (mfrag) mi] to [forall+ mi. epi_approx_band .. mi]
    // (each band is drained at the final pivot, so [mixed (mfrag) mi = epi_approx_band mi]),
    // then hand off to [epi_out_gather], whose body -- carrying the nonlinear
    // [ematrix_subtile] -- lives at top level, outside this [withlocal] block.
    forevery_map
      #(natlt (mfrag wm))
      (fun (mi : natlt (mfrag wm)) -> mixed (mfrag wm) mi)
      (fun (mi : natlt (mfrag wm)) ->
        epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD mi)
      fn mi {
        rewrite (mixed (mfrag wm) mi)
          as (epi_approx_band gD bm bn wm wn nblk nthr () bid tid rD mi);
      };
    epi_out_gather gD bm bn wm wn nblk nthr () bid tid rD;
    // [rD] is a block-local let for the coerced drain target; unfold it so the
    // produced [output_lane_approximates'] matches the fn's postcondition.
    rewrite each rD as
      (coerce_chest2_cols #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
        (chest_map post_map_r rAcc));
  };
}

#pop-options
