module Kuiops.SuperGEMM.Mm.Epi.EpilogueLemmas

(* Pure lemmas and ghost helpers of the epilogue of the C-combining
   ([D = comb C (A @ B^T)]) tensor-core GEMM.

   Split out of [Kuiops.SuperGEMM.Mm.Epi.Epilogue] for the same reason
   [Kuiops.SuperGEMM.Mm.KernelLemmas] was split out of [...Mm.Kernel]: the pure
   parts iterate in seconds, the Pulse parts do not.

   Everything here except [cband], [cband_approx], [comb_approx],
   [target_eq_chest_comb], [coerced_drain_target_eq_comb] and
   [subtile_comb_commute] is a verbatim copy of the C-independent helpers of
   [Kuiops.SuperGEMM.Mm.Epilogue], which has no interface exposing them.
   TODO(upstream): expose them there and drop the copies. *)

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
    cell_of_pos, aligned_strided_row_major }
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { mk2, acc2, chest2, chest_map, chest_comb }
open Kuiper.EMatrix { lemma_approximates_intro }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }

open Kuiper.Concrete { concrete_sz, concrete_sz_sz }
open Pulse.Lib.Trade

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
module SH = Kuiops.SuperGEMM.Mm.Shared

(* Value-preserving coercion between two [chest2] column extents that are
   provably equal (see [Kuiops.SuperGEMM.Mm.Epilogue.coerce_chest2_cols]). *)
inline_for_extraction noextract
let coerce_chest2_cols (#et : Type) (#r #c1 #c2 : nat)
  (_ : squash (c1 == c2)) (x : chest2 et r c1) : chest2 et r c2
= coerce_eq () x
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

(* ---------------------------------------------------------------------------
   The C view.

   C is read SCALARLY at global coordinates, so the epilogue only ever needs
   the [rows x cols] WINDOW of the C chest whose top-left corner is the drain's
   global origin.  [cband] names that window; its in-range obligation is an
   explicit [squash] so the window is total and call sites pass [()].
   --------------------------------------------------------------------------- *)

let cband (#et : Type0) (#d0 #d1 : nat) (eC : chest2 et d0 d1)
  (rows cols crb ccb : nat)
  (_ : squash (crb + rows <= d0 /\ ccb + cols <= d1))
  : chest2 et rows cols
= mk2 (fun (a : natlt rows) (b : natlt cols) -> acc2 eC (crb + a) (ccb + b))

(* A contiguous run in a C window is the same run at the corresponding global
   coordinates.  Isolating this definitional fact keeps the epilogue's large
   Pulse context out of the equality proof. *)
let cband_run_global
  (#et : Type0) (#d0 #d1 : nat) (eC : chest2 et d0 d1)
  (rows cols crb ccb : nat)
  (sq : squash (crb + rows <= d0 /\ ccb + cols <= d1))
  (row : natlt rows) (col : natlt cols) (width : pos)
  (global_row : natlt d0) (global_col : natlt d1)
  (_ : squash (col + width <= cols /\ global_col + width <= d1 /\
               crb + row == global_row /\ ccb + col == global_col))
  : Lemma (forall (x : natlt width).
      acc2 (cband eC rows cols crb ccb sq) row (col + x)
      == acc2 eC global_row (global_col + x))
= ()

let cband_approx (#et : Type0) {| scalar et, real_like et |}
  (#d0 #d1 : nat) (eC : chest2 et d0 d1) (rC : chest2 real d0 d1)
  (rows cols crb ccb : nat)
  (sq : squash (crb + rows <= d0 /\ ccb + cols <= d1))
  (_ : squash (eC %~ rC))
  : Lemma (cband eC rows cols crb ccb sq %~ cband rC rows cols crb ccb sq)
= introduce forall (a : natlt rows) (b : natlt cols).
    acc2 (cband eC rows cols crb ccb sq) a b
    %~ acc2 (cband rC rows cols crb ccb sq) a b
  with assert (acc2 eC (crb + a) (ccb + b) %~ acc2 rC (crb + a) (ccb + b));
  lemma_approximates_intro
    (cband eC rows cols crb ccb sq) (cband rC rows cols crb ccb sq)

(* Band [iv] of a warp's C window is itself a window, shifted down by
   [iv * frag] rows. *)
let cband_subtile (#et : Type0) (#d0 #d1 : nat) (eC : chest2 et d0 d1)
  (rows cnf frag wn : pos)
  (crb0 ccb0 iv : nat)
  (sq : squash (crb0 + rows <= d0 /\ ccb0 + cnf <= d1))
  (_ : squash (frag /? rows /\ cnf == wn /\ iv * frag + frag <= rows))
  : Lemma (ematrix_subtile (cband eC rows cnf crb0 ccb0 sq) frag wn iv 0
           == cband eC frag wn (crb0 + iv * frag) ccb0 ())
= let lhs = ematrix_subtile (cband eC rows cnf crb0 ccb0 sq) frag wn iv 0 in
  let rhs = cband eC frag wn (crb0 + iv * frag) ccb0 () in
  assert (forall (a : natlt frag) (b : natlt wn). acc2 lhs a b == acc2 rhs a b);
  Kuiper.Chest.ext lhs rhs

(* [comb] applied cellwise to two approximating pairs approximates [comb_r]
   applied cellwise -- the binary counterpart of [Mm.Epilogue.map_approx]. *)
let comb_approx
  (#et_c #et_s #et_d : Type0)
  {| scalar et_c, real_like et_c, scalar et_s, real_like et_s,
     scalar et_d, real_like et_d |}
  (rows cols : pos)
  (comb : et_c -> et_s -> et_d)
  (comb_r : real -> real -> real { approx2 comb comb_r })
  (eCb : chest2 et_c rows cols)
  (rCb : chest2 real rows cols)
  (e : chest2 et_s rows cols)
  (rBand : chest2 real rows cols)
  (_ : squash (eCb %~ rCb))
  (_ : squash (e %~ rBand))
  : Lemma (chest_comb comb eCb e %~ chest_comb comb_r rCb rBand)
= introduce forall (a : natlt rows) (b : natlt cols).
    acc2 (chest_comb comb eCb e) a b %~ acc2 (chest_comb comb_r rCb rBand) a b
  with begin
    assert (acc2 eCb a b %~ acc2 rCb a b);
    assert (acc2 e a b %~ acc2 rBand a b);
    assert (acc2 (chest_comb comb eCb e) a b == comb (acc2 eCb a b) (acc2 e a b));
    assert (acc2 (chest_comb comb_r rCb rBand) a b
            == comb_r (acc2 rCb a b) (acc2 rBand a b))
  end;
  lemma_approximates_intro (chest_comb comb eCb e) (chest_comb comb_r rCb rBand)

(* The [mk2] cell-wise form of the drain target is propositionally the
   [chest_comb] form; proven with literal [mk2] so [acc2_mk2] fires. *)
let target_eq_chest_comb
  (#et_c #et_acc #et_d : Type0)
  (comb : et_c -> et_acc -> et_d)
  (rows cols : nat)
  (eCb : chest2 et_c rows cols)
  (eAcc : chest2 et_acc rows cols)
  : Lemma
      (mk2 #et_d #rows #cols
        (fun (a : natlt rows) (b : natlt cols) -> comb (acc2 eCb a b) (acc2 eAcc a b))
       == chest_comb comb eCb eAcc)
= let lhs = mk2 #et_d #rows #cols
      (fun (a : natlt rows) (b : natlt cols) -> comb (acc2 eCb a b) (acc2 eAcc a b)) in
  introduce forall (a : natlt rows) (b : natlt cols).
    acc2 lhs a b == acc2 (chest_comb comb eCb eAcc) a b
  with begin
    assert (acc2 lhs a b == comb (acc2 eCb a b) (acc2 eAcc a b));
    assert (acc2 (chest_comb comb eCb eAcc) a b == comb (acc2 eCb a b) (acc2 eAcc a b))
  end;
  Kuiper.EMatrix.lemma_equal_intro lhs (chest_comb comb eCb eAcc);
  Kuiper.Chest.ext lhs (chest_comb comb eCb eAcc)

(* [chest_comb] commutes with the column coercion and the band subtiling on the
   drain target -- the binary counterpart of
   [Mm.Epilogue.coerced_drain_target_eq_cnf]. *)
let coerced_drain_target_eq_comb
  (comb_r : real -> real -> real)
  (rows wn frag cnf : pos)
  (rCw rAcc : chest2 real rows cnf)
  (iv : nat)
  (_ : squash (frag /? rows /\ cnf == wn /\ 1 * wn == wn /\ iv * frag + frag <= rows))
  : Lemma
      (chest_comb comb_r
         (ematrix_subtile rCw frag wn iv 0) (ematrix_subtile rAcc frag wn iv 0)
       == ematrix_subtile
            (coerce_chest2_cols #real #rows #cnf #(1 * wn) ()
              (chest_comb comb_r rCw rAcc))
            frag wn iv 0)
= let lhs = chest_comb comb_r
      (ematrix_subtile rCw frag wn iv 0) (ematrix_subtile rAcc frag wn iv 0) in
  let rhs = ematrix_subtile
              (coerce_chest2_cols #real #rows #cnf #(1 * wn) ()
                (chest_comb comb_r rCw rAcc))
              frag wn iv 0 in
  assert (forall (a : natlt frag) (b : natlt wn). acc2 lhs a b == acc2 rhs a b);
  Kuiper.Chest.ext lhs rhs

(* The warp tile [block_row, warp_m] fits inside [m]: element-coordinate form
   of [Shared.grow_bound]. *)
let window_bound (m bm wm block_row warp_m : nat)
  : Lemma (requires wm > 0 /\ bm > 0 /\ m > 0 /\ bm % wm == 0 /\ m % bm == 0 /\
                    block_row < m / bm /\ warp_m < bm / wm)
          (ensures block_row * bm + warp_m * wm + wm <= m)
= SH.grow_bound m bm wm block_row warp_m;
  SH.div_compose m bm wm;
  Kuiper.Divides.lemma_divides_trans wm bm m;
  let g : nat = block_row * (bm / wm) + warp_m in
  assert (g < m / wm);
  ML.lemma_div_exact m wm;
  ML.lemma_mult_le_right wm (g + 1) (m / wm);
  ML.swap_mul (m / wm) wm;
  assert ((g + 1) * wm <= m);
  ML.distributivity_add_left g 1 wm;
  ML.distributivity_add_left (block_row * (bm / wm)) warp_m wm;
  ML.paren_mul_right block_row (bm / wm) wm;
  ML.lemma_div_exact bm wm;
  ML.swap_mul (bm / wm) wm

(* A row inside a lane's [rows]-high fragment is still inside the enclosing
   block tile.  Keeping this nested flattening fact out of the Pulse drain
   avoids asking SMT to normalize two exact divisions in its large context. *)
let nested_row_bound (bm wm rows : pos) (warp_row lane_row row : nat)
  : Lemma
      (requires bm % (wm * rows) == 0 /\
                warp_row < bm / (wm * rows) /\ lane_row < wm /\ row < rows)
      (ensures warp_row * (wm * rows) + lane_row * rows + row < bm)
= ML.div_exact_r (wm * rows) rows;
  Kuiper.Divides.lemma_divides_trans rows (wm * rows) bm;
  SH.grow_bound bm (wm * rows) rows warp_row lane_row;
  let g = warp_row * wm + lane_row in
  assert (g < bm / rows);
  ML.lemma_mult_le_right rows (g + 1) (bm / rows);
  ML.lemma_div_exact bm rows;
  ML.distributivity_add_left g 1 rows;
  ML.distributivity_add_left (warp_row * wm) lane_row rows;
  ML.paren_mul_right warp_row wm rows

(* The lane's window of C: the [wm x wn] tile of C the lane's warp is
   responsible for, in the epilogue's reshaped dimensions.  Same [bid]/[tid]
   decoding as [Shared.lane_target], so [Epi.Shared.kpost1] and the epilogue's
   post agree on the C operand by construction. *)
let lane_c_target
  (#et : Type0)
  (#m #n : szp)
  (eC : chest2 et (SZ.v m) (SZ.v n))
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : chest2 et (mfrag wm * frag) (nfrag wn * frag)
=
  let num_n = SZ.v n / SZ.v bn in
  let block_row = bid / num_n in
  let block_col = bid % num_n in
  let wnn = SZ.v bn / SZ.v wn in
  let wid = tid / warp_size in
  let warp_m = wid / wnn in
  let warp_n = wid % wnn in
  SH.div_ub bid (SZ.v m / SZ.v bm) num_n;
  SH.div_ub wid (SZ.v bm / SZ.v wm) wnn;
  window_bound (SZ.v m) (SZ.v bm) (SZ.v wm) block_row warp_m;
  window_bound (SZ.v n) (SZ.v bn) (SZ.v wn) block_col warp_n;
  cband eC (mfrag wm * frag) (nfrag wn * frag)
    (block_row * SZ.v bm + warp_m * SZ.v wm)
    (block_col * SZ.v bn + warp_n * SZ.v wn) ()
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

(* [lane_c_target] in explicit-origin form.  [crb0]/[ccb0] and their in-range
   witness are lemma PARAMETERS so the ensures typechecks without the
   [bid]/[tid] decoding having to be re-derived by the elaborator. *)
let lane_c_target_eq
  (#et : Type0)
  (#m #n : szp)
  (eC : chest2 et (SZ.v m) (SZ.v n))
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  (crb0 ccb0 : nat)
  (sq : squash (crb0 + (mfrag wm * frag) <= SZ.v m /\
                ccb0 + (nfrag wn * frag) <= SZ.v n))
  (_ : squash (
    crb0 == (bid / (SZ.v n / SZ.v bn)) * SZ.v bm
            + ((tid / warp_size) / (SZ.v bn / SZ.v wn)) * SZ.v wm /\
    ccb0 == (bid % (SZ.v n / SZ.v bn)) * SZ.v bn
            + ((tid / warp_size) % (SZ.v bn / SZ.v wn)) * SZ.v wn))
  : Lemma (lane_c_target eC bm bn wm wn nblk nthr bid tid
           == cband eC (mfrag wm * frag) (nfrag wn * frag) crb0 ccb0 sq)
= let lhs = lane_c_target eC bm bn wm wn nblk nthr bid tid in
  let rhs = cband eC (mfrag wm * frag) (nfrag wn * frag) crb0 ccb0 sq in
  assert (forall (a : natlt (mfrag wm * frag)) (b : natlt (nfrag wn * frag)).
    acc2 lhs a b == acc2 rhs a b);
  Kuiper.Chest.ext lhs rhs
