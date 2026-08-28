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

open Kuiops.SuperGEMM.Mm.Epi.Drain { drain_band, store_band }
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

// A fragment row selected inside a warp window remains inside the global
// matrix.  Isolate the exact-division arithmetic from the Pulse loop context.
let band_row_bound
  (m bm wm band : pos) (block_row warp_row band_row : nat)
  : Lemma
      (requires bm % wm == 0 /\ m % bm == 0 /\ wm % band == 0 /\
                block_row < m / bm /\ warp_row < bm / wm /\
                band_row < wm / band)
      (ensures block_row * bm + warp_row * wm + band_row * band + band <= m)
= window_bound m bm wm block_row warp_row;
  ML.lemma_mult_le_right band (band_row + 1) (wm / band);
  ML.lemma_div_exact wm band;
  ML.distributivity_add_left band_row 1 band

#push-options " --z3rlimit 15"
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

  // ---- global origin of this warp's C window, and the [lane_c_target]
  // identity that lets the per-band drain target be read off as a [cband].
  window_bound (SZ.v m) (SZ.v bm) (SZ.v wm) (SZ.v mrow) (SZ.v warpRow);
  window_bound (SZ.v n) (SZ.v bn) (SZ.v wn) (SZ.v mcol) (SZ.v warpCol);
  let crb0 : (x:erased nat) = hide (SZ.v mrow * SZ.v bm + SZ.v warpRow * SZ.v wm);
  let ccb0 : (x:erased nat) =
    hide (SZ.v mcol * SZ.v bn + SZ.v warpCol * (1 * SZ.v wn) + 0 * SZ.v wn);
  assert pure (reveal crb0 ==
    SZ.v mrow * SZ.v bm + SZ.v warpRow * SZ.v wm);
  assert pure (reveal ccb0 ==
    SZ.v mcol * SZ.v bn + SZ.v warpCol * (1 * SZ.v wn) + 0 * SZ.v wn);
  assert pure (reveal crb0 + (mfrag wm * frag) <= SZ.v m);
  assert pure (reveal ccb0 + (nfrag wn * frag) <= SZ.v n);
  lane_c_target_eq rC bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid)
    (reveal crb0) (reveal ccb0) () ();

  // ---- skewed shared-tile leading dimension (row stride)
  let ld_sz : SZ.t = wn `SZ.add` chunk et_acc;
  assert pure (SZ.v ld_sz == lde et_acc wn);
  assert pure (SZ.fits (warps bm bn wm wn * frag * lde et_acc wn));
  assert pure (SZ.fits (warps bm bn wm wn * frag));
  assert pure (
    SZ.fits ((warps bm bn wm wn * frag) * (SZ.v wn + eskew et_acc)));
  assert pure (SZ.v ld_sz == SZ.v wn + eskew et_acc);
  let ld_sz2 : (ld : SZ.t {
      SZ.v ld == SZ.v wn + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_ /\
      SZ.fits (warps bm bn wm wn * frag) /\
      SZ.fits ((warps bm bn wm wn * frag) *
        (SZ.v wn + eskew et_acc #(Kuiper.Scalars.Base.is_sized #et_acc) #_)) }) = ld_sz;

  let zd : et_d = zero #et_d;
  // NOTE: the length must be written inline, not via a `let`-bound size: a
  // let-bound length extracts as a non-constant stack-array bound, which
  // KaRaMeL rejects.
  let mut obuf = [| zd; chunk et_d |];
  A.pts_to_len obuf;
  local_aligned16 #et_d obuf;
  {
    // ---- real drain target: [rAcc] retiled from [frag x frag] fragments to
    // [frag x wn] bands, combined with the C window by [comb_r], with the column coercion.
    let rD : chest2 real (mfrag wm * frag) (1 * SZ.v wn) =
      coerce_chest2_cols #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
        (chest_comb comb_r
          (lane_c_target rC bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid)) rAcc);

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
      // [comb_r (band iv of C) (band iv of rAcc)] == [band iv of rD].  Proven
      // up-front (pure, slprop-independent) where the proof state is light; the
      // nonlinear lemma-argument elaboration is fragile deeper in the block.
      coerced_drain_target_eq_comb comb_r (reveal rows_prod) (SZ.v wn) frag
        (reveal cnf_prod)
        (lane_c_target rC bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid))
        rAcc (SZ.v iv) ();
      ML.lemma_mult_le_right frag (SZ.v iv + 1) (mfrag wm);
      cband_subtile rC (reveal rows_prod) (reveal cnf_prod) frag (SZ.v wn)
        (reveal crb0) (reveal ccb0) (SZ.v iv) () ();
      let crb : (x:erased nat) = hide (reveal crb0 + SZ.v iv * frag);
      assert pure (reveal crb ==
        SZ.v mrow * SZ.v bm + SZ.v warpRow * SZ.v wm + SZ.v iv * frag);
      assert pure (
        chest_comb comb_r
          (cband rC frag (SZ.v wn) crb ccb0 ())
          (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v iv) 0)
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

      band_row_bound (SZ.v m) (SZ.v bm) (SZ.v wm) (SZ.v frag_sz)
        (SZ.v mrow) (SZ.v warpRow) (SZ.v iv);
      assert pure (reveal crb + SZ.v frag_sz <= SZ.v m);
      assert pure (reveal ccb0 + SZ.v wn <= SZ.v n);
      assert pure (SZ.v 1sz * SZ.v wn == SZ.v wn);
      assert pure (reveal ccb0 == SZ.v mcol * SZ.v bn
                     + SZ.v warpCol * (SZ.v 1sz * SZ.v wn)
                     + 0 * SZ.v wn);

      // drain band [iv]: read fp32 scratch, read C, [comb], store to D
      epilogue_band_fits et_ab et_acc bm bn bk wm wn skew;
      drain_band #et_c #et_acc #et_d
        gD gC rC comb comb_r obuf
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
        iv iv 0sz #_ crb ccb0 lane_sz ();

      // reshape the drained cells back to [approx_cell iv], i.e. [mixed (iv+1) iv].
      // First swap the (eD-independent) drain target from the [chest_comb . rAcc]
      // form to the [rD] form under the [exists*], then fold the sizet-coordinate
      // drained slprop into the opaque [approx_cell] (= [epi_approx_band]); the
      // sizet<->nat index equalities asserted above bridge the two, exactly as
      // the forward [live_cell] reshape at the top of the loop does.
      swap_lane_target #et_d
        (output_fragment' gD bm bn frag_sz wn mfrag_wm_sz 1sz
          (SZ.v bid_sz) (SZ.v wid_sz) (SZ.v iv) 0)
        (SZ.v lane_sz)
        (chest_comb comb_r (cband rC frag (SZ.v wn) crb ccb0 ())
          (ematrix_subtile rAcc frag (SZ.v wn) (SZ.v iv) 0))
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
        (chest_comb comb_r
          (lane_c_target rC bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid)) rAcc));
  };
}

#pop-options
