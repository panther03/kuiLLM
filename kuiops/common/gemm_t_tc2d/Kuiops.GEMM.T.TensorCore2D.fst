module Kuiops.GEMM.T.TensorCore2D

(* Implementation of the transposed-B ("TN") TensorCore2D GEMM launcher.
   See the interface for the design rationale.

   Modelled on [Kuiops.Matmuls.tc2d_to_async]; the only substantive
   differences are:
     - B is [l2_col_major shared cols], so its strided witness is
       [ColMajor.scm_l2_col_major] and its alignment obligation is closed by
       [ColMajor.lemma_aligned_scm_l2_col_major], which reduces to
       [chunk et_ab /?+ shared] (the leading dimension) rather than
       [chunk et_ab /?+ cols].
     - [scm_l2_col_major] is passed EXPLICITLY.  It is deliberately not an
       [instance], so that it cannot be confused with upstream's
       [strided_col_major_l2_col_major], whose offset/stride are abstract and
       against which the alignment fact is unprovable. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Divides
open Kuiper.Array2.Strided
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.TensorCore
open Kuiper.Tensor.Layout.Alg

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module KTN = Kuiops.GEMM.T.TensorCore2D.Kernel
module CM = Kuiops.Array2.Strided.ColMajor
module RO = Kuiper.TensorRO

#push-options "--split_queries always"
inline_for_extraction noextract
fn tc2d_tn_gen_async
  (et_ab et_acc et_cd : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
  (#_ : squash (chunk et_cd /?+ tn))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_size)) <= max_threads))
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (rows shared cols : szp)
  (gA : array2 et_ab (l2_row_major (SZ.v rows) (SZ.v shared)) { is_global gA })
  (gB : array2 et_ab (l2_col_major (SZ.v shared) (SZ.v cols)) { is_global gB })
  (#lC : RO.vlayout2 (SZ.v rows) (SZ.v cols))
  {| strC : strided_row_major lC |}
  (#_ : squash (aligned_strided_row_major (chunk et_cd) strC))
  (gC : RO.roarray2 et_cd lC { RO.is_global gC })
  (gD : array2 et_cd (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gD })
  (s : stream_t)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (RO.core gC)))
  (#_ : squash (aligned 16 (core gD)))
  (#_ : squash (chunk et_cd /?+ cols))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v shared) (SZ.v cols))
  (#eC : chest2 et_cd (SZ.v rows) (SZ.v cols))
  (#fA #fB #fC : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC) **
    on gpu_loc (live gD)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> Frac fC eC) **
        (exists* eD'. (gD |-> eD') **
          pure (eD' %~ MS.mmcomb comb_r
                  (to_real_matrix eC) (to_real_matrix eA) (to_real_matrix eB)))))
{
  tensor_pts_to_ref_located gA;
  tensor_pts_to_ref_located gB;
  RO.tensor_pts_to_ref_located gC;
  tensor_pts_to_ref_located gD;

  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dassert (bk %^ tk = 0sz);

  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);

  lemma_divides_chain (wm * tm) bm rows;
  lemma_divides_chain (wn * tn) bn cols;

  let nblk = rows/^bm *^ (cols/^bn);
  let nthr = bm/^(wm*^tm) *^ (bn/^(wn*^tn)) *^ warp_size;

  assert pure ((rows/bm) * (cols/bn) == nblk);
  assert pure ((rows/bm) * (cols/bn) <= max_blocks);
  dassert (nblk <=^ SZ.uint_to_t 2097152);
  assert pure (nblk <= max_blocks);

  dassert ((bm *^ bk) %^ (chunk et_ab *^ nthr) = 0sz);
  dassert ((bk *^ bn) %^ (chunk et_ab *^ nthr) = 0sz);

  (* A: row-major, ld = shared.  Unchanged from the NN launcher. *)
  lemma_divides_trans (chunk et_ab) bk shared;
  assert pure (chunk et_ab /?+ shared);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v rows) #(SZ.v shared) (chunk et_ab);

  (* B: column-major, ld = shared.  The alignment obligation therefore lands
     on [shared] (== k), not on [cols] (== n) as it does for NN.  This is the
     substantive constraint change of the whole TN design. *)
  CM.lemma_aligned_scm_l2_col_major
    #(SZ.v shared) #(SZ.v cols) (chunk et_ab);

  #set-options "--fuel 0 --ifuel 0 --z3refresh" {
  launch (
    KTN.mk_kernel comb comb_r
      gA #eA
      (CM.scm_l2_col_major #(SZ.v shared) #(SZ.v cols))
      gB #eB
      gC #_ #eC gD #eC
      bm bn bk tm tn tk wm wn
      nblk nthr
      #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_
      (to_real_matrix eA) (to_real_matrix eB) (to_real_matrix eC) #_ #_ ()
  ) s};
}
#pop-options

(* C is a dense row-major [rows x cols] matrix. *)
inline_for_extraction noextract
fn tc2d_tn_async
  (et_ab et_acc et_cd : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
  (#_ : squash (chunk et_cd /?+ tn))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_size)) <= max_threads))
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (rows shared cols : szp)
  (gA : array2 et_ab (l2_row_major (SZ.v rows) (SZ.v shared)) { is_global gA })
  (gB : array2 et_ab (l2_col_major (SZ.v shared) (SZ.v cols)) { is_global gB })
  (gC : RO.roarray2 et_cd
          (RO.vtlayout_of_tlayout (l2_row_major (SZ.v rows) (SZ.v cols)))
          { RO.is_global gC })
  (gD : array2 et_cd (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gD })
  (s : stream_t)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (RO.core gC)))
  (#_ : squash (aligned 16 (core gD)))
  (#_ : squash (chunk et_cd /?+ cols))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v shared) (SZ.v cols))
  (#eC : chest2 et_cd (SZ.v rows) (SZ.v cols))
  (#fA #fB #fC : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC) **
    on gpu_loc (live gD)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> Frac fC eC) **
        (exists* eD'. (gD |-> eD') **
          pure (eD' %~ MS.mmcomb comb_r
                  (to_real_matrix eC) (to_real_matrix eA) (to_real_matrix eB)))))
{
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v rows) #(SZ.v cols) (chunk et_cd);
  tc2d_tn_gen_async et_ab et_acc et_cd bm bn bk tm tn tk wm wn comb comb_r
    rows shared cols gA gB gC gD s
}

(* C is a length-[cols] bias vector broadcast down the rows, exactly as in
   [Kuiops.Matmuls.tc2d_to_bcast_async]: the epilogue reads one [cols]-wide
   run per block instead of a materialised [rows x cols] matrix. *)
inline_for_extraction noextract
fn tc2d_tn_bcast_async
  (et_ab et_acc et_cd : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
  (#_ : squash (chunk et_cd /?+ tn))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_size)) <= max_threads))
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (rows shared cols : szp)
  (gA : array2 et_ab (l2_row_major (SZ.v rows) (SZ.v shared)) { is_global gA })
  (gB : array2 et_ab (l2_col_major (SZ.v shared) (SZ.v cols)) { is_global gB })
  (gC : RO.roarray2 et_cd
          (RO.extended_layout
             (RO.vtlayout_of_tlayout (l1_forward (SZ.v cols))) (SZ.v rows))
          { RO.is_global gC })
  (gD : array2 et_cd (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gD })
  (s : stream_t)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (RO.core gC)))
  (#_ : squash (aligned 16 (core gD)))
  (#_ : squash (chunk et_cd /?+ cols))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v shared) (SZ.v cols))
  (#eC : chest2 et_cd (SZ.v rows) (SZ.v cols))
  (#fA #fB #fC : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC) **
    on gpu_loc (live gD)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> Frac fC eC) **
        (exists* eD'. (gD |-> eD') **
          pure (eD' %~ MS.mmcomb comb_r
                  (to_real_matrix eC) (to_real_matrix eA) (to_real_matrix eB)))))
{
  lemma_aligned_strided_row_major_bcast_l1
    #(SZ.v rows) #(SZ.v cols) (chunk et_cd);
  tc2d_tn_gen_async et_ab et_acc et_cd bm bn bk tm tn tk wm wn comb comb_r
    rows shared cols gA gB gC gD s
}
