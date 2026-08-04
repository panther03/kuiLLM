module Kuiops.Matmuls

(* Shared asynchronous GEMM launchers for the mm / addmm / bmm families.

   Both entry points [launch] their Kuiper [kernel_desc] on the caller's stream
   and do *not* sync, so they are legal under CUDA graph capture; ownership of
   the operands moves into the pledge redeemed when the stream's epoch
   completes. The C wrappers are the trusted boundary: they hand over buffers
   whose lifetime already outlives the launch, and Torch orders the consumer on
   the same stream. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Array2.Strided
open Kuiper.Array2.Strided.Slice
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Float.Casts { float_cast }
open Kuiper.TensorCore
open Kuiper.Tensor.Layout.Alg

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module KT = Kuiper.Kernel.GEMM.TensorCore2D
module KTT = Kuiper.Kernel.GEMM.TensorCore2D.To
module KB = Kuiper.Kernel.GEMM.BlockTiling2D

(* The batched row-major layout is affine in (page, row, col) with offset 0,
   page stride rows*cols and row stride cols; [l3_batched_row_major_imap] is the
   fsti-level characterization, so the instance can be built here rather than
   reaching into Kuiper's opaque one. *)
inline_for_extraction noextract
instance srm3_batched
  (batch : erased nat { SZ.fits batch })
  (rows : szp { SZ.fits (batch * rows) })
  (cols : szp { SZ.fits (rows * cols) /\ SZ.fits (batch * (rows * cols)) })
  : strided_row_major_3 (l3_batched_row_major batch rows cols) =
{
  offset3 = 0sz;
  pstride3 = rows *^ cols;
  rstride3 = cols;
  pf3 = (fun p i j ->
           l3_batched_row_major_imap batch rows cols
             (SZ.uint_to_t p) (SZ.uint_to_t i) (SZ.uint_to_t j));
}

(* Dimension factorization m == (m/bm)*((bm/tm)*tm) required by the batched
   BlockTiling2D index bijection. *)
let bt2d_dim_sq (m bm tm : szp)
  : Lemma (requires tm /?+ bm /\ bm /?+ m)
          (ensures SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm))
  = Kuiper.Divides.lemma_nat_divides_pos_divides bm m;
    Kuiper.Divides.lemma_nat_divides_pos_divides tm bm

(* Batched BlockTiling2D: one launch covers all [batch] pages. The rank-2 mm /
   addmm fallbacks use it at [batch = 1] -- an [m x n] row-major buffer *is* a
   [1 x m x n] batched row-major buffer, so the C wrapper hands over the same
   pointer and no rank-2/rank-3 bridging (which would have to happen inside the
   pledge) is needed. *)
#push-options "--z3rlimit 10"
inline_for_extraction noextract
fn bt2d_async
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : et -> et -> et)
  (bm bn bk : szp)
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp { tm /?+ bm })
  (tn : szp { tn /?+ bn /\ (bm/tm * (bn/tn) <= max_threads) })
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (batch m n k : szp)
  (#_ : squash (SZ.fits (m * k) /\ SZ.fits (k * n) /\ SZ.fits (m * n)))
  (#_ : squash (SZ.fits (batch * (m * k)) /\ SZ.fits (batch * (k * n))
             /\ SZ.fits (batch * (m * n))))
  (gA : array3 et (l3_batched_row_major (SZ.v batch) (SZ.v m) (SZ.v k)) { is_global gA })
  (gB : array3 et (l3_batched_row_major (SZ.v batch) (SZ.v k) (SZ.v n)) { is_global gB })
  (gC : array3 et (l3_batched_row_major (SZ.v batch) (SZ.v m) (SZ.v n)) { is_global gC })
  (s : stream_t)
  (#eA : chest3 et (SZ.v batch) (SZ.v m) (SZ.v k))
  (#eB : chest3 et (SZ.v batch) (SZ.v k) (SZ.v n))
  (#eC : chest3 et (SZ.v batch) (SZ.v m) (SZ.v n))
  (#fA #fB : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * ((m/bm) * (n/bn)) <= max_blocks) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> eC)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
                   (gC |-> MS.gbmmcomb (fun (x:et) -> x) (fun (x:et) -> x) comb eC eA eB)))
{
  tensor_pts_to_ref_located gA;
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;

  dassert (bm >^ 0sz);
  dassert (bn >^ 0sz);
  dassert (bk >^ 0sz);
  dassert (tm >^ 0sz);
  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dguard (m %^ bm = 0sz);
  dguard (k %^ bk = 0sz);
  dguard (n %^ bn = 0sz);

  lemma_divides_trans (chunk et) bk k;
  lemma_divides_trans (chunk et) bn n;
  lemma_divides_product_r (chunk et) (SZ.v m) (SZ.v k);
  lemma_divides_product_r (chunk et) (SZ.v k) (SZ.v n);

  let sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)) =
    bt2d_dim_sq m bm tm;
  let sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)) =
    bt2d_dim_sq n bn tn;

  launch (KB.bmk_kernel (fun (x:et) -> x) (fun (x:et) -> x) comb
            gA gB gC bm bn bk (l2_col_major _ _) (l2_row_major _ _) tm tn sq1 sq2
            (batch *^ (m/^bm *^ (n/^bn))) (bm/^tm *^ (bn/^tn)) ()) s;
}
#pop-options

(* TensorCore2D "to" GEMM: D = comb(C, A@B), with the accumulator type distinct
   from the C/D type. *)
#push-options "--split_queries always"
inline_for_extraction noextract
fn tc2d_to_async
  (et_ab et_acc et_cd : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
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
  (gB : array2 et_ab (l2_row_major (SZ.v shared) (SZ.v cols)) { is_global gB })
  (gC : array2 et_cd (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gC })
  (gD : array2 et_cd (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gD })
  (s : stream_t)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (core gC)))
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
  tensor_pts_to_ref_located gC;
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

  lemma_divides_trans (chunk et_ab) bk shared;
  assert pure (chunk et_ab /?+ shared);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v rows) #(SZ.v shared) (chunk et_ab);

  lemma_divides_trans (chunk et_ab) bn cols;
  assert pure (chunk et_ab /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v shared) #(SZ.v cols) (chunk et_ab);

  #set-options "--fuel 0 --ifuel 0 --z3refresh" {
  launch (
    KTT.mk_kernel comb comb_r
      gA #eA gB #eB
      gC #_ #eC gD #eC
      bm bn bk tm tn tk wm wn
      // #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_
      // #fA #fB #fC
      nblk nthr
      #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_
      (to_real_matrix eA) (to_real_matrix eB) (to_real_matrix eC) #_ #_ ()
  ) s};
}
#pop-options

#push-options "--split_queries always"
inline_for_extraction noextract
fn tc2d_async
  (et_ab et_acc et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_c, real_like et_c |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
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
  (comb : et_c -> et_acc -> et_c)
  (comb_r : binop real { approx2 comb comb_r })
  (rows shared cols : szp)
  (gA : array2 et_ab (l2_row_major (SZ.v rows) (SZ.v shared)) { is_global gA })
  (gB : array2 et_ab (l2_row_major (SZ.v shared) (SZ.v cols)) { is_global gB })
  (gC : array2 et_c (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gC })
  (s : stream_t)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v shared) (SZ.v cols))
  (#eC : chest2 et_c (SZ.v rows) (SZ.v cols))
  (#fA #fB : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> eC)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
        (exists* eC'. (gC |-> eC') **
          pure (eC' %~ MS.gmmcomb (fun (x:real) -> x) (fun (x:real) -> x) comb_r
            (to_real_matrix eC) (to_real_matrix eA) (to_real_matrix eB)))))
{
  tensor_pts_to_ref_located gA;
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;

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

  lemma_divides_trans (chunk et_ab) bk shared;
  assert pure (chunk et_ab /?+ shared);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v rows) #(SZ.v shared) (chunk et_ab);

  lemma_divides_trans (chunk et_ab) bn cols;
  assert pure (chunk et_ab /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v shared) #(SZ.v cols) (chunk et_ab);

  #set-options "--fuel 0 --ifuel 0 --z3refresh" {
  launch (
    KT.mk_kernel
      gA #eA gB #eB gC #_ #eC
      bm bn bk tm tn tk wm wn
      #_ #_ #_ #_ #_ #_ #_ #_
      #fA #fB
      nblk nthr
      (to_real_matrix eA) (to_real_matrix eB) (to_real_matrix eC)
      (fun (x:real) -> x) (fun (x:real) -> x) comb_r
      (fun (x:et_ab) -> x) (fun (x:et_ab) -> x) comb
      ()
  ) s};
}
#pop-options
