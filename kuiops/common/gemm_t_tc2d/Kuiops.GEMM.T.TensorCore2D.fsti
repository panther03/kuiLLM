module Kuiops.GEMM.T.TensorCore2D

(* Asynchronous launcher for the transposed-B ("TN") TensorCore2D GEMM:
   D = comb(C, A @ B), where B is stored COLUMN-major with leading dimension
   [shared] -- i.e. exactly the operand PyTorch hands to [aten.mm] for a frozen
   weight, [shape=(k,n)], [stride=(1,k)].  Using this kernel removes the
   [B.contiguous()] copy the NN kernel forces on every GEMM.

   The mathematical specification is IDENTICAL to the row-major launcher
   [Kuiops.Matmuls.tc2d_to_async]: the pledge still yields
   [eD' %~ MS.mmcomb comb_r rC rA rB].  Only B's layout differs; nothing is
   ever physically transposed.

   Constraint delta versus the NN launcher: the staging-copy divisibility
   lands on [shared] and [bk], NOT on [cols] and [bn].  [supported()] on the
   Python side must be updated accordingly.

   This interface deliberately declares exactly ONE [fn] and nothing else: it
   is imported by every JIT'd operator, so anything extra costs compile time
   on every kernel build. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.TensorCore
open Kuiper.Tensor.Layout.Alg
open Kuiper.Array2.Strided

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module RO = Kuiper.TensorRO

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

(* As [tc2d_tn_async], but C is a length-[cols] bias vector broadcast down
   the rows rather than a dense matrix -- the transposed-B counterpart of
   [Kuiops.Matmuls.tc2d_to_bcast_async], used by addmm with a rank-1 bias. *)
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
