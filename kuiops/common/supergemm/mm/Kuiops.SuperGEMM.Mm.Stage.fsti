module Kuiops.SuperGEMM.Mm.Stage
#lang-pulse

(* Module 4 -- Stage.  One thread's contribution to staging ONE k-tile of A
   and of B into one pair of shared buffers, using cp.async.

   Step 1: memory safety only.  All staged contents are existentially
   quantified. *)

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module FB = Kuiops.GEMM.T.FlipFlopBarrier2

open Pulse.Lib.Pledge

(* ---- tile geometry (shared with the implementation) ---- *)

(* [cols] is the k-tile width [bk]; [chunk] the 16-byte vector width; [nthr]
   the block thread count.  These are the divisibility facts staging needs. *)
let geo_ok (rows cols chunk nthr : pos) : prop =
  cols % chunk == 0 /\
  (chunk * nthr) % cols == 0 /\
  (rows * cols) % (chunk * nthr) == 0

unfold let g_row_step (cols chunk nthr : pos) : nat = (chunk * nthr) / cols
unfold let g_a_iters  (rows cols chunk nthr : pos) : nat = (rows * cols) / (chunk * nthr)
unfold let g_t_row (cols chunk nthr : pos) (tid : nat) : nat = (tid * chunk) / cols
unfold let g_t_col (cols chunk nthr : pos) (tid : nat) : nat = (tid * chunk) % cols

(* ---- staging one k-tile of A and of B into one pair of buffers ----

   The reference's hoisted addressing div/mod is computed ONCE by the caller
   and threaded in as [t_row]/[t_col]/[row_step]/[iters] for A and B; the inner
   per-copy loops only add [s*row_step].  [mAs]/[mBs] are the kt-th k-tile
   subviews of global A/B (extracted and permission-managed by the caller).

   Consumes writable per-thread ownership of both destination buffers plus a
   read share of each source subtile; issues [a_iters + b_iters] cp.asyncs then
   [pipeline_commit], returning the successor batch and a pledge that, once the
   batch is done, re-establishes writable ownership of both destinations (with
   unspecified staged contents) and the source read shares. *)
inline_for_extraction noextract
fn stage_tiles
  (#et : Type0) {| _sz : sized et |} {| _vc : has_vec_cpy et |}
  (#bm #bk : pos)
  (#ldA : layout2 bm bk)
  {| T.ctlayout ldA, dstrA : strided_row_major (vtlayout_of_tlayout ldA) |}
  (mAd : array2 et ldA)
  (#lsA : layout2 bm bk)
  {| T.ctlayout lsA, sstrA : strided_row_major (vtlayout_of_tlayout lsA) |}
  (mAs : array2 et lsA)
  (#bn : pos)
  (#ldB : layout2 bn bk)
  {| T.ctlayout ldB, dstrB : strided_row_major (vtlayout_of_tlayout ldB) |}
  (mBd : array2 et ldB)
  (#lsB : layout2 bn bk)
  {| T.ctlayout lsB, sstrB : strided_row_major (vtlayout_of_tlayout lsB) |}
  (mBs : array2 et lsB)
  (nthr : pos) (tid : natlt nthr)
  (sqA : squash (geo_ok bm bk (SZ.v (chunk et)) nthr))
  (sqB : squash (geo_ok bn bk (SZ.v (chunk et)) nthr))
  (#emAd : chest2 et bm bk) (fA : perm) (#eAs : chest2 et bm bk)
  (#emBd : chest2 et bn bk) (fB : perm) (#eBs : chest2 et bn bk)
  (a_t_row a_t_col a_row_step a_iters : SZ.t)
  (b_t_row b_t_col b_row_step b_iters : SZ.t)
  (b : Kuiper.PipelineCopy.pipeline_batch_t)
  (sqh : squash (
     SZ.v a_t_row == g_t_row bk (SZ.v (chunk et)) nthr tid /\
     SZ.v a_t_col == g_t_col bk (SZ.v (chunk et)) nthr tid /\
     SZ.v a_row_step == g_row_step bk (SZ.v (chunk et)) nthr /\
     SZ.v a_iters == g_a_iters bm bk (SZ.v (chunk et)) nthr /\
     SZ.v b_t_row == g_t_row bk (SZ.v (chunk et)) nthr tid /\
     SZ.v b_t_col == g_t_col bk (SZ.v (chunk et)) nthr tid /\
     SZ.v b_row_step == g_row_step bk (SZ.v (chunk et)) nthr /\
     SZ.v b_iters == g_a_iters bn bk (SZ.v (chunk et)) nthr /\
     SZ.fits bm /\ SZ.fits bn /\
     aligned 16 (T.core mAd) /\ aligned_strided_row_major (SZ.v (chunk et)) dstrA /\
     aligned 16 (T.core mAs) /\ aligned_strided_row_major (SZ.v (chunk et)) sstrA /\
     is_global mAs /\
     aligned 16 (T.core mBd) /\ aligned_strided_row_major (SZ.v (chunk et)) dstrB /\
     aligned 16 (T.core mBs) /\ aligned_strided_row_major (SZ.v (chunk et)) sstrB /\
     is_global mBs))
  ()
  preserves gpu
  requires FB.own_strided_chunks mAd emAd nthr tid ** (mAs |-> Frac fA eAs) **
           FB.own_strided_chunks mBd emBd nthr tid ** (mBs |-> Frac fB eBs) **
           Kuiper.PipelineCopy.batch_live b
  returns b' : Kuiper.PipelineCopy.pipeline_batch_t
  ensures
    pledge0 (Kuiper.PipelineCopy.batch_done b)
      (FB.own_strided_chunks mAd eAs nthr tid ** FB.own_strided_chunks mBd eBs nthr tid **
       (mAs |-> Frac fA eAs) ** (mBs |-> Frac fB eBs)) **
    Kuiper.PipelineCopy.batch_committed b **
    Kuiper.PipelineCopy.batch_live b' **
    pure (fst b' == fst b /\ snd b' > snd b)
