module Kuiops.SuperGEMM.Mm.SplitK

(* The single top-level polymorphic async launcher for the split-K variant of
   the software-pipelined tensor-core GEMM.  Computes D = A @ B^T with B
   supplied as an [(cols, shared)] = [(N, K)] row-major operand, exactly as
   [Kuiops.SuperGEMM.Mm.supergemm_mm_async] does; the split-K decomposition is
   an implementation strategy and is invisible in the specification.

   Two kernels are launched.  The first partitions the k range into [splits]
   contiguous, disjoint, covering blocks of [ks = shared / splits] and writes
   the [splits] fp32 partial products into the caller-supplied workspace [gW],
   viewed as a [(splits * rows, cols)] matrix.  The second sums the partials
   for each 128-bit granule of D and applies [post_map] on the way out.

   The two launches are chained on the stream with no synchronization between
   them: pass 2 consumes the pledge pass 1 produced, at the queue position pass
   1 left behind, which is exactly CUDA's guarantee that a kernel does not start
   until everything enqueued before it on the same stream has retired.  The
   operator therefore advances the epoch by two and is safe to capture in a
   CUDA graph.  Chaining is possible only because the reduce kernel's
   precondition keeps the workspace contents EXISTENTIAL -- pass 1 can name its
   output only up to [%~], and an existential cannot be pulled out from under a
   pledge; see [Kuiops.Approx.Share].

   The workspace [gW] is caller-allocated (Kuiper kernels never allocate) and
   comes back live but with unspecified contents.

   Specification.  The postcondition is character for character the one of
   [Kuiops.SuperGEMM.Mm.supergemm_mm_async] -- split-K is an implementation
   strategy and does not leak into the spec.  The mathematical core it rests on
   is [Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas] and
   [Kuiops.SuperGEMM.Mm.SplitK.Compose]. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Chest { chest_map }
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.TensorCore
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params

inline_for_extraction noextract
fn supergemm_mm_splitk_async
  (et_ab et_acc et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, has_vec_cpy et_acc, real_like et_acc |}
  {| scalar et_d,  has_vec_cpy et_d,  real_like et_d |}
  (bm bn bk wm wn skew group : szp)
  (#sqc : squash (P.constraints et_ab et_acc bm bn bk wm wn skew))
  (#sq_vf : squash (valid_frag_et_dims et_ab FragA frag frag frag /\
                valid_frag_et_dims et_ab FragB frag frag frag /\
                valid_frag_et_dims et_acc FragAcc frag frag frag /\
                valid_frag_et_comb et_ab et_acc /\
                SZ.fits (SZ.v wm / frag * (SZ.v wn / frag))))
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (rows shared cols : szp)
  (splits mws ks : szp)
  (#lA : layout2 (SZ.v rows) (SZ.v shared)) {| T.ctlayout lA |}
       {| strA : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA { is_global gA })
  (#lB : layout2 (SZ.v cols) (SZ.v shared)) {| T.ctlayout lB |}
       {| strB : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB { is_global gB })
  (#lD : layout2 (SZ.v rows) (SZ.v cols)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD { is_global gD })
  (#lW : layout2 (SZ.v mws) (SZ.v cols)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (gW : array2 et_acc lW { is_global gW })
  (s : stream_t)
  (#sq_split : squash (SZ.v mws == SZ.v splits * SZ.v rows /\
                SZ.v shared == SZ.v splits * SZ.v ks))
  (#sq_div : squash (SZ.v bm /?+ SZ.v rows /\ SZ.v bn /?+ SZ.v cols /\
                SZ.v bk /?+ SZ.v ks /\ SZ.v bk <= SZ.v ks /\
                SZ.v (chunk et_d) /?+ SZ.v cols /\
                SZ.v (chunk et_acc) /?+ SZ.v (chunk et_d)))
  (#sq_fits : squash (SZ.fits (SZ.v rows * SZ.v shared) /\
                SZ.fits (SZ.v cols * SZ.v shared) /\ SZ.fits (SZ.v rows * SZ.v cols) /\
                SZ.fits (SZ.v mws * SZ.v cols) /\
                SZ.fits lA.ulen /\ SZ.fits lB.ulen /\ SZ.fits lD.ulen /\ SZ.fits lW.ulen))
  (#sq_al16 : squash (aligned 16 (core gA) /\ aligned 16 (core gB) /\
                aligned 16 (core gD) /\ aligned 16 (core gW)))
  (#sq_asAB : squash (aligned_strided_row_major (SZ.v (chunk et_ab)) strA /\
                aligned_strided_row_major (SZ.v (chunk et_ab)) strB /\
                SZ.v (chunk et_ab) /?+ SZ.v ks))
  (#sq_asDW : squash (aligned_strided_row_major (SZ.v (chunk et_d)) strD /\
                aligned_strided_row_major (SZ.v (chunk et_acc)) strW))
  (#sq_cnbk : squash ((SZ.v (chunk et_ab) * P.nthr bm bn wm wn) % SZ.v bk == 0))
  (#sq_blk : squash ((SZ.v mws / SZ.v bm) * (SZ.v cols / SZ.v bn) <= SZ.v max_blocks))
  (#sq_job : squash (SZ.fits (SZ.v rows * (SZ.v cols / SZ.v (chunk et_d))) /\
                SZ.v rows * (SZ.v cols / SZ.v (chunk et_d))
                  <= SZ.v max_blocks * SZ.v max_threads))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v cols) (SZ.v shared))
  (#fA #fB : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (live gD) **
    on gpu_loc (live gW)
  ensures
    epoch_live s (epoch_next (epoch_next e)) **
    pledge0 (epoch_flushed s (epoch_next (epoch_next e)))
      (on gpu_loc
        (exists* (eW' : chest2 et_acc (SZ.v mws) (SZ.v cols))
                 (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)).
           (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
           (gW |-> eW') ** (gD |-> eD') **
           pure (eD' %~ chest_map post_map_r
                   (MS.matmul (to_real_matrix eA) (mtranspose (to_real_matrix eB))))))
