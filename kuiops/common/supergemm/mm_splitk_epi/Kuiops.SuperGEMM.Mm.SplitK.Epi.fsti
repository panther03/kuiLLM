module Kuiops.SuperGEMM.Mm.SplitK.Epi

(* The single top-level polymorphic async launcher for the split-K variant of
   the software-pipelined tensor-core GEMM WITH a C epilogue.  Computes

     D = comb C (A @ B^T)

   where B is supplied as an [(cols, shared)] = [(N, K)] row-major operand, and
   the k range is split [splits] ways.  This is the composition of
   [Kuiops.SuperGEMM.Mm.SplitK] and [Kuiops.SuperGEMM.Mm.Epi]: pass 1 and the
   entire workspace side are IMPORTED VERBATIM from the former, and only the
   reduce kernel differs.

   Why the epilogue lives in pass 2.  [comb] is affine in the accumulated
   value, so applying it inside pass 1 -- where each of the [splits] blocks
   holds only a PARTIAL k sum -- would add the [beta * C] term [splits] times
   instead of once.  Pass 2 is the first point at which a complete k reduction
   exists for an output element, so it is the only correct place for the
   epilogue.  [Kuiops.SuperGEMM.Mm.SplitK.Epi.Compose] is where that argument
   is discharged formally.

   C is an arbitrary read-only (M, N) *view*: it carries only a [vtlayout] with
   a concrete index map, with no injectivity, contiguity or alignment
   requirement.  A bias vector broadcast along rows and a dense row-major
   matrix are both instances of this one function; the epilogue reads C
   scalarly through the view's [imap], which is the definition and is correct
   for every view.  The D store stays 128-bit vectorized.

   [comb] is a general binary combiner rather than a fixed [alpha]/[beta] pair:
   [alpha * acc + beta * c] is the instantiation [MS.lincomb_to alpha beta] /
   [MS.rlincomb].  [comb_r] is tied to [comb] by [approx2 comb comb_r].

   Signature deviation.  Kuiper does not model the queue ordering of a CUDA
   stream, so a later launch cannot observe an earlier launch's writes.  The
   launcher therefore synchronizes the stream between the two launches: it
   CONSUMES [epoch_live s e] and RETURNS a fresh epoch [e'], rather than
   preserving one as [Kuiops.SuperGEMM.Mm.Epi.supergemm_mm_epi_async] does.
   The consequence is that this operator cannot run under CUDA graph capture.

   The workspace [gW] is caller-allocated (Kuiper kernels never allocate) and
   comes back live but with unspecified contents.

   Specification.  The postcondition is character for character the one of
   [Kuiops.SuperGEMM.Mm.Epi.supergemm_mm_epi_async] -- split-K is an
   implementation strategy and does not leak into the spec.

   This interface deliberately declares exactly ONE [fn] and nothing else: it
   is imported by every JIT'd operator, so anything extra costs compile time on
   every kernel build. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.TensorCore
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module RO = Kuiper.TensorRO
module P = Kuiops.SuperGEMM.Mm.Params

inline_for_extraction noextract
fn supergemm_mm_splitk_epi_async
  (et_ab et_acc et_c et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, has_vec_cpy et_acc, real_like et_acc |}
  {| scalar et_c, real_like et_c |}
  {| scalar et_d,  has_vec_cpy et_d,  real_like et_d |}
  (bm bn bk wm wn skew group : szp)
  (#sqc : squash (P.constraints et_ab et_acc bm bn bk wm wn skew))
  (#sq_vf : squash (valid_frag_et_dims et_ab FragA frag frag frag /\
                valid_frag_et_dims et_ab FragB frag frag frag /\
                valid_frag_et_dims et_acc FragAcc frag frag frag /\
                valid_frag_et_comb et_ab et_acc /\
                SZ.fits (SZ.v wm / frag * (SZ.v wn / frag))))
  (comb : et_c -> et_acc -> et_d)
  (comb_r : binop real { approx2 comb comb_r })
  (rows shared cols : szp)
  (splits mws ks : szp)
  (#lA : layout2 (SZ.v rows) (SZ.v shared)) {| T.ctlayout lA |}
       {| strA : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA { is_global gA })
  (#lB : layout2 (SZ.v cols) (SZ.v shared)) {| T.ctlayout lB |}
       {| strB : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB { is_global gB })
  (#lC : RO.vlayout2 (SZ.v rows) (SZ.v cols)) {| RO.cvtlayout lC |}
  (gC : RO.roarray2 et_c lC { RO.is_global gC })
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
  (#eC : chest2 et_c (SZ.v rows) (SZ.v cols))
  (#fA #fB #fC : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC) **
    on gpu_loc (live gD) **
    on gpu_loc (live gW)
  returns e' : epoch_t
  ensures
    epoch_live s e' **
    pledge0 (epoch_done s e')
      (on gpu_loc
        (exists* (eW' : chest2 et_acc (SZ.v mws) (SZ.v cols))
                 (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)).
           (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> Frac fC eC) **
           (gW |-> eW') ** (gD |-> eD') **
           pure (eD' %~ MS.mmcomb comb_r (to_real_matrix eC)
                   (to_real_matrix eA) (mtranspose (to_real_matrix eB)))))
