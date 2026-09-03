module Kuiops.SuperGEMM.Mm.Epi

(* The single top-level polymorphic async launcher for the software-pipelined
   tensor-core GEMM with a C epilogue.  Computes

     D = comb C (A @ B^T)

   where B is supplied as an [(cols, shared)] = [(N, K)] row-major operand (the
   weight PyTorch hands to [aten.addmm] for [C + A @ W^T]).

   A, B and C come back at their entry values: the kernel only ever reads them
   (under a [Frac] share).  D is pinned cell-for-cell -- [%~] on a [chest2] is
   elementwise, so no implementation can satisfy this while leaving part of D
   untouched.

   C is an arbitrary read-only (M, N) *view*: it carries only a [vtlayout] with
   a concrete index map, with no injectivity, contiguity or alignment
   requirement.  A bias vector broadcast along rows and a dense row-major
   matrix are both instances of this one function; the epilogue reads C
   scalarly through the view's [imap], which is the definition and is correct
   for every view.  The D store stays 128-bit vectorized.

   [comb] is a general binary combiner rather than a fixed [alpha]/[beta] pair:
   [alpha * acc + beta * c] is the instantiation [MS.lincomb_to alpha beta] /
   [MS.rlincomb], and the plain [post_map]-only kernel is the instantiation
   that ignores its C argument.  [comb_r] is tied to [comb] by
   [approx2 comb comb_r], exactly as [tc2d_to_async] does.

   The [mtranspose] in the specification is the same one [Kuiops.SuperGEMM.Mm]
   explains: B is [strided_row_major] over [(cols, shared)], so [eB] is
   [chest2 _ cols shared] and the transpose has to be written down.  It is a
   specification-level reindex only; nothing is physically transposed.

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
open Kuiops.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module RO = Kuiper.TensorRO
module P = Kuiops.SuperGEMM.Mm.Params

inline_for_extraction noextract
fn supergemm_mm_epi_async
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
  (s : stream_t)
  (#sq_div : squash (SZ.v bm /?+ SZ.v rows /\ SZ.v bn /?+ SZ.v cols /\
                SZ.v bk /?+ SZ.v shared /\ SZ.v bk <= SZ.v shared))
  (#sq_fits : squash (SZ.fits (SZ.v rows * SZ.v shared) /\
                SZ.fits (SZ.v cols * SZ.v shared) /\ SZ.fits (SZ.v rows * SZ.v cols)))
  (#sq_al16 : squash (aligned 16 (core gA) /\ aligned 16 (core gB) /\ aligned 16 (core gD)))
  (#sq_asAB : squash (aligned_strided_row_major (SZ.v (chunk et_ab)) strA /\
                aligned_strided_row_major (SZ.v (chunk et_ab)) strB))
  (#sq_asD : squash (aligned_strided_row_major (SZ.v (chunk et_d)) strD))
  (#sq_cnbk : squash ((SZ.v (chunk et_ab) * P.nthr bm bn wm wn) % SZ.v bk == 0))
  (#sq_blk : squash ((SZ.v rows / SZ.v bm) * (SZ.v cols / SZ.v bn) <= SZ.v max_blocks))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v cols) (SZ.v shared))
  (#eC : chest2 et_c (SZ.v rows) (SZ.v cols))
  (#fA #fB #fC : perm)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC) **
    on gpu_loc (live gD)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc
        ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> Frac fC eC) **
          (exists* (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)). (gD |-> eD') **
            pure (eD' %~ MS.mmcomb comb_r (to_real_matrix eC)
                    (to_real_matrix eA) (mtranspose (to_real_matrix eB))))))
