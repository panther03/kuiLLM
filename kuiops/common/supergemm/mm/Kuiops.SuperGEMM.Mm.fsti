module Kuiops.SuperGEMM.Mm

(* Module 8: the single top-level polymorphic async launcher for the
   software-pipelined tensor-core GEMM.  Computes D = A @ B^T, where B is
   supplied as an [(cols, shared)] = [(N, K)] row-major operand (i.e. the
   weight PyTorch hands to [aten.mm] for [A @ W^T]).  There is no C operand;
   [post_map : et_acc -> et_d] is applied elementwise to the accumulator on
   the way out (at the JIT instantiation this is [Kuiper.Float.Casts.fcast]).

   Step 1: memory-safety only.  The pledge hands D back at some unspecified
   value; A and B likewise come back existentially quantified (see the
   [Kernel]/[Shared] note -- a global read share is re-materialised through the
   cp.async pledge, which does not yet track content preservation).  The
   functional specification ([eD' %~ ...]) is step 4.

   Specification.  Where the other GEMMs state

     [eD' %~ MS.mmcomb comb_r (to_real_matrix eC) rA rB]

   this kernel has no C operand, so there is no [m0] to feed [mmcomb] and
   nothing for a [binop] combiner to combine with.  [mmcomb comb m0 m1 m2] is
   [chest_comb comb m0 (matmul m1 m2)]; with a combiner that ignores its first
   argument that degenerates to a map over [matmul], which is what is stated
   below.  Passing a dummy [m0] instead would be strictly worse: D is [live] on
   entry, so there is no value to name.

   The other deviation is [mtranspose].  [tc2d_tn] declares B at
   [l2_col_major (K, N)], so its [eB] is already [chest2 _ shared cols] and no
   spec-level transpose appears.  Here B is [strided_row_major] over
   [(cols, shared)] -- the (N, K) operand PyTorch actually hands us -- so [eB]
   is [chest2 _ cols shared] and the transpose has to be written down.  It is
   a specification-level reindex only: nothing is physically transposed, and
   [matmul rA (mtranspose rB)] is the (i,j) dot product of row i of A with row
   j of B, which is what the kernel computes.

   [post_map_r] is tied to [post_map] by [post_map %~ post_map_r], the unary
   analogue of the [approx2 comb comb_r] the other GEMMs use (via
   [Kuiper.Approximates.approx_function_can_approximate]).  At the JIT
   instantiation [post_map] is [fcast] and [post_map_r] is the identity.

   This interface deliberately declares exactly ONE [fn] and nothing else: it
   is imported by every JIT'd operator, so anything extra costs compile time on
   every kernel build. *)

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
fn supergemm_mm_async
  (et_ab et_acc et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, has_vec_cpy et_acc, real_like et_acc |}
  {| scalar et_d,  has_vec_cpy et_d,  real_like et_d |}
  (bm bn bk wm wn skew : szp)
  (#sqc : squash (P.constraints et_ab et_acc bm bn bk wm wn skew))
  (#sq_vf : squash (valid_frag_et_dims et_ab FragA frag frag frag /\
                valid_frag_et_dims et_ab FragB frag frag frag /\
                valid_frag_et_dims et_acc FragAcc frag frag frag /\
                valid_frag_et_comb et_ab et_acc /\
                SZ.fits (SZ.v wm / frag * (SZ.v wn / frag))))
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (rows shared cols : szp)
  (#lA : layout2 (SZ.v rows) (SZ.v shared)) {| T.ctlayout lA |}
       {| strA : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA { is_global gA })
  (#lB : layout2 (SZ.v cols) (SZ.v shared)) {| T.ctlayout lB |}
       {| strB : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB { is_global gB })
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
  (#fA #fB : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (live gD)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc
        ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
          (exists* (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)). (gD |-> eD') **
            pure (eD' %~ chest_map post_map_r
                    (MS.matmul (to_real_matrix eA) (mtranspose (to_real_matrix eB)))))))
