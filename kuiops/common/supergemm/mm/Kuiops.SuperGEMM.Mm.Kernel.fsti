module Kuiops.SuperGEMM.Mm.Kernel

(* Module 7 of the software-pipelined tensor-core GEMM (D = A @ B^T).

   [mk_kernel] assembles the shared-memory pre/post ([Shared]), the pipelined
   barrier ([Barrier]), the compute loop ([KLoop]) and the epilogue
   ([Epilogue]) into a single [kernel_desc].

   Step 1: memory safety only -- all data values are existentially quantified,
   no functional specification.  The whole-kernel post existentialises A, B
   and D (see [Shared.kpost1]). *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorCore

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Kernel.Desc { kernel_desc }
open Kuiops.SuperGEMM.Mm.Params

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
val mk_kernel
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc,
     scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA { is_global gA }) (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
       (#rA : chest2 real (SZ.v m) (SZ.v k) { eA %~ rA })
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB { is_global gB }) (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
       (#rB : chest2 real (SZ.v n) (SZ.v k) { eB %~ rB })
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD { is_global gD })
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (bm bn bk wm wn skew group : szp)
  (#sqc : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#sq_bmnk : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (#sq_fits : squash (SZ.fits (SZ.v m * SZ.v k) /\ SZ.fits (SZ.v n * SZ.v k) /\
                SZ.fits (SZ.v m * SZ.v n)))
  (#sq_al16 : squash (aligned 16 (core gA) /\ aligned 16 (core gB) /\ aligned 16 (core gD)))
  (#sq_asAB : squash (aligned_strided_row_major (SZ.v (chunk et_ab)) str_A /\
                aligned_strided_row_major (SZ.v (chunk et_ab)) str_B))
  (#sq_asD : squash (aligned_strided_row_major (SZ.v (chunk et_d)) strD))
  (#sq_cnbk : squash ((SZ.v (chunk et_ab) * P.nthr bm bn wm wn) % SZ.v bk == 0))
  (#sq_vf : squash (valid_frag_et_dims et_ab FragA frag frag frag /\
                valid_frag_et_dims et_ab FragB frag frag frag /\
                valid_frag_et_dims et_acc FragAcc frag frag frag /\
                valid_frag_et_comb et_ab et_acc /\
                SZ.fits (SZ.v wm / frag * (SZ.v wn / frag))))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (#_ : squash (SZ.v nblk <= SZ.v max_blocks))
  ()
  : kernel_desc
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       live gD)
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       (exists* (eD' : chest2 et_d (SZ.v m) (SZ.v n)).
          gD |-> eD' **
          pure (eD' %~ Kuiper.Chest.chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))))
