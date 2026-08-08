module Kuiops.SuperGEMM.Mm.KLoop

(* Module 5 of the software-pipelined tensor-core GEMM (D = A @ B^T).

   [subproducts] is one k-tile's fragment math for one warp, layout-generic:
   the A operand is [array2 et_ab lA] for an arbitrary [lA : layout2 bm bk]
   with a [strided_row_major] witness (instantiated at the skewed shared tile),
   and the B operand is [array2 et_ab lB] for an arbitrary [lB : layout2 bk bn]
   with a [strided_col_major] witness (instantiated at the [atranspose] of the
   skewed (bn, bk) B tile, so B^T is consumed for free by the tensor core via
   the FragLCM tag).  Fragment dims are the constants 16x16x16 ([frag]).

   Step 1 = memory safety only: the fragment arrays are existentially
   quantified ([live]); no real-number specification is carried. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Pulse.Lib.Array { length }

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params { frag }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

inline_for_extraction noextract
fn subproducts
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn : szp)
  (#_ : squash (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn /\ frag /?+ SZ.v bk /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn))
  (fmap : et_ab -> et_ab)
  (aFrags  : array (fragment et_ab FragA   frag frag frag FragLRM))
  (bFrags  : array (fragment et_ab FragB   frag frag frag FragLCM))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#lA : layout2 (SZ.v bm) (SZ.v bk)) {| T.ctlayout lA |}
       {| strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v bk) (SZ.v bn)) {| T.ctlayout lB |}
       {| strided_col_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB)
  (#eA : chest2 et_ab (SZ.v bm) (SZ.v bk))
  (#eB : chest2 et_ab (SZ.v bk) (SZ.v bn))
  (#fA #fB : perm)
  (warp_m : szlt (SZ.v bm / SZ.v wm))
  (warp_n : szlt (SZ.v bn / SZ.v wn))
  (#_ : squash (length aFrags == SZ.v wm / frag))
  (#_ : squash (length bFrags == SZ.v wn / frag))
  (#_ : squash (length accFrags == SZ.v wm / frag * (SZ.v wn / frag)))
  (#_ : squash (valid_frag_et_dims et_ab FragA frag frag frag))
  (#_ : squash (valid_frag_et_dims et_ab FragB frag frag frag))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc frag frag frag))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  ()
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  preserves
    live aFrags ** live bFrags ** live accFrags
