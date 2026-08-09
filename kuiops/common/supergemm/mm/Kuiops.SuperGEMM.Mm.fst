module Kuiops.SuperGEMM.Mm

(* Implementation of the top-level async launcher; see the interface for the
   contract.  The body is minimal: compute the grid/block dimensions once, then
   [launch] the assembled [kernel_desc] under the stream's epoch pledge. *)

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
open Kuiops.SuperGEMM.Mm.Kernel { mk_kernel }

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

inline_for_extraction noextract
fn supergemm_mm_async
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
        (exists* (eA' : chest2 et_ab (SZ.v rows) (SZ.v shared)). gA |-> Frac fA eA' **
          (exists* (eB' : chest2 et_ab (SZ.v cols) (SZ.v shared)). gB |-> Frac fB eB' **
            (exists* (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)). gD |-> eD'))))
{
  let nblk = (rows /^ bm) *^ (cols /^ bn);
  let nthr = (bm /^ wm) *^ (bn /^ wn) *^ warp_size;

  assert pure (SZ.v nblk == SZ.v rows / SZ.v bm * (SZ.v cols / SZ.v bn));
  assert pure (SZ.v nthr == P.nthr bm bn wm wn);

  launch (
    mk_kernel
      gA #eA gB #eB gD post_map
      bm bn bk wm wn skew group
      fA fB nblk nthr ()
  ) s;
}
