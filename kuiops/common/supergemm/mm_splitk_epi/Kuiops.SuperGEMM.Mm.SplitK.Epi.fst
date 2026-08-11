module Kuiops.SuperGEMM.Mm.SplitK.Epi

(* Implementation of the split-K + epilogue async launcher; see the interface
   for the contract.  Launch pass 1 (imported UNCHANGED from
   [Kuiops.SuperGEMM.Mm.SplitK.Kernel] -- it is byte-identical to the no-epi
   variant), synchronize the stream to obtain a fresh epoch under which pass
   1's workspace writes are visible, then launch the epilogue-aware reduction
   on that epoch.

   The only thing that differs from [Kuiops.SuperGEMM.Mm.SplitK] is the second
   launch: [comb] is affine in the accumulated value, so it may only be applied
   once a complete k reduction exists for an output element.  Pass 2 is that
   point; applying it in pass 1 would add the [beta * C] term once per split. *)

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
open Kuiops.SuperGEMM.Mm.SplitK.Kernel { mk_kernel }
open Kuiops.SuperGEMM.Mm.SplitK.Reduce { gran_ub_lemma, gcols }
open Kuiops.SuperGEMM.Mm.SplitK.WsLemmas { ws_target }
open Kuiops.SuperGEMM.Mm.SplitK.Epi.Reduce { mk_reduce_kernel }
open Kuiops.SuperGEMM.Mm.SplitK.Epi.ReduceLemmas { gran_target }
open Kuiops.SuperGEMM.Mm.SplitK.Epi.Compose { gran_target_mmcomb }

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module RO = Kuiper.TensorRO
module P = Kuiops.SuperGEMM.Mm.Params

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

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

{
  let nblk = (mws /^ bm) *^ (cols /^ bn);
  let nthr = (bm /^ wm) *^ (bn /^ wn) *^ warp_size;

  assert pure (SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v cols / SZ.v bn));
  assert pure (SZ.v nthr == P.nthr bm bn wm wn);

  launch (
    mk_kernel gA #eA gB #eB gW (to_real_matrix eA) (to_real_matrix eB)
      bm bn bk wm wn skew splits ks fA fB nblk nthr ()
  ) s;

  let e' = sync_stream s;
  redeem_pledge emp_inames (epoch_done s e)
    (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
                 (exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v cols)).
                    (gW |-> eW) **
                    pure (eW %~ ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                                  (to_real_matrix eA) (to_real_matrix eB) ()))));

  with eW. assert (on gpu_loc (gW |-> eW));

  let njobs = rows *^ (cols /^ chunk et_d);
  gran_ub_lemma (SZ.v rows) (SZ.v cols / SZ.v (chunk et_d));
  assert pure (SZ.v njobs == SZ.v rows * gcols et_d cols);

  launch (mk_reduce_kernel gW gC gD splits comb comb_r () njobs 1.0R eW fC eC
            (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
               (to_real_matrix eA) (to_real_matrix eB) ())
            (to_real_matrix eC) ()) s;

  gran_target_mmcomb (SZ.v mws) (SZ.v splits) (SZ.v ks)
    (to_real_matrix eC) (to_real_matrix eA) (to_real_matrix eB) comb_r ();

  return_pledge (epoch_done s e') (on gpu_loc (gA |-> Frac fA eA)) #solve;
  return_pledge (epoch_done s e') (on gpu_loc (gB |-> Frac fB eB)) #solve;
  join_pledge (on gpu_loc (gA |-> Frac fA eA)) (on gpu_loc (gB |-> Frac fB eB));
  join_pledge
    ((on gpu_loc (gA |-> Frac fA eA)) ** (on gpu_loc (gB |-> Frac fB eB)))
    (on gpu_loc ((gW |-> Frac 1.0R eW) ** (gC |-> Frac fC eC) **
                 (exists* (eD : chest2 et_d (SZ.v rows) (SZ.v cols)).
                    (gD |-> eD) **
                    pure (eD %~ gran_target (SZ.v rows) (SZ.v splits)
                                  (to_real_matrix eC)
                                  (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                                     (to_real_matrix eA) (to_real_matrix eB) ())
                                  comb_r))));
  rewrite_pledge
    (((on gpu_loc (gA |-> Frac fA eA)) ** (on gpu_loc (gB |-> Frac fB eB))) **
     (on gpu_loc ((gW |-> Frac 1.0R eW) ** (gC |-> Frac fC eC) **
                  (exists* (eD : chest2 et_d (SZ.v rows) (SZ.v cols)).
                     (gD |-> eD) **
                     pure (eD %~ gran_target (SZ.v rows) (SZ.v splits)
                                   (to_real_matrix eC)
                                   (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                                      (to_real_matrix eA) (to_real_matrix eB) ())
                                   comb_r)))))
    (on gpu_loc
      (exists* (eW' : chest2 et_acc (SZ.v mws) (SZ.v cols))
               (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)).
         (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> Frac fC eC) **
         (gW |-> eW') ** (gD |-> eD') **
         pure (eD' %~ MS.mmcomb comb_r (to_real_matrix eC)
                 (to_real_matrix eA) (mtranspose (to_real_matrix eB)))))
    #emp_inames fn () { () };
  drop_ (epoch_done s e);
  e'
}
