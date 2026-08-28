module Kuiops.SuperGEMM.Mm.SplitK

(* Implementation of the split-K async launcher; see the interface for the
   contract.  Launch pass 1, then launch the reduction directly on the pledge
   pass 1 produced -- no synchronization, so the pair is graph-safe. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Chest { chest_map }
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.TensorCore
open Kuiops.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.SplitK.Kernel { mk_kernel }
open Kuiops.SuperGEMM.Mm.SplitK.Reduce { mk_reduce_kernel, gran_ub_lemma, gcols }
open Kuiops.Approx.Share { approx_pts_to }
open Kuiops.Kernel.Frame { desc_frame }
open Kuiops.SuperGEMM.Mm.SplitK.WsLemmas { ws_target }
open Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas { gran_target }
open Kuiops.SuperGEMM.Mm.SplitK.Compose { gran_target_matmul }

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

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
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires
    Kuiops.Epoch.epoch_live s e **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (live gD) **
    on gpu_loc (live gW)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next (Kuiops.Epoch.epoch_next e)) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next (Kuiops.Epoch.epoch_next e)))
      (on gpu_loc
        (exists* (eW' : chest2 et_acc (SZ.v mws) (SZ.v cols))
                 (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)).
           (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
           (gW |-> eW') ** (gD |-> eD') **
           pure (eD' %~ chest_map post_map_r
                   (MS.matmul (to_real_matrix eA) (mtranspose (to_real_matrix eB))))))
{
  let nblk = (mws /^ bm) *^ (cols /^ bn);
  let nthr = (bm /^ wm) *^ (bn /^ wn) *^ warp_size;

  assert pure (SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v cols / SZ.v bn));
  assert pure (SZ.v nthr == P.nthr bm bn wm wn);

  Kuiops.Kernel.launch (
    mk_kernel gA #eA gB #eB gW (to_real_matrix eA) (to_real_matrix eB)
      bm bn bk wm wn skew splits ks fA fB nblk nthr ()
  ) s;

  let njobs = rows *^ (cols /^ chunk et_d);
  gran_ub_lemma (SZ.v rows) (SZ.v cols / SZ.v (chunk et_d));
  assert pure (SZ.v njobs == SZ.v rows * gcols et_d cols);

  (* Bring the reduce kernel's precondition to the queue position pass 1 left
     behind: [live gD] is owned outright, so it is injected with a trivial
     pledge and joined onto pass 1's.  A and B ride along as a frame -- pass 2
     never touches them, but [Kuiops.Kernel.launch_pledged] consumes the whole pledge. *)
  return_pledge (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e)) (on gpu_loc (live gD))
    #(is_send_placeless (on gpu_loc (live gD)) #(placeless_on gpu_loc (live gD)));
  join_pledge
    (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
                 (exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v cols)).
                    (gW |-> eW) **
                    pure (eW %~ ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                                  (to_real_matrix eA) (to_real_matrix eB) ()))))
    (on gpu_loc (live gD));
  rewrite_pledge
    ((on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
                  (exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v cols)).
                     (gW |-> eW) **
                     pure (eW %~ ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                                   (to_real_matrix eA) (to_real_matrix eB) ())))) **
     (on gpu_loc (live gD)))
    (on gpu_loc
      (((gA |-> Frac fA eA) ** (gB |-> Frac fB eB)) **
       ((exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v cols)).
           (gW |-> Frac 1.0R eW) **
           pure (eW %~ ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                         (to_real_matrix eA) (to_real_matrix eB) ())) **
        live gD)))
    #emp_inames fn () { () };

  Kuiops.Kernel.launch_pledged
    (desc_frame ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB))
       (mk_reduce_kernel gW gD splits post_map post_map_r () njobs 1.0R
          (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
             (to_real_matrix eA) (to_real_matrix eB) ()) ())) s;

  gran_target_matmul (SZ.v mws) (SZ.v splits) (SZ.v ks)
    (to_real_matrix eA) (to_real_matrix eB) post_map_r ();

  rewrite_pledge
    (on gpu_loc
      (((gA |-> Frac fA eA) ** (gB |-> Frac fB eB)) **
       ((exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v cols)).
           (gW |-> Frac 1.0R eW) **
           pure (eW %~ ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                         (to_real_matrix eA) (to_real_matrix eB) ())) **
        (exists* (eD : chest2 et_d (SZ.v rows) (SZ.v cols)).
           (gD |-> eD) **
           pure (eD %~ gran_target (SZ.v rows) (SZ.v splits)
                         (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks)
                            (to_real_matrix eA) (to_real_matrix eB) ())
                         post_map_r)))))
    (on gpu_loc
      (exists* (eW' : chest2 et_acc (SZ.v mws) (SZ.v cols))
               (eD' : chest2 et_d (SZ.v rows) (SZ.v cols)).
         (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
         (gW |-> eW') ** (gD |-> eD') **
         pure (eD' %~ chest_map post_map_r
                 (MS.matmul (to_real_matrix eA) (mtranspose (to_real_matrix eB))))))
    #emp_inames fn () { () };
}
