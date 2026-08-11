module Kuiops.SuperGEMM.Mm.Epi.KernelBody

(* Per-thread kernel body of the C-combining variant of the
   software-pipelined tensor-core GEMM:

     D = comb C (A @ B^T)

   Identical to [Kuiops.SuperGEMM.Mm.Kernel] except that a read-only C view is
   threaded from the launcher down to the epilogue: [setup] hands every thread
   a fractional read share of the whole C tensor (C may be read
   non-injectively, so no partition is possible), the epilogue combines it into
   the drain, and [teardown] gathers the shares back.

   The shared-memory pipeline, staging, barrier and k-loop are
   [Kuiops.SuperGEMM.Mm.*] verbatim. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.ForEvery
open Pulse.Lib.Array { length }

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Kernel.Desc { kernel_desc }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Output
  { output_lane_live', output_lane_approximates', output_fragment',
    split_output_to_lanes', gather_output_approximates' }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc { own_lane_cells, live_lane_cells }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.Barrier
  { skewed_view, pipe_live, pipe_q, pipe_contract, pipe_p_to_q_transform,
    pipe_contract_c, pipe_p_to_q_transform_c }
open Kuiops.SuperGEMM.Mm.Stage { geo_ok }
open Kuiops.SuperGEMM.Mm.KLoop { kloop, acc_len_reveal, acc_len_alloc }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KLoop { populate_acc_with_zero }
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul }
open Kuiops.SuperGEMM.Mm.KernelLemmas { mfrag_frag_eq, td_bounds }
open Kuiops.SuperGEMM.Mm.Epi.EpilogueLemmas { lane_c_target, coerce_chest2_cols }
open Kuiops.SuperGEMM.Mm.Epi.Epilogue { epilogue }
open Kuiops.SuperGEMM.Mm.Epi.Shared
  { lane_target_c, kpre1_c, kpost1_c, kpre_c, kpost_c, block_pre_c, block_post_c,
    block_setup_c, block_teardown_c, kpre_sendable_c, kpost_sendable_c }
open Kuiops.SuperGEMM.Mm.Epi.KernelLemmas { lane_target_c_is_subtile }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module RO = Kuiper.TensorRO
module B = Kuiper.Barrier
module P = Kuiops.SuperGEMM.Mm.Params
module SH = Kuiops.SuperGEMM.Mm.Shared
module MS = Kuiper.Spec.GEMM
module ML = FStar.Math.Lemmas

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

(* ---------------------------------------------------------------------- *)
(* setup : GPU-level pre transform                                        *)
(* ---------------------------------------------------------------------- *)

#push-options "--split_queries no"
ghost
fn setup
  (#et_ab #et_acc #et_c #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     scalar et_c, scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (core gD)))
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> Frac fC eC **
    live gD
  ensures
    (forall+ (bid : natlt nblk).
      block_pre_c gA eA gB eB gC eC gD bm bn wm wn fA fB fC nblk nthr bid) ** pure True
{
  let mfw = wm /^ frag_sz;
  assert pure (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn);
  assert pure (SZ.v mfw == mfrag wm);
  assert pure (SZ.v mfw * SZ.v frag_sz == SZ.v wm);
  assert pure (SZ.v 1sz * SZ.v wn == SZ.v wn);

  tensor_share_n gA (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_share_n gB (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  RO.tensor_share_n gC (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);

  split_output_to_lanes' gD bm bn frag_sz wn mfw 1sz nblk nthr ();

  rewrite each (SZ.v frag_sz) as frag;
  rewrite each (SZ.v mfw) as (mfrag wm);

  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      (gC |-> Frac (fC /. (nblk * nthr)) eC) **
      output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      (gB |-> Frac (fB /. (nblk * nthr)) eB) **
      (gC |-> Frac (fC /. (nblk * nthr)) eC) **
      output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid);

  forevery_map_2
    #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      (gA |-> Frac (fA /. (nblk * nthr)) eA) **
      ((gB |-> Frac (fB /. (nblk * nthr)) eB) **
       (gC |-> Frac (fC /. (nblk * nthr)) eC) **
       output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid))
    (fun bid tid ->
      kpre1_c gA eA gB eB gC eC gD bm bn wm wn fA fB fC nblk nthr bid tid)
    fn bid tid {
      fold SH.kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid;
    };
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* teardown : reverse of setup                                            *)
(* ---------------------------------------------------------------------- *)

(* Copy of [Mm.Kernel.lane_retarget] (private there): swap the
   proof-irrelevant real target of an [output_lane_approximates'] for a
   provably-equal one, lane by lane. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 30"
ghost
fn lane_retarget
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (#m #n : nat)
  (#lD : layout2 m n)
  (gD : array2 et lD)
  (bm bn tm tn wm wn : pos)
  (#sq : squash (bm /?+ m /\ bn /?+ n /\ wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  (rD1 rD2 : chest2 real (wm * tm) (wn * tn))
  (_ : squash (rD1 == rD2))
  requires output_lane_approximates' gD bm bn tm tn wm wn #sq bid tid rD1
  ensures  output_lane_approximates' gD bm bn tm tn wm wn #sq bid tid rD2
{
  unfold output_lane_approximates' gD bm bn tm tn wm wn #sq bid tid rD1;
  forevery_map_2
    #(natlt wm) #(natlt wn)
    (fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eD : chest2 et tm tn).
        own_lane_cells
          (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
          eD (tid % warp_size)
        ** pure (eD %~ ematrix_subtile rD1 tm tn mi nj))
    (fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eD : chest2 et tm tn).
        own_lane_cells
          (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
          eD (tid % warp_size)
        ** pure (eD %~ ematrix_subtile rD2 tm tn mi nj))
    fn mi nj {
      with eD. assert (own_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
        eD (tid % warp_size) ** pure (eD %~ ematrix_subtile rD1 tm tn mi nj));
      rewrite (pure (eD %~ ematrix_subtile rD1 tm tn mi nj))
        as (pure (eD %~ ematrix_subtile rD2 tm tn mi nj));
    };
  fold output_lane_approximates' gD bm bn tm tn wm wn #sq bid tid rD2;
}
#pop-options

#push-options "--split_queries no"
ghost
fn teardown
  (#et_ab #et_acc #et_c #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     scalar et_c, scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (comb_r : real -> real -> real)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  (fA fB fC : perm)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  ()
  requires
    (forall+ (bid : natlt nblk).
      block_post_c gA eA gB eB gC eC gD comb_r bm bn wm wn
        fA fB fC rC rA rB nblk nthr bid) ** pure True
  ensures
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (gC |-> Frac fC eC) **
    (exists* (eD' : chest2 et_d (SZ.v m) (SZ.v n)).
       gD |-> eD' **
       pure (eD' %~ MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB)))
{
  let mfw = wm /^ frag_sz;
  assert pure (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn);
  assert pure (SZ.v mfw == mfrag wm);
  assert pure (SZ.v mfw * SZ.v frag_sz == SZ.v wm);

  forevery_map_2
    #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      kpost1_c gA eA gB eB gC eC gD comb_r bm bn wm wn
        fA fB fC rC rA rB nblk nthr bid tid)
    (fun bid tid ->
      (gA |-> Frac (fA /. (nblk * nthr)) eA) **
      ((gB |-> Frac (fB /. (nblk * nthr)) eB) **
       ((gC |-> Frac (fC /. (nblk * nthr)) eC) **
        output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
          (ematrix_subtile
             (ematrix_subtile (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
               (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
             (mfrag wm * frag) (1 * SZ.v wn)
             ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
             ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))))))
    fn bid tid {
      td_bounds (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) (SZ.v wm) (SZ.v wn)
        (SZ.v nblk) (SZ.v nthr) bid tid;
      lane_target_c_is_subtile rC rA rB comb_r bm bn wm wn () nblk nthr bid tid
        (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn))
        ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
        ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)));
      assert (pure (mfrag wm * frag == SZ.v wm));
      assert (pure (1 * SZ.v wn == SZ.v wn));
      assert (pure (lane_target_c rC rA rB comb_r bm bn wm wn nblk nthr bid tid
        == ematrix_subtile
             (ematrix_subtile (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
               (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
             (mfrag wm * frag) (1 * SZ.v wn)
             ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
             ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))));
      lane_retarget gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
        (lane_target_c rC rA rB comb_r bm bn wm wn nblk nthr bid tid)
        (ematrix_subtile
           (ematrix_subtile (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
             (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
           (mfrag wm * frag) (1 * SZ.v wn)
           ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
           ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn))))
        ();
    };

  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      (gB |-> Frac (fB /. (nblk * nthr)) eB) **
      ((gC |-> Frac (fC /. (nblk * nthr)) eC) **
       output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
         (ematrix_subtile
            (ematrix_subtile (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
              (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
            (mfrag wm * frag) (1 * SZ.v wn)
            ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
            ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn))))));
  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      (gC |-> Frac (fC /. (nblk * nthr)) eC) **
      output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
        (ematrix_subtile
           (ematrix_subtile (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
             (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
           (mfrag wm * frag) (1 * SZ.v wn)
           ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
           ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))));
  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
        (ematrix_subtile
           (ematrix_subtile (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
             (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
           (mfrag wm * frag) (1 * SZ.v wn)
           ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
           ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))));

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_gather_n gA (nblk * nthr);

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_gather_n gB (nblk * nthr);

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  RO.tensor_gather_n gC (nblk * nthr);

  rewrite each (mfrag wm) as (SZ.v mfw);
  rewrite each frag as (SZ.v frag_sz);
  gather_output_approximates' gD bm bn frag_sz wn mfw 1sz nblk nthr
    (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB));
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* kf : one thread's kernel body                                          *)
(* ---------------------------------------------------------------------- *)

let div_ub (a b c : nat)
  : Lemma (requires c > 0 /\ a < b * c) (ensures a / c < b)
=
  ML.lemma_div_mod a c;
  ML.lemma_mult_lt_left c (a / c) b

let bok_bounds (m n bm bn nblk bid : nat)
  : Lemma
    (requires m > 0 /\ n > 0 /\ bm > 0 /\ bn > 0 /\
              m % bm == 0 /\ n % bn == 0 /\
              nblk == m / bm * (n / bn) /\ bid < nblk)
    (ensures  n / bn > 0 /\ m / bm > 0 /\
              bid / (n / bn) < m / bm /\ bid % (n / bn) < n / bn)
= ML.lemma_div_mod m bm;
  ML.lemma_div_mod n bn;
  div_ub bid (m / bm) (n / bn);
  ML.lemma_mod_lt bid (n / bn)

let bcontract
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\ SZ.v bk /?+ SZ.v k))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (bid : natlt nblk)
  (ptrs : c_shmems (SH.shmems_desc et_ab et_acc bm bn bk wm wn skew))
  : B.contract (SZ.v nthr)
= let num_n = SZ.v n / SZ.v bn in
  div_ub bid (SZ.v m / SZ.v bm) num_n;
  pipe_contract_c m n k bm bn bk skew eA eB (bid / num_n) (bid % num_n)
    (SH.sar_a0 bm bn bk wm wn skew ptrs) (SH.sar_a1 bm bn bk wm wn skew ptrs)
    (SH.sar_b0 bm bn bk wm wn skew ptrs) (SH.sar_b1 bm bn bk wm wn skew ptrs)
    (SZ.v nthr) ()

(* C-combining counterpart of [Mm.Kernel.lane_fold_bridge]: the epilogue's real
   target IS [lane_target_c], definitionally, once the block/warp decode is
   written in [bid]/[tid] nat div/mod form. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let lane_fold_bridge_c
  (#m #n #k : szp)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (comb_r : real -> real -> real)
  (bm bn wm wn : szp)
  (sq : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (sq2 : squash (SZ.v m % (mfrag wm * frag) == 0 /\ SZ.v n % (nfrag wn * frag) == 0))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  (grow_v : natlt (SZ.v m / SZ.v wm))
  (gcol_v : natlt (SZ.v n / SZ.v wn))
  (rAcc : chest2 real (mfrag wm * frag) (nfrag wn * frag))
  : Lemma
    (requires
      grow_v == (bid / (SZ.v n / SZ.v bn)) * (SZ.v bm / SZ.v wm)
                 + (tid / warp_size) / (SZ.v bn / SZ.v wn) /\
      gcol_v == (bid % (SZ.v n / SZ.v bn)) * (SZ.v bn / SZ.v wn)
                 + (tid / warp_size) % (SZ.v bn / SZ.v wn) /\
      rAcc
        == warp_matmul rA rB (mfrag wm * frag) (nfrag wn * frag) grow_v gcol_v)
    (ensures
      coerce_chest2_cols
        #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
        (Kuiper.Chest.chest_comb comb_r
          (lane_c_target rC bm bn wm wn #sq nblk nthr bid tid) rAcc)
      == lane_target_c rC rA rB comb_r bm bn wm wn #sq nblk nthr bid tid)
  = ()
#pop-options

#push-options "--split_queries no --z3rlimit 15"
inline_for_extraction noextract
fn kf
  (#et_ab #et_acc #et_c #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc,
     scalar et_c, real_like et_c,
     scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA) (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
       (#rA : chest2 real (SZ.v m) (SZ.v k) { eA %~ rA })
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB) (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
       (#rB : chest2 real (SZ.v n) (SZ.v k) { eB %~ rB })
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n)) {| RO.cvtlayout lC |}
  (gC : RO.roarray2 et_c lC) (#eC : chest2 et_c (SZ.v m) (SZ.v n))
       (#rC : chest2 real (SZ.v m) (SZ.v n) { eC %~ rC })
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (comb : et_c -> et_acc -> et_d)
  (comb_r : real -> real -> real { approx2 comb comb_r })
  (bm bn bk wm wn skew group : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v k) /\ SZ.fits (SZ.v n * SZ.v k) /\
                SZ.fits (SZ.v m * SZ.v n)))
  (#_ : squash (is_global gA /\ is_global gB))
  (#_ : squash (aligned_strided_row_major (SZ.v (chunk et_ab)) str_A /\
                aligned_strided_row_major (SZ.v (chunk et_ab)) str_B))
  (#_ : squash (aligned_strided_row_major (SZ.v (chunk et_d)) strD))
  (#_ : squash (geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn) /\
                geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA frag frag frag /\
                valid_frag_et_dims et_ab FragB frag frag frag /\
                valid_frag_et_dims et_acc FragAcc frag frag frag /\
                valid_frag_et_comb et_ab et_acc /\
                SZ.fits (SZ.v wm / frag * (SZ.v wn / frag))))
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (SH.shmems_desc et_ab et_acc bm bn bk wm wn skew) { c_shmems_inv sh })
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre_c gA eA gB eB gC eC gD bm bn bk wm wn skew fA fB fC nblk nthr sh bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok (bcontract eA eB bm bn bk wm wn skew nthr nblk (SZ.v bid) sh) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew
      fA fB fC rC rA rB nblk nthr sh bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok (bcontract eA eB bm bn bk wm wn skew nthr nblk (SZ.v bid) sh) **
    B.barrier_state (SZ.v k / SZ.v bk)
{
  unfold SH.kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid;
  unfold SH.kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid;
  unfold SH.shared_thread_live bm bn bk wm wn skew sh nthr (SZ.v tid);

  SH.shared_buffers_aligned16 bm bn bk wm wn skew sh ();

  assert pure (mfrag wm * frag == SZ.v wm);

  let num_n = n /^ bn;
  assert pure (SZ.v num_n == SZ.v n / SZ.v bn /\ SZ.v num_n > 0);
  let num_m = m /^ bm;
  assert pure (SZ.v num_m == SZ.v m / SZ.v bm /\ SZ.v num_m > 0);
  div_ub (SZ.v bid) (SZ.v num_m) (SZ.v num_n);
  let block_row : szlt (SZ.v m / SZ.v bm) = bid /^ num_n;
  let block_col : szlt (SZ.v n / SZ.v bn) = bid %^ num_n;

  let wid = tid /^ warp_size;
  assert pure (SZ.v nthr == warps bm bn wm wn * SZ.v warp_size);
  div_ub (SZ.v tid) (warps bm bn wm wn) (SZ.v warp_size);
  assert pure (SZ.v wid < warps bm bn wm wn);
  let wnn = bn /^ wn;
  assert pure (SZ.v wnn == SZ.v bn / SZ.v wn /\ SZ.v wnn > 0);
  div_ub (SZ.v wid) (SZ.v bm / SZ.v wm) (SZ.v wnn);
  let warp_m : szlt (SZ.v bm / SZ.v wm) = wid /^ wnn;
  let warp_n : szlt (SZ.v bn / SZ.v wn) = wid %^ wnn;

  let nthrc = P.nthr_sz bm bn wm wn;
  let ch : szp = chunk et_ab;
  let a_t_row   = (tid *^ ch) /^ bk;
  let a_t_col   = (tid *^ ch) %^ bk;
  let a_row_step = (ch *^ nthrc) /^ bk;
  let a_iters   = (bm *^ bk) /^ (ch *^ nthrc);
  let b_iters   = (bn *^ bk) /^ (ch *^ nthrc);

  let accFrags = __alloc_array_fragment et_acc FragAcc frag_sz frag_sz frag_sz FragLAcc ((wm /^ frag_sz) *^ (wn /^ frag_sz));
  acc_len_alloc wm wn;
  acc_len_reveal wm wn;
  assert pure (length accFrags == SZ.v wm / frag * (SZ.v wn / frag));

  populate_acc_with_zero #et_acc frag_sz frag_sz frag_sz (wm /^ frag_sz) (wn /^ frag_sz) accFrags;
  rewrite each SZ.v (wm /^ frag_sz) as (SZ.v wm / frag);
  rewrite each SZ.v (wn /^ frag_sz) as (SZ.v wn / frag);

  SH.grow_bound (SZ.v m) (SZ.v bm) (SZ.v wm) (SZ.v block_row) (SZ.v warp_m);
  SH.grow_bound (SZ.v n) (SZ.v bn) (SZ.v wn) (SZ.v block_col) (SZ.v warp_n);
  let grow : (g:erased nat { reveal g < SZ.v m / SZ.v wm }) =
    hide (SZ.v block_row * (SZ.v bm / SZ.v wm) + SZ.v warp_m);
  let gcol : (g:erased nat { reveal g < SZ.v n / SZ.v wn }) =
    hide (SZ.v block_col * (SZ.v bn / SZ.v wn) + SZ.v warp_n);

  assert pure (SZ.v block_row == SZ.v bid / (SZ.v n / SZ.v bn));
  assert pure (SZ.v block_col == SZ.v bid % (SZ.v n / SZ.v bn));
  rewrite (B.barrier_tok (bcontract eA eB bm bn bk wm wn skew nthr nblk (SZ.v bid) sh))
       as (B.barrier_tok
             (pipe_contract_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
                (SH.sar_a0 bm bn bk wm wn skew sh) (SH.sar_a1 bm bn bk wm wn skew sh)
                (SH.sar_b0 bm bn bk wm wn skew sh) (SH.sar_b1 bm bn bk wm wn skew sh)
                (SZ.v nthr) ()));

  let (sA0, (sA1, (sB0, (sB1, srest)))) = sh;
  assert rewrites_to sA0 (SH.sar_a0 bm bn bk wm wn skew sh);
  assert rewrites_to sA1 (SH.sar_a1 bm bn bk wm wn skew sh);
  assert rewrites_to sB0 (SH.sar_b0 bm bn bk wm wn skew sh);
  assert rewrites_to sB1 (SH.sar_b1 bm bn bk wm wn skew sh);

  assert pure (SZ.v k > 0);
  Kuiper.Divides.lemma_divides_trans (SZ.v wm) (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_divides_trans (SZ.v wn) (SZ.v bn) (SZ.v n);
  assert pure (SZ.v wm /?+ SZ.v m);
  assert pure (SZ.v wn /?+ SZ.v n);
  kloop #et_ab #et_acc bm bn bk wm wn skew
    gA #eA gB #eB
    sA0 sA1 sB0 sB1
    accFrags
    (fun x -> x)
    (fA /. (nblk * nthr)) (fB /. (nblk * nthr))
    nthr tid block_row block_col warp_m warp_n
    rA rB grow gcol
    a_t_row a_t_col a_row_step a_iters
    a_t_row a_t_col a_row_step b_iters
    () () () () () () () () () () ();

  rewrite (SH.scratch_tile_live bm bn bk wm wn skew
             (sA0, (sA1, (sB0, (sB1, srest)))) nthr (SZ.v tid))
       as (SH.scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid));

  rewrite (B.barrier_tok
             (pipe_contract_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
                sA0 sA1 sB0 sB1 (SZ.v nthr) ()))
       as (B.barrier_tok (bcontract eA eB bm bn bk wm wn skew nthr nblk (SZ.v bid) sh));

  assert pure (mfrag wm * frag == SZ.v wm);
  assert pure (nfrag wn * frag == SZ.v wn);
  assert pure (SZ.v m % (mfrag wm * frag) == 0);
  assert pure (SZ.v n % (nfrag wn * frag) == 0);
  let rAcc : chest2 real (mfrag wm * frag) (nfrag wn * frag) =
    warp_matmul rA rB (mfrag wm * frag) (nfrag wn * frag) (reveal grow) (reveal gcol);
  rewrite (fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
             (warp_matmul rA rB (SZ.v wm) (SZ.v wn) (reveal grow) (reveal gcol)))
       as (fragarrayAcc_approximates (mfrag wm) (nfrag wn) accFrags rAcc);
  epilogue #et_ab #et_c #et_acc #et_d gD gC rC comb comb_r
    bm bn bk wm wn skew nblk nthr sh accFrags rAcc bid tid ();

  unfold (fragarrayAcc_approximates (mfrag wm) (nfrag wn) accFrags rAcc);
  with ems'. assert (accFrags |-> ems');
  drop_ (accFrags |-> ems');

  lane_fold_bridge_c #m #n #k rC rA rB comb_r bm bn wm wn () ()
    nblk nthr (SZ.v bid) (SZ.v tid) (reveal grow) (reveal gcol) rAcc;
  rewrite (output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
             (SZ.v bid) (SZ.v tid)
             (coerce_chest2_cols
               #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
               (Kuiper.Chest.chest_comb comb_r
                 (lane_c_target rC bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid)) rAcc)))
       as (output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
             (SZ.v bid) (SZ.v tid)
             (lane_target_c rC rA rB comb_r bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid)));
  fold SH.shared_thread_final bm bn bk wm wn skew sh nthr (SZ.v k / SZ.v bk) (SZ.v tid);
}
#pop-options

