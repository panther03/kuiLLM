module Kuiops.SuperGEMM.Mm.Epi.Kernel

(* Kernel body and [kernel_desc] of the C-combining variant of the
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

open Kuiops.SuperGEMM.Mm.Epi.KernelBody { setup, teardown, kf, bok_bounds, bcontract }

(* ---------------------------------------------------------------------- *)
(* geo/tiling facts (copies of the [Mm.Kernel] private helpers)            *)
(* ---------------------------------------------------------------------- *)

let geo_facts
  (et_ab et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  : Lemma
      (requires
        P.constraints et_ab et_acc bm bn bk wm wn skew /\
        (SZ.v (chunk et_ab) * P.nthr bm bn wm wn) % SZ.v bk == 0)
      (ensures
        geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn) /\
        geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn))
=
  P.nthr_pos bm bn wm wn;
  P.chunk_nthr_divides_ab et_ab et_acc bm bn bk wm wn skew

#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
let output_tiling_divides (bm bn wm wn m n : pos)
  : Lemma
      (requires bm /?+ m /\ bn /?+ n /\ wm /?+ bm /\ wn /?+ bn /\ frag /?+ wm)
      (ensures  bm /? m /\ bn /? n /\ wn /? bn /\
                (wm / frag * frag) == wm /\
                (wm / frag * frag) /? bm /\ frag /? (wm / frag * frag))
=
  ML.euclidean_division_definition wm frag;
  Kuiper.Divides.lemma_nat_divides_pos_divides bm m;
  Kuiper.Divides.lemma_nat_divides_pos_divides bn n;
  Kuiper.Divides.lemma_nat_divides_pos_divides wn bn;
  Kuiper.Divides.lemma_nat_divides_pos_divides wm bm;
  Kuiper.Divides.lemma_nat_divides_pos_divides frag wm
#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
let div_lt_of_lt_mul (x p q : nat)
  : Lemma (requires q > 0 /\ x < p * q) (ensures x / q < p)
=
  ML.lemma_div_mod x q;
  if x / q >= p then ML.lemma_mult_le_left q p (x / q)

let output_tiling_bounds (bm bn wm wn m n : pos)
  : Lemma
      (requires bm /?+ m /\ bn /?+ n /\ wm /?+ bm /\ wn /?+ bn /\ frag /?+ wm)
      (ensures
        (forall (x : nat{x < (m / bm) * (n / bn)}). {:pattern (x / (n / bn))}
           x / (n / bn) < m / bm) /\
        (forall (x : nat{x < (bm / (wm / frag * frag)) * (bn / wn)}).
           {:pattern (x / (bn / wn))}
           x / (bn / wn) < bm / (wm / frag * frag)))
=
  ML.euclidean_division_definition wm frag;
  Kuiper.Divides.lemma_nat_divides_pos_divides bn n;
  Kuiper.Divides.lemma_nat_divides_pos_divides wn bn;
  Kuiper.Divides.lemma_nat_divides_pos_divides wm bm;
  ML.lemma_div_le bn n bn;
  ML.lemma_div_le wn bn wn;
  introduce forall (x : nat{x < (m / bm) * (n / bn)}).
      x / (n / bn) < m / bm
  with div_lt_of_lt_mul x (m / bm) (n / bn);
  introduce forall (x : nat{x < (bm / (wm / frag * frag)) * (bn / wn)}).
      x / (bn / wn) < bm / (wm / frag * frag)
  with div_lt_of_lt_mul x (bm / (wm / frag * frag)) (bn / wn)
#pop-options

(* ---------------------------------------------------------------------- *)
(* mk_kernel                                                              *)
(* ---------------------------------------------------------------------- *)

(* [--split_queries always] is a 40x slowdown here (~1000 sub-queries) AND
   loses hypotheses in the field-type checks; one query per field is both
   faster and complete. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_acc #et_c #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc,
     scalar et_c, real_like et_c,
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
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n)) {| RO.cvtlayout lC |}
  (gC : RO.roarray2 et_c lC { RO.is_global gC }) (#eC : chest2 et_c (SZ.v m) (SZ.v n))
       (#rC : chest2 real (SZ.v m) (SZ.v n) { eC %~ rC })
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD { is_global gD })
  (comb : et_c -> et_acc -> et_d)
  (comb_r : real -> real -> real { approx2 comb comb_r })
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
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (#_ : squash (SZ.v nblk <= SZ.v max_blocks))
  ()
  : kernel_desc
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       gC |-> Frac fC eC **
       live gD)
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       gC |-> Frac fC eC **
       (exists* (eD' : chest2 et_d (SZ.v m) (SZ.v n)).
          gD |-> eD' **
          pure (eD' %~ MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))))
=
  geo_facts et_ab et_acc bm bn bk wm wn skew;
  P.nthr_pos bm bn wm wn;
  P.nthr_le_max_threads et_ab et_acc bm bn bk wm wn skew;
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  let sq_bmn : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n) = () in
  let sq_bcon : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                        SZ.v bk /?+ SZ.v k) = () in
  let sq_blk6 : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                        SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                        frag /?+ SZ.v wm /\ frag /?+ SZ.v wn) = () in
  let sq_mn : squash (SZ.fits (SZ.v m * SZ.v n)) = () in
  let sq_geo : squash
      (geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn) /\
       geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn)) = () in
  let sq_glob : squash (is_global gA /\ is_global gB) = () in
  output_tiling_divides (SZ.v bm) (SZ.v bn) (SZ.v wm) (SZ.v wn) (SZ.v m) (SZ.v n);
  output_tiling_bounds (SZ.v bm) (SZ.v bn) (SZ.v wm) (SZ.v wn) (SZ.v m) (SZ.v n);
  {
    nblk;
    nthr;

    shmems_desc = SH.shmems_desc et_ab et_acc bm bn bk wm wn skew;

    kpre  = (fun sh bid tid ->
      kpre_c gA eA gB eB gC eC gD bm bn bk wm wn skew #sqc #sq_bmn
        fA fB fC nblk nthr sh bid tid);
    kpost = (fun sh bid tid ->
      kpost_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew #sqc #sq_bmnk
        fA fB fC rC rA rB nblk nthr sh bid tid);

    barrier_contract = (fun _bid ptrs ->
      bcontract eA eB bm bn bk wm wn skew #sqc #sq_bcon nthr nblk _bid ptrs);
    barrier_count = (fun _bid -> SZ.v k / SZ.v bk);
    barrier_ok = (fun _bid ptrs ->
      let num_n = SZ.v n / SZ.v bn in
      bok_bounds (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) (SZ.v nblk) _bid;
      pipe_p_to_q_transform_c m n k bm bn bk skew eA eB (_bid / num_n) (_bid % num_n)
        (SH.sar_a0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_a1 bm bn bk wm wn skew #sqc ptrs)
        (SH.sar_b0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_b1 bm bn bk wm wn skew #sqc ptrs)
        (SZ.v nthr) ());

    frame = pure True;

    block_pre  = (fun bid ->
      block_pre_c gA eA gB eB gC eC gD bm bn wm wn #sq_blk6 fA fB fC nblk nthr bid);
    block_post = (fun bid ->
      block_post_c gA eA gB eB gC eC gD comb_r bm bn wm wn #sq_blk6
        fA fB fC rC rA rB nblk nthr bid);

    setup = setup #_ #et_acc #_ #_ gA eA gB eB gC eC gD bm bn bk wm wn skew
              #sqc #sq_bmn fA fB fC nblk nthr;
    teardown = teardown #_ #et_acc #_ #_ gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew
              #sqc #sq_bmn #sq_mn fA fB fC rC rA rB nblk nthr;

    block_frame = (fun ptrs _bid -> SH.block_frame bm bn bk wm wn skew #sqc ptrs);
    block_setup = (fun sh bid ->
      block_setup_c gA eA gB eB gC eC gD bm bn bk wm wn skew #sqc #sq_bmn
        fA fB fC nblk nthr sh bid);
    block_teardown = (fun sh bid ->
      block_teardown_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew #sqc #sq_bmnk
        fA fB fC rC rA rB nblk nthr sh bid);

    f = kf gA #eA gB #eB gC #eC gD comb comb_r bm bn bk wm wn skew group
          #sqc #sq_bmnk #sq_fits #sq_glob #sq_asAB #sq_asD #sq_geo #sq_vf
          fA fB fC nblk nthr;

    block_pre_sendable = solve;
    block_post_sendable = solve;
    kpre_sendable = (fun sh sh_inv bid tid ->
      kpre_sendable_c gA eA gB eB gC eC gD bm bn bk wm wn skew #sqc #sq_bmn
        fA fB fC nblk nthr sh #sh_inv bid tid);
    kpost_sendable = (fun sh sh_inv bid tid ->
      kpost_sendable_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew #sqc #sq_bmnk
        fA fB fC rC rA rB nblk nthr sh #sh_inv bid tid);
  }
#pop-options

