module Kuiops.SuperGEMM.Mm.Kernel

(* Module 7 of the software-pipelined tensor-core GEMM (D = A @ B^T).

   Ties the shared-memory pre/post ([Shared]), the pipelined barrier
   ([Barrier]), the compute loop ([KLoop]) and the epilogue ([Epilogue]) into a
   single [kernel_desc].  Step 1: memory safety only -- all data values are
   existentially quantified, no functional specification.

   The whole-kernel post existentialises A and B (not pinned to [eA]/[eB]):
   [kloop] re-materialises each fractional global read share through the
   cp.async batch pledge, which in step 1 does not track content preservation
   of a global read.  See [Shared.kpost1]. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiops.Array2.Strided
open Kuiper.TensorCore
open Kuiper.ForEvery
open Pulse.Lib.Array { length }

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Kernel.Desc { kernel_desc }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Output
  { output_lane_live', output_lane_approximates', output_fragment',
    split_output_to_lanes', gather_output_live', gather_output_approximates' }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc { own_lane_cells, live_lane_cells }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.Barrier
  { skewed_view, pipe_live, pipe_q, pipe_contract, pipe_p_to_q_transform,
    pipe_contract_c, pipe_p_to_q_transform_c }
open Kuiops.SuperGEMM.Mm.Stage { geo_ok, g_t_row, g_t_col, g_row_step, g_a_iters }
open Kuiops.SuperGEMM.Mm.KLoop { kloop, acc_len_reveal, acc_len_alloc }
open Kuiops.Kernel.GEMM.TensorCore2D.To.KLoop { populate_acc_with_zero }
open Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul, warp_matmul_is_subtile, mtranspose_subtile }
open Kuiops.SuperGEMM.Mm.Epilogue { epilogue }
open Kuiops.SuperGEMM.Mm.KernelLemmas
  { map_subtile_commute, subtile_subtile_compose, coerce_subtile_col,
    coerce_wm_nested, mfrag_frag_eq, lane_target_is_subtile, td_bounds }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module B = Kuiper.Barrier
module P = Kuiops.SuperGEMM.Mm.Params
module SH = Kuiops.SuperGEMM.Mm.Shared
module MS = Kuiper.Spec.GEMM
module ML = FStar.Math.Lemmas

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

(* ---------------------------------------------------------------------- *)
(* setup : GPU-level pre transform                                        *)
(* ---------------------------------------------------------------------- *)

#push-options ""
ghost
fn setup
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (core gD)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    live gD
  ensures
    (forall+ (bid : natlt nblk).
      SH.block_pre gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid) ** pure True
{
  (* frag divides wm/wn: reconcile the band tiling arguments. *)
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

  split_output_to_lanes' gD bm bn frag_sz wn mfw 1sz nblk nthr ();

  (* bridge the output tiling row index to [kpre1]'s exact form *)
  rewrite each (SZ.v frag_sz) as frag;
  rewrite each (SZ.v mfw) as (mfrag wm);

  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid);

  forevery_map_2
    #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      (gB |-> Frac (fB /. (nblk * nthr)) eB **
       output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid))
    (fun bid tid ->
      SH.kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid)
    fn bid tid {
      fold SH.kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid;
    };
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* Weaken the functional output-lane predicate to its liveness counterpart.*)
(* Throwaway bridge for step 1: [kpost1] is functional ([output_lane_      *)
(* approximates']) but the whole-matrix teardown gather is still           *)
(* [gather_output_live'] (the functional gather lands in a later pass), so *)
(* forget the approximation content per lane, keeping ownership.           *)
(* ---------------------------------------------------------------------- *)
ghost
fn lane_approx_to_live'
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (#m #n : nat) (#lD : layout2 m n)
  (gD : array2 et lD)
  (bm bn tm tn wm wn : pos)
  (#sq : squash (bm /?+ m /\ bn /?+ n /\ wm * tm /?+ bm /\ wn * tn /?+ bn))
  (bid : natlt (m / bm * (n / bn)))
  (tid : natlt (bm / (wm * tm) * (bn / (wn * tn)) * warp_size))
  (rD : chest2 real (wm * tm) (wn * tn))
  requires output_lane_approximates' gD bm bn tm tn wm wn #sq bid tid rD
  ensures  output_lane_live' gD bm bn tm tn wm wn #sq bid tid
{
  unfold output_lane_approximates' gD bm bn tm tn wm wn #sq bid tid rD;
  forevery_map_2
    #(natlt wm) #(natlt wn)
    (fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eD : chest2 et tm tn).
        own_lane_cells
          (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
          eD (tid % warp_size)
        ** pure (eD %~ ematrix_subtile rD tm tn mi nj))
    (fun (mi : natlt wm) (nj : natlt wn) ->
      live_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
        (tid % warp_size))
    fn mi nj {
      with eD. assert (own_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
        eD (tid % warp_size) ** pure (eD %~ ematrix_subtile rD tm tn mi nj));
      drop_ (pure (eD %~ ematrix_subtile rD tm tn mi nj));
      fold live_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid (tid / warp_size) mi nj)
        (tid % warp_size);
    };
  fold output_lane_live' gD bm bn tm tn wm wn #sq bid tid;
}

(* Pure chest/subtile algebra for the functional teardown lives in
   [Kuiops.SuperGEMM.Mm.KernelLemmas]. *)


(* ---------------------------------------------------------------------- *)
(* teardown : reverse of setup                                            *)
(* ---------------------------------------------------------------------- *)

(* Swap the (proof-irrelevant) real target of an [output_lane_approximates']
   for a provably-equal one.  A raw [rewrite] at the [output_lane_approximates']
   level forces Pulse to re-elaborate the heavy nested-subtile term (and its
   [tid]-refinement / divisor side-conditions) under no binder; doing it lane
   by lane after [unfold] discharges those obligations with [mi:natlt wm],
   [nj:natlt wn] in scope instead. *)
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

#push-options ""
ghost
fn teardown
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (post_map_r : real -> real)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  (fA fB : perm)
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  ()
  requires
    (forall+ (bid : natlt nblk).
      SH.block_post gA eA gB eB gD post_map_r bm bn wm wn fA fB rA rB nblk nthr bid) ** pure True
  ensures
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (exists* (eD' : chest2 et_d (SZ.v m) (SZ.v n)).
       gD |-> eD' **
       pure (eD' %~ chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB))))
{
  let mfw = wm /^ frag_sz;
  assert pure (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn);
  assert pure (SZ.v mfw == mfrag wm);
  assert pure (SZ.v mfw * SZ.v frag_sz == SZ.v wm);

  (* [forall+ bid. block_post] unfolds to [forall+ bid tid. kpost1]; keep the
     A/B pinning to [eA]/[eB] and rewrite each functional output lane from
     [lane_target] into the doubly-nested-subtile form of the whole real target
     that [gather_output_approximates'] consumes. *)
  forevery_map_2
    #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      SH.kpost1 gA eA gB eB gD post_map_r bm bn wm wn fA fB rA rB nblk nthr bid tid)
    (fun bid tid ->
      (gA |-> Frac (fA /. (nblk * nthr)) eA) **
      ((gB |-> Frac (fB /. (nblk * nthr)) eB) **
       output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
         (ematrix_subtile
            (ematrix_subtile (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))
              (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
            (mfrag wm * frag) (1 * SZ.v wn)
            ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
            ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn))))))
    fn bid tid {
      unfold SH.kpost1 gA eA gB eB gD post_map_r bm bn wm wn fA fB rA rB nblk nthr bid tid;
      td_bounds (SZ.v m) (SZ.v n) (SZ.v bm) (SZ.v bn) (SZ.v wm) (SZ.v wn)
        (SZ.v nblk) (SZ.v nthr) bid tid;
      lane_target_is_subtile rA rB post_map_r bm bn wm wn () nblk nthr bid tid
        (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn))
        ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
        ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)));
      assert (pure (mfrag wm * frag == SZ.v wm));
      assert (pure (1 * SZ.v wn == SZ.v wn));
      assert (pure (SZ.v n / SZ.v bn > 0));
      assert (pure (SZ.v bn / (1 * SZ.v wn) > 0));
      assert (pure (bid / (SZ.v n / SZ.v bn) < SZ.v m / SZ.v bm));
      assert (pure (bid % (SZ.v n / SZ.v bn) < SZ.v n / SZ.v bn));
      assert (pure ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)) < SZ.v bm / (mfrag wm * frag)));
      assert (pure ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)) < SZ.v bn / (1 * SZ.v wn)));
      assert (pure (SH.lane_target rA rB post_map_r bm bn wm wn nblk nthr bid tid
        == ematrix_subtile
             (ematrix_subtile (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))
               (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
             (mfrag wm * frag) (1 * SZ.v wn)
             ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
             ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))));
      lane_retarget gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
        (SH.lane_target rA rB post_map_r bm bn wm wn nblk nthr bid tid)
        (ematrix_subtile
           (ematrix_subtile (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))
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
      output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
        (ematrix_subtile
           (ematrix_subtile (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))
             (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
           (mfrag wm * frag) (1 * SZ.v wn)
           ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
           ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))));
  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
        (ematrix_subtile
           (ematrix_subtile (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))
             (SZ.v bm) (SZ.v bn) (bid / (SZ.v n / SZ.v bn)) (bid % (SZ.v n / SZ.v bn)))
           (mfrag wm * frag) (1 * SZ.v wn)
           ((tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)))
           ((tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))));

  (* gather A functionally: each read share re-materialises at [eA] *)
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_gather_n gA (nblk * nthr);

  (* gather B functionally *)
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_gather_n gB (nblk * nthr);

  (* gather D functionally: bridge the band indices back to the [split]/[gather]
     szp form, then reassemble the whole output certified against the real
     product. *)
  rewrite each (mfrag wm) as (SZ.v mfw);
  rewrite each frag as (SZ.v frag_sz);
  gather_output_approximates' gD bm bn frag_sz wn mfw 1sz nblk nthr
    (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)));
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* kf : one thread's kernel body                                          *)
(* ---------------------------------------------------------------------- *)

(* a < b*c /\ c > 0 ==> a/c < b. *)
let div_ub (a b c : nat)
  : Lemma (requires c > 0 /\ a < b * c) (ensures a / c < b)
=
  FStar.Math.Lemmas.lemma_div_mod a c;
  FStar.Math.Lemmas.lemma_mult_lt_left c (a / c) b

(* Both block-tile decode bounds in one standalone lemma: keeps the nonlinear
   [bid < (m/bm)*(n/bn)] discharge out of the giant [mk_kernel] record context,
   where the [barrier_ok] lambda's inline [div_ub] is Z3-flaky. *)
let bok_bounds (m n bm bn nblk bid : nat)
  : Lemma
    (requires m > 0 /\ n > 0 /\ bm > 0 /\ bn > 0 /\
              m % bm == 0 /\ n % bn == 0 /\
              nblk == m / bm * (n / bn) /\ bid < nblk)
    (ensures  n / bn > 0 /\ m / bm > 0 /\
              bid / (n / bn) < m / bm /\ bid % (n / bn) < n / bn)
= FStar.Math.Lemmas.lemma_div_mod m bm;
  FStar.Math.Lemmas.lemma_div_mod n bn;
  div_ub bid (m / bm) (n / bn);
  FStar.Math.Lemmas.lemma_mod_lt bid (n / bn)

(* ---- content-carrying block barrier contract ----
   The functional [kloop] consumes a CONTENT-carrying barrier ([pipe_contract_c],
   which pins each staged shared tile to the block's global A/B subtile) rather
   than the memory-safety [pipe_contract].  Naming it through this helper keeps
   the [block_row]/[block_col] refinement ([< m/bm], [< n/bn]) discharged ONCE
   (via [div_ub]) here, so neither [kf]'s contract nor the [barrier_contract]
   field re-derives the nonlinear bound. *)
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

#push-options " --z3rlimit 15"

(* Pure bridge for the [kf] post fold: the [rD] the epilogue produces
   ([chest_map post_map_r] over the coerced warp accumulator) is definitionally
   the [lane_target] [kpost1] expects, once the szt block/warp decode is written
   in [bid]/[tid] nat div/mod form.  Proving this in a pure lemma keeps the
   [lane_target] unfold (and its bound lemmas) out of the Pulse VC. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let lane_fold_bridge
  (#m #n #k : szp)
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (post_map_r : real -> real)
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
      Kuiops.SuperGEMM.Mm.Epilogue.coerce_chest2_cols
        #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
        (chest_map post_map_r rAcc)
      == SH.lane_target rA rB post_map_r bm bn wm wn #sq nblk nthr bid tid)
  = ()
#pop-options

inline_for_extraction noextract
fn kf
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc,
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
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
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
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (SH.shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
  requires
    gpu **
    pure (c_shmems_inv sh) **
    SH.kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok (bcontract eA eB bm bn bk wm wn skew nthr nblk (SZ.v bid) sh) **
    B.barrier_state 0
  ensures
    gpu **
    SH.kpost gA eA gB eB gD post_map_r bm bn bk wm wn skew fA fB rA rB nblk nthr sh bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok (bcontract eA eB bm bn bk wm wn skew nthr nblk (SZ.v bid) sh) **
    B.barrier_state (SZ.v k / SZ.v bk)
{
  unfold SH.kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid;
  unfold SH.kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid;
  unfold SH.shared_thread_live bm bn bk wm wn skew sh nthr (SZ.v tid);

  (* shared-buffer 16-byte alignment for cp.async *)
  SH.shared_buffers_aligned16 bm bn bk wm wn skew sh ();

  (* frag divides wm/wn *)
  assert pure (mfrag wm * frag == SZ.v wm);

  (* ---- index decode (plain row-major) ----
     Block [bid] owns the row-major tile [(bid / num_n, bid % num_n)], which is
     exactly the D ownership partition ([Output.block_tile] decodes [bid] the
     same way).  The [group] parameter is retained on [mk_kernel] (an inert L2
     swizzle knob) but is NOT applied here: a swizzled COMPUTE decode against a
     row-major OWNERSHIP partition would write each block's product to the wrong
     tile -- silent numerical garbage that the safety proof cannot see.  The
     functional postcondition requires compute and ownership to agree, so the
     decode is row-major.  Re-wiring the swizzle correctly means reindexing
     ownership too (see [Swizzle.sw_lin]); that is deferred. *)
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

  (* ---- hoisted staging addressing (computed ONCE) ----
     [nthrc] recomputes the thread count from the (compile-time) tiling so the
     staging trip counts / row step constant-fold at codegen, instead of being
     laundered through the runtime [nthr] kernel argument.  [SZ.v nthrc ==
     SZ.v nthr] definitionally (both reveal to [P.nthr bm bn wm wn]), so every
     downstream [squash] stated on [SZ.v nthr] is unaffected. *)
  let nthrc = P.nthr_sz bm bn wm wn;
  let ch : szp = chunk et_ab;
  let a_t_row   = (tid *^ ch) /^ bk;
  let a_t_col   = (tid *^ ch) %^ bk;
  let a_row_step = (ch *^ nthrc) /^ bk;
  let a_iters   = (bm *^ bk) /^ (ch *^ nthrc);
  let b_iters   = (bn *^ bk) /^ (ch *^ nthrc);

  (* ---- allocate the per-warp accumulator fragment array ONCE ----
     Hoisted out of [kloop] into [kf] (repo convention: kernels take their
     output buffers as arguments) so its KPR_INIT_ARR lands at this
     unconditional declaration site instead of flowing out of kloop's parity
     branch as an uninitialised C++ reference. *)
  let accFrags = __alloc_array_fragment et_acc FragAcc frag_sz frag_sz frag_sz FragLAcc ((wm /^ frag_sz) *^ (wn /^ frag_sz));
  acc_len_alloc wm wn;
  acc_len_reveal wm wn;
  assert pure (length accFrags == SZ.v wm / frag * (SZ.v wn / frag));

  (* zero the accumulator before the k-loop (route 1); bound folds to a
     compile-time literal since [wm]/[wn] are closure-captured szp params. *)
  populate_acc_with_zero #et_acc frag_sz frag_sz frag_sz (wm /^ frag_sz) (wn /^ frag_sz) accFrags;
  rewrite each SZ.v (wm /^ frag_sz) as (SZ.v wm / frag);
  rewrite each SZ.v (wn /^ frag_sz) as (SZ.v wn / frag);

  (* per-warp real output target, in [warp_matmul] form: the combined
     warp-row/col indices [block_row*(bm/wm)+warp_m] / [block_col*(bn/wn)+warp_n]
     are valid [natlt (m/wm)]/[natlt (n/wn)] (grow_bound). *)
  SH.grow_bound (SZ.v m) (SZ.v bm) (SZ.v wm) (SZ.v block_row) (SZ.v warp_m);
  SH.grow_bound (SZ.v n) (SZ.v bn) (SZ.v wn) (SZ.v block_col) (SZ.v warp_n);
  let grow : (g:erased nat { reveal g < SZ.v m / SZ.v wm }) =
    hide (SZ.v block_row * (SZ.v bm / SZ.v wm) + SZ.v warp_m);
  let gcol : (g:erased nat { reveal g < SZ.v n / SZ.v wn }) =
    hide (SZ.v block_col * (SZ.v bn / SZ.v wn) + SZ.v warp_n);

  (* ---- pipeline buffers ----
     Destructure [sh] as a tuple so KaRaMeL emits the cast
     `(et * ) KPR_SHMEM_AT(..)` on each shared pointer; the [sar_*] accessors (fst/snd
     projections) bypass that path and leave `void *` initialisers that nvcc
     rejects.  The [rewrites_to] lines reconcile the destructured names with
     the [sar_*] forms appearing in the pipe_live/pipe_q slprops. *)
  (* bridge the content-carrying block barrier to the [pipe_contract_c] form the
     functional [kloop] consumes: [bcontract]'s [block_row]/[block_col] are the
     nat div/mod of [bid]; reconcile them with the szt decode. *)
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

  (* ---- pipelined main loop ----
     Prime Z3 with the divides-transitivity facts [kloop]'s [sq_pc] needs
     ([wm | m], [wn | n] from [wm | bm | m]); without these hints the squash
     argument's SMT discharge fails and the whole application cores ill-typed. *)
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

  (* Reconcile [sh]: the [let]-pattern above substituted [sh] with the
     destructured tuple in the ambient slprops (e.g. [scratch_tile_live]); the
     epilogue's [preserves] are stated over the whole [sh].  Rewrite the tuple
     back to [sh] (surjective pairing discharges the equality). *)
  rewrite (SH.scratch_tile_live bm bn bk wm wn skew
             (sA0, (sA1, (sB0, (sB1, srest)))) nthr (SZ.v tid))
       as (SH.scratch_tile_live bm bn bk wm wn skew sh nthr (SZ.v tid));

  (* restore the block barrier to [bcontract] form for [kf]'s ensures *)
  rewrite (B.barrier_tok
             (pipe_contract_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
                sA0 sA1 sB0 sB1 (SZ.v nthr) ()))
       as (B.barrier_tok (bcontract eA eB bm bn bk wm wn skew nthr nblk (SZ.v bid) sh));

  (* ---- epilogue: drain accumulator into D ----
     Bridge kloop's per-warp product to the epilogue's [rAcc] parameter:
     [warp_matmul ... : chest2 real (SZ.v wm) (SZ.v wn)] reshaped to the
     epilogue's [(mfrag wm * frag, nfrag wn * frag)] tiling (both == wm/wn). *)
  assert pure (mfrag wm * frag == SZ.v wm);
  assert pure (nfrag wn * frag == SZ.v wn);
  assert pure (SZ.v m % (mfrag wm * frag) == 0);
  assert pure (SZ.v n % (nfrag wn * frag) == 0);
  let rAcc : chest2 real (mfrag wm * frag) (nfrag wn * frag) =
    warp_matmul rA rB (mfrag wm * frag) (nfrag wn * frag) (reveal grow) (reveal gcol);
  rewrite (fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
             (warp_matmul rA rB (SZ.v wm) (SZ.v wn) (reveal grow) (reveal gcol)))
       as (fragarrayAcc_approximates (mfrag wm) (nfrag wn) accFrags rAcc);
  epilogue gD post_map post_map_r bm bn bk wm wn skew nblk nthr sh accFrags rAcc bid tid ();

  (* dispose accumulator fragments *)
  unfold (fragarrayAcc_approximates (mfrag wm) (nfrag wn) accFrags rAcc);
  with ems'. assert (accFrags |-> ems');
  drop_ (accFrags |-> ems');

  (* ---- fold the post ----
     Bridge the epilogue's [rD] (chest_map over the coerced accumulator) to
     [kpost1]'s [lane_target] via the pure [lane_fold_bridge] (keeps the
     [lane_target] unfold out of the Pulse VC), then rewrite the slprop target
     by the resulting chest equality (congruence only). *)
  lane_fold_bridge #m #n #k rA rB post_map_r bm bn wm wn () ()
    nblk nthr (SZ.v bid) (SZ.v tid) (reveal grow) (reveal gcol) rAcc;
  rewrite (output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
             (SZ.v bid) (SZ.v tid)
             (Kuiops.SuperGEMM.Mm.Epilogue.coerce_chest2_cols
               #real #(mfrag wm * frag) #(nfrag wn * frag) #(1 * SZ.v wn) ()
               (chest_map post_map_r rAcc)))
       as (output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1
             (SZ.v bid) (SZ.v tid)
             (SH.lane_target rA rB post_map_r bm bn wm wn nblk nthr (SZ.v bid) (SZ.v tid)));
  fold SH.kpost1 gA eA gB eB gD post_map_r bm bn wm wn fA fB rA rB nblk nthr bid tid;
  fold SH.shared_thread_final bm bn bk wm wn skew sh nthr (SZ.v k / SZ.v bk) (SZ.v tid);
  fold SH.kpost gA eA gB eB gD post_map_r bm bn bk wm wn skew fA fB rA rB nblk nthr sh bid tid;
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* geo_ok facts: derive both staging geometries from [constraints] plus    *)
(* the one modular fact not implied by it, [(chunk*nthr) % bk == 0].        *)
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

(* Divides facts required when [output_lane_live'] unfolds inside the record
   field-type checks: [block_tile]/[warp_tile]/[subtile_layout] all demand the
   [/?] (divides) relation, but we only carry [/?+] (mod). Bridge them once,
   with an explicit proof, so the record queries have them as hypotheses. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
let output_tiling_divides (bm bn wm wn m n : pos)
  : Lemma
      (requires bm /?+ m /\ bn /?+ n /\ wm /?+ bm /\ wn /?+ bn /\ frag /?+ wm)
      (ensures  bm /? m /\ bn /? n /\ wn /? bn /\
                (wm / frag * frag) == wm /\
                (wm / frag * frag) /? bm /\ frag /? (wm / frag * frag))
=
  FStar.Math.Lemmas.euclidean_division_definition wm frag;
  Kuiper.Divides.lemma_nat_divides_pos_divides bm m;
  Kuiper.Divides.lemma_nat_divides_pos_divides bn n;
  Kuiper.Divides.lemma_nat_divides_pos_divides wn bn;
  Kuiper.Divides.lemma_nat_divides_pos_divides wm bm;
  Kuiper.Divides.lemma_nat_divides_pos_divides frag wm
#pop-options

(* [x < p*q ==> x/q < p]: the quotient bound that [block_tile_idx_rows] and
   [warp_tile_idx_rows] rely on for their [enatlt] return types. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
let div_lt_of_lt_mul (x p q : nat)
  : Lemma (requires q > 0 /\ x < p * q) (ensures x / q < p)
=
  FStar.Math.Lemmas.lemma_div_mod x q;
  if x / q >= p then FStar.Math.Lemmas.lemma_mult_le_left q p (x / q)

(* Quotient bounds for the output block/warp row indices, quantified over the
   (field-bound) block/warp ids so the record queries can instantiate them. *)
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
  FStar.Math.Lemmas.euclidean_division_definition wm frag;
  Kuiper.Divides.lemma_nat_divides_pos_divides bn n;
  Kuiper.Divides.lemma_nat_divides_pos_divides wn bn;
  Kuiper.Divides.lemma_nat_divides_pos_divides wm bm;
  (* [n/bn > 0], [bn/wn > 0]: divisors are <= dividends (SMTPat lemma_divides_le)
     and division is monotone. *)
  FStar.Math.Lemmas.lemma_div_le bn n bn;
  FStar.Math.Lemmas.lemma_div_le wn bn wn;
  introduce forall (x : nat{x < (m / bm) * (n / bn)}).
      x / (n / bn) < m / bm
  with div_lt_of_lt_mul x (m / bm) (n / bn);
  introduce forall (x : nat{x < (bm / (wm / frag * frag)) * (bn / wn)}).
      x / (bn / wn) < bm / (wm / frag * frag)
  with div_lt_of_lt_mul x (bm / (wm / frag * frag)) (bn / wn)
#pop-options

(* ---------------------------------------------------------------------- *)
(* mk_kernel : assemble the [kernel_desc]                                  *)
(* ---------------------------------------------------------------------- *)

(* [] is a 25x slowdown here (~1000 sub-queries) AND
   loses hypotheses in the field-type checks; one query per field is both
   faster and complete. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
inline_for_extraction noextract
let mk_kernel
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
          pure (eD' %~ chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))))
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
  (* Bridge the [/?+] (mod) divisibility facts we carry to the [/?] (divides)
     relation that [block_tile]/[warp_tile]/[subtile_layout] require when
     [output_lane_live'] unfolds inside the field-type checks. *)
  output_tiling_divides (SZ.v bm) (SZ.v bn) (SZ.v wm) (SZ.v wn) (SZ.v m) (SZ.v n);
  output_tiling_bounds (SZ.v bm) (SZ.v bn) (SZ.v wm) (SZ.v wn) (SZ.v m) (SZ.v n);
  {
    nblk;
    nthr;

    shmems_desc = SH.shmems_desc et_ab et_acc bm bn bk wm wn skew;

    kpre  = (fun sh bid tid ->
      SH.kpre gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn fA fB nblk nthr sh bid tid);
    kpost = (fun sh bid tid ->
      SH.kpost gA eA gB eB gD post_map_r bm bn bk wm wn skew #sqc #sq_bmnk fA fB rA rB nblk nthr sh bid tid);

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
      SH.block_pre gA eA gB eB gD bm bn wm wn #sq_blk6 fA fB nblk nthr bid);
    block_post = (fun bid ->
      SH.block_post gA eA gB eB gD post_map_r bm bn wm wn #sq_blk6 fA fB rA rB nblk nthr bid);

    setup = setup #_ #et_acc #_ gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn fA fB nblk nthr;
    teardown = teardown #_ #et_acc #_ gA eA gB eB gD post_map_r bm bn bk wm wn skew #sqc #sq_bmn #sq_mn fA fB rA rB nblk nthr;

    block_frame = (fun ptrs _bid -> SH.block_frame bm bn bk wm wn skew #sqc ptrs);
    block_setup = (fun sh bid ->
      SH.block_setup gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn fA fB nblk nthr sh bid);
    block_teardown = (fun sh bid ->
      SH.block_teardown gA eA gB eB gD post_map_r bm bn bk wm wn skew #sqc #sq_bmnk fA fB rA rB nblk nthr sh bid);

    f = kf gA #eA gB #eB gD post_map post_map_r bm bn bk wm wn skew group
          #sqc #sq_bmnk #sq_fits #sq_glob #sq_asAB #sq_asD #sq_geo #sq_vf
          fA fB nblk nthr;

    block_pre_sendable = solve;
    block_post_sendable = solve;
    kpre_sendable = (fun sh sh_inv bid tid ->
      SH.kpre_sendable gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn
        fA fB nblk nthr sh #sh_inv bid tid);
    kpost_sendable = (fun sh sh_inv bid tid ->
      SH.kpost_sendable gA eA gB eB gD post_map_r bm bn bk wm wn skew #sqc #sq_bmnk
        fA fB rA rB nblk nthr sh #sh_inv bid tid);
  }
#pop-options
