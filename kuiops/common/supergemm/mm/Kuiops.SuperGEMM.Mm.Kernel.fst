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
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.ForEvery
open Pulse.Lib.Array { length }

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Kernel.Desc { kernel_desc }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', split_output_to_lanes', gather_output_live' }
open Kuiops.SuperGEMM.Mm.Barrier
  { skewed_view, pipe_live, pipe_q, pipe_contract, pipe_p_to_q_transform }
open Kuiops.SuperGEMM.Mm.Stage { geo_ok, g_t_row, g_t_col, g_row_step, g_a_iters }
open Kuiops.SuperGEMM.Mm.KLoop { kloop, acc_len_reveal }
open Kuiops.SuperGEMM.Mm.Epilogue { epilogue }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module B = Kuiper.Barrier
module P = Kuiops.SuperGEMM.Mm.Params
module SH = Kuiops.SuperGEMM.Mm.Shared

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

(* ---------------------------------------------------------------------- *)
(* setup : GPU-level pre transform                                        *)
(* ---------------------------------------------------------------------- *)

#push-options "--split_queries no"
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
(* teardown : reverse of setup                                            *)
(* ---------------------------------------------------------------------- *)

#push-options "--split_queries no"
ghost
fn teardown
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
  (#_ : squash (SZ.fits (SZ.v m * SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  ()
  requires
    (forall+ (bid : natlt nblk).
      SH.block_post gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid) ** pure True
  ensures
    (exists* (eA' : chest2 et_ab (SZ.v m) (SZ.v k)). gA |-> Frac fA eA') **
    (exists* (eB' : chest2 et_ab (SZ.v n) (SZ.v k)). gB |-> Frac fB eB') **
    (exists* (eD' : chest2 et_d (SZ.v m) (SZ.v n)). gD |-> eD')
{
  let mfw = wm /^ frag_sz;
  assert pure (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn);
  assert pure (SZ.v mfw == mfrag wm);
  assert pure (SZ.v mfw * SZ.v frag_sz == SZ.v wm);

  (* [forall+ bid. block_post] unfolds to [forall+ bid tid. kpost1];
     unfold each kpost1 into its components. *)
  forevery_map_2
    #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      SH.kpost1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid)
    (fun bid tid ->
      (exists* (eA' : chest2 et_ab (SZ.v m) (SZ.v k)). gA |-> Frac (fA /. (nblk * nthr)) eA') **
      ((exists* (eB' : chest2 et_ab (SZ.v n) (SZ.v k)). gB |-> Frac (fB /. (nblk * nthr)) eB') **
       output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid))
    fn bid tid {
      unfold SH.kpost1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid;
    };

  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> exists* (eA' : chest2 et_ab (SZ.v m) (SZ.v k)). gA |-> Frac (fA /. (nblk * nthr)) eA')
    (fun bid tid ->
      (exists* (eB' : chest2 et_ab (SZ.v n) (SZ.v k)). gB |-> Frac (fB /. (nblk * nthr)) eB') **
      output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid);
  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> exists* (eB' : chest2 et_ab (SZ.v n) (SZ.v k)). gB |-> Frac (fB /. (nblk * nthr)) eB')
    (fun bid tid ->
      output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid);

  (* gather A *)
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> exists* (eA' : chest2 et_ab (SZ.v m) (SZ.v k)). gA |-> Frac (fA /. (nblk * nthr)) eA');
  tensor_gather_n_underspec gA (nblk * nthr);

  (* gather B *)
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> exists* (eB' : chest2 et_ab (SZ.v n) (SZ.v k)). gB |-> Frac (fB /. (nblk * nthr)) eB');
  tensor_gather_n_underspec gB (nblk * nthr);

  (* gather D: bridge the band indices back to the [split]/[gather] szp form *)
  rewrite each (mfrag wm) as (SZ.v mfw);
  rewrite each frag as (SZ.v frag_sz);
  gather_output_live' gD bm bn frag_sz wn mfw 1sz nblk nthr ();
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

#push-options "--split_queries no --z3rlimit 15"


fn kf
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA) (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB) (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD)
  (post_map : et_acc -> et_d)
  (bm bn bk wm wn skew : szp)
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
  (sh : c_shmems (SH.shmems_desc et_ab et_acc bm bn bk wm wn skew) { c_shmems_inv sh })
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
  requires
    gpu **
    SH.kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok
      (pipe_contract bm bn bk skew
        (SH.sar_a0 bm bn bk wm wn skew sh) (SH.sar_a1 bm bn bk wm wn skew sh)
        (SH.sar_b0 bm bn bk wm wn skew sh) (SH.sar_b1 bm bn bk wm wn skew sh)
        (SZ.v nthr) (SZ.v k / SZ.v bk)) **
    B.barrier_state 0
  ensures
    gpu **
    SH.kpost gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok
      (pipe_contract bm bn bk skew
        (SH.sar_a0 bm bn bk wm wn skew sh) (SH.sar_a1 bm bn bk wm wn skew sh)
        (SH.sar_b0 bm bn bk wm wn skew sh) (SH.sar_b1 bm bn bk wm wn skew sh)
        (SZ.v nthr) (SZ.v k / SZ.v bk)) **
    B.barrier_state (SZ.v k / SZ.v bk)
{
  unfold SH.kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid;
  unfold SH.kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid;
  unfold SH.shared_thread_live bm bn bk wm wn skew sh nthr (SZ.v tid);

  (* shared-buffer 16-byte alignment for cp.async *)
  SH.shared_buffers_aligned16 bm bn bk wm wn skew sh ();

  (* frag divides wm/wn *)
  assert pure (mfrag wm * frag == SZ.v wm);

  (* ---- index decode ----
     TODO(perf): the reference applies a GROUP-based L2 swizzle here -- a pure
     bijection on the linear block index `bid` (regrouping blocks so that
     concurrently-resident CTAs share L2 working sets) before the div/mod
     decode below. It is a bijection on block indices, so it changes neither
     the ownership partition nor any proof obligation, and can be slotted in
     at exactly this point (rewrite `bid` -> `swizzle bid`) without touching
     anything downstream. Deliberately omitted for step 1. *)
  let num_n = n /^ bn;
  assert pure (SZ.v num_n == SZ.v n / SZ.v bn /\ SZ.v num_n > 0);
  div_ub (SZ.v bid) (SZ.v m / SZ.v bm) (SZ.v num_n);
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

  (* ---- hoisted staging addressing (computed ONCE) ---- *)
  let ch = chunk et_ab;
  let a_t_row   = (tid *^ ch) /^ bk;
  let a_t_col   = (tid *^ ch) %^ bk;
  let a_row_step = (ch *^ nthr) /^ bk;
  let a_iters   = (bm *^ bk) /^ (ch *^ nthr);
  let b_iters   = (bn *^ bk) /^ (ch *^ nthr);

  (* ---- pipelined main loop ---- *)
  let accFrags =
    kloop #et_ab #et_acc bm bn bk wm wn skew
      gA #eA gB #eB
      (SH.sar_a0 bm bn bk wm wn skew sh) (SH.sar_a1 bm bn bk wm wn skew sh)
      (SH.sar_b0 bm bn bk wm wn skew sh) (SH.sar_b1 bm bn bk wm wn skew sh)
      (fun x -> x)
      (fA /. (nblk * nthr)) (fB /. (nblk * nthr))
      nthr tid block_row block_col warp_m warp_n
      a_t_row a_t_col a_row_step a_iters
      a_t_row a_t_col a_row_step b_iters
      () () () () () () ();

  (* ---- epilogue: drain accumulator into D ---- *)
  with ems. assert (accFrags |-> ems);
  acc_len_reveal wm wn;
  epilogue gD post_map bm bn bk wm wn skew nblk nthr sh accFrags (SZ.v bid) (SZ.v tid) ();

  (* dispose accumulator fragments *)
  with ems'. assert (accFrags |-> ems');
  drop_ (accFrags |-> ems');

  (* ---- fold the post ---- *)
  fold SH.kpost1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid;
  fold SH.shared_thread_final bm bn bk wm wn skew sh nthr (SZ.v k / SZ.v bk) (SZ.v tid);
  fold SH.kpost gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid;
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

#push-options "--fuel 1 --ifuel 1 --z3rlimit 15 --split_queries always"
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_acc #et_d : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA { is_global gA }) (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB { is_global gB }) (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gD : array2 et_d lD { is_global gD })
  (post_map : et_acc -> et_d)
  (bm bn bk wm wn skew : szp)
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
      (exists* (eA' : chest2 et_ab (SZ.v m) (SZ.v k)). gA |-> Frac fA eA' **
        (exists* (eB' : chest2 et_ab (SZ.v n) (SZ.v k)). gB |-> Frac fB eB' **
          (exists* (eD' : chest2 et_d (SZ.v m) (SZ.v n)). gD |-> eD')))
=
  geo_facts et_ab et_acc bm bn bk wm wn skew;
  P.nthr_pos bm bn wm wn;
  P.nthr_le_max_threads et_ab et_acc bm bn bk wm wn skew;
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  let sq_bmn : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n) = () in
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
      SH.kpost gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmnk fA fB nblk nthr sh bid tid);

    barrier_contract = (fun _bid ptrs ->
      pipe_contract bm bn bk skew
        (SH.sar_a0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_a1 bm bn bk wm wn skew #sqc ptrs)
        (SH.sar_b0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_b1 bm bn bk wm wn skew #sqc ptrs)
        (SZ.v nthr) (SZ.v k / SZ.v bk));
    barrier_count = (fun _bid -> SZ.v k / SZ.v bk);
    barrier_ok = (fun _bid ptrs ->
      pipe_p_to_q_transform bm bn bk skew
        (SH.sar_a0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_a1 bm bn bk wm wn skew #sqc ptrs)
        (SH.sar_b0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_b1 bm bn bk wm wn skew #sqc ptrs)
        (SZ.v nthr) (SZ.v k / SZ.v bk));

    frame = pure True;

    block_pre  = (fun bid ->
      SH.block_pre gA eA gB eB gD bm bn wm wn #sq_blk6 fA fB nblk nthr bid);
    block_post = (fun bid ->
      SH.block_post gA eA gB eB gD bm bn wm wn #sq_blk6 fA fB nblk nthr bid);

    setup = setup #_ #et_acc #_ gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn fA fB nblk nthr;
    teardown = teardown #_ #et_acc #_ gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn #sq_mn fA fB nblk nthr;

    block_frame = (fun ptrs _bid -> SH.block_frame bm bn bk wm wn skew #sqc ptrs);
    block_setup = (fun sh bid ->
      SH.block_setup gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn fA fB nblk nthr sh bid);
    block_teardown = (fun sh bid ->
      SH.block_teardown gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmnk fA fB nblk nthr sh bid);

    f = kf gA #eA gB #eB gD post_map bm bn bk wm wn skew
          #sqc #sq_bmnk #sq_fits #sq_glob #sq_asAB #sq_asD #sq_geo #sq_vf
          fA fB nblk nthr;

    block_pre_sendable = solve;
    block_post_sendable = solve;
    kpre_sendable = (fun sh sh_inv bid tid ->
      SH.kpre_sendable gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmn
        fA fB nblk nthr sh #sh_inv bid tid);
    kpost_sendable = (fun sh sh_inv bid tid ->
      SH.kpost_sendable gA eA gB eB gD bm bn bk wm wn skew #sqc #sq_bmnk
        fA fB nblk nthr sh #sh_inv bid tid);
  }
#pop-options
