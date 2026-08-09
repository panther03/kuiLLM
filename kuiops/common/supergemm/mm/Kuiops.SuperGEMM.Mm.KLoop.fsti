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
open Kuiops.SuperGEMM.Mm.Params { frag, ldt }
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_live, pipe_q, pipe_contract }
open Kuiops.SuperGEMM.Mm.Stage { geo_ok, g_row_step, g_a_iters, g_t_row, g_t_col }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module B = Kuiper.Barrier
module P = Kuiops.SuperGEMM.Mm.Params

(* Opaque length of the per-warp accumulator fragment array ([mfrag x nfrag]).
   Kept opaque so [kloop]'s postcondition carries no [SZ.v]-mul/div arithmetic
   into its (rlimit-bounded) branch VCs; callers bridge with [acc_len_reveal]. *)
[@@"opaque_to_smt"]
let acc_len (wm wn : szp) : nat = SZ.v wm / frag * (SZ.v wn / frag)

val acc_len_reveal (wm wn : szp)
  : Lemma (acc_len wm wn == SZ.v wm / frag * (SZ.v wn / frag))

val acc_len_alloc (wm wn : szp)
  : Lemma (requires SZ.fits (SZ.v (wm /^ P.frag_sz) * SZ.v (wn /^ P.frag_sz)))
          (ensures  acc_len wm wn == SZ.v ((wm /^ P.frag_sz) *^ (wn /^ P.frag_sz)))

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
  (#_ : squash (length accFrags == acc_len wm wn))
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

(* [kloop] is the software-pipelined main loop for one thread block: it stages
   k-tile 0 into buffer 0, then for each k-tile [kt] waits on the prior
   cp.async batch, waits the pipeline barrier once, stages the NEXT k-tile into
   the opposite buffer (double-buffered, buffer = kt & 1), and runs [subproducts]
   on the current buffer.  The stage-into-the-other-buffer happens BEFORE the
   math so global->shared transfer overlaps tensor-core compute.

   Step 1 = memory safety only: the returned accumulator fragments are only
   [live] (values existentially quantified); no real-number specification.

   IMPORTANT PRECONDITION FOR THE KERNEL AGENT: [kloop] requires 4x [pipe_live]
   -- writable chunks of ALL FOUR shared buffers (sarA0, sarA1, sarB0, sarB1).
   The double buffers are staged/consumed by parity, and the writable
   ([pipe_live], = [own_strided_chunks]) capability is what [stage_tiles] needs
   for its cp.async destination.  If the caller holds read shares
   ([pipe_sharing], = the barrier's post-wait product), it MUST convert
   [pipe_sharing] -> [pipe_live] for all four buffers before calling [kloop].
   On exit the thread holds [pipe_q ... (k/bk-1)] (read shares of the last-used
   buffer plus live chunks of the other), [barrier_state (k/bk)], the two global
   operands back (existential values), and [live accFrags].

   The addressing arguments [a_t_row]/[a_t_col]/[a_row_step]/[a_iters] (and the
   [b_*] mirror) are the hoisted per-thread staging geometry (div/mod computed
   once by the caller); [sq_a] pins them to the [g_*] spec.  [barrier_count] for
   the [pipe_contract] is [k/bk] (one barrier per k-tile). *)
inline_for_extraction noextract
fn kloop
  (#et_ab #et_acc : Type0)
  {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  {| _sac : scalar et_acc |} {| _vac : has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (bm bn bk wm wn skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| _clA : T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| _clB : T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (fmap : et_ab -> et_ab)
  (fA fB : perm)
  (nthr : szp { SZ.v nthr == P.nthr bm bn wm wn })
  (tid : szlt nthr)
  (block_row : szlt (SZ.v m / SZ.v bm))
  (block_col : szlt (SZ.v n / SZ.v bn))
  (warp_m : szlt (SZ.v bm / SZ.v wm))
  (warp_n : szlt (SZ.v bn / SZ.v wn))
  (a_t_row a_t_col a_row_step a_iters : SZ.t)
  (b_t_row b_t_col b_row_step b_iters : SZ.t)
  (sq_c : squash (P.constraints et_ab et_acc bm bn bk wm wn skew))
  (sq_g : squash (geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) /\
                  geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr)))
  (sq_d : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\ SZ.v bk /?+ SZ.v k /\
                  SZ.v bk <= SZ.v k /\ SZ.fits (SZ.v m * SZ.v k) /\ SZ.fits (SZ.v n * SZ.v k)))
  (sq_a : squash (
     SZ.v a_t_row == g_t_row (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) (SZ.v tid) /\
     SZ.v a_t_col == g_t_col (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) (SZ.v tid) /\
     SZ.v a_row_step == g_row_step (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) /\
     SZ.v a_iters == g_a_iters (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) /\
     SZ.v b_t_row == g_t_row (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) (SZ.v tid) /\
     SZ.v b_t_col == g_t_col (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) (SZ.v tid) /\
     SZ.v b_row_step == g_row_step (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) /\
     SZ.v b_iters == g_a_iters (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr)))
  (sq_al : squash (
     aligned 16 (T.core gA) /\ aligned 16 (T.core gB) /\
     aligned_strided_row_major (SZ.v (chunk et_ab)) str_A /\
     aligned_strided_row_major (SZ.v (chunk et_ab)) str_B /\
     is_global gA /\ is_global gB /\
     aligned 16 (T.core (skewed_view bm bk skew sarA0)) /\
     aligned 16 (T.core (skewed_view bm bk skew sarA1)) /\
     aligned 16 (T.core (skewed_view bn bk skew sarB0)) /\
     aligned 16 (T.core (skewed_view bn bk skew sarB1))))
  (sq_v : squash (
     valid_frag_et_dims et_ab FragA frag frag frag /\
     valid_frag_et_dims et_ab FragB frag frag frag /\
     valid_frag_et_dims et_acc FragAcc frag frag frag /\
     valid_frag_et_comb et_ab et_acc /\
     SZ.fits (SZ.v wm / frag * (SZ.v wn / frag)) /\
     length accFrags == acc_len wm wn))
  ()
  preserves gpu
  preserves thread_id (SZ.v nthr) (SZ.v tid)
  preserves live accFrags
  preserves B.barrier_tok
    (pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk))
  requires
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
    B.barrier_state 0 **
    pipe_live (skewed_view bm bk skew sarA0) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bm bk skew sarA1) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bn bk skew sarB0) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bn bk skew sarB1) (SZ.v nthr) (SZ.v tid)
  ensures
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    B.barrier_state (SZ.v k / SZ.v bk) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)
           (SZ.v k / SZ.v bk - 1) (SZ.v tid)
