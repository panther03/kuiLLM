module Kuiops.GEMM.T.TensorCore2D.KLoop

(* Fork of [Kuiper.Kernel.GEMM.TensorCore2D.To.KLoop] for the transposed-B
   ("TN") GEMM: global B is column-major (ld = k) and the shared-memory B
   tile is stored column-major in the same flat allocation.

   Delta versus upstream:
     - shmem B tile viewed as [cm bk bn] instead of [rm bk bn]
     - B fragments are [FragLCM] (see Kuiops.GEMM.T.TensorCore2D.Fragments)
     - [str_B] is an EXPLICIT [strided_col_major lB] argument, not an
       instance.  It must be [Kuiops.Array2.Strided.ColMajor.scm_l2_col_major]:
       upstream's [strided_col_major_l2_col_major] is an [instance val], so
       its offset/stride are abstract outside [Kuiper.Array2.Strided] and the
       alignment obligation below is unprovable against it.
     - the barrier is [Kuiops.GEMM.T.FlipFlopBarrier2], whose B side names the
       COLUMN-major chunk partition
     - [copy_tiles_out_of_matrices_vec] is inlined: its A block verbatim, and
       a B block that copies at dims (bn, bk) over transposed views.

   TODO(upstream): the reason the copy wrapper had to be inlined is that
   [Kuiper.Kernel.GEMM.Tiled.Common.Vec.copy_tiles_out_of_matrices_vec] pins
   [slB : layout2 bk bn], i.e. it forces both operands onto the same tile
   dims.  The real upstream fix is to let it take independent dims per
   operand, after which this file could call it again. *)

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiops.GEMM.T.FlipFlopBarrier2
module CV2 = Kuiper.Kernel.GEMM.Copy.Vec2
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc


open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState
open Kuiops.GEMM.T.TensorCore2D.Fragments
open Kuiops.Tensor.Transpose2
open Kuiops.Array2.Strided.ColMajor
open Kuiper.Kernel.GEMM.TensorCore2D.To.KLoopInvariant
open Kuiper.TensorRO { vtlayout_of_tlayout }

(* Folds the row-major chunk ownership of the transposed view back into the
   column-major predicate.  [own_strided_chunks_cm m em] is *definitionally*
   [own_strided_chunks (atranspose m) (ctranspose em)], so the only work here
   is rewriting the chest, which is done by [rewrite each] on a variable
   (reliable) rather than on a compound term (not). *)
ghost
fn fold_own_strided_chunks_cm
  (#et : Type0) {| sized et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (em : chest2 et rows cols)
  (em' : chest2 et cols rows)
  (nthr : nat)
  (tid : natlt nthr)
  requires
    CV2.own_strided_chunks (atranspose m) em' nthr tid **
    pure (em' == ctranspose em)
  ensures
    FB.own_strided_chunks_cm m em nthr tid
{
  rewrite each em' as (ctranspose em);
  rewrite CV2.own_strided_chunks (atranspose m) (ctranspose em) nthr tid
    as FB.own_strided_chunks_cm m em nthr tid;
}

(* Local copy of the divisibility helper from
   [Kuiper.Kernel.GEMM.Tiled.Common.Vec.fst]; it is private to that module's
   implementation and so is not importable.  Identical to upstream. *)
let divides_helper
  (d : pos)
  (a b r c : int)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
  = Kuiper.Divides.lemma_divides_product_l d b r;
    Kuiper.Divides.lemma_divides_sum d a (b * r);
    Kuiper.Divides.lemma_divides_sum d (a + b * r) c;
    ()

inline_for_extraction noextract
fn populate_acc_with_zero
  (#et : Type0) {| sc : scalar et, real_like et |}
  (tm tn tk wm wn : szp)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length accumFrags == wm*wn))
requires
  live accumFrags
ensures
  fragarrayAcc_approximates wm wn accumFrags (const _ 0.0R)
{
  array_fragment_pts_to_ref accumFrags;

  let mut fi : sz = 0sz;
  while (!fi <^ wm*^wn)
    invariant
      live fi **
      (exists* (eAcc : seq (chest2 et tm tn)).
        accumFrags |-> eAcc **
        pure (
          Seq.length eAcc == wm*wn /\ !fi <= wm*wn  /\
          forall (i : natlt !fi).
            Seq.index eAcc i %~ const _ 0.0R))
    decreases (wm*^wn - !fi)
  {
    array_fragment_pts_to_ref accumFrags;
    array_fragment_extract accumFrags !fi;
    mma_fill accumFrags.(!fi) sc.zero;

    Pulse.Lib.Forall.elim_forall (fill_value sc.zero);
    ambig_trade_elim();

    fi := !fi +^ 1sz;
  };
  fold fragarrayAcc_approximates wm wn accumFrags (const _ 0.0R);
  ()
}

(* TODO(upstream): [Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Epilogue]
   has this lemma but does not export it from its interface. *)
let ematrix_subtile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (e : chest2 et rows cols)
  (r : chest2 real rows cols)
  (trows : pos{trows /?+ rows})
  (tcols : pos{tcols /?+ cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma
      (requires e %~ r)
      (ensures
        ematrix_subtile e trows tcols tr tc
          %~ ematrix_subtile r trows tcols tr tc)
= ()

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn k_loop_step
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  {| real_like et_ab, real_like et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (str_B : strided_col_major (vtlayout_of_tlayout lB))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_col_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab k n)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (wm * tm /?+ m /\ wn * tn /?+ n))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#_ : squash (eA %~ rA /\ eB %~ rB))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (cm bk bn))
  (#fA #fB : perm)
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n / bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm / (wm * tm)) {
    SZ.v warpRow == (SZ.v tid / warp_size) /
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (warpCol : szlt (bn / (wn * tn)) {
    SZ.v warpCol == (SZ.v tid / warp_size) %
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (gwRow : enatlt (m / (wm * tm)) {
    gwRow == mrow * (bm / (wm * tm)) + warpRow})
  (gwCol : enatlt (n / (wn * tn)) {
    gwCol == mcol * (bn / (wn * tn)) + warpCol})
  (rAcc0 : chest2 real (wm * tm) (wn * tn) {
    rAcc0 == const _ 0.0R})
  (aFrags : array (fragment et_ab FragA tm tn tk FragLRM))
  (bFrags : array (fragment et_ab FragB tm tn tk FragLCM))
  (accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc))
  (#_ : squash (Pulse.Lib.Array.length aFrags == wm))
  (#_ : squash (Pulse.Lib.Array.length bFrags == wn))
  (#_ : squash (Pulse.Lib.Array.length accFrags == wm * wn))
  (v : szle (k / bk))
  (#_ : squash (SZ.v v < k / bk))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (cm bk bn)
        (core sA) (core sB) nthr bid)
  requires
    fragarrayAcc_approximates wm wn accFrags
      (tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol v) **
    live aFrags **
    live bFrags **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (barrier_iteration v)
  returns vnext : szle (k / bk)
  ensures
    pure (SZ.v vnext == SZ.v v + 1) **
    fragarrayAcc_approximates wm wn accFrags
      (tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol (v + 1)) **
    live aFrags **
    live bFrags **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (barrier_iteration vnext)
{
  even_2x v;
  assert pure ((2 * v % 2 = 0) == true);
  assert pure (even (2 * v));

  FB.fold_barrier_p_even eA eB sA sB nthr bid v tid;
  rewrite (FB.barrier_p eA eB sA sB nthr bid) (2 * v) tid
    as (FB.contract eA eB (rm bm bk) (cm bk bn)
      (core sA) (core sB) nthr bid).rin (2 * v) tid;

  B.barrier_wait ();

  rewrite (FB.contract eA eB (rm bm bk) (cm bk bn)
      (core sA) (core sB) nthr bid).rout (2 * v) tid
    as (FB.barrier_q eA eB sA sB nthr bid) (2 * v) tid;
  FB.unfold_barrier_q_even eA eB sA sB nthr bid v tid;

  unfold FB.live_strided_chunks sA nthr tid;
  with eA0. assert (FB.own_strided_chunks sA eA0 nthr tid);
  rewrite FB.own_strided_chunks sA eA0 nthr tid
    as CV2.own_strided_chunks sA eA0 nthr tid;
  fold CV2.live_strided_chunks sA nthr tid;
  unfold FB.live_strided_chunks_cm sB nthr tid;
  unfold FB.live_strided_chunks (atranspose sB) nthr tid;
  with eB0. assert (FB.own_strided_chunks (atranspose sB) eB0 nthr tid);
  rewrite FB.own_strided_chunks (atranspose sB) eB0 nthr tid
    as CV2.own_strided_chunks (atranspose sB) eB0 nthr tid;
  fold CV2.live_strided_chunks (atranspose sB) nthr tid;

  (* The copies take the thread count as the literal tile expression, not the
     [nthr] parameter: [nthr] is opaque at extraction, so the copy loop's stride
     stays a runtime value and nvcc can neither unroll it nor fold the index
     arithmetic.  Upstream's [copy_tiles_out_of_matrices_vec] call does the
     same. *)

  (* ---- A tile: verbatim from upstream copy_tiles_out_of_matrices_vec ---- *)
  {
    unfold CV2.live_strided_chunks sA nthr tid;
    let tileA = array2_extract_tile_ro' gA
      (SZ.v bm) (SZ.v bk) (SZ.v mrow) (SZ.v v);

    Kuiper.Divides.lemma_divides_product_l (chunk et_ab) str_A.stride (mrow * bm);
    Kuiper.Divides.lemma_divides_product_r (chunk et_ab) v bk;
    divides_helper (chunk et_ab) str_A.offset str_A.stride (mrow * bm) (v * bk);

    cp_array2_vec bm bk tileA sA
      (bm /^ (wm *^ tm) *^ (bn /^ (wn *^ tn)) *^ warp_size) tid;

    elim_trade _ _;
  };

  (* ---- B tile: COLUMN-major.
     The shmem tile's physical address is addr(kk,nn) = nn*bk + kk.
     [CV2.in_chunk] flattens as ij._1 * cols + ij._2, so the partition
     coincides with the physical address ONLY at dims (bn, bk) over the
     transposed view -- hence [cp_array2_vec bn bk], not [bk bn].  The
     (bk, bn) ordering type-checks silently and is fully uncoalesced.
     Self-check: [cp_array2_vec] demands [chunk et /?+ cols], and below
     that obligation lands on bk, matching the barrier fork. ---- *)
  {
    unfold CV2.live_strided_chunks (atranspose sB) nthr tid;
    atranspose_fwd gB;
    let tileB = array2_extract_tile_ro' (atranspose gB)
      (SZ.v bn) (SZ.v bk) (SZ.v mcol) (SZ.v v);

    lemma_aligned_srm_of_scm str_B (chunk et_ab);
    Kuiper.Divides.lemma_divides_product_l (chunk et_ab) str_B.stride (mcol * bn);
    Kuiper.Divides.lemma_divides_product_r (chunk et_ab) v bk;
    divides_helper (chunk et_ab) str_B.offset str_B.stride (mcol * bn) (v * bk);

    cp_array2_vec bn bk tileB (atranspose sB)
      (bm /^ (wm *^ tm) *^ (bn /^ (wn *^ tn)) *^ warp_size) tid;

    elim_trade _ _;
    atranspose_back gB;
    lemma_ctranspose_involutive eB;
    rewrite each (ctranspose (ctranspose eB)) as eB;
  };

  rewrite
    CV2.own_strided_chunks sA (ematrix_subtile eA bm bk mrow v) nthr tid
    as FB.own_strided_chunks sA (ematrix_subtile eA bm bk mrow v) nthr tid;
  lemma_ctranspose_subtile eB (SZ.v bk) (SZ.v bn) (SZ.v v) (SZ.v mcol);
  fold_own_strided_chunks_cm sB
    (ematrix_subtile eB (SZ.v bk) (SZ.v bn) (SZ.v v) (SZ.v mcol))
    (ematrix_subtile (ctranspose eB) (SZ.v bn) (SZ.v bk) (SZ.v mcol) (SZ.v v))
    (SZ.v nthr) (SZ.v tid);

  odd_2x1 v;
  assert pure (odd (2 * v + 1));
  FB.fold_barrier_p_odd eA eB sA sB nthr bid mrow mcol v tid;
  rewrite (FB.barrier_p eA eB sA sB nthr bid) (2 * v + 1) tid
    as (FB.contract eA eB (rm bm bk) (cm bk bn)
      (core sA) (core sB) nthr bid).rin (2 * v + 1) tid;

  B.barrier_wait ();

  even_2x (v + 1);
  assert pure (2 * (v + 1) == 2 * v + 2);
  assert pure (odd (2 * v + 1));
  (* Upstream discharges this by SMT alone; the extra pure facts introduced by
     the column-major staging path above make the nonlinear step unreliable, so
     spell it out: bk divides k, hence (2*k)/bk == 2*(k/bk). *)
  FStar.Math.Lemmas.multiple_division_lemma (2 * (SZ.v k / SZ.v bk)) (SZ.v bk);
  assert pure (2 * SZ.v k == (2 * (SZ.v k / SZ.v bk)) * SZ.v bk);
  assert pure ((2 * v + 1) < 2 * k / bk);
  assert pure (even (2 * v + 2));
  rewrite (FB.contract eA eB (rm bm bk) (cm bk bn)
      (core sA) (core sB) nthr bid).rout (2 * v + 1) tid
    as (FB.barrier_q eA eB sA sB nthr bid) (2 * v + 1) tid;
  FB.unfold_barrier_q_odd eA eB sA sB nthr bid mrow mcol v tid;

  unfold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow v) nthr;
  unfold FB.bp_sharing sB (ematrix_subtile eB bk bn v mcol) nthr;

  let rA_sub = ematrix_subtile rA bm bk mrow v;
  let rB_sub = ematrix_subtile rB bk bn v mcol;
  ematrix_subtile_approximates eA rA (SZ.v bm) (SZ.v bk) (SZ.v mrow) (SZ.v v);
  ematrix_subtile_approximates eB rB (SZ.v bk) (SZ.v bn) (SZ.v v) (SZ.v mcol);
  with rAcc. assert fragarrayAcc_approximates wm wn accFrags rAcc;
  subproducts_tc_2d bm bn bk tm tn tk wm wn
    aFrags bFrags accFrags sA sB
    rA_sub rB_sub rAcc warpRow warpCol;

  loop_invariant_lemma
    m n k bm bn bk tm tn tk wm wn
    mrow mcol warpRow warpCol gwRow gwCol v
    rA rB rAcc0 rAcc rA_sub rB_sub;
  assert pure (
    rAcc `matplus`
      matmul (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
             (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol)
    ==
    tiled_partial_matmul rAcc0
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol (v + 1));
  rewrite_fragarrayAcc wm wn accFrags
    (rAcc `matplus`
      matmul (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
             (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol))
    (tiled_partial_matmul rAcc0
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol (v + 1));

  fold FB.bp_sharing sA (ematrix_subtile eA bm bk mrow v) nthr;
  fold FB.bp_sharing sB (ematrix_subtile eB bk bn v mcol) nthr;
  v +^ 1sz
}
#pop-options

inline_for_extraction noextract
fn compute_acc
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc |}
  {| real_like et_ab, real_like et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (str_B : strided_col_major (vtlayout_of_tlayout lB))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_col_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab k n)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (bm bn bk
   tm tn tk
   wm wn : szp { constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#_ : squash (eA %~ rA /\ eB %~ rB))
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (sA : array2 et_ab (rm bm bk))
  (sB : array2 et_ab (cm bk bn))
  (#fA #fB : perm)
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm) {
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) })
  (mcol : szlt (n / bn) {
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) })
  (warpRow : szlt (bm / (wm * tm)) {
    SZ.v warpRow == (SZ.v tid / warp_size) /
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  (warpCol : szlt (bn / (wn * tn)) {
    SZ.v warpCol == (SZ.v tid / warp_size) %
      (SZ.v bn / (SZ.v wn * SZ.v tn)) })
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (cm bk bn)
        (core sA) (core sB) nthr bid)
  requires
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state 0
  returns accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc)
  ensures
    pure (Pulse.Lib.Array.length accFrags == wm * wn) **
    fragarrayAcc_approximates wm wn accFrags
      (kernel_warp_matmul rA rB
        bm bn bk tm tn tk wm wn nthr bid tid) **
    (exists* em1. FB.bp_sharing sA em1 nthr) **
    (exists* em2. FB.bp_sharing sB em2 nthr) **
    B.barrier_state (tile_barrier_iteration k bk)
{
  let num_k_tiles = k /^ bk;
  let gwRow : enatlt (m / (wm * tm)) =
    mrow * (bm / (wm * tm)) + warpRow;
  let gwCol : enatlt (n / (wn * tn)) =
    mcol * (bn / (wn * tn)) + warpCol;

  let aFrags =
    __alloc_array_fragment et_ab FragA tm tn tk FragLRM wm;
  let bFrags =
    __alloc_array_fragment et_ab FragB tm tn tk FragLCM wn;
  let accFrags =
    __alloc_array_fragment et_acc FragAcc tm tn tk FragLAcc (wm *^ wn);

  populate_acc_with_zero tm tn tk wm wn accFrags;
  let rAcc0 : chest2 real (wm * tm) (wn * tn) = const _ 0.0R;
  assert (rewrites_to rAcc0 (const _ 0.0R));
  rewrite fragarrayAcc_approximates wm wn accFrags rAcc0
    as fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol 0);

  let mut bkIdx : szle num_k_tiles = 0sz;
  while (!bkIdx <^ num_k_tiles)
    invariant
      exists* (vbkIdx : szle num_k_tiles).
        bkIdx |-> vbkIdx **
        fragarrayAcc_approximates wm wn accFrags
          (__gmatmul_single rAcc0 matmul matplus
            (ematrix_tiled rA (wm * tm) bk)
            (ematrix_tiled rB bk (wn * tn))
            gwRow gwCol !bkIdx)
    invariant live aFrags ** live bFrags
    invariant
      (exists* em1. FB.bp_sharing sA em1 nthr) **
      (exists* em2. FB.bp_sharing sB em2 nthr) **
      B.barrier_state (2 * !bkIdx)
    decreases (num_k_tiles - !bkIdx)
  {
    let vnext = k_loop_step gA str_B gB
      bm bn bk tm tn tk wm wn
      rA rB nthr sA sB
      bid tid mrow mcol warpRow warpCol gwRow gwCol rAcc0
      aFrags bFrags accFrags !bkIdx;
    rewrite each (SZ.v !bkIdx + 1) as SZ.v vnext;
    bkIdx := !bkIdx +^ 1sz;
  };

  assert
    fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol (k / bk));
  assert pure (
    gwRow == warp_tile_i #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size));
  assert pure (
    gwCol == warp_tile_j #m #n bm bn bk tm tn tk wm wn
      nthr bid (tid / warp_size));

  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
    (wm * tm) (wn * tn) bk rAcc0 rA rB gwRow gwCol;
  let rAcc' : chest2 real (wm * tm) (wn * tn) =
    gmatmul_single rAcc0 matmul matplus
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol;
  assert pure (
    (__gmatmul_single rAcc0 matmul matplus
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol !bkIdx) == rAcc');

  let rAcc : chest2 real (wm * tm) (wn * tn) =
    MS.matmul
      (ematrix_subtile rA (wm * tm) k
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)) 0)
      (ematrix_subtile rB k (wn * tn) 0
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)));
  assert pure (matplus (const _ 0.0R) rAcc `equal` rAcc);
  assert pure (rAcc' == rAcc);
  rewrite
    fragarrayAcc_approximates wm wn accFrags
      (__gmatmul_single rAcc0 matmul matplus
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol !bkIdx)
    as fragarrayAcc_approximates wm wn accFrags rAcc;

  with vaFrags. assert aFrags |-> vaFrags;
  drop_ (aFrags |-> vaFrags);
  with vbFrags. assert bFrags |-> vbFrags;
  drop_ (bFrags |-> vbFrags);
  accFrags
}
