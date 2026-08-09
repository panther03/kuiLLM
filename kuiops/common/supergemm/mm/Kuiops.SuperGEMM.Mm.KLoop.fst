module Kuiops.SuperGEMM.Mm.KLoop

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.TensorCore
open Kuiper.Chest { chest_map }
open Pulse.Lib.Array { length }
open Pulse.Lib.Array.PtsTo { op_Array_Access }
open Pulse.Lib.Trade
open Pulse.Lib.Pledge

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params { frag, frag_sz, ldt }
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_live, pipe_sharing, pipe_p, pipe_q,
                                    pipe_contract }
open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major, srm_l2_skewed_row_major,
                                    c_l2_skewed_row_major, lemma_aligned_srm_l2_skewed_row_major }
open Kuiops.SuperGEMM.Mm.Stage { geo_ok, g_row_step, g_a_iters, g_t_row, g_t_col, stage_tiles }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module B = Kuiper.Barrier
module FB = Kuiops.GEMM.T.FlipFlopBarrier2
module PC = Kuiper.PipelineCopy
module D = Kuiper.Divides
module TR = Kuiops.Tensor.Transpose2
module P = Kuiops.SuperGEMM.Mm.Params

let acc_len_reveal wm wn = reveal_opaque (`%acc_len) acc_len

let acc_len_alloc wm wn = reveal_opaque (`%acc_len) acc_len

(* [warp_m * mfrag + i] indexes a 16-row band of the [bm x bk] A tile. *)
let a_tile_bound (bm wm : pos) (warp_m i : nat)
  : Lemma (requires frag /?+ wm /\ wm /?+ bm /\
                    warp_m < bm / wm /\ i < wm / frag)
          (ensures warp_m * (wm / frag) + i < bm / frag)
= let q1 = bm / wm in
  let q2 = wm / frag in
  Kuiper.Divides.lemma_nat_divides_pos_divides wm bm;
  Kuiper.Divides.lemma_nat_divides_pos_divides frag wm;
  let g1 = Kuiper.Divides.get_factor wm bm in   // wm * g1 == bm, g1 == q1
  let g2 = Kuiper.Divides.get_factor frag wm in  // frag * g2 == wm, g2 == q2
  FStar.Math.Lemmas.multiple_division_lemma q1 wm;
  FStar.Math.Lemmas.multiple_division_lemma q2 frag;
  // bm == q1 * q2 * frag, so bm / frag == q1 * q2
  FStar.Math.Lemmas.paren_mul_right q1 q2 frag;
  FStar.Math.Lemmas.multiple_division_lemma (q1 * q2) frag;
  ()

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn sp_load_a
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (bm bk wm : szp)
  (#_ : squash (frag /?+ SZ.v wm /\ frag /?+ SZ.v bk /\ SZ.v wm /?+ SZ.v bm))
  (fmap : et_ab -> et_ab)
  (aFrags : array (fragment et_ab FragA frag frag frag FragLRM))
  (#lA : layout2 (SZ.v bm) (SZ.v bk)) {| T.ctlayout lA |}
       {| strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab (SZ.v bm) (SZ.v bk))
  (#fA : perm)
  (warp_m : szlt (SZ.v bm / SZ.v wm))
  (ks : szlt (SZ.v bk / frag))
  (#_ : squash (length aFrags == SZ.v wm / frag))
  (#_ : squash (valid_frag_et_dims et_ab FragA frag frag frag))
  ()
  preserves gpu ** gA |-> Frac fA eA
  preserves live aFrags
{
  array_fragment_pts_to_ref aFrags;
  let mfrag = wm /^ frag_sz;
  let mut i = 0sz;
  while (!i <^ mfrag)
    invariant live i
    invariant
      exists* ems.
        array_fragment_pts_to aFrags ems **
        pure (Seq.length ems == SZ.v wm / frag /\ SZ.v !i <= SZ.v wm / frag)
    decreases (SZ.v wm / frag - SZ.v !i)
  {
    a_tile_bound (SZ.v bm) (SZ.v wm) (SZ.v warp_m) (SZ.v !i);
    Kuiper.Divides.lemma_nat_divides_pos_divides frag (SZ.v wm);
    Kuiper.Divides.lemma_nat_divides_pos_divides frag (SZ.v bk);
    Kuiper.Divides.lemma_nat_divides_pos_divides (SZ.v wm) (SZ.v bm);
    Kuiper.Divides.lemma_divides_trans frag (SZ.v wm) (SZ.v bm);
    let arow = warp_m *^ mfrag +^ !i;
    let atile =
      array2_extract_tile_ro' gA frag frag (SZ.v arow) (SZ.v ks);
    array_fragment_pts_to_ref aFrags;
    array_fragment_extract aFrags !i;
    let a_frag = aFrags.(!i);
    mma_loadA_map fmap a_frag atile;
    with v. assert a_frag |-> v;
    Pulse.Lib.Forall.elim_forall #_ v;
    Kuiper.TradeHelpers.ambig_trade_elim ();
    Kuiper.TradeHelpers.ambig_trade_elim ();
    i := !i +^ 1sz;
  };
}
#pop-options

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn sp_load_b
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (bn bk wn : szp)
  (#_ : squash (frag /?+ SZ.v wn /\ frag /?+ SZ.v bk /\ SZ.v wn /?+ SZ.v bn))
  (fmap : et_ab -> et_ab)
  (bFrags : array (fragment et_ab FragB frag frag frag FragLCM))
  (#lB : layout2 (SZ.v bk) (SZ.v bn)) {| T.ctlayout lB |}
       {| strided_col_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab (SZ.v bk) (SZ.v bn))
  (#fB : perm)
  (warp_n : szlt (SZ.v bn / SZ.v wn))
  (ks : szlt (SZ.v bk / frag))
  (#_ : squash (length bFrags == SZ.v wn / frag))
  (#_ : squash (valid_frag_et_dims et_ab FragB frag frag frag))
  ()
  preserves gpu ** gB |-> Frac fB eB
  preserves live bFrags
{
  array_fragment_pts_to_ref bFrags;
  let nfrag = wn /^ frag_sz;
  let mut j = 0sz;
  while (!j <^ nfrag)
    invariant live j
    invariant
      exists* ems.
        array_fragment_pts_to bFrags ems **
        pure (Seq.length ems == SZ.v wn / frag /\ SZ.v !j <= SZ.v wn / frag)
    decreases (SZ.v wn / frag - SZ.v !j)
  {
    a_tile_bound (SZ.v bn) (SZ.v wn) (SZ.v warp_n) (SZ.v !j);
    Kuiper.Divides.lemma_nat_divides_pos_divides frag (SZ.v wn);
    Kuiper.Divides.lemma_nat_divides_pos_divides frag (SZ.v bk);
    Kuiper.Divides.lemma_nat_divides_pos_divides (SZ.v wn) (SZ.v bn);
    Kuiper.Divides.lemma_divides_trans frag (SZ.v wn) (SZ.v bn);
    let bcol = warp_n *^ nfrag +^ !j;
    let btile =
      array2_extract_tile_ro' gB frag frag (SZ.v ks) (SZ.v bcol);
    array_fragment_pts_to_ref bFrags;
    array_fragment_extract bFrags !j;
    let b_frag = bFrags.(!j);
    mma_loadB_map_cm fmap b_frag btile;
    with v. assert b_frag |-> v;
    Pulse.Lib.Forall.elim_forall #_ v;
    Kuiper.TradeHelpers.ambig_trade_elim ();
    Kuiper.TradeHelpers.ambig_trade_elim ();
    j := !j +^ 1sz;
  };
}
#pop-options

#push-options "--z3rlimit 40 --split_queries always"
inline_for_extraction noextract
fn sp_mma
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (wm wn : szp)
  (#_ : squash (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (aFrags  : array (fragment et_ab FragA   frag frag frag FragLRM))
  (bFrags  : array (fragment et_ab FragB   frag frag frag FragLCM))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (#_ : squash (length aFrags == SZ.v wm / frag))
  (#_ : squash (length bFrags == SZ.v wn / frag))
  (#_ : squash (length accFrags == SZ.v wm / frag * (SZ.v wn / frag)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  ()
  preserves live aFrags ** live bFrags ** live accFrags
{
  let mfrag = wm /^ frag_sz;
  let nfrag = wn /^ frag_sz;
  array_fragment_pts_to_ref aFrags;
  array_fragment_pts_to_ref bFrags;
  array_fragment_pts_to_ref accFrags;
  let mut ri = 0sz;
  while (!ri <^ mfrag)
    invariant live ri
    invariant live aFrags ** live bFrags
    invariant
      exists* eAcc.
        array_fragment_pts_to accFrags eAcc **
        pure (Seq.length eAcc == SZ.v wm / frag * (SZ.v wn / frag) /\
              SZ.v !ri <= SZ.v wm / frag)
    decreases (SZ.v wm / frag - SZ.v !ri)
  {
    let mut rj = 0sz;
    while (!rj <^ nfrag)
      invariant live rj
      invariant live aFrags ** live bFrags
      invariant
        exists* eAcc.
          array_fragment_pts_to accFrags eAcc **
          pure (Seq.length eAcc == SZ.v wm / frag * (SZ.v wn / frag) /\
                SZ.v !ri < SZ.v wm / frag /\ SZ.v !rj <= SZ.v wn / frag)
      decreases (SZ.v wn / frag - SZ.v !rj)
    {
      array_fragment_pts_to_ref aFrags;
      array_fragment_pts_to_ref bFrags;
      array_fragment_pts_to_ref accFrags;
      array_fragment_extract_ro aFrags !ri;
      array_fragment_extract_ro bFrags !rj;
      let idx = !ri *^ nfrag +^ !rj;
      array_fragment_extract accFrags idx;
      let a_frag = aFrags.(!ri);
      let b_frag = bFrags.(!rj);
      let acc_frag = accFrags.(idx);
      mma_sync' a_frag b_frag acc_frag;
      Kuiper.TradeHelpers.ambig_trade_elim ();
      Kuiper.TradeHelpers.ambig_trade_elim ();
      with v. assert acc_frag |-> v;
      Pulse.Lib.Forall.elim_forall #_ v;
      Kuiper.TradeHelpers.ambig_trade_elim ();
      rj := !rj +^ 1sz;
    };
    ri := !ri +^ 1sz;
  };
}
#pop-options

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
{
  acc_len_reveal wm wn;
  let kstep = bk /^ frag_sz;
  let mut ks = 0sz;
  while (!ks <^ kstep)
    invariant live ks
    invariant
      gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB **
      live aFrags ** live bFrags ** live accFrags
    invariant pure (SZ.v !ks <= SZ.v bk / frag)
    decreases (SZ.v bk / frag - SZ.v !ks)
  {
    sp_load_a bm bk wm fmap aFrags gA warp_m !ks ();
    sp_load_b bn bk wn fmap bFrags gB warp_n !ks ();
    sp_mma wm wn aFrags bFrags accFrags ();
    ks := !ks +^ 1sz;
  };
}

(* ---- subtile alignment inheritance (pure) ---- *)
#push-options "--fuel 0 --ifuel 0"
let lemma_subtile_aligned
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr : erased int {0 <= tr /\ tr < rows / trows})
  (tc : erased int {0 <= tc /\ tc < cols / tcols})
  {| concrete_sz trows, concrete_sz tcols, concrete_sz tr, concrete_sz tc |}
  (n : pos)
  : Lemma (requires SZ.fits (l.ulen) /\ aligned_strided_row_major n sub /\ n /?+ tcols)
          (ensures aligned_strided_row_major n
                     (strided_row_major_subtile l trows tcols tr tc))
= let s = strided_row_major_subtile l trows tcols tr tc in
  assert (SZ.v s.offset == sub.offset + sub.stride * (tr * trows) + tc * tcols);
  assert (s.stride == sub.stride);
  D.lemma_nat_divides_pos_divides n sub.offset;
  D.lemma_nat_divides_pos_divides n sub.stride;
  D.lemma_nat_divides_pos_divides n tcols;
  D.lemma_divides_product_l n sub.stride (tr * trows);
  D.lemma_divides_product_r n tc tcols;
  D.lemma_divides_sum n sub.offset (sub.stride * (tr * trows));
  D.lemma_divides_sum n (sub.offset + sub.stride * (tr * trows)) (tc * tcols);
  D.lemma_nat_divides_pos_divides n (SZ.v s.offset);
  ()
#pop-options

(* ---- parity fold/unfold of the pipe contract ---- *)
#push-options "--fuel 2 --ifuel 1"
ghost
fn fold_pipe_p_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (it : nat) (tid : natlt nthr)
  requires
    pure (it % 2 == 0 /\ it < ktiles) **
    pipe_live (skewed_view bm bk skew sarA0) nthr tid **
    pipe_live (skewed_view bn bk skew sarB0) nthr tid **
    (if it = 0 then
       pipe_live (skewed_view bm bk skew sarA1) nthr tid **
       pipe_live (skewed_view bn bk skew sarB1) nthr tid
     else
       pipe_sharing (skewed_view bm bk skew sarA1) nthr **
       pipe_sharing (skewed_view bn bk skew sarB1) nthr)
  ensures
    pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
{
  rewrite
    (pipe_live (skewed_view bm bk skew sarA0) nthr tid **
     pipe_live (skewed_view bn bk skew sarB0) nthr tid **
     (if it = 0 then
        pipe_live (skewed_view bm bk skew sarA1) nthr tid **
        pipe_live (skewed_view bn bk skew sarB1) nthr tid
      else
        pipe_sharing (skewed_view bm bk skew sarA1) nthr **
        pipe_sharing (skewed_view bn bk skew sarB1) nthr))
  as
    (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid);
}

ghost
fn fold_pipe_p_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (it : nat) (tid : natlt nthr)
  requires
    pure (it % 2 == 1 /\ it < ktiles) **
    pipe_live (skewed_view bm bk skew sarA1) nthr tid **
    pipe_live (skewed_view bn bk skew sarB1) nthr tid **
    pipe_sharing (skewed_view bm bk skew sarA0) nthr **
    pipe_sharing (skewed_view bn bk skew sarB0) nthr
  ensures
    pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
{
  rewrite
    (pipe_live (skewed_view bm bk skew sarA1) nthr tid **
     pipe_live (skewed_view bn bk skew sarB1) nthr tid **
     pipe_sharing (skewed_view bm bk skew sarA0) nthr **
     pipe_sharing (skewed_view bn bk skew sarB0) nthr)
  as
    (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid);
}

ghost
fn unfold_pipe_q_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (it : nat) (tid : natlt nthr)
  requires
    pure (it % 2 == 0 /\ it < ktiles) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
  ensures
    pipe_sharing (skewed_view bm bk skew sarA0) nthr **
    pipe_sharing (skewed_view bn bk skew sarB0) nthr **
    pipe_live (skewed_view bm bk skew sarA1) nthr tid **
    pipe_live (skewed_view bn bk skew sarB1) nthr tid
{
  rewrite
    (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
  as
    (pipe_sharing (skewed_view bm bk skew sarA0) nthr **
     pipe_sharing (skewed_view bn bk skew sarB0) nthr **
     pipe_live (skewed_view bm bk skew sarA1) nthr tid **
     pipe_live (skewed_view bn bk skew sarB1) nthr tid);
}

ghost
fn unfold_pipe_q_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (it : nat) (tid : natlt nthr)
  requires
    pure (it % 2 == 1 /\ it < ktiles) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
  ensures
    pipe_sharing (skewed_view bm bk skew sarA1) nthr **
    pipe_sharing (skewed_view bn bk skew sarB1) nthr **
    pipe_live (skewed_view bm bk skew sarA0) nthr tid **
    pipe_live (skewed_view bn bk skew sarB0) nthr tid
{
  rewrite
    (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
  as
    (pipe_sharing (skewed_view bm bk skew sarA1) nthr **
     pipe_sharing (skewed_view bn bk skew sarB1) nthr **
     pipe_live (skewed_view bm bk skew sarA0) nthr tid **
     pipe_live (skewed_view bn bk skew sarB0) nthr tid);
}

ghost
fn fold_pipe_q_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (it : nat) (tid : natlt nthr)
  requires
    pure (it % 2 == 0 /\ it < ktiles) **
    pipe_sharing (skewed_view bm bk skew sarA0) nthr **
    pipe_sharing (skewed_view bn bk skew sarB0) nthr **
    pipe_live (skewed_view bm bk skew sarA1) nthr tid **
    pipe_live (skewed_view bn bk skew sarB1) nthr tid
  ensures
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
{
  rewrite
    (pipe_sharing (skewed_view bm bk skew sarA0) nthr **
     pipe_sharing (skewed_view bn bk skew sarB0) nthr **
     pipe_live (skewed_view bm bk skew sarA1) nthr tid **
     pipe_live (skewed_view bn bk skew sarB1) nthr tid)
  as
    (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid);
}

ghost
fn fold_pipe_q_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (it : nat) (tid : natlt nthr)
  requires
    pure (it % 2 == 1 /\ it < ktiles) **
    pipe_sharing (skewed_view bm bk skew sarA1) nthr **
    pipe_sharing (skewed_view bn bk skew sarB1) nthr **
    pipe_live (skewed_view bm bk skew sarA0) nthr tid **
    pipe_live (skewed_view bn bk skew sarB0) nthr tid
  ensures
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
{
  rewrite
    (pipe_sharing (skewed_view bm bk skew sarA1) nthr **
     pipe_sharing (skewed_view bn bk skew sarB1) nthr **
     pipe_live (skewed_view bm bk skew sarA0) nthr tid **
     pipe_live (skewed_view bn bk skew sarB0) nthr tid)
  as
    (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid);
}
#pop-options


(* ---- carry predicate: the parity/k-tile-dependent tail of the k-loop
   invariant, packaged so the loop matcher only unifies its [vkt] argument
   (never a buffer identity buried inside a pledge/trade body). ---- *)
#push-options "--fuel 1 --ifuel 1"
let pending
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  : slprop
= pledge0 (PC.batch_done b_c)
    (FB.live_strided_chunks
       (skewed_view bm bk skew (if vkt % 2 = 0 then sarA0 else sarA1)) nthr tid **
     FB.live_strided_chunks
       (skewed_view bn bk skew (if vkt % 2 = 0 then sarB0 else sarB1)) nthr tid **
     (exists* eA eB.
        (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA eA) **
        (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB eB))) **
  (if vkt = 0 then
     pipe_live (skewed_view bm bk skew (if vkt % 2 = 0 then sarA1 else sarA0)) nthr tid **
     pipe_live (skewed_view bn bk skew (if vkt % 2 = 0 then sarB1 else sarB0)) nthr tid
   else
     pipe_sharing (skewed_view bm bk skew (if vkt % 2 = 0 then sarA1 else sarA0)) nthr **
     pipe_sharing (skewed_view bn bk skew (if vkt % 2 = 0 then sarB1 else sarB0)) nthr) **
  (exists* (emA : chest2 et_ab (SZ.v m) (SZ.v k)).
     forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
       (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
         @==> gA |-> Frac fA (update_tile emA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
  (exists* (emB : chest2 et_ab (SZ.v n) (SZ.v k)).
     forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
       (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
         @==> gB |-> Frac fB (update_tile emB (SZ.v bn) (SZ.v bk) block_col vkt tm'))

(* the pledge/other/wands bundle at a fixed current/other buffer pair *)
unfold
let pending_body
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (curA othA : larray et_ab (SZ.v bm * ldt bk skew))
  (curB othB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (othlive : bool)
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  : slprop
= pledge0 (PC.batch_done b_c)
    (FB.live_strided_chunks (skewed_view bm bk skew curA) nthr tid **
     FB.live_strided_chunks (skewed_view bn bk skew curB) nthr tid **
     (exists* eA eB.
        (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA eA) **
        (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB eB))) **
  (if othlive then
     pipe_live (skewed_view bm bk skew othA) nthr tid **
     pipe_live (skewed_view bn bk skew othB) nthr tid
   else
     pipe_sharing (skewed_view bm bk skew othA) nthr **
     pipe_sharing (skewed_view bn bk skew othB) nthr) **
  (exists* (emA : chest2 et_ab (SZ.v m) (SZ.v k)).
     forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
       (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
         @==> gA |-> Frac fA (update_tile emA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
  (exists* (emB : chest2 et_ab (SZ.v n) (SZ.v k)).
     forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
       (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
         @==> gB |-> Frac fB (update_tile emB (SZ.v bn) (SZ.v bk) block_col vkt tm'))
#pop-options

(* ---- parity fold/unfold of the carry predicate ---- *)
#push-options "--fuel 1 --ifuel 2 --z3rlimit 15"
ghost
fn fold_pending_even
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  requires
    pure (vkt % 2 = 0) **
    pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt
  ensures
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
{
  rewrite
    (pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt)
  as
    (pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt);
}

ghost
fn fold_pending_even_pos
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  requires
    pure (vkt % 2 = 0 /\ vkt <> 0) **
    pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col
      b_c sq false vkt
  ensures
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
{
  rewrite
    (pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col
      b_c sq false vkt)
  as
    (pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt);
}

ghost
fn fold_pending_odd
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  requires
    pure (vkt % 2 = 1) **
    pending_body bm bn bk skew gA gB sarA1 sarA0 sarB1 sarB0 fA fB nthr tid block_row block_col
      b_c sq false vkt
  ensures
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
{
  rewrite
    (pending_body bm bn bk skew gA gB sarA1 sarA0 sarB1 sarB0 fA fB nthr tid block_row block_col
      b_c sq false vkt)
  as
    (pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt);
}

ghost
fn unfold_pending_even
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  requires
    pure (vkt % 2 = 0) **
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
  ensures
    pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt
{
  rewrite
    (pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt)
  as
    (pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt);
}

ghost
fn unfold_pending_odd
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  requires
    pure (vkt % 2 = 1) **
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
  ensures
    pending_body bm bn bk skew gA gB sarA1 sarA0 sarB1 sarB0 fA fB nthr tid block_row block_col
      b_c sq false vkt
{
  rewrite
    (pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt)
  as
    (pending_body bm bn bk skew gA gB sarA1 sarA0 sarB1 sarB0 fA fB nthr tid block_row block_col
      b_c sq false vkt);
}
#pop-options

(* ---- parity-independent per-k-tile compute + next-tile staging ----
   [cur*] are the buffers just read (held read-shared, consumed only by
   [subproducts]); [oth*] are the now-writable buffers into which k-tile
   [kt1 = kt+1] is staged. *)
#push-options "--z3rlimit 15 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn body_compute
  (#et_ab #et_acc : Type0)
  {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  {| _sac : scalar et_acc |} {| _vac : has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (bm bn bk wm wn skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| _clA : T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| _clB : T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB)
  (curbufA othbufA : larray et_ab (SZ.v bm * ldt bk skew))
  (curbufB othbufB : larray et_ab (SZ.v bn * ldt bk skew))
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
  (ldsz : szp { SZ.v ldsz == ldt bk skew })
  (kt1 : szlt (SZ.v k / SZ.v bk))
  (b_l : PC.pipeline_batch_t)
  (aFrags  : array (fragment et_ab  FragA   frag frag frag FragLRM))
  (bFrags  : array (fragment et_ab  FragB   frag frag frag FragLCM))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (sq_c : squash (P.constraints et_ab et_acc bm bn bk wm wn skew))
  (sq_g : squash (geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr) /\
                  geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (SZ.v nthr)))
  (sq_d : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\ SZ.v bk /?+ SZ.v k /\
                  SZ.fits (SZ.v m * SZ.v k) /\ SZ.fits (SZ.v n * SZ.v k)))
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
     aligned 16 (T.core (skewed_view bm bk skew othbufA)) /\
     aligned 16 (T.core (skewed_view bn bk skew othbufB))))
  (sq_v : squash (
     valid_frag_et_dims et_ab FragA frag frag frag /\
     valid_frag_et_dims et_ab FragB frag frag frag /\
     valid_frag_et_dims et_acc FragAcc frag frag frag /\
     valid_frag_et_comb et_ab et_acc /\
     length aFrags == SZ.v wm / frag /\ length bFrags == SZ.v wn / frag /\
     length accFrags == acc_len wm wn))
  ()
  requires
    gpu **
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr) **
    pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr) **
    pipe_live (skewed_view bm bk skew othbufA) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bn bk skew othbufB) (SZ.v nthr) (SZ.v tid) **
    live aFrags ** live bFrags ** live accFrags **
    PC.batch_live b_l
  returns b_l' : PC.pipeline_batch_t
  ensures
    gpu **
    pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr) **
    pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr) **
    PC.batch_committed b_l ** PC.batch_live b_l' **
    pledge0 (PC.batch_done b_l)
      (FB.live_strided_chunks (skewed_view bm bk skew othbufA) (SZ.v nthr) (SZ.v tid) **
       FB.live_strided_chunks (skewed_view bn bk skew othbufB) (SZ.v nthr) (SZ.v tid) **
       (exists* eA eB.
          (array2_subtile gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt1) |-> Frac fA eA) **
          (array2_subtile gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt1) |-> Frac fB eB))) **
    (exists* (emA : chest2 et_ab (SZ.v m) (SZ.v k)).
       forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
         (array2_subtile gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt1) |-> Frac fA tm')
           @==> gA |-> Frac fA (update_tile emA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt1) tm')) **
    (exists* (emB : chest2 et_ab (SZ.v n) (SZ.v k)).
       forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
         (array2_subtile gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt1) |-> Frac fB tm')
           @==> gB |-> Frac fB (update_tile emB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt1) tm')) **
    live aFrags ** live bFrags ** live accFrags
{
  (* ---- 6. extract next k-tile subviews of global A/B ---- *)
  with egA. assert (gA |-> Frac fA egA);
  let tileA1 = array2_extract_tile_st gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt1);
  with egB. assert (gB |-> Frac fB egB);
  let tileB1 = array2_extract_tile_st gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt1);

  (* ---- 7. stage k-tile [kt1] into the now-writable OTHER buffer ---- *)
  unfold (pipe_live (skewed_view bm bk skew othbufA) (SZ.v nthr) (SZ.v tid));
  unfold (FB.live_strided_chunks (skewed_view bm bk skew othbufA) (SZ.v nthr) (SZ.v tid));
  with emAd. assert (FB.own_strided_chunks (skewed_view bm bk skew othbufA) emAd (SZ.v nthr) (SZ.v tid));
  unfold (pipe_live (skewed_view bn bk skew othbufB) (SZ.v nthr) (SZ.v tid));
  unfold (FB.live_strided_chunks (skewed_view bn bk skew othbufB) (SZ.v nthr) (SZ.v tid));
  with emBd. assert (FB.own_strided_chunks (skewed_view bn bk skew othbufB) emBd (SZ.v nthr) (SZ.v tid));

  lemma_subtile_aligned lA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt1) (SZ.v (chunk et_ab));
  lemma_subtile_aligned lB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt1) (SZ.v (chunk et_ab));
  lemma_aligned_srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz (SZ.v (chunk et_ab));
  lemma_aligned_srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz (SZ.v (chunk et_ab));

  let b_l' = stage_tiles
    #et_ab #_ #_ #_ #_
    #(l2_skewed_row_major (SZ.v bm) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bm bk skew othbufA)
    tileA1
    #_
    #(l2_skewed_row_major (SZ.v bn) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bn bk skew othbufB)
    tileB1
    (SZ.v nthr) (SZ.v tid) () ()
    fA fB
    a_t_row a_t_col a_row_step a_iters
    b_t_row b_t_col b_row_step b_iters
    b_l () ();

  rewrite each tileA1 as array2_subtile gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt1);
  rewrite each tileB1 as array2_subtile gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt1);

  (* ---- 8. fragment math on the CURRENT (read-shared) buffer ---- *)
  unfold (pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr));
  with emCA. assert (FB.bp_sharing (skewed_view bm bk skew curbufA) emCA (SZ.v nthr));
  unfold (FB.bp_sharing (skewed_view bm bk skew curbufA) emCA (SZ.v nthr));
  unfold (pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr));
  with emCB. assert (FB.bp_sharing (skewed_view bn bk skew curbufB) emCB (SZ.v nthr));
  unfold (FB.bp_sharing (skewed_view bn bk skew curbufB) emCB (SZ.v nthr));

  TR.atranspose_fwd (skewed_view bn bk skew curbufB);

  subproducts bm bn bk wm wn fmap aFrags bFrags accFrags
    #(l2_skewed_row_major (SZ.v bm) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bm bk skew curbufA)
    #(TR.ltranspose (l2_skewed_row_major (SZ.v bn) (SZ.v bk) (SZ.v skew)))
    #(TR.ctlayout_ltranspose_inst
        #(SZ.v bn) #(SZ.v bk)
        #(l2_skewed_row_major (SZ.v bn) (SZ.v bk) (SZ.v skew))
        #(c_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz) #())
    #(TR.scm_of_srm (srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz))
    (TR.atranspose (skewed_view bn bk skew curbufB))
    warp_m warp_n ();

  TR.atranspose_back (skewed_view bn bk skew curbufB);

  fold (FB.bp_sharing (skewed_view bm bk skew curbufA) emCA (SZ.v nthr));
  fold (pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr));
  fold (FB.bp_sharing (skewed_view bn bk skew curbufB) (TR.ctranspose (TR.ctranspose emCB)) (SZ.v nthr));
  fold (pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr));

  b_l'
}
#pop-options

(* ---- fragment math on the CURRENT (read-shared) buffer, no staging;
   used by the k-loop's peeled tail (last k-tile). ---- *)
#push-options "--z3rlimit 15 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn subproducts_buf
  (#et_ab #et_acc : Type0)
  {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  {| _sac : scalar et_acc |} {| _vac : has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (curbufA : larray et_ab (SZ.v bm * ldt bk skew))
  (curbufB : larray et_ab (SZ.v bn * ldt bk skew))
  (fmap : et_ab -> et_ab)
  (nthr : szp { SZ.v nthr == P.nthr bm bn wm wn })
  (warp_m : szlt (SZ.v bm / SZ.v wm))
  (warp_n : szlt (SZ.v bn / SZ.v wn))
  (ldsz : szp { SZ.v ldsz == ldt bk skew })
  (aFrags  : array (fragment et_ab  FragA   frag frag frag FragLRM))
  (bFrags  : array (fragment et_ab  FragB   frag frag frag FragLCM))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (sq_c : squash (P.constraints et_ab et_acc bm bn bk wm wn skew))
  (sq_v : squash (
     valid_frag_et_dims et_ab FragA frag frag frag /\
     valid_frag_et_dims et_ab FragB frag frag frag /\
     valid_frag_et_dims et_acc FragAcc frag frag frag /\
     valid_frag_et_comb et_ab et_acc /\
     length aFrags == SZ.v wm / frag /\ length bFrags == SZ.v wn / frag /\
     length accFrags == acc_len wm wn))
  ()
  requires
    gpu **
    pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr) **
    pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr) **
    live aFrags ** live bFrags ** live accFrags
  ensures
    gpu **
    pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr) **
    pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr) **
    live aFrags ** live bFrags ** live accFrags
{
  unfold (pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr));
  with emCA. assert (FB.bp_sharing (skewed_view bm bk skew curbufA) emCA (SZ.v nthr));
  unfold (FB.bp_sharing (skewed_view bm bk skew curbufA) emCA (SZ.v nthr));
  unfold (pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr));
  with emCB. assert (FB.bp_sharing (skewed_view bn bk skew curbufB) emCB (SZ.v nthr));
  unfold (FB.bp_sharing (skewed_view bn bk skew curbufB) emCB (SZ.v nthr));

  TR.atranspose_fwd (skewed_view bn bk skew curbufB);

  subproducts bm bn bk wm wn fmap aFrags bFrags accFrags
    #(l2_skewed_row_major (SZ.v bm) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bm bk skew curbufA)
    #(TR.ltranspose (l2_skewed_row_major (SZ.v bn) (SZ.v bk) (SZ.v skew)))
    #(TR.ctlayout_ltranspose_inst
        #(SZ.v bn) #(SZ.v bk)
        #(l2_skewed_row_major (SZ.v bn) (SZ.v bk) (SZ.v skew))
        #(c_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz) #())
    #(TR.scm_of_srm (srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz))
    (TR.atranspose (skewed_view bn bk skew curbufB))
    warp_m warp_n ();

  TR.atranspose_back (skewed_view bn bk skew curbufB);

  fold (FB.bp_sharing (skewed_view bm bk skew curbufA) emCA (SZ.v nthr));
  fold (pipe_sharing (skewed_view bm bk skew curbufA) (SZ.v nthr));
  fold (FB.bp_sharing (skewed_view bn bk skew curbufB) (TR.ctranspose (TR.ctranspose emCB)) (SZ.v nthr));
  fold (pipe_sharing (skewed_view bn bk skew curbufB) (SZ.v nthr));
}
#pop-options

(* ---- reconstitute global A/B from a staged subtile + its restore wands ---- *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
ghost
fn restore_globals
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (fA fB : perm)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  requires
    (exists* eA eB.
       (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA eA) **
       (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB eB)) **
    (exists* (emA : chest2 et_ab (SZ.v m) (SZ.v k)).
       forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
         (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
           @==> gA |-> Frac fA (update_tile emA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
    (exists* (emB : chest2 et_ab (SZ.v n) (SZ.v k)).
       forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
         (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
           @==> gB |-> Frac fB (update_tile emB (SZ.v bn) (SZ.v bk) block_col vkt tm'))
  ensures
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e)
{
  with eAr. assert (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA eAr);
  with emAw. assert (forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
      (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
        @==> gA |-> Frac fA (update_tile emAw (SZ.v bm) (SZ.v bk) block_row vkt tm'));
  Pulse.Lib.Forall.elim_forall #_ eAr;
  Pulse.Lib.Trade.elim_trade _ _;
  with eBr. assert (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB eBr);
  with emBw. assert (forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
      (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
        @==> gB |-> Frac fB (update_tile emBw (SZ.v bn) (SZ.v bk) block_col vkt tm'));
  Pulse.Lib.Forall.elim_forall #_ eBr;
  Pulse.Lib.Trade.elim_trade _ _;
}
#pop-options

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1"
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
{
  let num_k_tiles = k /^ bk;
  FStar.Math.Lemmas.lemma_mult_le_right (ldt bk skew) 1 (SZ.v bm);
  assert pure (SZ.fits (ldt bk skew));
  let ldsz = P.ldt_sz bk skew;

  let aFrags   = __alloc_array_fragment et_ab  FragA   frag_sz frag_sz frag_sz FragLRM  (wm /^ frag_sz);
  let bFrags   = __alloc_array_fragment et_ab  FragB   frag_sz frag_sz frag_sz FragLCM  (wn /^ frag_sz);

  (* ---- prologue: stage k-tile 0 into buffer 0 ---- *)
  let b0 = PC.get_batch ();
  let tileA0 = array2_extract_tile_st gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v 0sz);
  let tileB0 = array2_extract_tile_st gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v 0sz);

  unfold (pipe_live (skewed_view bm bk skew sarA0) (SZ.v nthr) (SZ.v tid));
  unfold (FB.live_strided_chunks (skewed_view bm bk skew sarA0) (SZ.v nthr) (SZ.v tid));
  with emAd0. assert (FB.own_strided_chunks (skewed_view bm bk skew sarA0) emAd0 (SZ.v nthr) (SZ.v tid));
  unfold (pipe_live (skewed_view bn bk skew sarB0) (SZ.v nthr) (SZ.v tid));
  unfold (FB.live_strided_chunks (skewed_view bn bk skew sarB0) (SZ.v nthr) (SZ.v tid));
  with emBd0. assert (FB.own_strided_chunks (skewed_view bn bk skew sarB0) emBd0 (SZ.v nthr) (SZ.v tid));

  lemma_subtile_aligned lA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v 0sz) (SZ.v (chunk et_ab));
  lemma_subtile_aligned lB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v 0sz) (SZ.v (chunk et_ab));
  lemma_aligned_srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz (SZ.v (chunk et_ab));
  lemma_aligned_srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz (SZ.v (chunk et_ab));

  let b0' = stage_tiles
    #et_ab #_ #_ #_ #_
    #(l2_skewed_row_major (SZ.v bm) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bm bk skew sarA0)
    tileA0
    #_
    #(l2_skewed_row_major (SZ.v bn) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bn bk skew sarB0)
    tileB0
    (SZ.v nthr) (SZ.v tid) () ()
    fA fB
    a_t_row a_t_col a_row_step a_iters
    b_t_row b_t_col b_row_step b_iters
    b0 () ();

  rewrite each tileA0 as array2_subtile gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v 0sz);
  rewrite each tileB0 as array2_subtile gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v 0sz);


  fold_pending_even bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
    (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) b0 () 0;

  let mut idx = 0sz;
  FStar.Math.Lemmas.lemma_div_le (SZ.v bk) (SZ.v k) (SZ.v bk);
  assert pure (SZ.v bk / SZ.v bk == 1);
  assert pure (1 <= SZ.v k / SZ.v bk);
  while (!idx <^ (num_k_tiles -^ 1sz))
    invariant
      exists* (vkt : SZ.t { SZ.v vkt <= SZ.v k / SZ.v bk - 1 }) (b_c b_l : PC.pipeline_batch_t).
        (idx |-> vkt) **
        gpu ** thread_id (SZ.v nthr) (SZ.v tid) **
        B.barrier_tok
          (pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)) **
        B.barrier_state (SZ.v vkt) **
        live aFrags ** live bFrags ** live accFrags **
        PC.batch_committed b_c ** PC.batch_live b_l **
        pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
          (SZ.v block_row) (SZ.v block_col) b_c () (SZ.v vkt)
    decreases (SZ.v num_k_tiles - SZ.v !idx)
  {
    with vkt b_c b_l. _;
    let kti = !idx;
    let par = Kuiper.SizeT.sizet_and kti 1sz;
    Kuiper.SizeT.sizet_and_div_pow2 kti 2sz 1;
    let kt1 : szlt (SZ.v k / SZ.v bk) = kti +^ 1sz;
    (* discharge the parity conclusions once, in the light pre-branch context,
       so the (heavy, post-body_compute) branch VCs only need modus ponens *)
    assert pure ((SZ.v par = 0 ==> SZ.v kt1 % 2 = 1) /\
                 (SZ.v par <> 0 ==> SZ.v kt1 % 2 = 0));

    if (par = 0sz) {
      (* ---- EVEN: current = buffer 0, other = buffer 1 ---- *)
      unfold_pending_even bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_c) () (SZ.v vkt);
      with bd. assert (PC.batch_committed bd);
      PC.pipeline_wait_all_prior #bd;
      redeem_pledge emp_inames (PC.batch_done bd) _;
      drop_ (PC.batch_done bd);
      restore_globals bm bn bk gA gB fA fB (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt);
      fold (pipe_live (skewed_view bm bk skew sarA0) (SZ.v nthr) (SZ.v tid));
      fold (pipe_live (skewed_view bn bk skew sarB0) (SZ.v nthr) (SZ.v tid));
      fold_pipe_p_even bm bn bk skew sarA0 sarA1 sarB0 sarB1
        (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
      rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
        as ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rin (SZ.v vkt) (SZ.v tid));
      B.barrier_wait () #_ #_ #(SZ.v vkt) #(SZ.v tid);
      rewrite ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rout (SZ.v vkt) (SZ.v tid))
        as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid));
      unfold_pipe_q_even bm bn bk skew sarA0 sarA1 sarB0 sarB1
        (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
      let b_l2 = body_compute bm bn bk wm wn skew gA gB
        sarA0 sarA1 sarB0 sarB1 fmap fA fB nthr tid block_row block_col warp_m warp_n
        a_t_row a_t_col a_row_step a_iters b_t_row b_t_col b_row_step b_iters
        ldsz kt1 (reveal b_l) aFrags bFrags accFrags () () () () () () ();
      fold_pending_odd bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v kt1);
      idx := kt1;
    } else {
      (* ---- ODD: current = buffer 1, other = buffer 0 ---- *)
      unfold_pending_odd bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_c) () (SZ.v vkt);
      with bd. assert (PC.batch_committed bd);
      PC.pipeline_wait_all_prior #bd;
      redeem_pledge emp_inames (PC.batch_done bd) _;
      drop_ (PC.batch_done bd);
      restore_globals bm bn bk gA gB fA fB (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt);
      fold (pipe_live (skewed_view bm bk skew sarA1) (SZ.v nthr) (SZ.v tid));
      fold (pipe_live (skewed_view bn bk skew sarB1) (SZ.v nthr) (SZ.v tid));
      fold_pipe_p_odd bm bn bk skew sarA0 sarA1 sarB0 sarB1
        (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
      rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
        as ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rin (SZ.v vkt) (SZ.v tid));
      B.barrier_wait () #_ #_ #(SZ.v vkt) #(SZ.v tid);
      rewrite ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rout (SZ.v vkt) (SZ.v tid))
        as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid));
      unfold_pipe_q_odd bm bn bk skew sarA0 sarA1 sarB0 sarB1
        (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
      let b_l2 = body_compute bm bn bk wm wn skew gA gB
        sarA1 sarA0 sarB1 sarB0 fmap fA fB nthr tid block_row block_col warp_m warp_n
        a_t_row a_t_col a_row_step a_iters b_t_row b_t_col b_row_step b_iters
        ldsz kt1 (reveal b_l) aFrags bFrags accFrags () () () () () () ();
      assert pure (SZ.v kt1 % 2 = 0);
      assert pure (SZ.v kt1 <> 0);
      fold_pending_even_pos bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v kt1);
      idx := kt1;
    }
  };

  (* ---- peeled tail: last k-tile [ktiles-1], math only, no staging ---- *)
  with xtra vkt b_c b_l. _;
  let ktL = !idx;
  let parL = Kuiper.SizeT.sizet_and ktL 1sz;
  Kuiper.SizeT.sizet_and_div_pow2 ktL 2sz 1;

  if (parL = 0sz) {
    unfold_pending_even bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
      (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_c) () (SZ.v vkt);
    with bd. assert (PC.batch_committed bd);
    PC.pipeline_wait_all_prior #bd;
    redeem_pledge emp_inames (PC.batch_done bd) _;
    drop_ (PC.batch_done bd);
    restore_globals bm bn bk gA gB fA fB (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt);
    fold (pipe_live (skewed_view bm bk skew sarA0) (SZ.v nthr) (SZ.v tid));
    fold (pipe_live (skewed_view bn bk skew sarB0) (SZ.v nthr) (SZ.v tid));
    fold_pipe_p_even bm bn bk skew sarA0 sarA1 sarB0 sarB1
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
    rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
      as ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rin (SZ.v vkt) (SZ.v tid));
    B.barrier_wait () #_ #_ #(SZ.v vkt) #(SZ.v tid);
    rewrite ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rout (SZ.v vkt) (SZ.v tid))
      as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid));
    unfold_pipe_q_even bm bn bk skew sarA0 sarA1 sarB0 sarB1
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
    subproducts_buf bm bn bk wm wn skew sarA0 sarB0 fmap nthr warp_m warp_n ldsz
      aFrags bFrags accFrags () () ();
    fold_pipe_q_even bm bn bk skew sarA0 sarA1 sarB0 sarB1
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
    assert pure (SZ.v vkt == SZ.v k / SZ.v bk - 1);
    rewrite (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
      as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v k / SZ.v bk - 1) (SZ.v tid));
    rewrite (B.barrier_state (SZ.v vkt + 1))
      as (B.barrier_state (SZ.v k / SZ.v bk));
    with lb. assert (PC.batch_live lb);
    drop_ (PC.batch_live lb);
    with va. assert (aFrags |-> va);
    drop_ (aFrags |-> va);
    with vb. assert (bFrags |-> vb);
    drop_ (bFrags |-> vb);
    ()
  } else {
    unfold_pending_odd bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
      (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_c) () (SZ.v vkt);
    with bd. assert (PC.batch_committed bd);
    PC.pipeline_wait_all_prior #bd;
    redeem_pledge emp_inames (PC.batch_done bd) _;
    drop_ (PC.batch_done bd);
    restore_globals bm bn bk gA gB fA fB (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt);
    fold (pipe_live (skewed_view bm bk skew sarA1) (SZ.v nthr) (SZ.v tid));
    fold (pipe_live (skewed_view bn bk skew sarB1) (SZ.v nthr) (SZ.v tid));
    fold_pipe_p_odd bm bn bk skew sarA0 sarA1 sarB0 sarB1
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
    rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
      as ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rin (SZ.v vkt) (SZ.v tid));
    B.barrier_wait () #_ #_ #(SZ.v vkt) #(SZ.v tid);
    rewrite ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rout (SZ.v vkt) (SZ.v tid))
      as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid));
    unfold_pipe_q_odd bm bn bk skew sarA0 sarA1 sarB0 sarB1
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
    subproducts_buf bm bn bk wm wn skew sarA1 sarB1 fmap nthr warp_m warp_n ldsz
      aFrags bFrags accFrags () () ();
    fold_pipe_q_odd bm bn bk skew sarA0 sarA1 sarB0 sarB1
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid);
    assert pure (SZ.v vkt == SZ.v k / SZ.v bk - 1);
    rewrite (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
      as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v k / SZ.v bk - 1) (SZ.v tid));
    rewrite (B.barrier_state (SZ.v vkt + 1))
      as (B.barrier_state (SZ.v k / SZ.v bk));
    with lb. assert (PC.batch_live lb);
    drop_ (PC.batch_live lb);
    with va. assert (aFrags |-> va);
    drop_ (aFrags |-> va);
    with vb. assert (bFrags |-> vb);
    drop_ (bFrags |-> vb);
    ()
  }
}
#pop-options
