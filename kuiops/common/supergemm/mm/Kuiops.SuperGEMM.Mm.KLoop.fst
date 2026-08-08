module Kuiops.SuperGEMM.Mm.KLoop

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.TensorCore
open Kuiper.Chest { chest_map }
open Pulse.Lib.Array { length }
open Pulse.Lib.Array.PtsTo { op_Array_Access }
open Pulse.Lib.Trade

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params { frag, frag_sz }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

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
  (#_ : squash (length accFrags == SZ.v wm / frag * (SZ.v wn / frag)))
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
