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

(* Parity-flip buffer selection: the buffer holding k-tile [r+1] under the
   double-buffer scheme is the opposite of the one holding k-tile [r].  Proven
   once as a pure lemma so the k-loop body need not do fragile inline [ite]
   arithmetic. *)
let sel_flip (#a:Type0) (r:nat) (x y:a)
  : Lemma ((if (r + 1) % 2 = 0 then x else y) == (if r % 2 = 0 then y else x))
  = FStar.Math.Lemmas.lemma_mod_add_distr 1 r 2

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
#pop-options

(* ---- parity-independent per-k-tile compute + next-tile staging ----
   [cur*] are the buffers just read (held read-shared, consumed only by
   [subproducts]); [oth*] are the now-writable buffers into which k-tile
   [kt1 = kt+1] is staged. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 15"
(* ---- the "cur-buffer pledge + global-restore wands" half of [pending],
   named so [stage_next] can produce it and the k-loop can recombine it with
   the read-share of the just-consumed buffer.  [pending_body cur oth false]
   is exactly [staged_half cur ** pipe_sharing oth]. ---- *)
let staged_half
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (curA : larray et_ab (SZ.v bm * ldt bk skew))
  (curB : larray et_ab (SZ.v bn * ldt bk skew))
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
    (FB.live_strided_chunks (skewed_view bm bk skew curA) nthr tid **
     FB.live_strided_chunks (skewed_view bn bk skew curB) nthr tid **
     (exists* eA eB.
        (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA eA) **
        (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB eB))) **
  (exists* (emA : chest2 et_ab (SZ.v m) (SZ.v k)).
     forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
       (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
         @==> gA |-> Frac fA (update_tile emA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
  (exists* (emB : chest2 et_ab (SZ.v n) (SZ.v k)).
     forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
       (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
         @==> gB |-> Frac fB (update_tile emB (SZ.v bn) (SZ.v bk) block_col vkt tm'))
#pop-options

(* ---- stage one k-tile [kt] of the globals into the writable buffers
   [dstA]/[dstB], leaving [staged_half] (the pledge + global-restore wands)
   plus the committed/successor batches.  This is the ONLY copy of the
   cp.async staging wrapper: it is called both by the prologue (k-tile 0 into
   buffer 0) and by the single k-loop body (k-tile kt+1 into the opposite
   buffer). ---- *)
#push-options "--z3rlimit 15 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn stage_next
  (#et_ab : Type0)
  {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| _clA : T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| _clB : T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB)
  (dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : szp)
  (tid : szlt nthr)
  (block_row : szlt (SZ.v m / SZ.v bm))
  (block_col : szlt (SZ.v n / SZ.v bn))
  (a_t_row a_t_col a_row_step a_iters : SZ.t)
  (b_t_row b_t_col b_row_step b_iters : SZ.t)
  (ldsz : szp { SZ.v ldsz == ldt bk skew })
  (kt : SZ.t)
  (b : PC.pipeline_batch_t)
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
     aligned 16 (T.core (skewed_view bm bk skew dstA)) /\
     aligned 16 (T.core (skewed_view bn bk skew dstB))))
  (sq_dd : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                   SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (sq_ch : squash (SZ.v (chunk et_ab) /?+ SZ.v bk /\ SZ.v (chunk et_ab) /?+ SZ.v skew /\
                   SZ.v kt < SZ.v k / SZ.v bk))
  ()
  preserves gpu
  requires
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    pipe_live (skewed_view bm bk skew dstA) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bn bk skew dstB) (SZ.v nthr) (SZ.v tid) **
    PC.batch_live b
  returns b' : PC.pipeline_batch_t
  ensures
    staged_half bm bn bk skew gA gB dstA dstB fA fB (SZ.v nthr) (SZ.v tid)
      (SZ.v block_row) (SZ.v block_col) b () (SZ.v kt) **
    PC.batch_committed b ** PC.batch_live b'
{
  with egA. assert (gA |-> Frac fA egA);
  let tileA1 = array2_extract_tile_st gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt);
  with egB. assert (gB |-> Frac fB egB);
  let tileB1 = array2_extract_tile_st gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt);

  unfold (pipe_live (skewed_view bm bk skew dstA) (SZ.v nthr) (SZ.v tid));
  unfold (FB.live_strided_chunks (skewed_view bm bk skew dstA) (SZ.v nthr) (SZ.v tid));
  with emAd. assert (FB.own_strided_chunks (skewed_view bm bk skew dstA) emAd (SZ.v nthr) (SZ.v tid));
  unfold (pipe_live (skewed_view bn bk skew dstB) (SZ.v nthr) (SZ.v tid));
  unfold (FB.live_strided_chunks (skewed_view bn bk skew dstB) (SZ.v nthr) (SZ.v tid));
  with emBd. assert (FB.own_strided_chunks (skewed_view bn bk skew dstB) emBd (SZ.v nthr) (SZ.v tid));

  lemma_subtile_aligned lA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt) (SZ.v (chunk et_ab));
  lemma_subtile_aligned lB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt) (SZ.v (chunk et_ab));
  lemma_aligned_srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz (SZ.v (chunk et_ab));
  lemma_aligned_srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz (SZ.v (chunk et_ab));

  let b' = stage_tiles
    #et_ab #_ #_ #_ #_
    #(l2_skewed_row_major (SZ.v bm) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bm) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bm bk skew dstA)
    tileA1
    #_
    #(l2_skewed_row_major (SZ.v bn) (SZ.v bk) (SZ.v skew))
    #(c_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz)
    #(srm_l2_skewed_row_major #(SZ.v bn) #(SZ.v bk) #(SZ.v skew) ldsz)
    (skewed_view bn bk skew dstB)
    tileB1
    (SZ.v nthr) (SZ.v tid) () ()
    fA fB
    a_t_row a_t_col a_row_step a_iters
    b_t_row b_t_col b_row_step b_iters
    b () ();

  rewrite each tileA1 as array2_subtile gA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v kt);
  rewrite each tileB1 as array2_subtile gB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v kt);

  fold (staged_half bm bn bk skew gA gB dstA dstB fA fB (SZ.v nthr) (SZ.v tid)
          (SZ.v block_row) (SZ.v block_col) b () (SZ.v kt));
  b'
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

(* =====================================================================
   Single-body k-loop scaffolding.

   The k-loop runs ONE concrete body [ktiles] times; buffer parity and the
   last-tile case are handled in ghost steps.  [srcA]/[srcB] name the buffer
   read this iteration (= [buf]), [dstA]/[dstB] the opposite buffer staged into
   (= [buf ^ 1]); they are selected by a runtime [if] (a value-select => a CUDA
   ternary, not duplicated code).  The generic reconciliation helpers below
   fold/unfold the parity-generic [pending]/[pipe_p]/[pipe_q] predicates against
   [src*]/[dst*] using the pure equations [srcA == (if vkt%2=0 then sarA0 else
   sarA1)] etc., so no even/odd branching is needed. ===================== *)

#push-options "--fuel 1 --ifuel 2 --z3rlimit 15"

(* [pending vkt] (over the physical buffers) -> [pending_body] over the
   parity-selected [src*]/[dst*] (uniform). *)
ghost
fn unfold_pending_g
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (srcA dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (srcB dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  (sq_sel : squash (
     srcA == (if vkt % 2 = 0 then sarA0 else sarA1) /\
     dstA == (if vkt % 2 = 0 then sarA1 else sarA0) /\
     srcB == (if vkt % 2 = 0 then sarB0 else sarB1) /\
     dstB == (if vkt % 2 = 0 then sarB1 else sarB0)))
  requires
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
  ensures
    pending_body bm bn bk skew gA gB srcA dstA srcB dstB fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt
{
  rewrite
    (pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt)
  as
    (pending_body bm bn bk skew gA gB srcA dstA srcB dstB fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt);
}

(* [live src ** (ite oth)] over [src*]/[dst*] -> [pipe_p vkt]. *)
ghost
fn fold_pipe_p_g
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (srcA dstA : larray et (SZ.v bm * ldt bk skew))
  (srcB dstB : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (vkt : nat) (tid : natlt nthr)
  (sq_sel : squash (
     srcA == (if vkt % 2 = 0 then sarA0 else sarA1) /\
     dstA == (if vkt % 2 = 0 then sarA1 else sarA0) /\
     srcB == (if vkt % 2 = 0 then sarB0 else sarB1) /\
     dstB == (if vkt % 2 = 0 then sarB1 else sarB0)))
  requires
    pure (vkt < ktiles) **
    pipe_live (skewed_view bm bk skew srcA) nthr tid **
    pipe_live (skewed_view bn bk skew srcB) nthr tid **
    (if vkt = 0 then
       pipe_live (skewed_view bm bk skew dstA) nthr tid **
       pipe_live (skewed_view bn bk skew dstB) nthr tid
     else
       pipe_sharing (skewed_view bm bk skew dstA) nthr **
       pipe_sharing (skewed_view bn bk skew dstB) nthr)
  ensures
    pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles vkt tid
{
  rewrite
    (pipe_live (skewed_view bm bk skew srcA) nthr tid **
     pipe_live (skewed_view bn bk skew srcB) nthr tid **
     (if vkt = 0 then
        pipe_live (skewed_view bm bk skew dstA) nthr tid **
        pipe_live (skewed_view bn bk skew dstB) nthr tid
      else
        pipe_sharing (skewed_view bm bk skew dstA) nthr **
        pipe_sharing (skewed_view bn bk skew dstB) nthr))
  as
    (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles vkt tid);
}

(* [pipe_q vkt] -> [sharing src ** live dst] over [src*]/[dst*]. *)
ghost
fn unfold_pipe_q_g
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (srcA dstA : larray et (SZ.v bm * ldt bk skew))
  (srcB dstB : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (vkt : nat) (tid : natlt nthr)
  (sq_sel : squash (
     srcA == (if vkt % 2 = 0 then sarA0 else sarA1) /\
     dstA == (if vkt % 2 = 0 then sarA1 else sarA0) /\
     srcB == (if vkt % 2 = 0 then sarB0 else sarB1) /\
     dstB == (if vkt % 2 = 0 then sarB1 else sarB0)))
  requires
    pure (vkt < ktiles) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles vkt tid
  ensures
    pipe_sharing (skewed_view bm bk skew srcA) nthr **
    pipe_sharing (skewed_view bn bk skew srcB) nthr **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid
{
  rewrite
    (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles vkt tid)
  as
    (pipe_sharing (skewed_view bm bk skew srcA) nthr **
     pipe_sharing (skewed_view bn bk skew srcB) nthr **
     pipe_live (skewed_view bm bk skew dstA) nthr tid **
     pipe_live (skewed_view bn bk skew dstB) nthr tid);
}

(* [sharing src ** live dst] over [src*]/[dst*] -> [pipe_q vkt] (terminal). *)
ghost
fn fold_pipe_q_g
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (srcA dstA : larray et (SZ.v bm * ldt bk skew))
  (srcB dstB : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos) (ktiles : nat)
  (vkt : nat) (tid : natlt nthr)
  (sq_sel : squash (
     srcA == (if vkt % 2 = 0 then sarA0 else sarA1) /\
     dstA == (if vkt % 2 = 0 then sarA1 else sarA0) /\
     srcB == (if vkt % 2 = 0 then sarB0 else sarB1) /\
     dstB == (if vkt % 2 = 0 then sarB1 else sarB0)))
  requires
    pure (vkt < ktiles) **
    pipe_sharing (skewed_view bm bk skew srcA) nthr **
    pipe_sharing (skewed_view bn bk skew srcB) nthr **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid
  ensures
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles vkt tid
{
  rewrite
    (pipe_sharing (skewed_view bm bk skew srcA) nthr **
     pipe_sharing (skewed_view bn bk skew srcB) nthr **
     pipe_live (skewed_view bm bk skew dstA) nthr tid **
     pipe_live (skewed_view bn bk skew dstB) nthr tid)
  as
    (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles vkt tid);
}

(* [staged_half dst kt1 ** sharing src] -> [pending kt1] (kt1 = vkt+1, kt1<>0). *)
ghost
fn fold_pending_g
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (sarA0 sarA1 : larray et_ab (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et_ab (SZ.v bn * ldt bk skew))
  (srcA dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (srcB dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (b_c : PC.pipeline_batch_t)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (kt1 : nat { kt1 < SZ.v k / SZ.v bk })
  (sq_sel : squash (
     kt1 <> 0 /\
     dstA == (if kt1 % 2 = 0 then sarA0 else sarA1) /\
     srcA == (if kt1 % 2 = 0 then sarA1 else sarA0) /\
     dstB == (if kt1 % 2 = 0 then sarB0 else sarB1) /\
     srcB == (if kt1 % 2 = 0 then sarB1 else sarB0)))
  requires
    pending_body bm bn bk skew gA gB dstA srcA dstB srcB fA fB nthr tid block_row block_col
      b_c sq false kt1
  ensures
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq kt1
{
  rewrite
    (pending_body bm bn bk skew gA gB dstA srcA dstB srcB fA fB nthr tid block_row block_col
      b_c sq false kt1)
  as
    (pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq kt1);
}
#pop-options

(* ---- the k-loop invariant tail: [pending vkt] while staging, [pipe_q] once
   done ([vkt == ktiles]).  A single conditional slprop keyed on [vkt < ktiles]
   so ONE loop over [ktiles] iterations subsumes the (old) peeled last tile. --*)
#push-options "--fuel 1 --ifuel 2 --z3rlimit 15"
let kcarry
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
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt <= SZ.v k / SZ.v bk })
  : slprop
= if vkt < SZ.v k / SZ.v bk then
    (exists* (b_c b_l : PC.pipeline_batch_t).
      PC.batch_committed b_c ** PC.batch_live b_l **
      pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt)
  else
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr (SZ.v k / SZ.v bk)
           (SZ.v k / SZ.v bk - 1) tid

ghost
fn unfold_kcarry_live
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
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt <= SZ.v k / SZ.v bk })
  (sq_lt : squash (vkt < SZ.v k / SZ.v bk))
  requires
    kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt
  ensures
    exists* (b_c b_l : PC.pipeline_batch_t).
      PC.batch_committed b_c ** PC.batch_live b_l **
      pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
{
  rewrite
    (kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt)
  as
    (exists* (b_c b_l : PC.pipeline_batch_t).
      PC.batch_committed b_c ** PC.batch_live b_l **
      pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt);
}

ghost
fn fold_kcarry_live
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
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt <= SZ.v k / SZ.v bk })
  (b_c : PC.pipeline_batch_t)
  (sq_lt : squash (vkt < SZ.v k / SZ.v bk))
  requires
    PC.batch_committed b_c ** (exists* (bl : PC.pipeline_batch_t). PC.batch_live bl) **
    pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
  ensures
    kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt
{
  with (b_l : PC.pipeline_batch_t). assert (PC.batch_live b_l);
  introduce
    exists* (c l : PC.pipeline_batch_t).
      PC.batch_committed c ** PC.batch_live l **
      pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col c sq vkt
  with b_c b_l;
  rewrite
    (exists* (c l : PC.pipeline_batch_t).
      PC.batch_committed c ** PC.batch_live l **
      pending bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col c sq vkt)
  as
    (kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt);
}

ghost
fn fold_kcarry_done
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
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt <= SZ.v k / SZ.v bk })
  (sq_ge : squash (~(vkt < SZ.v k / SZ.v bk)))
  requires
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr (SZ.v k / SZ.v bk)
           (SZ.v k / SZ.v bk - 1) tid
  ensures
    kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt
{
  rewrite
    ((exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
     pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr (SZ.v k / SZ.v bk)
            (SZ.v k / SZ.v bk - 1) tid)
  as
    (kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt);
}
#pop-options

(* ---- the guard output: the "dst-side" resource produced by the single
   staging guard, carried across the [subproducts] math (which round-trips only
   [pipe_sharing src]).  [stg = kt1 < ktiles]. ---- *)
#push-options "--fuel 1 --ifuel 2 --z3rlimit 15"
let cstage
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (b_l : PC.pipeline_batch_t)
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  : slprop
= if vkt + 1 < SZ.v k / SZ.v bk then
    PC.batch_committed b_l ** (exists* b_l'. PC.batch_live b_l') **
    staged_half bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col b_l sq (vkt + 1)
  else
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid **
    PC.batch_live b_l

ghost
fn fold_cstage_stage
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (b_l b_l' : PC.pipeline_batch_t)
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  (sq_stg : squash (vkt + 1 < SZ.v k / SZ.v bk))
  requires
    PC.batch_committed b_l ** PC.batch_live b_l' **
    staged_half bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col b_l sq (vkt + 1)
  ensures
    cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt
{
  introduce exists* (bl : PC.pipeline_batch_t). PC.batch_live bl with b_l';
  rewrite
    (PC.batch_committed b_l ** (exists* (bl : PC.pipeline_batch_t). PC.batch_live bl) **
     staged_half bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col b_l sq (vkt + 1))
  as
    (cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt);
}

ghost
fn fold_cstage_nostage
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (b_l : PC.pipeline_batch_t)
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  (sq_nostg : squash (~(vkt + 1 < SZ.v k / SZ.v bk)))
  requires
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid **
    PC.batch_live b_l
  ensures
    cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt
{
  rewrite
    ((exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
     pipe_live (skewed_view bm bk skew dstA) nthr tid **
     pipe_live (skewed_view bn bk skew dstB) nthr tid **
     PC.batch_live b_l)
  as
    (cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt);
}

ghost
fn unfold_cstage_stage
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (b_l : PC.pipeline_batch_t)
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  (sq_stg : squash (vkt + 1 < SZ.v k / SZ.v bk))
  requires
    cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt
  ensures
    PC.batch_committed b_l ** (exists* b_l'. PC.batch_live b_l') **
    staged_half bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col b_l sq (vkt + 1)
{
  rewrite
    (cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt)
  as
    (PC.batch_committed b_l ** (exists* b_l'. PC.batch_live b_l') **
     staged_half bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col b_l sq (vkt + 1));
}

ghost
fn unfold_cstage_nostage
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (dstA : larray et_ab (SZ.v bm * ldt bk skew))
  (dstB : larray et_ab (SZ.v bn * ldt bk skew))
  (fA fB : perm)
  (nthr : pos) (tid : natlt nthr)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (b_l : PC.pipeline_batch_t)
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  (sq_nostg : squash (~(vkt + 1 < SZ.v k / SZ.v bk)))
  requires
    cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt
  ensures
    (exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid **
    PC.batch_live b_l
{
  rewrite
    (cstage bm bn bk skew gA gB dstA dstB fA fB nthr tid block_row block_col sq b_l vkt)
  as
    ((exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
     pipe_live (skewed_view bm bk skew dstA) nthr tid **
     pipe_live (skewed_view bn bk skew dstB) nthr tid **
     PC.batch_live b_l);
}
#pop-options


#push-options "--z3rlimit 15 --fuel 1 --ifuel 2"
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

  (* ---- fold the prologue's [pending 0] + batches into the loop carry ---- *)
  fold_kcarry_live bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
    (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () 0 b0 ();

  let mut idx = 0sz;
  FStar.Math.Lemmas.lemma_div_le (SZ.v bk) (SZ.v k) (SZ.v bk);
  assert pure (SZ.v bk / SZ.v bk == 1);
  assert pure (1 <= SZ.v k / SZ.v bk);

  (* ---- single-body software-pipelined k-loop.  ONE concrete body runs
     [ktiles] times; buffer parity and the last-tile (no-stage) case are
     handled entirely in ghost steps.  [srcA]/[srcB] = buffer read this
     iteration ([buf]); [dstA]/[dstB] = opposite buffer staged into
     ([buf ^ 1]); they are runtime value-selects (=> CUDA ternaries, not
     duplicated statements). ---- *)
  while (!idx <^ num_k_tiles)
    invariant
      exists* (vkt : SZ.t { SZ.v vkt <= SZ.v k / SZ.v bk }).
        gpu ** thread_id (SZ.v nthr) (SZ.v tid) **
        B.barrier_tok
          (pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)) **
        B.barrier_state (SZ.v vkt) **
        live aFrags ** live bFrags ** live accFrags **
        kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
          (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt) **
        (idx |-> vkt)
    decreases (SZ.v num_k_tiles - SZ.v !idx)
  {
    with vkt. _;
    let kti = !idx;
    let par = Kuiper.SizeT.sizet_and kti 1sz;
    Kuiper.SizeT.sizet_and_div_pow2 kti 2sz 1;
    assert pure (SZ.v num_k_tiles == SZ.v k / SZ.v bk);
    assert pure (SZ.v kti < SZ.v k / SZ.v bk);
    let kt1 : (x:SZ.t{SZ.v x <= SZ.v k / SZ.v bk}) = kti +^ 1sz;

    (* CONCRETE buffer value-selects: extract to CUDA ternaries. *)
    let srcA = if (par = 0sz) { sarA0 } else { sarA1 };
    let srcB = if (par = 0sz) { sarB0 } else { sarB1 };
    let dstA = if (par = 0sz) { sarA1 } else { sarA0 };
    let dstB = if (par = 0sz) { sarB1 } else { sarB0 };

    (* the buffer-selection equations, keyed on [vkt % 2], feeding the generic
       reconciliation helpers.  Established once in the light pre-body context. *)
    assert pure (SZ.v kti == SZ.v vkt);
    assert pure (SZ.v vkt < SZ.v k / SZ.v bk);
    assert pure (SZ.v par == SZ.v vkt % 2);
    assert pure (SZ.v kt1 == SZ.v vkt + 1 /\ SZ.v kt1 <= SZ.v k / SZ.v bk);
    assert pure (
      srcA == (if SZ.v vkt % 2 = 0 then sarA0 else sarA1) /\
      dstA == (if SZ.v vkt % 2 = 0 then sarA1 else sarA0) /\
      srcB == (if SZ.v vkt % 2 = 0 then sarB0 else sarB1) /\
      dstB == (if SZ.v vkt % 2 = 0 then sarB1 else sarB0));

    (* ---- ghost: expose the current-tile pledge over [src*]/[dst*] ---- *)
    unfold_kcarry_live bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
      (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt) ();
    with b_c b_l. _;
    unfold_pending_g bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB fA fB
      (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_c) () (SZ.v vkt) ();

    (* ---- CONCRETE: __pipeline_wait_prior(0) then redeem the staged pledge ---- *)
    with bd. assert (PC.batch_committed bd);
    PC.pipeline_wait_all_prior #bd;
    redeem_pledge emp_inames (PC.batch_done bd) _;
    drop_ (PC.batch_done bd);
    restore_globals bm bn bk gA gB fA fB (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt);
    fold (pipe_live (skewed_view bm bk skew srcA) (SZ.v nthr) (SZ.v tid));
    fold (pipe_live (skewed_view bn bk skew srcB) (SZ.v nthr) (SZ.v tid));

    (* ---- CONCRETE: __syncthreads() (barrier) advancing the pipe contract ---- *)
    fold_pipe_p_g bm bn bk skew sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid) ();
    rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
      as ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rin (SZ.v vkt) (SZ.v tid));
    B.barrier_wait () #_ #_ #(SZ.v vkt) #(SZ.v tid);
    rewrite ((pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)).B.rout (SZ.v vkt) (SZ.v tid))
      as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid));
    unfold_pipe_q_g bm bn bk skew sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB
      (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid) ();

    (* ---- the ONLY copy of the staging code: guarded on [kt+1 < ktiles] ---- *)
    if (kt1 <^ num_k_tiles) {
      let b_l2 = stage_next bm bn bk skew gA gB dstA dstB fA fB
        nthr tid block_row block_col
        a_t_row a_t_col a_row_step a_iters b_t_row b_t_col b_row_step b_iters
        ldsz kt1 (reveal b_l) () () () () () () ();
      rewrite (staged_half bm bn bk skew gA gB dstA dstB fA fB (SZ.v nthr) (SZ.v tid)
                (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v kt1))
        as (staged_half bm bn bk skew gA gB dstA dstB fA fB (SZ.v nthr) (SZ.v tid)
                (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v vkt + 1));
      fold_cstage_stage bm bn bk skew gA gB dstA dstB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) b_l2 (SZ.v vkt) ();
    } else {
      fold_cstage_nostage bm bn bk skew gA gB dstA dstB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) (SZ.v vkt) ();
    };

    (* ---- the ONLY copy of the fragment math, over the read buffer [src*] ---- *)
    subproducts_buf bm bn bk wm wn skew srcA srcB fmap nthr warp_m warp_n ldsz
      aFrags bFrags accFrags () () ();

    (* ---- ghost: reconcile the guard output back into the loop carry ---- *)
    sel_flip (SZ.v vkt) sarA0 sarA1;
    sel_flip (SZ.v vkt) sarA1 sarA0;
    sel_flip (SZ.v vkt) sarB0 sarB1;
    sel_flip (SZ.v vkt) sarB1 sarB0;
    assert pure (
      (SZ.v vkt + 1) <> 0 /\
      dstA == (if (SZ.v vkt + 1) % 2 = 0 then sarA0 else sarA1) /\
      srcA == (if (SZ.v vkt + 1) % 2 = 0 then sarA1 else sarA0) /\
      dstB == (if (SZ.v vkt + 1) % 2 = 0 then sarB0 else sarB1) /\
      srcB == (if (SZ.v vkt + 1) % 2 = 0 then sarB1 else sarB0));
    if (kt1 <^ num_k_tiles) {
      unfold_cstage_stage bm bn bk skew gA gB dstA dstB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) (SZ.v vkt) ();
      unfold (staged_half bm bn bk skew gA gB dstA dstB fA fB (SZ.v nthr) (SZ.v tid)
                (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v vkt + 1));
      fold_pending_g bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v vkt + 1) ();
      fold_kcarry_live bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt + 1) (reveal b_l) ();
    } else {
      assert pure (SZ.v num_k_tiles == SZ.v k / SZ.v bk);
      assert pure (SZ.v kt1 == SZ.v vkt + 1 /\ SZ.v vkt < SZ.v k / SZ.v bk);
      unfold_cstage_nostage bm bn bk skew gA gB dstA dstB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) (SZ.v vkt) ();
      fold_pipe_q_g bm bn bk skew sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB
        (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid) ();
      assert pure (SZ.v vkt == SZ.v k / SZ.v bk - 1);
      rewrite (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
        as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v k / SZ.v bk - 1) (SZ.v tid));
      with lb. assert (PC.batch_live lb);
      drop_ (PC.batch_live lb);
      fold_kcarry_done bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt + 1) ();
    };
    rewrite
      (kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
        (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt + 1))
    as
      (kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
        (SZ.v block_row) (SZ.v block_col) () (SZ.v kt1));
    rewrite (B.barrier_state (SZ.v vkt + 1)) as (B.barrier_state (SZ.v kt1));
    idx := !idx +^ 1sz;
  };

  (* ---- loop exit: [vkt == ktiles], so [kcarry] is its "done" branch ---- *)
  with vkt. assert (idx |-> vkt);
  assert pure (SZ.v vkt == SZ.v k / SZ.v bk);
  rewrite
    (kcarry bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
      (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt))
  as
    ((exists* e. gA |-> Frac fA e) ** (exists* e. gB |-> Frac fB e) **
     pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)
            (SZ.v k / SZ.v bk - 1) (SZ.v tid));
  rewrite (B.barrier_state (SZ.v vkt))
    as (B.barrier_state (SZ.v k / SZ.v bk));
  with va. assert (aFrags |-> va);
  drop_ (aFrags |-> va);
  with vb. assert (bFrags |-> vb);
  drop_ (bFrags |-> vb);
  ()
}
#pop-options
