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
open Kuiper.Chest { chest_map, equal, chest2, acc2, const }
open Kuiper.EMatrix { mtranspose }
open Kuiper.EMatrix.Tiling { update_tile_self }
open Kuiper.Spec.GEMM { matmul, matplus, __gmatmul_single, __gmatmul_single_lemma,
                        gmatmul_single, matmul_tiles_lemma }
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul, mtranspose_subtile }
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }
open Pulse.Lib.Array { length }
open Pulse.Lib.Array.PtsTo { op_Array_Access }
open Pulse.Lib.Trade
open Pulse.Lib.Pledge

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Params { frag, frag_sz, ldt }
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_live, pipe_sharing, pipe_p, pipe_q,
                                    pipe_contract,
                                    pipe_live_c, pipe_sharing_c, pipe_p_c, pipe_q_c,
                                    pipe_contract_c, pipe_p_to_q_transform_c, pipe_q_c_forget,
                                    unfold_pipe_q_c_even, unfold_pipe_q_c_odd,
                                    fold_pipe_p_c_even, fold_pipe_p_c_odd }
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

(* Last-k-tile arithmetic, isolated as a pure lemma so the resource-heavy
   k-loop body does not spend its rlimit budget re-deriving trivial linear
   facts inside a large SMT context. *)
let ktiles_last_arith (kt1 vkt kk : nat) (bkv : pos) (nkt : nat)
  : Lemma (requires nkt == kk / bkv /\ kt1 == vkt + 1 /\ vkt < kk / bkv)
          (ensures (kt1 < nkt) \/ (kt1 == kk / bkv /\ vkt == kk / bkv - 1))
  = ()

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

(* ----------------------------------------------------------------------------
   Functional (real-approximation) layer for one warp's k-tile of fragment
   math.  Mirrors Kuiper.Kernel.GEMM.TensorCore2D.To.Fragments, specialized to
   the constant 16x16x16 fragment dims and to B's FragLCM tag (B consumed
   transposed for free by the tensor core).  Everything here is ghost/spec;
   the executable structure of [sp_load_a]/[sp_load_b]/[sp_mma] is unchanged.
   ---------------------------------------------------------------------------- *)

(* [frag] divides [w] -> the fragment count times [frag] recovers [w]. *)
let frag_div_mul (w : pos)
  : Lemma (requires frag /?+ w)
          (ensures (w / frag) * frag == w /\ frag * (w / frag) == w)
= D.lemma_nat_divides_pos_divides frag w;
  let g = D.get_factor frag w in
  FStar.Math.Lemmas.multiple_division_lemma g frag

(* Applying a pointwise-identity elementwise map to a chest is the identity. *)
let chest2_map_id (#et:Type) (#rows #cols : nat)
  (f : et -> et)
  (m0 : chest2 et rows cols)
  : Lemma (requires forall (x:et). f x == x)
          (ensures chest_map f m0 == m0)
= assert (equal (chest_map f m0) m0)

(* Elementwise elimination of a chest approximation at a single index. *)
let elim_approx (#et:Type0) {| scalar et, real_like et |} (#rows #cols:nat)
  (m1 : chest2 et rows cols) (m2 : chest2 real rows cols)
  (i:natlt rows) (j:natlt cols)
  : Lemma (requires m1 %~ m2) (ensures acc2 m1 i j %~ acc2 m2 i j)
= ()

(* The [i]-th 16x16 A row-fragment loaded from the global tile approximates the
   [i]-th sub-fragment of the per-warp A tile of [rA]. *)
let sp_a_tile_approx
  (#et:Type0) {| scalar et, real_like et |}
  (bm bk wm : pos)
  (eA : chest2 et bm bk) (rA : chest2 real bm bk)
  (warp_m ks i : nat)
  (_ : squash (frag /?+ wm /\ frag /?+ bk /\ frag /?+ bm /\ wm /?+ bm /\
               warp_m < bm/wm /\ ks < bk/frag /\ i < wm/frag /\ eA %~ rA))
  : Lemma (ensures
      ematrix_subtile eA frag frag (warp_m*(wm/frag)+i) ks
        %~ ematrix_subtile (ematrix_subtile rA wm frag warp_m ks) frag frag i 0)
= frag_div_mul wm; frag_div_mul bm; frag_div_mul bk;
  D.lemma_nat_divides_pos_divides frag wm;
  D.lemma_nat_divides_pos_divides frag bk;
  D.lemma_nat_divides_pos_divides frag bm;
  D.lemma_nat_divides_pos_divides wm bm;
  a_tile_bound bm wm warp_m i;
  let lhs = ematrix_subtile eA frag frag (warp_m*(wm/frag)+i) ks in
  let rhs = ematrix_subtile (ematrix_subtile rA wm frag warp_m ks) frag frag i 0 in
  introduce forall (i':natlt frag) (j':natlt frag). acc2 lhs i' j' %~ acc2 rhs i' j'
  with (
    FStar.Math.Lemmas.paren_mul_right warp_m (wm/frag) frag;
    let r = (warp_m*(wm/frag)+i)*frag + i' in
    let c = ks*frag + j' in
    FStar.Math.Lemmas.lemma_mult_le_right frag (warp_m*(wm/frag)+i+1) (bm/frag);
    FStar.Math.Lemmas.lemma_mult_le_right frag (ks+1) (bk/frag);
    elim_approx eA rA r c
  );
  lemma_approximates_intro lhs rhs

(* The [j]-th 16x16 B col-fragment (FragLCM) loaded from the global tile
   approximates the [j]-th sub-fragment of the per-warp B tile of [rB]. *)
let sp_b_tile_approx
  (#et:Type0) {| scalar et, real_like et |}
  (bn bk wn : pos)
  (eB : chest2 et bk bn) (rB : chest2 real bk bn)
  (warp_n ks j : nat)
  (_ : squash (frag /?+ wn /\ frag /?+ bk /\ frag /?+ bn /\ wn /?+ bn /\
               warp_n < bn/wn /\ ks < bk/frag /\ j < wn/frag /\ eB %~ rB))
  : Lemma (ensures
      ematrix_subtile eB frag frag ks (warp_n*(wn/frag)+j)
        %~ ematrix_subtile (ematrix_subtile rB frag wn ks warp_n) frag frag 0 j)
= frag_div_mul wn; frag_div_mul bn; frag_div_mul bk;
  D.lemma_nat_divides_pos_divides frag wn;
  D.lemma_nat_divides_pos_divides frag bk;
  D.lemma_nat_divides_pos_divides frag bn;
  D.lemma_nat_divides_pos_divides wn bn;
  a_tile_bound bn wn warp_n j;
  let lhs = ematrix_subtile eB frag frag ks (warp_n*(wn/frag)+j) in
  let rhs = ematrix_subtile (ematrix_subtile rB frag wn ks warp_n) frag frag 0 j in
  introduce forall (i':natlt frag) (j':natlt frag). acc2 lhs i' j' %~ acc2 rhs i' j'
  with (
    FStar.Math.Lemmas.paren_mul_right warp_n (wn/frag) frag;
    let r = ks*frag + i' in
    let c = (warp_n*(wn/frag)+j)*frag + j' in
    FStar.Math.Lemmas.lemma_mult_le_right frag (warp_n*(wn/frag)+j+1) (bn/frag);
    FStar.Math.Lemmas.lemma_mult_le_right frag (ks+1) (bk/frag);
    elim_approx eB rB r c
  );
  lemma_approximates_intro lhs rhs

(* ---- nested-subtile composition + approximation lemmas (pure/ghost),
   proven by chest2 cell extensionality.  Used to relate the block-warp matmul
   [subproducts_buf] advances by to the [__gmatmul_single] step of the k-loop
   accumulator invariant. ---- *)

(* generic tile index bound: obr*(oR/iR)+ibr < R/iR *)
let lemma_tile_index_bound (bigR oR iR : pos) (obr ibr : nat)
  : Lemma (requires iR /?+ oR /\ oR /?+ bigR /\ obr < bigR/oR /\ ibr < oR/iR)
          (ensures obr*(oR/iR)+ibr < bigR/iR)
= let q1 = bigR / oR in
  let q2 = oR / iR in
  D.lemma_nat_divides_pos_divides oR bigR;
  D.lemma_nat_divides_pos_divides iR oR;
  let g1 = D.get_factor oR bigR in
  let g2 = D.get_factor iR oR in
  FStar.Math.Lemmas.multiple_division_lemma q1 oR;
  FStar.Math.Lemmas.multiple_division_lemma q2 iR;
  FStar.Math.Lemmas.paren_mul_right q1 q2 iR;
  FStar.Math.Lemmas.multiple_division_lemma (q1 * q2) iR;
  ()

(* nested subtile composition (general): the inner tile of an outer tile is one
   subtile at the fine granularity with a composed index. *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 30 --split_queries always"
let ematrix_subtile_compose
  (#et : Type) (#bigR #bigC : nat)
  (m : chest2 et bigR bigC)
  (oR oC iR iC : pos)
  (obr : natlt (bigR/oR)) (obc : natlt (bigC/oC))
  (ibr : natlt (oR/iR)) (ibc : natlt (oC/iC))
  (tr : natlt (bigR/iR)) (tc : natlt (bigC/iC))
  (sq : squash (oR /? bigR /\ oC /? bigC /\ iR /? oR /\ iC /? oC /\
                tr == obr*(oR/iR)+ibr /\ tc == obc*(oC/iC)+ibc))
  : Lemma (ematrix_subtile (ematrix_subtile m oR oC obr obc) iR iC ibr ibc
           == ematrix_subtile m iR iC tr tc)
= D.lemma_nat_divides_pos_divides iR oR;
  D.lemma_nat_divides_pos_divides oR bigR;
  let gr = D.get_factor iR oR in
  let gc = D.get_factor iC oC in
  FStar.Math.Lemmas.multiple_division_lemma gr iR;
  FStar.Math.Lemmas.multiple_division_lemma gc iC;
  let lhs = ematrix_subtile (ematrix_subtile m oR oC obr obc) iR iC ibr ibc in
  let rhs = ematrix_subtile m iR iC tr tc in
  introduce forall (i:natlt iR) (j:natlt iC). acc2 lhs i j == acc2 rhs i j
  with (
    FStar.Math.Lemmas.distributivity_add_left obr (oR/iR) iR;
    FStar.Math.Lemmas.distributivity_add_left obc (oC/iC) iC;
    FStar.Math.Lemmas.paren_mul_right obr (oR/iR) iR;
    FStar.Math.Lemmas.paren_mul_right obc (oC/iC) iC;
    ()
  );
  assert (equal lhs rhs)
#pop-options

(* a subtile of a constant chest is the constant chest at the tile shape; the
   source dimensions are irrelevant.  Used to bridge the zeroed accumulator
   across the [(wm/frag)*frag] vs [wm] shapes of [fragarrayAcc_approximates]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 30"
let ematrix_subtile_const
  (#rows #cols : nat) (v : real)
  (tr : pos { tr /? rows }) (tc : pos { tc /? cols })
  (i : natlt (rows / tr)) (j : natlt (cols / tc))
  : Lemma
      (ensures
        ematrix_subtile #real #rows #cols (const _ v) tr tc i j
        `equal` (const _ v <: chest2 real tr tc))
      [SMTPat (ematrix_subtile #real #rows #cols (const _ v) tr tc i j)]
= assert (equal (ematrix_subtile #real #rows #cols (const _ v) tr tc i j)
                (const _ v <: chest2 real tr tc))
#pop-options

(* subtile of an approximation is an approximation *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let lemma_subtile_approx
  (#et:Type0) {| scalar et, real_like et |} (#rows #cols:nat)
  (m1 : chest2 et rows cols) (m2 : chest2 real rows cols)
  (trows : pos {trows /? rows}) (tcols : pos {tcols /? cols})
  (tr : natlt (rows/trows)) (tc : natlt (cols/tcols))
  (_ : squash (m1 %~ m2))
  : Lemma (ematrix_subtile m1 trows tcols tr tc %~ ematrix_subtile m2 trows tcols tr tc)
= D.lemma_nat_divides_pos_divides trows rows;
  D.lemma_nat_divides_pos_divides tcols cols;
  let gr = D.get_factor trows rows in
  let gc = D.get_factor tcols cols in
  FStar.Math.Lemmas.multiple_division_lemma gr trows;
  FStar.Math.Lemmas.multiple_division_lemma gc tcols;
  let lhs = ematrix_subtile m1 trows tcols tr tc in
  let rhs = ematrix_subtile m2 trows tcols tr tc in
  introduce forall (i:natlt trows) (j:natlt tcols). acc2 lhs i j %~ acc2 rhs i j
  with (
    FStar.Math.Lemmas.lemma_mult_le_right trows (tr+1) (rows/trows);
    FStar.Math.Lemmas.lemma_mult_le_right tcols (tc+1) (cols/tcols);
    assert (tr*trows+i < rows);
    assert (tc*tcols+j < cols);
    elim_approx m1 m2 (tr*trows+i) (tc*tcols+j)
  );
  lemma_approximates_intro lhs rhs
#pop-options

(* ctranspose of an approximation approximates the mtranspose *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 20"
let lemma_ctranspose_approx
  (#et:Type0) {| scalar et, real_like et |} (#rows #cols:nat)
  (m1 : chest2 et rows cols) (m2 : chest2 real rows cols)
  (_ : squash (m1 %~ m2))
  : Lemma (TR.ctranspose m1 %~ mtranspose m2)
= let lhs = TR.ctranspose m1 in
  let rhs = mtranspose m2 in
  introduce forall (i:natlt cols) (j:natlt rows). acc2 lhs i j %~ acc2 rhs i j
  with (
    TR.lemma_ctranspose_acc m1 i j;
    assert (acc2 rhs i j == acc2 m2 j i);
    elim_approx m1 m2 j i
  );
  lemma_approximates_intro lhs rhs
#pop-options

(* A-fragment approximation (FragLRM): [arr] holds [wm] row-fragments of the
   per-warp A tile [rm : chest2 real (wm*frag) frag]. *)
let sg_fragA_approx
  (#et:Type0) {| scalar et, real_like et |}
  (wm : nat)
  (arr : array (fragment et FragA frag frag frag FragLRM) { length arr == wm })
  (rm : chest2 real (wm*frag) frag)
  : slprop
= exists* (eAs : seq (chest2 et frag frag)).
    arr |-> eAs **
    pure ((Seq.length eAs == wm) /\
      (forall (i : natlt wm). (Seq.index eAs i) %~ (ematrix_subtile rm frag frag i 0)))

(* B-fragment approximation (FragLCM): [arr] holds [wn] col-fragments of the
   per-warp B tile [rm : chest2 real frag (wn*frag)], (k,n)-oriented. *)
let sg_fragB_approx
  (#et:Type0) {| scalar et, real_like et |}
  (wn : nat)
  (arr : array (fragment et FragB frag frag frag FragLCM) { length arr == wn })
  (rm : chest2 real frag (wn*frag))
  : slprop
= exists* (eBs : seq (chest2 et frag frag)).
    arr |-> eBs **
    pure ((Seq.length eBs == wn) /\
      (forall (i : natlt wn). (Seq.index eBs i) %~ (ematrix_subtile rm frag frag 0 i)))

(* Grid invariant for the [sp_mma] double loop (mirror of [arrayfragments_fade]):
   accumulator tile (i,j) has one extra matmul added once the sweep has passed
   it, otherwise it is unchanged. *)
let arrayfragments_fade
  (wm_e wn_e : nat)
  (_ : squash (frag /?+ wm_e /\ frag /?+ wn_e))
  (i : natlt (wm_e / frag))
  (j : natlt (wn_e / frag))
  (resIdxM : natle (wm_e / frag))
  (resIdxN : natle (wn_e / frag))
  (rA : chest2 real wm_e frag)
  (rB : chest2 real frag wn_e)
  (rAcc : chest2 real wm_e wn_e)
: chest2 real frag frag
=
  D.lemma_nat_divides_pos_divides frag wm_e;
  D.lemma_nat_divides_pos_divides frag wn_e;
  if i < resIdxM || (i = resIdxM && j < resIdxN)
  then ematrix_subtile rAcc frag frag i j `matplus`
    (matmul (ematrix_subtile rA frag frag i 0) (ematrix_subtile rB frag frag 0 j))
  else ematrix_subtile rAcc frag frag i j

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn sp_load_a
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  (bm bk wm : szp)
  (#_ : squash (frag /?+ SZ.v wm /\ frag /?+ SZ.v bk /\ SZ.v wm /?+ SZ.v bm))
  (fmap : et_ab -> et_ab)
  (aFrags : array (fragment et_ab FragA frag frag frag FragLRM))
  (#lA : layout2 (SZ.v bm) (SZ.v bk)) {| T.ctlayout lA |}
       {| strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab (SZ.v bm) (SZ.v bk))
  (rA : chest2 real (SZ.v bm) (SZ.v bk) { eA %~ rA })
  (#fA : perm)
  (warp_m : szlt (SZ.v bm / SZ.v wm))
  (ks : szlt (SZ.v bk / frag))
  (#_ : squash (length aFrags == SZ.v wm / frag))
  (#_ : squash (valid_frag_et_dims et_ab FragA frag frag frag))
  (fmap_id : squash (forall (x : et_ab). fmap x == x))
  ()
  preserves gpu ** gA |-> Frac fA eA
  requires live aFrags
  ensures
    sg_fragA_approx (SZ.v wm / frag) aFrags
      (ematrix_subtile rA (SZ.v wm) frag (SZ.v warp_m) (SZ.v ks))
{
  array_fragment_pts_to_ref aFrags;
  let mfrag = wm /^ frag_sz;
  assert (pure (SZ.v mfrag == SZ.v wm / frag));
  let mut i = 0sz;
  while (!i <^ mfrag)
    invariant live i
    invariant
      exists* ems.
        array_fragment_pts_to aFrags ems **
        pure (Seq.length ems == SZ.v wm / frag /\ SZ.v !i <= SZ.v wm / frag /\
          (forall (t : natlt (SZ.v wm / frag)). SZ.v !i > t ==>
            (Seq.index ems t) %~
              (ematrix_subtile
                 (ematrix_subtile rA (SZ.v wm) frag (SZ.v warp_m) (SZ.v ks))
                 frag frag t 0)))
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
    Pulse.Lib.Forall.elim_forall
      (chest_map fmap (ematrix_subtile eA frag frag (SZ.v arow) (SZ.v ks)));
    Kuiper.TradeHelpers.ambig_trade_elim ();
    Kuiper.TradeHelpers.ambig_trade_elim ();
    chest2_map_id fmap (ematrix_subtile eA frag frag (SZ.v arow) (SZ.v ks));
    sp_a_tile_approx (SZ.v bm) (SZ.v bk) (SZ.v wm) eA rA
      (SZ.v warp_m) (SZ.v ks) (SZ.v !i) ();
    assert (pure (SZ.v arow == SZ.v warp_m * (SZ.v wm / frag) + SZ.v !i));
    i := !i +^ 1sz;
  };
  frag_div_mul (SZ.v wm);
  assert (pure (SZ.v !i == SZ.v wm / frag));
  fold sg_fragA_approx (SZ.v wm / frag) aFrags
         (ematrix_subtile rA (SZ.v wm) frag (SZ.v warp_m) (SZ.v ks));
}
#pop-options

#push-options "--z3rlimit 30"
inline_for_extraction noextract
fn sp_load_b
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  (bn bk wn : szp)
  (#_ : squash (frag /?+ SZ.v wn /\ frag /?+ SZ.v bk /\ SZ.v wn /?+ SZ.v bn))
  (fmap : et_ab -> et_ab)
  (bFrags : array (fragment et_ab FragB frag frag frag FragLCM))
  (#lB : layout2 (SZ.v bk) (SZ.v bn)) {| T.ctlayout lB |}
       {| strided_col_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab (SZ.v bk) (SZ.v bn))
  (rB : chest2 real (SZ.v bk) (SZ.v bn) { eB %~ rB })
  (#fB : perm)
  (warp_n : szlt (SZ.v bn / SZ.v wn))
  (ks : szlt (SZ.v bk / frag))
  (#_ : squash (length bFrags == SZ.v wn / frag))
  (#_ : squash (valid_frag_et_dims et_ab FragB frag frag frag))
  (fmap_id : squash (forall (x : et_ab). fmap x == x))
  ()
  preserves gpu ** gB |-> Frac fB eB
  requires live bFrags
  ensures
    sg_fragB_approx (SZ.v wn / frag) bFrags
      (ematrix_subtile rB frag (SZ.v wn) (SZ.v ks) (SZ.v warp_n))
{
  array_fragment_pts_to_ref bFrags;
  let nfrag = wn /^ frag_sz;
  assert (pure (SZ.v nfrag == SZ.v wn / frag));
  let mut j = 0sz;
  while (!j <^ nfrag)
    invariant live j
    invariant
      exists* ems.
        array_fragment_pts_to bFrags ems **
        pure (Seq.length ems == SZ.v wn / frag /\ SZ.v !j <= SZ.v wn / frag /\
          (forall (t : natlt (SZ.v wn / frag)). SZ.v !j > t ==>
            (Seq.index ems t) %~
              (ematrix_subtile
                 (ematrix_subtile rB frag (SZ.v wn) (SZ.v ks) (SZ.v warp_n))
                 frag frag 0 t)))
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
    Pulse.Lib.Forall.elim_forall
      (chest_map fmap (ematrix_subtile eB frag frag (SZ.v ks) (SZ.v bcol)));
    Kuiper.TradeHelpers.ambig_trade_elim ();
    Kuiper.TradeHelpers.ambig_trade_elim ();
    chest2_map_id fmap (ematrix_subtile eB frag frag (SZ.v ks) (SZ.v bcol));
    sp_b_tile_approx (SZ.v bn) (SZ.v bk) (SZ.v wn) eB rB
      (SZ.v warp_n) (SZ.v ks) (SZ.v !j) ();
    assert (pure (SZ.v bcol == SZ.v warp_n * (SZ.v wn / frag) + SZ.v !j));
    j := !j +^ 1sz;
  };
  frag_div_mul (SZ.v wn);
  assert (pure (SZ.v !j == SZ.v wn / frag));
  fold sg_fragB_approx (SZ.v wn / frag) bFrags
         (ematrix_subtile rB frag (SZ.v wn) (SZ.v ks) (SZ.v warp_n));
}
#pop-options

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn sp_mma
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     real_like et_ab, real_like et_acc |}
  (wm wn : szp)
  (#_ : squash (frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (aFrags  : array (fragment et_ab FragA   frag frag frag FragLRM))
  (bFrags  : array (fragment et_ab FragB   frag frag frag FragLCM))
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (rA : chest2 real (SZ.v wm) frag)
  (rB : chest2 real frag (SZ.v wn))
  (rAcc : chest2 real (SZ.v wm) (SZ.v wn))
  (#_ : squash (length aFrags == SZ.v wm / frag))
  (#_ : squash (length bFrags == SZ.v wn / frag))
  (#_ : squash (length accFrags == SZ.v wm / frag * (SZ.v wn / frag)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  ()
  requires
    sg_fragA_approx (SZ.v wm / frag) aFrags rA **
    sg_fragB_approx (SZ.v wn / frag) bFrags rB **
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags rAcc
  ensures
    live aFrags ** live bFrags **
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
      (rAcc `matplus` (matmul rA rB))
{
  let mfrag = wm /^ frag_sz;
  let nfrag = wn /^ frag_sz;
  assert (pure (SZ.v mfrag == SZ.v wm / frag /\ SZ.v nfrag == SZ.v wn / frag));

  unfold sg_fragA_approx (SZ.v wm / frag) aFrags rA;
  unfold sg_fragB_approx (SZ.v wn / frag) bFrags rB;
  unfold fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags rAcc;

  with eAs. assert aFrags |-> eAs;
  with eBs. assert bFrags |-> eBs;

  let mut resIdxM = 0sz;
  while (!resIdxM <^ mfrag)
    invariant live resIdxM
    invariant
      exists* (eAcc : seq (chest2 et_acc frag frag)).
        accFrags |-> eAcc **
        pure (
          SZ.v !resIdxM <= SZ.v wm / frag /\
          (Seq.length eAcc == SZ.v wm / frag * (SZ.v wn / frag)) /\
          (forall (i : natlt (SZ.v wm / frag)) (j : natlt (SZ.v wn / frag)).
            (Seq.index eAcc (i * (SZ.v wn / frag) + j)) %~
              (arrayfragments_fade (SZ.v wm) (SZ.v wn) () i j (SZ.v !resIdxM) 0 rA rB rAcc)))
    decreases (SZ.v wm / frag - SZ.v !resIdxM)
  {
    let mut resIdxN = 0sz;
    while (!resIdxN <^ nfrag)
      invariant live resIdxN
      invariant
        exists* (eAcc : seq (chest2 et_acc frag frag)).
          accFrags |-> eAcc **
          pure (
            SZ.v !resIdxN <= SZ.v wn / frag /\
            SZ.v !resIdxM < SZ.v wm / frag /\
            (Seq.length eAcc == SZ.v wm / frag * (SZ.v wn / frag)) /\
            (forall (i : natlt (SZ.v wm / frag)) (j : natlt (SZ.v wn / frag)).
              (Seq.index eAcc (i * (SZ.v wn / frag) + j)) %~
                (arrayfragments_fade (SZ.v wm) (SZ.v wn) () i j (SZ.v !resIdxM) (SZ.v !resIdxN) rA rB rAcc)))
      decreases (SZ.v wn / frag - SZ.v !resIdxN)
    {
      with eAccs. assert accFrags |-> eAccs;

      array_fragment_pts_to_ref aFrags;
      array_fragment_pts_to_ref bFrags;
      array_fragment_pts_to_ref accFrags;

      array_fragment_extract_ro aFrags !resIdxM;
      array_fragment_extract_ro bFrags !resIdxN;
      array_fragment_extract accFrags (!resIdxM * nfrag + !resIdxN);

      let a_frag = aFrags.(!resIdxM);
      let b_frag = bFrags.(!resIdxN);
      let acc_frag = accFrags.(!resIdxM *^ nfrag +^ !resIdxN);

      with eAt. assert a_frag |-> eAt;
      with eBt. assert b_frag |-> eBt;
      with eAcct. assert acc_frag |-> eAcct;
      assert pure (eAt %~ (ematrix_subtile rA frag frag (SZ.v !resIdxM) 0));
      assert pure (eBt %~ (ematrix_subtile rB frag frag 0 (SZ.v !resIdxN)));
      assert pure (eAcct %~ (ematrix_subtile rAcc frag frag (SZ.v !resIdxM) (SZ.v !resIdxN)));

      mma_sync' a_frag b_frag acc_frag;

      Kuiper.TensorCore.Base.emma_approx_lemma eAcct eAt eBt
        (ematrix_subtile rAcc frag frag (SZ.v !resIdxM) (SZ.v !resIdxN))
        (ematrix_subtile rA frag frag (SZ.v !resIdxM) 0)
        (ematrix_subtile rB frag frag 0 (SZ.v !resIdxN));

      Kuiper.TradeHelpers.ambig_trade_elim ();
      Kuiper.TradeHelpers.ambig_trade_elim ();

      with v. assert acc_frag `fragment_pts_to` v;
      Pulse.Lib.Forall.elim_forall v;

      Kuiper.TradeHelpers.ambig_trade_elim ();

      assert array_fragment_pts_to accFrags (Seq.Base.upd eAccs
            (SZ.v !resIdxM * (SZ.v wn / frag) + SZ.v !resIdxN)
            (emma (Seq.index eAccs (SZ.v !resIdxM * (SZ.v wn / frag) + SZ.v !resIdxN))
                (Seq.index eAs (SZ.v !resIdxM))
                (Seq.index eBs (SZ.v !resIdxN))));

      resIdxN := !resIdxN +^ 1sz;
    };

    resIdxM := !resIdxM +^ 1sz;
  };

  with eAcc. assert accFrags |-> eAcc;
  assert pure (
    forall (i : natlt (SZ.v wm / frag)) (j : natlt (SZ.v wn / frag)).
      (Seq.index eAcc (i * (SZ.v wn / frag) + j)) %~
        (arrayfragments_fade (SZ.v wm) (SZ.v wn) () i j (SZ.v mfrag) 0 rA rB rAcc));
  assert pure (
    forall (i : natlt (SZ.v wm / frag)) (j : natlt (SZ.v wn / frag)).
      (Seq.index eAcc (i * (SZ.v wn / frag) + j)) %~
        (ematrix_subtile rAcc frag frag i j `matplus`
          (matmul (ematrix_subtile rA frag frag i 0) (ematrix_subtile rB frag frag 0 j))));
  assert pure (
    forall (i : natlt (SZ.v wm / frag)) (j : natlt (SZ.v wn / frag)).
      (Seq.index eAcc (i * (SZ.v wn / frag) + j)) %~
        (ematrix_subtile (rAcc `matplus` (matmul rA rB)) frag frag i j));
  assert pure (Seq.length eAcc == SZ.v wm / frag * (SZ.v wn / frag));

  fold fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
    (rAcc `matplus` (matmul rA rB));
}
#pop-options

(* ---- accumulator bookkeeping (pure/ghost; mirror of To.Fragments) ---- *)

#push-options "--z3rlimit 60"
let subproducts_step_eq
  (#bm #bk #bn : nat)
  (wm_tm tk_dim wn_tn : pos)
  (_ : squash (wm_tm /? bm /\ tk_dim /? bk /\ wn_tn /? bn))
  (rAcc : chest2 real wm_tm wn_tn)
  (rA : chest2 real bm bk) (rB : chest2 real bk bn)
  (row : natlt (bm/wm_tm)) (col : natlt (bn/wn_tn))
  (k : natlt (bk/tk_dim))
  : Lemma (
      matplus (__gmatmul_single rAcc matmul matplus
              (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
              row col k)
          (matmul (ematrix_subtile rA wm_tm tk_dim row k)
                  (ematrix_subtile rB tk_dim wn_tn k col))
      ==
      __gmatmul_single rAcc matmul matplus
          (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
          row col (k + 1))
  = __gmatmul_single_lemma rAcc matmul matplus
      (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
      row col (k + 1);
    macc_ematrix_tiled rA wm_tm tk_dim row k;
    macc_ematrix_tiled rB tk_dim wn_tn k col;
    assert (equal
      (matplus (__gmatmul_single rAcc matmul matplus
               (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
               row col k)
          (matmul (ematrix_subtile rA wm_tm tk_dim row k)
                  (ematrix_subtile rB tk_dim wn_tn k col)))
      (__gmatmul_single rAcc matmul matplus
          (ematrix_tiled rA wm_tm tk_dim) (ematrix_tiled rB tk_dim wn_tn)
          row col (k + 1)))
#pop-options

#push-options "--z3rlimit 60"
noextract
ghost fn rewrite_fragarrayAcc_step
  (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (#bm #bk #bn : pos)
  (wm wn : pos)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc)
                { length accumFrags == wm*wn })
  (rAcc : chest2 real (wm*tm) (wn*tn))
  (rA : chest2 real bm bk) (rB : chest2 real bk bn)
  (arow : natlt (bm/(wm*tm))) (bcol : natlt (bn/(wn*tn)))
  (k : natlt (bk/tk))
  (#_ : squash ((wm*tm) /? bm /\ tk /? bk /\ (wn*tn) /? bn))
  requires fragarrayAcc_approximates wm wn accumFrags
    (matplus (__gmatmul_single rAcc matmul matplus
              (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
              arow bcol k)
            (matmul (ematrix_subtile rA (wm*tm) tk arow k)
                    (ematrix_subtile rB tk (wn*tn) k bcol)))
  ensures fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
      arow bcol (k + 1))
{
  subproducts_step_eq (wm*tm) tk (wn*tn) () rAcc rA rB arow bcol k;
  unfold fragarrayAcc_approximates wm wn accumFrags;
  fold fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
      arow bcol (k + 1));
}
#pop-options

noextract
ghost fn rewrite_fragarrayAcc_tiles
  (#et:Type0) {| scalar et, real_like et |}
  (#tm #tn #tk : pos)
  (#bm #bk #bn : pos)
  (wm wn : pos)
  (accumFrags : array (fragment et FragAcc tm tn tk FragLAcc)
                { length accumFrags == wm*wn })
  (rAcc : chest2 real (wm*tm) (wn*tn))
  (rA : chest2 real bm bk) (rB : chest2 real bk bn)
  (arow : natlt (bm/(wm*tm))) (bcol : natlt (bn/(wn*tn)))
  (#_ : squash ((wm*tm) /? bm /\ tk /? bk /\ (wn*tn) /? bn))
  requires fragarrayAcc_approximates wm wn accumFrags
    (__gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (wm*tm) tk) (ematrix_tiled rB tk (wn*tn))
      arow bcol (bk/tk))
  ensures fragarrayAcc_approximates wm wn accumFrags
    (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                           (ematrix_subtile rB bk (wn*tn) 0 bcol))
{
  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
    (wm*tm) (wn*tn) tk rAcc rA rB arow bcol;
  unfold fragarrayAcc_approximates wm wn accumFrags;
  fold fragarrayAcc_approximates wm wn accumFrags
    (rAcc `matplus` matmul (ematrix_subtile rA (wm*tm) bk arow 0)
                           (ematrix_subtile rB bk (wn*tn) 0 bcol));
}

inline_for_extraction noextract
fn subproducts
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc,
     real_like et_ab, real_like et_acc |}
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
  (rA : chest2 real (SZ.v bm) (SZ.v bk) { eA %~ rA })
  (rB : chest2 real (SZ.v bk) (SZ.v bn) { eB %~ rB })
  (rAcc : chest2 real (SZ.v wm) (SZ.v wn))
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
  (fmap_id : squash (forall (x : et_ab). fmap x == x))
  ()
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  preserves
    live aFrags ** live bFrags
  requires
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags rAcc
  ensures
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
      (rAcc `matplus`
        (matmul (ematrix_subtile rA (SZ.v wm) (SZ.v bk) (SZ.v warp_m) 0)
                (ematrix_subtile rB (SZ.v bk) (SZ.v wn) 0 (SZ.v warp_n))))
{
  acc_len_reveal wm wn;
  frag_div_mul (SZ.v wm);
  frag_div_mul (SZ.v wn);

  rewrite each rAcc
  as __gmatmul_single rAcc matmul matplus
      (ematrix_tiled rA (SZ.v wm) frag) (ematrix_tiled rB frag (SZ.v wn))
      (SZ.v warp_m) (SZ.v warp_n) 0;

  let kstep = bk /^ frag_sz;
  let mut ks = 0sz;
  while (!ks <^ kstep)
    invariant live aFrags ** live bFrags
    invariant
      gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB
    invariant
      exists* (vki : sz { SZ.v vki <= SZ.v bk / frag }).
        ks |-> vki **
        fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
          (__gmatmul_single rAcc matmul matplus
            (ematrix_tiled rA (SZ.v wm) frag) (ematrix_tiled rB frag (SZ.v wn))
            (SZ.v warp_m) (SZ.v warp_n) (SZ.v !ks))
    decreases (SZ.v bk / frag - SZ.v !ks)
  {
    sp_load_a bm bk wm fmap aFrags gA rA warp_m !ks fmap_id ();
    sp_load_b bn bk wn fmap bFrags gB rB warp_n !ks fmap_id ();
    sp_mma wm wn aFrags bFrags accFrags
      (ematrix_subtile rA (SZ.v wm) frag (SZ.v warp_m) (SZ.v !ks))
      (ematrix_subtile rB frag (SZ.v wn) (SZ.v !ks) (SZ.v warp_n))
      (__gmatmul_single rAcc matmul matplus
        (ematrix_tiled rA (SZ.v wm) frag) (ematrix_tiled rB frag (SZ.v wn))
        (SZ.v warp_m) (SZ.v warp_n) (SZ.v !ks))
      ();

    rewrite_fragarrayAcc_step (SZ.v wm / frag) (SZ.v wn / frag) accFrags
      rAcc rA rB (SZ.v warp_m) (SZ.v warp_n) !ks;

    ks := !ks +^ 1sz;
  };

  with vki. assert (ks |-> vki);
  assert pure (SZ.v vki == SZ.v bk / frag);
  rewrite each (SZ.v vki) as (SZ.v bk / frag);

  rewrite_fragarrayAcc_tiles (SZ.v wm / frag) (SZ.v wn / frag) accFrags
    rAcc rA rB (SZ.v warp_m) (SZ.v warp_n);
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    (FB.own_strided_chunks
       (skewed_view bm bk skew (if vkt % 2 = 0 then sarA0 else sarA1))
       (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt) nthr tid **
     FB.own_strided_chunks
       (skewed_view bn bk skew (if vkt % 2 = 0 then sarB0 else sarB1))
       (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt) nthr tid **
     (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |->
        Frac fA (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt)) **
     (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |->
        Frac fB (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt))) **
  (if vkt = 0 then
     pipe_live (skewed_view bm bk skew (if vkt % 2 = 0 then sarA1 else sarA0)) nthr tid **
     pipe_live (skewed_view bn bk skew (if vkt % 2 = 0 then sarB1 else sarB0)) nthr tid
   else
     pipe_sharing (skewed_view bm bk skew (if vkt % 2 = 0 then sarA1 else sarA0)) nthr **
     pipe_sharing (skewed_view bn bk skew (if vkt % 2 = 0 then sarB1 else sarB0)) nthr) **
  (forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
     (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
       @==> gA |-> Frac fA (update_tile eA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
  (forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
     (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
       @==> gB |-> Frac fB (update_tile eB (SZ.v bn) (SZ.v bk) block_col vkt tm'))

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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    (FB.own_strided_chunks (skewed_view bm bk skew curA)
       (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt) nthr tid **
     FB.own_strided_chunks (skewed_view bn bk skew curB)
       (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt) nthr tid **
     (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |->
        Frac fA (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt)) **
     (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |->
        Frac fB (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt))) **
  (if othlive then
     pipe_live (skewed_view bm bk skew othA) nthr tid **
     pipe_live (skewed_view bn bk skew othB) nthr tid
   else
     pipe_sharing (skewed_view bm bk skew othA) nthr **
     pipe_sharing (skewed_view bn bk skew othB) nthr) **
  (forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
     (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
       @==> gA |-> Frac fA (update_tile eA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
  (forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
     (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
       @==> gB |-> Frac fB (update_tile eB (SZ.v bn) (SZ.v bk) block_col vkt tm'))
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 eA eB fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt
  ensures
    pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
{
  rewrite
    (pending_body bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 eA eB fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt)
  as
    (pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt);
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    (FB.own_strided_chunks (skewed_view bm bk skew curA)
       (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt) nthr tid **
     FB.own_strided_chunks (skewed_view bn bk skew curB)
       (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt) nthr tid **
     (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |->
        Frac fA (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt)) **
     (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |->
        Frac fB (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt))) **
  (forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
     (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
       @==> gA |-> Frac fA (update_tile eA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
  (forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
     (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
       @==> gB |-> Frac fB (update_tile eB (SZ.v bn) (SZ.v bk) block_col vkt tm'))
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
  (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
    pipe_live (skewed_view bm bk skew dstA) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bn bk skew dstB) (SZ.v nthr) (SZ.v tid) **
    PC.batch_live b
  returns b' : PC.pipeline_batch_t
  ensures
    staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB (SZ.v nthr) (SZ.v tid)
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

  fold (staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB (SZ.v nthr) (SZ.v tid)
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
  {| _rab : real_like et_ab |} {| _rac : real_like et_acc |}
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
  (#eA_tile : chest2 et_ab (SZ.v bm) (SZ.v bk))
  (#eB_tile : chest2 et_ab (SZ.v bn) (SZ.v bk))
  (rA : chest2 real (SZ.v bm) (SZ.v bk) { eA_tile %~ rA })
  (rB_phys : chest2 real (SZ.v bn) (SZ.v bk) { eB_tile %~ rB_phys })
  (rAcc : chest2 real (SZ.v wm) (SZ.v wn))
  (sq_c : squash (P.constraints et_ab et_acc bm bn bk wm wn skew))
  (sq_v : squash (
     valid_frag_et_dims et_ab FragA frag frag frag /\
     valid_frag_et_dims et_ab FragB frag frag frag /\
     valid_frag_et_dims et_acc FragAcc frag frag frag /\
     valid_frag_et_comb et_ab et_acc /\
     length aFrags == SZ.v wm / frag /\ length bFrags == SZ.v wn / frag /\
     length accFrags == acc_len wm wn))
  (fmap_id : squash (forall (x : et_ab). fmap x == x))
  ()
  requires
    gpu **
    pipe_sharing_c (skewed_view bm bk skew curbufA) eA_tile (SZ.v nthr) **
    pipe_sharing_c (skewed_view bn bk skew curbufB) eB_tile (SZ.v nthr) **
    live aFrags ** live bFrags **
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags rAcc
  ensures
    gpu **
    pipe_sharing_c (skewed_view bm bk skew curbufA) eA_tile (SZ.v nthr) **
    pipe_sharing_c (skewed_view bn bk skew curbufB) eB_tile (SZ.v nthr) **
    live aFrags ** live bFrags **
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
      (rAcc `matplus`
        (matmul (ematrix_subtile rA (SZ.v wm) (SZ.v bk) (SZ.v warp_m) 0)
                (ematrix_subtile (mtranspose rB_phys) (SZ.v bk) (SZ.v wn) 0 (SZ.v warp_n))))
{
  unfold (pipe_sharing_c (skewed_view bm bk skew curbufA) eA_tile (SZ.v nthr));
  unfold (FB.bp_sharing (skewed_view bm bk skew curbufA) eA_tile (SZ.v nthr));
  unfold (pipe_sharing_c (skewed_view bn bk skew curbufB) eB_tile (SZ.v nthr));
  unfold (FB.bp_sharing (skewed_view bn bk skew curbufB) eB_tile (SZ.v nthr));

  TR.atranspose_fwd (skewed_view bn bk skew curbufB);

  lemma_ctranspose_approx eB_tile rB_phys ();

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
    #eA_tile
    #(TR.ctranspose eB_tile)
    rA
    (mtranspose rB_phys)
    rAcc
    #(1.0R /. SZ.v nthr) #(1.0R /. SZ.v nthr)
    warp_m warp_n
    () () ;

  TR.atranspose_back (skewed_view bn bk skew curbufB);
  TR.lemma_ctranspose_involutive eB_tile;

  fold (FB.bp_sharing (skewed_view bm bk skew curbufA) eA_tile (SZ.v nthr));
  fold (pipe_sharing_c (skewed_view bm bk skew curbufA) eA_tile (SZ.v nthr));
  fold (FB.bp_sharing (skewed_view bn bk skew curbufB) eB_tile (SZ.v nthr));
  fold (pipe_sharing_c (skewed_view bn bk skew curbufB) eB_tile (SZ.v nthr));
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (fA fB : perm)
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (vkt : nat { vkt < SZ.v k / SZ.v bk })
  requires
    (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |->
       Frac fA (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt)) **
    (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |->
       Frac fB (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt)) **
    (forall* (tm' : chest2 et_ab (SZ.v bm) (SZ.v bk)).
       (array2_subtile gA (SZ.v bm) (SZ.v bk) block_row vkt |-> Frac fA tm')
         @==> gA |-> Frac fA (update_tile eA (SZ.v bm) (SZ.v bk) block_row vkt tm')) **
    (forall* (tm' : chest2 et_ab (SZ.v bn) (SZ.v bk)).
       (array2_subtile gB (SZ.v bn) (SZ.v bk) block_col vkt |-> Frac fB tm')
         @==> gB |-> Frac fB (update_tile eB (SZ.v bn) (SZ.v bk) block_col vkt tm'))
  ensures
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB)
{
  Pulse.Lib.Forall.elim_forall #_ (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt);
  Pulse.Lib.Trade.elim_trade _ _;
  update_tile_self eA (SZ.v bm) (SZ.v bk) block_row vkt;
  rewrite (gA |-> Frac fA (update_tile eA (SZ.v bm) (SZ.v bk) block_row vkt
             (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt)))
    as (gA |-> Frac fA eA);
  Pulse.Lib.Forall.elim_forall #_ (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt);
  Pulse.Lib.Trade.elim_trade _ _;
  update_tile_self eB (SZ.v bn) (SZ.v bk) block_col vkt;
  rewrite (gB |-> Frac fB (update_tile eB (SZ.v bn) (SZ.v bk) block_col vkt
             (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt)))
    as (gB |-> Frac fB eB);
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
  ensures
    pending_body bm bn bk skew gA gB srcA dstA srcB dstB eA eB fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt
{
  rewrite
    (pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt)
  as
    (pending_body bm bn bk skew gA gB srcA dstA srcB dstB eA eB fA fB nthr tid block_row block_col
      b_c sq (vkt = 0) vkt);
}

(* [live_c src ** (ite oth)] over [src*]/[dst*] -> [pipe_p_c vkt] (content). *)
ghost
fn fold_pipe_p_g
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (srcA dstA : larray et (SZ.v bm * ldt bk skew))
  (srcB dstB : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (vkt : nat) (tid : natlt nthr)
  (sq_bc : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                   SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (sq_sel : squash (
     vkt < SZ.v k / SZ.v bk /\
     srcA == (if vkt % 2 = 0 then sarA0 else sarA1) /\
     dstA == (if vkt % 2 = 0 then sarA1 else sarA0) /\
     srcB == (if vkt % 2 = 0 then sarB0 else sarB1) /\
     dstB == (if vkt % 2 = 0 then sarB1 else sarB0)))
  requires
    pipe_live_c (skewed_view bm bk skew srcA)
       (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt) nthr tid **
    pipe_live_c (skewed_view bn bk skew srcB)
       (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt) nthr tid **
    (if vkt = 0 then
       pipe_live (skewed_view bm bk skew dstA) nthr tid **
       pipe_live (skewed_view bn bk skew dstB) nthr tid
     else
       pipe_sharing (skewed_view bm bk skew dstA) nthr **
       pipe_sharing (skewed_view bn bk skew dstB) nthr)
  ensures
    pipe_p_c m n k bm bn bk skew eA eB block_row block_col
             sarA0 sarA1 sarB0 sarB1 nthr sq_bc vkt tid
{
  rewrite
    (pipe_live_c (skewed_view bm bk skew srcA)
       (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt) nthr tid **
     pipe_live_c (skewed_view bn bk skew srcB)
       (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt) nthr tid **
     (if vkt = 0 then
        pipe_live (skewed_view bm bk skew dstA) nthr tid **
        pipe_live (skewed_view bn bk skew dstB) nthr tid
      else
        pipe_sharing (skewed_view bm bk skew dstA) nthr **
        pipe_sharing (skewed_view bn bk skew dstB) nthr))
  as
    (pipe_p_c m n k bm bn bk skew eA eB block_row block_col
              sarA0 sarA1 sarB0 sarB1 nthr sq_bc vkt tid);
}

(* [pipe_q_c vkt] -> [sharing_c src (content) ** live dst] over [src*]/[dst*]. *)
ghost
fn unfold_pipe_q_g
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (srcA dstA : larray et (SZ.v bm * ldt bk skew))
  (srcB dstB : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (vkt : nat) (tid : natlt nthr)
  (sq_bc : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                   SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (sq_sel : squash (
     vkt < SZ.v k / SZ.v bk /\
     srcA == (if vkt % 2 = 0 then sarA0 else sarA1) /\
     dstA == (if vkt % 2 = 0 then sarA1 else sarA0) /\
     srcB == (if vkt % 2 = 0 then sarB0 else sarB1) /\
     dstB == (if vkt % 2 = 0 then sarB1 else sarB0)))
  requires
    pipe_q_c m n k bm bn bk skew eA eB block_row block_col
             sarA0 sarA1 sarB0 sarB1 nthr sq_bc vkt tid
  ensures
    pipe_sharing_c (skewed_view bm bk skew srcA)
       (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt) nthr **
    pipe_sharing_c (skewed_view bn bk skew srcB)
       (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt) nthr **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid
{
  rewrite
    (pipe_q_c m n k bm bn bk skew eA eB block_row block_col
              sarA0 sarA1 sarB0 sarB1 nthr sq_bc vkt tid)
  as
    (pipe_sharing_c (skewed_view bm bk skew srcA)
       (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row vkt) nthr **
     pipe_sharing_c (skewed_view bn bk skew srcB)
       (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col vkt) nthr **
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    pending_body bm bn bk skew gA gB dstA srcA dstB srcB eA eB fA fB nthr tid block_row block_col
      b_c sq false kt1
  ensures
    pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq kt1
{
  rewrite
    (pending_body bm bn bk skew gA gB dstA srcA dstB srcB eA eB fA fB nthr tid block_row block_col
      b_c sq false kt1)
  as
    (pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq kt1);
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
      pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt)
  else
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr (SZ.v k / SZ.v bk)
           (SZ.v k / SZ.v bk - 1) tid

ghost
fn unfold_kcarry_live
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt
  ensures
    exists* (b_c b_l : PC.pipeline_batch_t).
      PC.batch_committed b_c ** PC.batch_live b_l **
      pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
{
  rewrite
    (kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt)
  as
    (exists* (b_c b_l : PC.pipeline_batch_t).
      PC.batch_committed b_c ** PC.batch_live b_l **
      pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt);
}

ghost
fn fold_kcarry_live
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col b_c sq vkt
  ensures
    kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt
{
  with (b_l : PC.pipeline_batch_t). assert (PC.batch_live b_l);
  introduce
    exists* (c l : PC.pipeline_batch_t).
      PC.batch_committed c ** PC.batch_live l **
      pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col c sq vkt
  with b_c b_l;
  rewrite
    (exists* (c l : PC.pipeline_batch_t).
      PC.batch_committed c ** PC.batch_live l **
      pending bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col c sq vkt)
  as
    (kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt);
}

ghost
fn fold_kcarry_done
  (#et_ab : Type0) {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  (#m #n #k : szp)
  (bm bn bk skew : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) (gA : array2 et_ab lA)
  (#lB : layout2 (SZ.v n) (SZ.v k)) (gB : array2 et_ab lB)
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
    pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr (SZ.v k / SZ.v bk)
           (SZ.v k / SZ.v bk - 1) tid
  ensures
    kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt
{
  rewrite
    ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
     pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr (SZ.v k / SZ.v bk)
            (SZ.v k / SZ.v bk - 1) tid)
  as
    (kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB nthr tid block_row block_col sq vkt);
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col b_l sq (vkt + 1)
  else
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col b_l sq (vkt + 1)
  ensures
    cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt
{
  introduce exists* (bl : PC.pipeline_batch_t). PC.batch_live bl with b_l';
  rewrite
    (PC.batch_committed b_l ** (exists* (bl : PC.pipeline_batch_t). PC.batch_live bl) **
     staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col b_l sq (vkt + 1))
  as
    (cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt);
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid **
    PC.batch_live b_l
  ensures
    cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt
{
  rewrite
    ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
     pipe_live (skewed_view bm bk skew dstA) nthr tid **
     pipe_live (skewed_view bn bk skew dstB) nthr tid **
     PC.batch_live b_l)
  as
    (cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt);
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt
  ensures
    PC.batch_committed b_l ** (exists* b_l'. PC.batch_live b_l') **
    staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col b_l sq (vkt + 1)
{
  rewrite
    (cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt)
  as
    (PC.batch_committed b_l ** (exists* b_l'. PC.batch_live b_l') **
     staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col b_l sq (vkt + 1));
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
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
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
    cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt
  ensures
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
    pipe_live (skewed_view bm bk skew dstA) nthr tid **
    pipe_live (skewed_view bn bk skew dstB) nthr tid **
    PC.batch_live b_l
{
  rewrite
    (cstage bm bn bk skew gA gB dstA dstB eA eB fA fB nthr tid block_row block_col sq b_l vkt)
  as
    ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
     pipe_live (skewed_view bm bk skew dstA) nthr tid **
     pipe_live (skewed_view bn bk skew dstB) nthr tid **
     PC.batch_live b_l);
}
#pop-options


(* ---- k-loop accumulator advance/collapse (pure/ghost).  The accumulator
   invariant is keyed on [__gmatmul_single ... vkt]; each iteration advances it
   by the block-warp matmul [subproducts_buf] produced, and at loop exit it
   collapses to [warp_matmul].  Mirrors To.KLoop's [k_loop_step]/[compute_acc]
   exit, but with tiling granularity [bk] (not the fragment [tk]). ---- *)

(* nested-subtile identity: the block-then-warp subtile equals the global-warp
   subtile at the composed index (A: row composition; B: col composition after
   [mtranspose_subtile]). *)
#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let kloop_acc_identity
  (#m #n #k : nat) (bm bn bk wm wn : pos)
  (rA : chest2 real m k) (rB : chest2 real n k)
  (block_row block_col warp_m warp_n grow gcol vkt : nat)
  (sq : squash (
     bm /? m /\ bn /? n /\ bk /? k /\ wm /? bm /\ wn /? bn /\
     block_row < m / bm /\ block_col < n / bn /\
     warp_m < bm / wm /\ warp_n < bn / wn /\ vkt < k / bk /\
     grow == block_row * (bm / wm) + warp_m /\
     gcol == block_col * (bn / wn) + warp_n))
  : Lemma (
      matmul (ematrix_subtile (ematrix_subtile rA bm bk block_row vkt) wm bk warp_m 0)
             (ematrix_subtile (mtranspose (ematrix_subtile rB bn bk block_col vkt)) bk wn 0 warp_n)
      ==
      matmul (ematrix_subtile rA wm bk grow vkt)
             (ematrix_subtile (mtranspose rB) bk wn vkt gcol))
= lemma_tile_index_bound m bm wm block_row warp_m;
  lemma_tile_index_bound n bn wn block_col warp_n;
  ematrix_subtile_compose rA bm bk wm bk block_row vkt warp_m 0 grow vkt ();
  ematrix_subtile_compose (mtranspose rB) bk bn bk wn vkt block_col 0 warp_n vkt gcol ()

#pop-options

(* one k-tile: advance the [__gmatmul_single ... vkt] accumulator to [vkt+1]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
noextract
ghost
fn kloop_acc_step
  (#et_acc : Type0) {| scalar et_acc |} {| real_like et_acc |}
  (#m #n #k : nat) (bm bn bk wm wn : pos)
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (rA : chest2 real m k) (rB : chest2 real n k)
  (block_row block_col warp_m warp_n grow gcol vkt : nat)
  (sq : squash (
     length accFrags == (wm / frag) * (wn / frag) /\
     frag /? wm /\ frag /? wn /\
     bm /? m /\ bn /? n /\ bk /? k /\ wm /? bm /\ wn /? bn /\
     block_row < m / bm /\ block_col < n / bn /\
     warp_m < bm / wm /\ warp_n < bn / wn /\ vkt < k / bk /\
     grow == block_row * (bm / wm) + warp_m /\
     gcol == block_col * (bn / wn) + warp_n))
  requires
    fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
      (matplus
        (__gmatmul_single (const _ 0.0R) matmul matplus
           (ematrix_tiled rA wm bk) (ematrix_tiled (mtranspose rB) bk wn) grow gcol vkt)
        (matmul (ematrix_subtile (ematrix_subtile rA bm bk block_row vkt) wm bk warp_m 0)
                (ematrix_subtile (mtranspose (ematrix_subtile rB bn bk block_col vkt)) bk wn 0 warp_n)))
  ensures
    fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
      (__gmatmul_single (const _ 0.0R) matmul matplus
         (ematrix_tiled rA wm bk) (ematrix_tiled (mtranspose rB) bk wn) grow gcol (vkt + 1))
{
  lemma_tile_index_bound m bm wm block_row warp_m;
  lemma_tile_index_bound n bn wn block_col warp_n;
  kloop_acc_identity bm bn bk wm wn rA rB block_row block_col warp_m warp_n grow gcol vkt ();
  subproducts_step_eq #m #k #n wm bk wn () (const _ 0.0R) rA (mtranspose rB) grow gcol vkt;
  unfold fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
    (matplus
      (__gmatmul_single (const _ 0.0R) matmul matplus
         (ematrix_tiled rA wm bk) (ematrix_tiled (mtranspose rB) bk wn) grow gcol vkt)
      (matmul (ematrix_subtile (ematrix_subtile rA bm bk block_row vkt) wm bk warp_m 0)
              (ematrix_subtile (mtranspose (ematrix_subtile rB bn bk block_col vkt)) bk wn 0 warp_n)));
  fold fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
    (__gmatmul_single (const _ 0.0R) matmul matplus
       (ematrix_tiled rA wm bk) (ematrix_tiled (mtranspose rB) bk wn) grow gcol (vkt + 1));
}
#pop-options

(* loop exit: collapse [__gmatmul_single ... (k/bk)] to [warp_matmul]. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
noextract
ghost
fn kloop_acc_collapse
  (#et_acc : Type0) {| scalar et_acc |} {| real_like et_acc |}
  (#m #n #k : nat) (bm bn bk wm wn : pos)
  (accFrags : array (fragment et_acc FragAcc frag frag frag FragLAcc))
  (rA : chest2 real m k) (rB : chest2 real n k)
  (grow gcol : nat)
  (sq : squash (
     length accFrags == (wm / frag) * (wn / frag) /\
     frag /? wm /\ frag /? wn /\ wm /?+ m /\ wn /?+ n /\ bk /? k /\ k > 0 /\
     grow < m / wm /\ gcol < n / wn))
  requires
    fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
      (__gmatmul_single (const _ 0.0R) matmul matplus
         (ematrix_tiled rA wm bk) (ematrix_tiled (mtranspose rB) bk wn) grow gcol (k / bk))
  ensures
    fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
      (warp_matmul rA rB wm wn grow gcol)
{
  matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
    #m #n #k wm wn bk (const _ 0.0R) rA (mtranspose rB) grow gcol;
  // matmul_tiles_lemma gives (in gmatmul_single form, unfolds to __gmatmul_single ...(k/bk)):
  //   __gmatmul...(k/bk) == matplus (const 0) (matmul A' B')
  //     with A' = ematrix_subtile rA wm k grow 0, B' = ematrix_subtile (mtranspose rB) k wn 0 gcol
  // matplus (const 0) X == X (equal + ext SMTPat):
  assert pure (matplus (const _ 0.0R)
      (matmul (ematrix_subtile rA wm k grow 0) (ematrix_subtile (mtranspose rB) k wn 0 gcol))
      `equal`
      (matmul (ematrix_subtile rA wm k grow 0) (ematrix_subtile (mtranspose rB) k wn 0 gcol)));
  // warp_matmul rA rB wm wn grow gcol == matmul A' (mtranspose (ematrix_subtile rB wn k gcol 0))
  //   and mtranspose_subtile (SMTPat) turns the second factor into B'.  Chain everything:
  assert pure (__gmatmul_single (const _ 0.0R) matmul matplus
      (ematrix_tiled rA wm bk) (ematrix_tiled (mtranspose rB) bk wn) grow gcol (k / bk)
      == warp_matmul rA rB wm wn grow gcol);
  unfold fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
    (__gmatmul_single (const _ 0.0R) matmul matplus
       (ematrix_tiled rA wm bk) (ematrix_tiled (mtranspose rB) bk wn) grow gcol (k / bk));
  fold fragarrayAcc_approximates (wm / frag) (wn / frag) accFrags
    (warp_matmul rA rB wm wn grow gcol);
}
#pop-options


#push-options "--z3rlimit 15 --fuel 1 --ifuel 2"
inline_for_extraction noextract
fn kloop
  (#et_ab #et_acc : Type0)
  {| _sab : scalar et_ab |} {| _vab : has_vec_cpy et_ab |}
  {| _sac : scalar et_acc |} {| _vac : has_vec_cpy et_acc |}
  {| _rab : real_like et_ab |} {| _rac : real_like et_acc |}
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
  (rA : chest2 real (SZ.v m) (SZ.v k) { eA %~ rA })
  (rB : chest2 real (SZ.v n) (SZ.v k) { eB %~ rB })
  (grow : erased nat { reveal grow < SZ.v m / SZ.v wm })
  (gcol : erased nat { reveal gcol < SZ.v n / SZ.v wn })
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
  (sq_bc : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                   SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (sq_gc : squash (
     reveal grow == SZ.v block_row * (SZ.v bm / SZ.v wm) + SZ.v warp_m /\
     reveal gcol == SZ.v block_col * (SZ.v bn / SZ.v wn) + SZ.v warp_n))
  (sq_pc : squash (SZ.v k > 0 /\ frag /?+ SZ.v wm /\ frag /?+ SZ.v wn /\
                   SZ.v wm /?+ SZ.v m /\ SZ.v wn /?+ SZ.v n))
  (fmap_id : squash (forall (x : et_ab). fmap x == x))
  ()
  preserves gpu
  preserves thread_id (SZ.v nthr) (SZ.v tid)
  preserves B.barrier_tok
    (pipe_contract_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
       sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) sq_bc)
  requires
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags (const _ 0.0R) **
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
    B.barrier_state 0 **
    pipe_live (skewed_view bm bk skew sarA0) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bm bk skew sarA1) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bn bk skew sarB0) (SZ.v nthr) (SZ.v tid) **
    pipe_live (skewed_view bn bk skew sarB1) (SZ.v nthr) (SZ.v tid)
  ensures
    fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
      (warp_matmul rA rB (SZ.v wm) (SZ.v wn) (reveal grow) (reveal gcol)) **
    (gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
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

  P.chunk_divides_ldt et_ab et_acc bm bn bk wm wn skew;
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


  fold_pending_even bm bn bk skew gA gB sarA0 sarA1 sarB0 sarB1 eA eB fA fB
    (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) b0 () 0;

  (* ---- fold the prologue's [pending 0] + batches into the loop carry ---- *)
  fold_kcarry_live bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB
    (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () 0 b0 ();

  (* ---- accumulator: reindex the (zeroed) fragments as the empty partial
     ([__gmatmul_single ... 0] over the (wm,bk)x(bk,wn) tilings).  The requires
     hands us [const 0] at the [(wm/frag)*frag] shape forced by
     [fragarrayAcc_approximates]; bridge it to a [wm]-shaped zero [rAcc0] (a
     subtile of a constant is the constant tile, so the pure content-facts are
     preserved across the shape mismatch), then rewrite by the zero-lemma. ---- *)
  frag_div_mul (SZ.v wm);
  frag_div_mul (SZ.v wn);
  unfold fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags (const _ 0.0R);
  Kuiper.Spec.GEMM.__gmatmul_single_zero_lemma (const _ 0.0R) matmul matplus
    (ematrix_tiled rA (SZ.v wm) (SZ.v bk))
    (ematrix_tiled (mtranspose rB) (SZ.v bk) (SZ.v wn))
    (reveal grow) (reveal gcol);
  (* the zeroed [__gmatmul_single ... 0] and the WMP-shaped [const 0] have the
     same per-tile content (all zero), so [fragarrayAcc]'s content-facts carry
     across; make the tile equality explicit so the fold discharges. *)
  assert pure (forall (i : natlt (SZ.v wm / frag)) (j : natlt (SZ.v wn / frag)).
    ematrix_subtile
      (__gmatmul_single (const _ 0.0R) matmul matplus
         (ematrix_tiled rA (SZ.v wm) (SZ.v bk))
         (ematrix_tiled (mtranspose rB) (SZ.v bk) (SZ.v wn))
         (reveal grow) (reveal gcol) 0)
      frag frag i j
    `equal`
    ematrix_subtile
      (const _ 0.0R <: chest2 real ((SZ.v wm / frag) * frag) ((SZ.v wn / frag) * frag))
      frag frag i j);
  fold fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
    (__gmatmul_single (const _ 0.0R) matmul matplus
       (ematrix_tiled rA (SZ.v wm) (SZ.v bk))
       (ematrix_tiled (mtranspose rB) (SZ.v bk) (SZ.v wn))
       (reveal grow) (reveal gcol) 0);

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
          (pipe_contract_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
             sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) sq_bc) **
        B.barrier_state (SZ.v vkt) **
        live aFrags ** live bFrags **
        fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
          (__gmatmul_single (const _ 0.0R) matmul matplus
             (ematrix_tiled rA (SZ.v wm) (SZ.v bk))
             (ematrix_tiled (mtranspose rB) (SZ.v bk) (SZ.v wn))
             (reveal grow) (reveal gcol) (SZ.v vkt)) **
        kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
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
    (* Isolate the last-k-tile linear arithmetic in the light pre-body context:
       either [kt1] is not the last tile, or it is and [vkt == k/bk - 1].  The
       resource-heavy else-branch below then only picks a disjunct from the
       [kt1 <^ num_k_tiles] guard, never re-deriving the div/mod. *)
    ktiles_last_arith (SZ.v kt1) (SZ.v vkt) (SZ.v k) (SZ.v bk) (SZ.v num_k_tiles);
    assert pure (
      srcA == (if SZ.v vkt % 2 = 0 then sarA0 else sarA1) /\
      dstA == (if SZ.v vkt % 2 = 0 then sarA1 else sarA0) /\
      srcB == (if SZ.v vkt % 2 = 0 then sarB0 else sarB1) /\
      dstB == (if SZ.v vkt % 2 = 0 then sarB1 else sarB0));

    (* ---- ghost: expose the current-tile pledge over [src*]/[dst*] ---- *)
    unfold_kcarry_live bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB
      (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt) ();
    with b_c b_l. _;
    unfold_pending_g bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB fA fB
      (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_c) () (SZ.v vkt) ();

    (* ---- CONCRETE: __pipeline_wait_prior(0) then redeem the staged pledge ---- *)
    with bd. assert (PC.batch_committed bd);
    PC.pipeline_wait_all_prior #bd;
    redeem_pledge emp_inames (PC.batch_done bd) _;
    drop_ (PC.batch_done bd);
    restore_globals bm bn bk gA gB eA eB fA fB (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt);
    fold (pipe_live_c (skewed_view bm bk skew srcA)
            (ematrix_subtile eA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v vkt)) (SZ.v nthr) (SZ.v tid));
    fold (pipe_live_c (skewed_view bn bk skew srcB)
            (ematrix_subtile eB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v vkt)) (SZ.v nthr) (SZ.v tid));

    (* ---- CONCRETE: __syncthreads() (barrier) advancing the pipe contract ---- *)
    fold_pipe_p_g m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
      sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB
      (SZ.v nthr) (SZ.v vkt) (SZ.v tid) sq_bc ();
    rewrite (pipe_p_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
               sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) sq_bc (SZ.v vkt) (SZ.v tid))
      as ((pipe_contract_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
             sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) sq_bc).B.rin (SZ.v vkt) (SZ.v tid));
    B.barrier_wait () #_ #_ #(SZ.v vkt) #(SZ.v tid);
    rewrite ((pipe_contract_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
                sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) sq_bc).B.rout (SZ.v vkt) (SZ.v tid))
      as (pipe_q_c m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
            sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) sq_bc (SZ.v vkt) (SZ.v tid));
    unfold_pipe_q_g m n k bm bn bk skew eA eB (SZ.v block_row) (SZ.v block_col)
      sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB
      (SZ.v nthr) (SZ.v vkt) (SZ.v tid) sq_bc ();

    (* ---- the ONLY copy of the staging code: guarded on [kt+1 < ktiles] ---- *)
    if (kt1 <^ num_k_tiles) {
      let b_l2 = stage_next bm bn bk skew gA gB dstA dstB fA fB
        nthr tid block_row block_col
        a_t_row a_t_col a_row_step a_iters b_t_row b_t_col b_row_step b_iters
        ldsz kt1 (reveal b_l) () () () () () () ();
      rewrite (staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB (SZ.v nthr) (SZ.v tid)
                (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v kt1))
        as (staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB (SZ.v nthr) (SZ.v tid)
                (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v vkt + 1));
      fold_cstage_stage bm bn bk skew gA gB dstA dstB eA eB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) b_l2 (SZ.v vkt) ();
    } else {
      fold_cstage_nostage bm bn bk skew gA gB dstA dstB eA eB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) (SZ.v vkt) ();
    };

    (* ---- the ONLY copy of the fragment math, over the read buffer [src*] ---- *)
    lemma_subtile_approx eA rA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v vkt) ();
    lemma_subtile_approx eB rB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v vkt) ();
    subproducts_buf bm bn bk wm wn skew srcA srcB fmap nthr warp_m warp_n ldsz
      aFrags bFrags accFrags
      (ematrix_subtile rA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v vkt))
      (ematrix_subtile rB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v vkt))
      (__gmatmul_single (const _ 0.0R) matmul matplus
         (ematrix_tiled rA (SZ.v wm) (SZ.v bk))
         (ematrix_tiled (mtranspose rB) (SZ.v bk) (SZ.v wn))
         (reveal grow) (reveal gcol) (SZ.v vkt))
      () () () ();
    (* advance the [__gmatmul_single ... vkt] accumulator to [vkt+1]. *)
    acc_len_reveal wm wn;
    kloop_acc_step (SZ.v bm) (SZ.v bn) (SZ.v bk) (SZ.v wm) (SZ.v wn) accFrags
      rA rB (SZ.v block_row) (SZ.v block_col) (SZ.v warp_m) (SZ.v warp_n)
      (reveal grow) (reveal gcol) (SZ.v vkt) ();
    (* forget the read buffer's published content: the reconciled loop carry
       ([pending]/[pipe_q]) uses the content-free [pipe_sharing]. *)
    unfold (pipe_sharing_c (skewed_view bm bk skew srcA)
              (ematrix_subtile eA (SZ.v bm) (SZ.v bk) (SZ.v block_row) (SZ.v vkt)) (SZ.v nthr));
    fold (pipe_sharing (skewed_view bm bk skew srcA) (SZ.v nthr));
    unfold (pipe_sharing_c (skewed_view bn bk skew srcB)
              (ematrix_subtile eB (SZ.v bn) (SZ.v bk) (SZ.v block_col) (SZ.v vkt)) (SZ.v nthr));
    fold (pipe_sharing (skewed_view bn bk skew srcB) (SZ.v nthr));

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
      unfold_cstage_stage bm bn bk skew gA gB dstA dstB eA eB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) (SZ.v vkt) ();
      unfold (staged_half bm bn bk skew gA gB dstA dstB eA eB fA fB (SZ.v nthr) (SZ.v tid)
                (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v vkt + 1));
      fold_pending_g bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) (reveal b_l) () (SZ.v vkt + 1) ();
      fold_kcarry_live bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt + 1) (reveal b_l) ();
    } else {
      unfold_cstage_nostage bm bn bk skew gA gB dstA dstB eA eB fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (reveal b_l) (SZ.v vkt) ();
      fold_pipe_q_g bm bn bk skew sarA0 sarA1 sarB0 sarB1 srcA dstA srcB dstB
        (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid) ();
      rewrite (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v vkt) (SZ.v tid))
        as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk) (SZ.v k / SZ.v bk - 1) (SZ.v tid));
      with lb. assert (PC.batch_live lb);
      drop_ (PC.batch_live lb);
      fold_kcarry_done bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB
        (SZ.v nthr) (SZ.v tid) (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt + 1) ();
    };
    rewrite
      (kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
        (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt + 1))
    as
      (kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
        (SZ.v block_row) (SZ.v block_col) () (SZ.v kt1));
    rewrite
      (fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
        (__gmatmul_single (const _ 0.0R) matmul matplus
           (ematrix_tiled rA (SZ.v wm) (SZ.v bk))
           (ematrix_tiled (mtranspose rB) (SZ.v bk) (SZ.v wn))
           (reveal grow) (reveal gcol) (SZ.v vkt + 1)))
    as
      (fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
        (__gmatmul_single (const _ 0.0R) matmul matplus
           (ematrix_tiled rA (SZ.v wm) (SZ.v bk))
           (ematrix_tiled (mtranspose rB) (SZ.v bk) (SZ.v wn))
           (reveal grow) (reveal gcol) (SZ.v kt1)));
    rewrite (B.barrier_state (SZ.v vkt + 1)) as (B.barrier_state (SZ.v kt1));
    idx := !idx +^ 1sz;
  };

  (* ---- loop exit: [vkt == ktiles], so [kcarry] is its "done" branch ---- *)
  with vkt. assert (idx |-> vkt);
  assert pure (SZ.v num_k_tiles == SZ.v k / SZ.v bk);
  assert pure (SZ.v vkt >= SZ.v num_k_tiles /\ SZ.v vkt <= SZ.v k / SZ.v bk);
  assert pure (SZ.v vkt == SZ.v k / SZ.v bk);
  rewrite
    (kcarry bm bn bk skew gA gB eA eB sarA0 sarA1 sarB0 sarB1 fA fB (SZ.v nthr) (SZ.v tid)
      (SZ.v block_row) (SZ.v block_col) () (SZ.v vkt))
  as
    ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
     pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 (SZ.v nthr) (SZ.v k / SZ.v bk)
            (SZ.v k / SZ.v bk - 1) (SZ.v tid));
  rewrite (B.barrier_state (SZ.v vkt))
    as (B.barrier_state (SZ.v k / SZ.v bk));
  (* collapse the accumulator [__gmatmul_single ... (k/bk)] to [warp_matmul]. *)
  acc_len_reveal wm wn;
  kloop_acc_collapse (SZ.v bm) (SZ.v bn) (SZ.v bk) (SZ.v wm) (SZ.v wn) accFrags
    rA rB (reveal grow) (reveal gcol) ();
  with va. assert (aFrags |-> va);
  drop_ (aFrags |-> va);
  with vb. assert (bFrags |-> vb);
  drop_ (bFrags |-> vb);
  ()
}
#pop-options
