module Kuiops.Sdpa

open Kuiper
open Kuiper.Shape
open Kuiper.Tensor.Layout { tlayout, ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.Bijection
open Kuiper.Kernel.SDPA.Naive

module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

inline_for_extraction noextract
let unfold3 (d0 d1 d2 : szp)
  (i : szlt (d0 * d1) & (szlt d2 & unit))
  : Tot (szlt d0 & (szlt d1 & (szlt d2 & unit))) =
  let ih, (i2, ()) = i in
  (ih /^ d1, (ih %^ d1, (i2, ())))

inline_for_extraction noextract
let fold3 (d0 d1 d2 : szp)
  (lin : tlayout (d0 @| d1 @| d2 @| INil)) {| c : ctlayout lin |}
  : ctlayout (tlayout_fold_outer lin) =
  ctlayout_bij (fold_bij #3 #(d0 @| d1 @| d2 @| INil))
    (unfold3 d0 d1 d2)
    (fun x -> fold_bij_gg (up x); ())
    lin

inline_for_extraction noextract
let unfold4 (d0 d1 d2 d3 : szp)
  (i : szlt (d0 * d1) & (szlt d2 & (szlt d3 & unit)))
  : Tot (szlt d0 & (szlt d1 & (szlt d2 & (szlt d3 & unit)))) =
  let ih, (i2, (i3, ())) = i in
  (ih /^ d1, (ih %^ d1, (i2, (i3, ()))))

inline_for_extraction noextract
let fold4 (d0 d1 d2 d3 : szp)
  (lin : tlayout (d0 @| d1 @| d2 @| d3 @| INil)) {| c : ctlayout lin |}
  : ctlayout (tlayout_fold_outer lin) =
  ctlayout_bij (fold_bij #4 #(d0 @| d1 @| d2 @| d3 @| INil))
    (unfold4 d0 d1 d2 d3)
    (fun x -> fold_bij_gg (up x); ())
    lin

#push-options "--fuel 2 --ifuel 2 --z3rlimit 80 --split_queries always"
// Fixed-rank scalar index functions keep erased dependent tuples out of Karamel.
inline_for_extraction noextract
let fold4_rm_cimap
  (d0 : szp)
  (d1 : szp { SZ.fits (d0 * d1) })
  (d2 : szp {
    SZ.fits (d1 * d2) /\
    SZ.fits (d0 * (d1 * d2)) })
  (d3 : szp {
    SZ.fits (d2 * d3) /\
    SZ.fits (d1 * (d2 * d3)) /\
    SZ.fits (d0 * (d1 * (d2 * d3))) })
  (ih : szlt (d0 * d1))
  (i2 : szlt d2)
  (i3 : szlt d3)
  : r : SZ.t {
      let idx : conc ((d0 * d1) @| d2 @| d3 @| INil) =
        (ih, (i2, (i3, ()))) in
      SZ.v r ==
        (tlayout_fold_outer (
          l4_batched_row_major d0 d1 d2 d3)).imap.f (up idx) } =
  let d01 = d0 *^ d1 in
  let idx : conc ((d0 * d1) @| d2 @| d3 @| INil) =
    (ih, (i2, (i3, ()))) in
  let unfolded :
    szlt d0 & (szlt d1 & (szlt d2 & (szlt d3 & unit))) =
    (ih /^ d1, (ih %^ d1, (i2, (i3, ())))) in
  tlayout_bij_imap
    (fold_bij #4 #(d0 @| d1 @| d2 @| d3 @| INil))
    (l4_batched_row_major d0 d1 d2 d3)
    (up idx);
  fold_bij_gg #4 #(d0 @| d1 @| d2 @| d3 @| INil) (up idx);
  SZ.s_divmod_inv_1 d1 ih;
  assert (
    up unfolded ==
    unfold_index #4 #(d0 @| d1 @| d2 @| d3 @| INil) (up idx));
  let result =
    (c_l3_batched_row_major (SZ.v d01) d2 d3).cimap idx in
  l4_batched_row_major_imap
    (SZ.v d0) d1 d2 d3 (ih /^ d1) (ih %^ d1) i2 i3;
  l3_batched_row_major_imap (SZ.v d01) d2 d3 ih i2 i3;
  assert (
    SZ.v (ih /^ d1) * SZ.v d1 + SZ.v (ih %^ d1) == SZ.v ih);
  ML.paren_mul_right
    (SZ.v (ih /^ d1)) (SZ.v d1) (SZ.v d2 * SZ.v d3);
  ML.distributivity_add_left
    (SZ.v (ih /^ d1) * SZ.v d1) (SZ.v (ih %^ d1))
    (SZ.v d2 * SZ.v d3);
  calc (==) {
    SZ.v (ih /^ d1) * (SZ.v d1 * (SZ.v d2 * SZ.v d3)) +
      SZ.v (ih %^ d1) * (SZ.v d2 * SZ.v d3);
    == {}
    (SZ.v (ih /^ d1) * SZ.v d1 + SZ.v (ih %^ d1)) *
      (SZ.v d2 * SZ.v d3);
    == {}
    SZ.v ih * (SZ.v d2 * SZ.v d3);
  };
  assert (
    (tlayout_fold_outer (
      l4_batched_row_major d0 d1 d2 d3)).imap.f (up idx) ==
    SZ.v result);
  result

inline_for_extraction noextract
let fold_transpose4_rm_cimap
  (d0 : szp)
  (d1 : szp { SZ.fits (d0 * d1) })
  (d2 : szp {
    SZ.fits (d1 * d2) /\
    SZ.fits (d0 * (d1 * d2)) })
  (d3 : szp {
    SZ.fits (d2 * d3) /\
    SZ.fits (d1 * (d2 * d3)) /\
    SZ.fits (d0 * (d1 * (d2 * d3))) })
  (ih : szlt (d0 * d1))
  (i3 : szlt d3)
  (i2 : szlt d2)
  : r : SZ.t {
      let idx : conc ((d0 * d1) @| d3 @| d2 @| INil) =
        (ih, (i3, (i2, ()))) in
      SZ.v r ==
        (tlayout_fold_outer (
          tlayout_bij (transpose4_2 d0 d1 d2 d3)
            (l4_batched_row_major d0 d1 d2 d3))).imap.f (up idx) } =
  let d01 = d0 *^ d1 in
  let idx : conc ((d0 * d1) @| d3 @| d2 @| INil) =
    (ih, (i3, (i2, ()))) in
  let unfolded :
    szlt d0 & (szlt d1 & (szlt d3 & (szlt d2 & unit))) =
    (ih /^ d1, (ih %^ d1, (i3, (i2, ())))) in
  tlayout_bij_imap
    (fold_bij #4 #(d0 @| d1 @| d3 @| d2 @| INil))
    (tlayout_bij (transpose4_2 d0 d1 d2 d3)
      (l4_batched_row_major d0 d1 d2 d3))
    (up idx);
  fold_bij_gg #4 #(d0 @| d1 @| d3 @| d2 @| INil) (up idx);
  SZ.s_divmod_inv_1 d1 ih;
  assert (
    up unfolded ==
    unfold_index #4 #(d0 @| d1 @| d3 @| d2 @| INil) (up idx));
  tlayout_bij_imap
    (transpose4_2 d0 d1 d2 d3)
    (l4_batched_row_major d0 d1 d2 d3)
    (up unfolded);
  transpose4_2_conc_correct
    #(SZ.v d0) #(SZ.v d1) #(SZ.v d2) #(SZ.v d3) unfolded;
  let result =
    (c_l3_batched_col_major (SZ.v d01) d3 d2).cimap idx in
  l4_batched_row_major_imap
    (SZ.v d0) d1 d2 d3 (ih /^ d1) (ih %^ d1) i2 i3;
  l3_batched_col_major_imap (SZ.v d01) d3 d2 ih i3 i2;
  assert (
    SZ.v (ih /^ d1) * SZ.v d1 + SZ.v (ih %^ d1) == SZ.v ih);
  ML.paren_mul_right
    (SZ.v (ih /^ d1)) (SZ.v d1) (SZ.v d2 * SZ.v d3);
  ML.distributivity_add_left
    (SZ.v (ih /^ d1) * SZ.v d1) (SZ.v (ih %^ d1))
    (SZ.v d2 * SZ.v d3);
  calc (==) {
    SZ.v (ih /^ d1) * (SZ.v d1 * (SZ.v d2 * SZ.v d3)) +
      SZ.v (ih %^ d1) * (SZ.v d2 * SZ.v d3);
    == {}
    (SZ.v (ih /^ d1) * SZ.v d1 + SZ.v (ih %^ d1)) *
      (SZ.v d2 * SZ.v d3);
    == {}
    SZ.v ih * (SZ.v d2 * SZ.v d3);
  };
  assert (
    (tlayout_fold_outer (
      tlayout_bij (transpose4_2 d0 d1 d2 d3)
        (l4_batched_row_major d0 d1 d2 d3))).imap.f (up idx) ==
    SZ.v result);
  result
#pop-options
