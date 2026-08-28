module Kuiops.Common.Band.Test
#lang-pulse

(* Pins the [Kuiops.Common.Band] interface against the shape a proportional
   split-K [Kernel.fst] would use it in: a global, layout-generic,
   [strided_row_major] A operand sliced to the k-range
   [[kt0*bk, kt1*bk)] of split [z], where [kt0 = ktiles*z/splits] is not a
   multiple of the band width and so is out of reach of [array2_extract_tile_ro'].

   Checks that the extracted band still carries the [ctlayout] and
   [strided_row_major] instances the k loop demands, that alignment survives,
   and that the trade returns the operand. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.Array2.Strided
open Kuiper.TradeHelpers
open Kuiops.Common.Band
open Pulse.Lib.Trade
module SZ = Kuiper.SizeT

let div_ub (a b c : nat)
  : Lemma (requires c > 0 /\ a < b * c) (ensures a / c < b)
= if a / c >= b then begin
    FStar.Math.Lemmas.lemma_div_mod a c;
    FStar.Math.Lemmas.lemma_mult_le_right c b (a / c)
  end

inline_for_extraction noextract
fn band_client
  (#et : Type0)
  (m k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  {| cA : ctlayout lA |}
  {| sA : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et lA { is_global gA })
  (bk : szp)
  (splits : szp)
  (z : szlt (SZ.v splits))
  (#eA : chest2 et (SZ.v m) (SZ.v k))
  (#f : perm)
  requires
    gA |-> Frac f eA **
    pure (SZ.fits lA.ulen /\ SZ.v m > 0 /\ SZ.v bk > 0 /\
          SZ.v k == (SZ.v k / SZ.v bk) * SZ.v bk /\
          SZ.v splits <= SZ.v k / SZ.v bk /\
          SZ.fits ((SZ.v k / SZ.v bk) * SZ.v splits) /\
          aligned_strided_row_major 8 sA /\ 8 /?+ SZ.v bk)
  ensures
    gA |-> Frac f eA
{
  let ktiles : szp = k /^ bk;
  FStar.Math.Lemmas.lemma_mult_le_left (SZ.v ktiles) (SZ.v z + 1) (SZ.v splits);
  div_ub (SZ.v ktiles * SZ.v z) (SZ.v ktiles) (SZ.v splits);
  let kt0 : szlt (SZ.v ktiles) = (ktiles *^ z) /^ splits;
  FStar.Math.Lemmas.lemma_div_le (SZ.v ktiles * (SZ.v z + 1)) (SZ.v ktiles * SZ.v splits) (SZ.v splits);
  FStar.Math.Lemmas.cancel_mul_div (SZ.v ktiles) (SZ.v splits);
  let kt1 : (x:sz { SZ.v kt0 < SZ.v x /\ SZ.v x <= SZ.v ktiles }) =
    (ktiles *^ (z +^ 1sz)) /^ splits;

  let coff : erased nat = hide (SZ.v kt0 * SZ.v bk);
  let clen : erased nat = hide ((SZ.v kt1 - SZ.v kt0) * SZ.v bk);
  assert pure (reveal coff + reveal clen <= SZ.v k);
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v bk) (SZ.v kt0) (SZ.v ktiles);
  assert pure (SZ.fits (SZ.v kt0 * SZ.v bk));

  let gA' = array2_extract_band_ro' gA coff clen;

  (* the k loop needs both instances on the band *)
  let _ : ctlayout (band_layout lA coff clen) =
    c_band_layout lA coff #(concrete_sz_mul (SZ.v kt0) (SZ.v bk)) clen;
  let _ : strided_row_major (vtlayout_of_tlayout (band_layout lA coff clen)) =
    strided_row_major_band lA coff #(concrete_sz_mul (SZ.v kt0) (SZ.v bk)) clen;

  Kuiper.Divides.lemma_nat_divides_pos_divides 8 (SZ.v bk);
  Kuiper.Divides.lemma_divides_product_r 8 (SZ.v kt0) (SZ.v bk);
  Kuiper.Divides.lemma_nat_divides_pos_divides 8 (SZ.v kt0 * SZ.v bk);
  lemma_aligned_strided_row_major_band lA coff #(concrete_sz_mul (SZ.v kt0) (SZ.v bk)) clen 8;
  assert pure (aligned_strided_row_major 8
                 (strided_row_major_band lA coff
                    #(concrete_sz_mul (SZ.v kt0) (SZ.v bk)) clen));
  assert pure (is_global (array2_band gA coff clen));

  elim_trade _ _;
}
