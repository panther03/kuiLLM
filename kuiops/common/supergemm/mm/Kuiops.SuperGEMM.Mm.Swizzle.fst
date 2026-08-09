module Kuiops.SuperGEMM.Mm.Swizzle

open Kuiper.Common { natlt }

module ML = FStar.Math.Lemmas

#set-options "--z3rlimit 15 --fuel 1 --ifuel 1"

(* [rows]: length of the current scheduling strip.  For a valid strip
   ([gid*g < nm]) this is [min g (nm - gid*g)]; out of that range the value is
   irrelevant, so we return a positive dummy to keep the type [pos] (division
   and modulo by [rows] then need no side condition). *)
inline_for_extraction noextract
let sw_rows (nm g : pos) (gid : nat) : pos =
  let d = nm - gid * g in
  if 1 <= d && d < g then d else g

let sw_rows_spec (nm g : pos) (gid : nat)
  : Lemma (requires gid * g < nm)
          (ensures sw_rows nm g gid > 0 /\
                   sw_rows nm g gid <= g /\
                   sw_rows nm g gid <= nm - gid * g /\
                   (sw_rows nm g gid == g \/ sw_rows nm g gid == nm - gid * g))
  = ()

(* The scheduling strip [gid = bid / (g*nn)] fits: [gid*g < nm]. *)
let gid_g_lt (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (ensures (bid / (g * nn)) * g < nm /\
                   (bid / (g * nn)) * (g * nn) <= bid)
  = let gg = g * nn in
    ML.division_propriety bid gg;
    let gid = bid / gg in
    ML.paren_mul_right gid g nn;
    ML.multiplication_order_lemma (gid * g) nm nn

(* [rem = bid % (g*nn) < rows * nn], the fact that makes [sw_col < nn]. *)
let rem_lt_rows (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (ensures bid % (g * nn) < sw_rows nm g (bid / (g * nn)) * nn)
  = let gg = g * nn in
    let gid = bid / gg in
    gid_g_lt nm nn g bid;
    sw_rows_spec nm g gid;
    ML.modulo_range_lemma bid gg;
    ML.euclidean_division_definition bid gg;
    ML.paren_mul_right gid g nn;
    ML.distributivity_sub_left nm (gid * g) nn

(* [a < b*c ==> a/b < c]. *)
let div_lt_mul (a : nat) (b : pos) (c : nat)
  : Lemma (requires a < b * c) (ensures a / b < c)
  = ML.division_propriety a b;
    if a / b >= c then begin
      ML.lemma_mult_le_right b c (a / b);
      ML.swap_mul b c
    end else ()

(* [r - gid*g < rows] for the inverse, with [gid = r / g]. *)
let r_sub_lt_rows (nm g : pos) (r : natlt nm)
  : Lemma (ensures (r / g) * g <= r /\
                   (r / g) * g < nm /\
                   r - (r / g) * g < sw_rows nm g (r / g))
  = let gid = r / g in
    ML.division_propriety r g;
    sw_rows_spec nm g gid;
    ML.euclidean_division_definition r g;
    ML.modulo_range_lemma r g

(* The inverse's value is in range. *)
let inv_bound (nm nn g : pos) (r : natlt nm) (c : natlt nn)
  : Lemma
      (ensures (let gid = r / g in let rows = sw_rows nm g gid in
                gid * g * nn + (r - gid * g) + c * rows < nm * nn))
  = let gid = r / g in
    r_sub_lt_rows nm g r;
    sw_rows_spec nm g gid;
    let rows = sw_rows nm g gid in
    ML.lemma_mult_le_right rows c (nn - 1);
    ML.distributivity_sub_left nn 1 rows;
    ML.swap_mul (gid * g) nn;
    ML.distributivity_add_right nn (gid * g) rows;
    ML.lemma_mult_le_left nn (gid * g + rows) nm;
    ML.swap_mul nn nm

(* ---- exported functions ---- *)

inline_for_extraction noextract
let sw_row (nm nn g : pos) (bid : natlt (nm * nn)) : natlt nm =
  let gg = g * nn in
  let gid = bid / gg in
  let rem = bid % gg in
  gid_g_lt nm nn g bid;
  sw_rows_spec nm g gid;
  let rows = sw_rows nm g gid in
  ML.modulo_range_lemma rem rows;
  gid * g + rem % rows

inline_for_extraction noextract
let sw_col (nm nn g : pos) (bid : natlt (nm * nn)) : natlt nn =
  let gg = g * nn in
  let gid = bid / gg in
  let rem = bid % gg in
  let rows = sw_rows nm g gid in
  rem_lt_rows nm nn g bid;
  div_lt_mul rem rows nn;
  rem / rows

inline_for_extraction noextract
let sw_inv (nm nn g : pos) (r : natlt nm) (c : natlt nn) : natlt (nm * nn) =
  let gid = r / g in
  let rows = sw_rows nm g gid in
  r_sub_lt_rows nm g r;
  inv_bound nm nn g r c;
  gid * g * nn + (r - gid * g) + c * rows

(* Value characterizations so downstream proofs see the clean expressions. *)
let sw_row_eq (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (sw_row nm nn g bid ==
           (bid / (g * nn)) * g + (bid % (g * nn)) % sw_rows nm g (bid / (g * nn)))
          [SMTPat (sw_row nm nn g bid)]
  = ()

let sw_col_eq (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (sw_col nm nn g bid ==
           (bid % (g * nn)) / sw_rows nm g (bid / (g * nn)))
          [SMTPat (sw_col nm nn g bid)]
  = ()

let sw_inv_eq (nm nn g : pos) (r : natlt nm) (c : natlt nn)
  : Lemma (sw_inv nm nn g r c ==
           (r / g) * g * nn + (r - (r / g) * g) + c * sw_rows nm g (r / g))
          [SMTPat (sw_inv nm nn g r c)]
  = ()

(* ---- bijectivity ---- *)

let sw_decode_spec (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (let pg   = g * nn in
           let gid  = bid / pg in
           let rem  = bid - gid * pg in
           let d    = nm - gid * g in
           let rows = if d < g then d else g in
           rows > 0 /\
           gid * g < nm /\
           gid * (g * nn) <= bid /\
           gid * g + rem % rows < nm /\
           rem / rows < nn /\
           sw_row nm nn g bid == gid * g + rem % rows /\
           sw_col nm nn g bid == rem / rows)
  = let pg = g * nn in
    let gid = bid / pg in
    gid_g_lt nm nn g bid;
    sw_rows_spec nm g gid;
    let rows = sw_rows nm g gid in
    (* [rem = bid - gid*pg = bid % pg] and, since [d >= 1], the caller's
       [if d < g then d else g] coincides with [sw_rows]. *)
    ML.euclidean_division_definition bid pg;
    assert (bid - gid * pg == bid % pg);
    assert (rows == (let d = nm - gid * g in if d < g then d else g));
    ML.modulo_range_lemma (bid % pg) rows;
    rem_lt_rows nm nn g bid;
    div_lt_mul (bid % pg) rows nn

let sw_inv_correct (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (sw_inv nm nn g (sw_row nm nn g bid) (sw_col nm nn g bid) == bid)
  = let gg = g * nn in
    let gid = bid / gg in
    let rem = bid % gg in
    gid_g_lt nm nn g bid;
    sw_rows_spec nm g gid;
    let rows = sw_rows nm g gid in
    ML.modulo_range_lemma rem rows;
    let r = gid * g + rem % rows in
    (* r / g = gid *)
    ML.small_division_lemma_1 (rem % rows) g;
    ML.lemma_div_plus (rem % rows) gid g;
    (* the inverse expression collapses to gid*gg + rem *)
    ML.euclidean_division_definition rem rows;
    ML.paren_mul_right gid g nn;
    ML.euclidean_division_definition bid gg

let sw_surjective (nm nn g : pos) (r : natlt nm) (c : natlt nn)
  : Lemma
      (ensures sw_row nm nn g (sw_inv nm nn g r c) == r /\
               sw_col nm nn g (sw_inv nm nn g r c) == c)
      [SMTPat (sw_inv nm nn g r c)]
  = let gg = g * nn in
    let gid = r / g in
    r_sub_lt_rows nm g r;
    sw_rows_spec nm g gid;
    let rows = sw_rows nm g gid in
    let t = r - gid * g in
    let br = t + c * rows in
    (* [br < gg]: [t < rows] and [c*rows <= (nn-1)*rows], so [br < nn*rows <= nn*g = gg]. *)
    ML.lemma_mult_le_right rows c (nn - 1);
    ML.distributivity_sub_left nn 1 rows;
    ML.lemma_mult_le_left nn rows g;
    ML.swap_mul nn g;
    assert (br < gg);
    (* the inverse's value is [gid*gg + br] *)
    ML.paren_mul_right gid g nn;
    let bid = sw_inv nm nn g r c in
    assert (bid == gid * gg + br);
    (* decode the forward strip and remainder of [bid] *)
    ML.small_division_lemma_1 br gg;
    ML.lemma_div_plus br gid gg;
    ML.lemma_mod_plus br gid gg;
    ML.small_modulo_lemma_1 br gg;
    assert (bid / gg == gid);
    assert (bid % gg == br);
    (* block_col = br / rows = c *)
    ML.small_division_lemma_1 t rows;
    ML.lemma_div_plus t c rows;
    assert (br / rows == c);
    assert (sw_col nm nn g bid == c);
    (* block_row = gid*g + br % rows = gid*g + t = r *)
    ML.small_modulo_lemma_1 t rows;
    ML.lemma_mod_plus t c rows;
    assert (br % rows == t);
    assert (sw_row nm nn g bid == r)
