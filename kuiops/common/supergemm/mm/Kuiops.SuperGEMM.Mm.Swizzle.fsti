module Kuiops.SuperGEMM.Mm.Swizzle

(* GROUP-based L2 block-index swizzle of the software-pipelined tensor-core GEMM.

   A grid of [nm * nn] block indices [bid] is permuted into (block_row,
   block_col) coordinates so that consecutively scheduled blocks form a
   [g]-tall, [nn]-wide column-major strip rather than a full block row.  Blocks
   in a strip share B columns and reuse A rows, so a scheduling wave's working
   set stays resident in L2.  [rows] handles the final, possibly short, strip
   when [g] does not divide [nm]; there is NO divisibility precondition.

   This is a pure permutation of the linear block index -- any bijection is
   correct, [g] only affects locality.  [sw_inv] is the explicit inverse;
   [sw_inv_correct] and [sw_surjective] together establish bijectivity, which
   the whole-matrix coverage proof of the launcher needs. *)

open Kuiper.Common { natlt }
open Kuiper.Bijection { bijection, ( =~ ) }

module ML = FStar.Math.Lemmas

(* block_row: [gid * g + rem % rows], where the scheduling strip [gid] is
   [bid / (g * nn)], [rem = bid % (g * nn)] and [rows = min g (nm - gid*g)]. *)
inline_for_extraction noextract
val sw_row (nm nn g : pos) (bid : natlt (nm * nn)) : natlt nm

(* block_col: [rem / rows]. *)
inline_for_extraction noextract
val sw_col (nm nn g : pos) (bid : natlt (nm * nn)) : natlt nn

(* Explicit inverse: [gid*g*nn + (r - gid*g) + c*rows] with [gid = r / g] and
   [rows = min g (nm - gid*g)]. *)
inline_for_extraction noextract
val sw_inv (nm nn g : pos) (r : natlt nm) (c : natlt nn) : natlt (nm * nn)

(* The controlled window onto [sw_row]/[sw_col] for the kernel: it computes the
   decode in machine arithmetic ([bid /^ pg], [bid -^ gid *^ pg], ...) and this
   lemma discharges every resulting refinement while tying the machine values
   to the abstract [sw_row]/[sw_col].  [rem] is stated as [bid - gid*pg] (equal
   to [bid % pg]) and [rows] as [if d < g then d else g] to mirror a SizeT
   computation exactly. *)
val sw_decode_spec (nm nn g : pos) (bid : natlt (nm * nn))
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

val sw_inv_correct (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (sw_inv nm nn g (sw_row nm nn g bid) (sw_col nm nn g bid) == bid)

val sw_surjective (nm nn g : pos) (r : natlt nm) (c : natlt nn)
  : Lemma
      (ensures sw_row nm nn g (sw_inv nm nn g r c) == r /\
               sw_col nm nn g (sw_inv nm nn g r c) == c)
      [SMTPat (sw_inv nm nn g r c)]

(* ---- self-bijection on the linear block index ----

   The block ownership partition is reindexed so that block [bid] owns the tile
   it actually computes.  [sw_lin bid] is the row-major linear index
   [sw_row * nn + sw_col] of that tile -- a [Tot] value carrying the [natlt]
   bound in its type, used directly in every ownership predicate, squash, and as
   the epilogue's data-ownership index.  [sw_bij] is the corresponding
   self-bijection, whose forward map [gg] is [sw_lin]; [forevery_iso] shifts the
   whole block predicate family through it, and [sw_bij_gg] bridges the two. *)
inline_for_extraction noextract
val sw_lin (nm nn g : pos) (bid : natlt (nm * nn)) : natlt (nm * nn)

(* [sw_lin] is [sw_row]/[sw_col] in the two base-[nn] digits, so a decode
   [sw_lin bid / nn] reduces to [sw_row bid] (and [% nn] to [sw_col bid]).
   No SMTPat: the kernel's field-type decode obligations are discharged by
   [output_tiling_bounds]; an SMTPat here fires on every [sw_lin] occurrence
   and floods the record queries.  Exported for explicit use. *)
val sw_lin_div (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (sw_lin nm nn g bid / nn == sw_row nm nn g bid)

val sw_lin_mod (nm nn g : pos) (bid : natlt (nm * nn))
  : Lemma (sw_lin nm nn g bid % nn == sw_col nm nn g bid)

(* [nb] is the concrete block count ([SZ.v nblk] at the call site), passed
   explicitly so the bijection's carrier is definitionally [natlt nb] --
   [forevery_iso] needs the family index type to match exactly, and [natlt nb]
   is not judgmentally [natlt (nm*nn)] even though [nb == nm*nn]. *)
[@@erasable]
val sw_bij (nm nn g : pos) (nb : pos { nb == nm * nn }) : (natlt nb =~ natlt nb)

(* The bridge: [forevery_iso] emits the ghost projection [(sw_bij ...).gg bid],
   which this lemma re-expresses as the [Tot] ownership index [sw_lin ... bid]. *)
val sw_bij_gg (nm nn g : pos) (nb : pos { nb == nm * nn }) (bid : natlt nb)
  : Lemma ((sw_bij nm nn g nb).gg bid == sw_lin nm nn g bid)
