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
