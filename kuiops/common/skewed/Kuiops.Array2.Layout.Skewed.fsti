module Kuiops.Array2.Layout.Skewed

(* A row-major [rows x cols] matrix whose backing rows are [cols + pad]
   elements apart: the "skewed" (padded) shared tile every tensor-core GEMM
   stages its operands into.

   The padding is not cosmetic.  A shared row stride of exactly [cols] makes
   the stride a multiple of the 32 four-byte banks whenever [cols] is, so the
   [FRAG] rows one [wmma::load_matrix_sync] touches all land in the same banks
   and the load serialises; padding by a chunk makes the stride odd in units of
   banks.  Measured in isolation on sm_86 this is the difference between 103
   and 306 TF/s on the inner loop.

   Unlike [l2_row_major], this layout is NOT full: its [ulen] is
   [rows * (cols + pad)] while it only maps [rows * cols] cells.  The unmapped
   cells are handed back separately by [skew_split] as [skew_residual], and
   [skew_join] consumes them again.  Both sides are stated over an existentially
   quantified chest, because the pre-staging content of a shared tile is never
   observed.

   TODO(upstream): belongs next to [l2_row_major] in
   [Kuiper.Tensor.Layout.Alg] / [Kuiper.Array2.Strided]. *)

#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiops.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }

module SZ = Kuiper.SizeT

(* [(i, j) |-> i * (cols + pad) + j]. *)
[@@erasable]
val l2_skewed_row_major (rows cols pad : nat)
  : l : tlayout (rows @| cols @| INil) { l.ulen == rows * (cols + pad) }

val lemma_skewed_imap (rows cols pad : nat) (i : natlt rows) (j : natlt cols)
  : Lemma ((l2_skewed_row_major rows cols pad).imap.f (idx2 i j)
             == i * (cols + pad) + j)
          [SMTPat ((l2_skewed_row_major rows cols pad).imap.f (idx2 i j))]

inline_for_extraction noextract
val c_l2_skewed_row_major
  (#rows : erased nat) (#cols #pad : erased nat)
  (ld : SZ.t { SZ.v ld == cols + pad /\ SZ.fits rows /\ SZ.fits (rows * (cols + pad)) })
  : ctlayout (l2_skewed_row_major rows cols pad)

(* Offset 0, stride [cols + pad]. *)
inline_for_extraction noextract
val srm_l2_skewed_row_major
  (#rows : erased nat) (#cols #pad : erased nat)
  (ld : SZ.t { SZ.v ld == cols + pad })
  : strided_row_major (vtlayout_of_tlayout (l2_skewed_row_major rows cols pad))

val lemma_srm_l2_skewed_stride
  (#rows : erased nat) (#cols : erased nat) (#pad : erased nat)
  (ld : SZ.t { SZ.v ld == cols + pad })
  : Lemma (SZ.v (srm_l2_skewed_row_major #rows #cols #pad ld).stride == SZ.v ld)
          [SMTPat (SZ.v (srm_l2_skewed_row_major #rows #cols #pad ld).stride)]

val lemma_aligned_srm_l2_skewed_row_major
  (#rows : erased nat) (#cols #pad : erased nat)
  (ld : SZ.t { SZ.v ld == cols + pad })
  (n : pos)
  : Lemma (requires n /?+ (cols + pad))
          (ensures aligned_strided_row_major n
                     (srm_l2_skewed_row_major #rows #cols #pad ld))

(* The [pad] trailing cells of every row, which the layout does not map. *)
val skew_residual
  (#et : Type0)
  (p : array et)
  (rows cols pad : nat)
  : slprop

(* Both directions are stated over an existential chest: a shared tile's
   content before it is staged into is never observed, and after the last
   k-step it is dead. *)
ghost
fn skew_split
  (#et : Type0)
  (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  requires
    (exists* (v : Seq.seq et). pts_to p v) **
    pure (SZ.fits (rows * (cols + pad)))
  ensures
    (exists* (em : chest2 et rows cols).
       from_array (l2_skewed_row_major rows cols pad) p |-> em) **
    skew_residual p rows cols pad

ghost
fn skew_join
  (#et : Type0)
  (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  requires
    (exists* (em : chest2 et rows cols).
       from_array (l2_skewed_row_major rows cols pad) p |-> em) **
    skew_residual p rows cols pad **
    pure (SZ.fits (rows * (cols + pad)))
  ensures exists* (v : Seq.seq et). pts_to p v
