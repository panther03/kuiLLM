module Kuiops.Array2.Strided.ColMajor

(* Column-major counterpart to [lemma_aligned_strided_row_major_l2_row_major].

   Kuiper defines the [strided_col_major] class, the [aligned_strided_col_major]
   predicate and the [strided_col_major_l2_col_major] instance, but ships no
   alignment lemma for that instance. The instance is declared [instance val],
   so its [offset]/[stride] are abstract outside [Kuiper.Array2.Strided] and the
   lemma cannot be proven downstream. We therefore restate the witness here,
   where its fields are transparent, and prove alignment against it.

   Belongs upstream in [Kuiper.Array2.Strided]; once there, drop this module and
   use [strided_col_major_l2_col_major] directly. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Array2.Strided
module SZ = Kuiper.SizeT

(* [l2_col_major rows cols] maps (i,j) to [j * rows + i]: leading dimension
   [rows], zero offset. *)
inline_for_extraction noextract
val scm_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  : strided_col_major (l2_col_major rows cols)

(* Alignment reduces to [n] dividing the leading dimension, since the offset is
   zero. *)
val lemma_aligned_scm_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  (n : pos)
  : Lemma (requires n /?+ rows)
          (ensures aligned_strided_col_major n
                     (scm_l2_col_major #rows #cols #_ #d))
