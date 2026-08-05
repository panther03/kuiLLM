module Kuiops.Array2.Strided.ColMajor

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Array2.Strided
module SZ = Kuiper.SizeT
open Kuiper.TensorRO { vtlayout_of_tlayout }

inline_for_extraction noextract
let scm_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  : strided_col_major (vtlayout_of_tlayout (l2_col_major rows cols)) =
{
  offset = 0sz;
  stride = concr' d;
  pf = ez;
}

let lemma_aligned_scm_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  (n : pos)
  : Lemma (requires n /?+ rows)
          (ensures aligned_strided_col_major n
                     (scm_l2_col_major #rows #cols #_ #d))
  = ()
