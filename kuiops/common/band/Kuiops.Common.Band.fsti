module Kuiops.Common.Band
#lang-pulse

(* An (offset, length) column-band view of a matrix.

   TODO(upstream): this belongs next to [Kuiper.Tensor.Tiling].  It generalizes
   [subtile_layout] / [array2_extract_tile_ro'] in the column direction: those
   take a tile width [tcols] with [tcols /? cols] and index by tile number
   [tc : natlt (cols / tcols)], so the column range they name always starts at
   a multiple of its own width.  A band starts at an arbitrary [coff] and has
   an arbitrary [clen], with only [coff + clen <= cols] required.  Nothing else
   in Kuiper offers this: [stride_subtile_layout] is an interleaved tile and
   also requires divisibility, and [Slice.slice_of_3] slices the page
   dimension.

   A split-K GEMM that divides its k tiles proportionally rather than uniformly
   needs exactly this to hand a per-split operand view to the k loop. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Injection
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.Array2.Strided
open Kuiper.TradeHelpers { factored }
module SZ = Kuiper.SizeT

(* The band of [em] holding columns [coff .. coff + clen). *)
let ematrix_band
  (#et : _)
  (#rows #cols : _)
  (em : chest2 et rows cols)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  : chest2 et rows clen
=
  mk2 fun i j -> acc2 em i (coff + j)

let band_inj_f
  (#rows #cols : nat)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  : abs ((rows @| clen @| INil)) -> abs ((rows @| cols @| INil))
=
  (fun (i, (j, ())) -> (i, (coff + j, ())))

let band_inj
  (#rows #cols : nat)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  : (abs ((rows @| clen @| INil)) @~> abs ((rows @| cols @| INil)))
= {
   f = band_inj_f #rows #cols coff clen;
}

let band_layout
  (#rows #cols : nat)
  (l : layout2 rows cols)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  : layout2 rows clen =
  {
    ulen = l.ulen;
    imap = inj_comp (band_inj #rows #cols coff clen) l.imap;
  }

inline_for_extraction noextract
instance val c_band_layout
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| cc : ctlayout l |}
  (coff : erased nat)
  {| concrete_sz coff |}
  (clen : erased nat { coff + clen <= cols })
  : ctlayout (band_layout l coff clen)

inline_for_extraction noextract
val array2_band
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : erased nat)
  (clen : erased nat { coff + clen <= cols })
  : Tot (array2 et (band_layout l coff clen))

val array2_band_core
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : erased nat)
  (clen : erased nat { coff + clen <= cols })
  : Lemma (ensures core (array2_band gm coff clen) == core gm)
          [SMTPat (core (array2_band gm coff clen))]

val array2_band_is_global
  (#et : _)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : erased nat)
  (clen : erased nat { coff + clen <= cols })
  : Lemma (ensures is_global (array2_band gm coff clen) <==> is_global gm)
          [SMTPat (is_global (array2_band gm coff clen))]

(* [rows > 0 /\ clen > 0] is what makes the band name a cell of the parent, so
   that its offset is bounded by [l.ulen] and fits in a [sz].  Both hold of any
   band a kernel actually takes. *)
inline_for_extraction noextract
instance val strided_row_major_band
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen /\ rows > 0))
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (coff : erased nat)
  {| c_coff : concrete_sz coff |}
  (clen : erased nat { 0 < clen /\ coff + clen <= cols })
  : strided_row_major (vtlayout_of_tlayout (band_layout l coff clen))

val lemma_band_strided_row_major_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (coff : erased nat)
  {| c_coff : concrete_sz coff |}
  (clen : erased nat { 0 < clen /\ coff + clen <= cols })
  : Lemma (requires SZ.fits l.ulen /\ rows > 0)
          (ensures SZ.v (strided_row_major_band l coff clen).offset
                   == sub.offset + coff)
          [SMTPat (strided_row_major_band l coff clen).offset]

val lemma_band_strided_row_major_stride
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (coff : erased nat)
  {| c_coff : concrete_sz coff |}
  (clen : erased nat { 0 < clen /\ coff + clen <= cols })
  : Lemma (requires SZ.fits l.ulen /\ rows > 0)
          (ensures (strided_row_major_band l coff clen).stride == sub.stride)
          [SMTPat (strided_row_major_band l coff clen).stride]

val lemma_aligned_strided_row_major_band
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (coff : erased nat)
  {| c_coff : concrete_sz coff |}
  (clen : erased nat { 0 < clen /\ coff + clen <= cols })
  (n : pos)
  : Lemma (requires SZ.fits l.ulen /\ rows > 0 /\
                    aligned_strided_row_major n sub /\ n /?+ coff)
          (ensures aligned_strided_row_major n
                     (strided_row_major_band l coff clen))

ghost
fn array2_extract_band_ro
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : nat)
  (clen : nat { coff + clen <= cols })
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em ** pure (SZ.fits l.ulen)
  ensures
    factored
      (array2_band gm coff clen |-> Frac f (ematrix_band em coff clen))
      (gm |-> Frac f em)

inline_for_extraction noextract
fn array2_extract_band_ro'
  (#et : Type0)
  (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (coff : erased nat)
  (clen : erased nat { coff + clen <= cols })
  (#em : chest2 et rows cols)
  (#f : perm)
  requires
    gm |-> Frac f em ** pure (SZ.fits l.ulen)
  returns gm' : array2 et (band_layout l coff clen)
  ensures
    rewrites_to gm' (array2_band gm coff clen) **
    factored
      (gm' |-> Frac f (ematrix_band em coff clen))
      (gm |-> Frac f em)
