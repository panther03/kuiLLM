module Kuiops.Array2.Strided

#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Tensor { array2, layout2, full_layout2 }
open Kuiper.Tensor.Layout.Alg
open Kuiper.TensorRO { vlayout1, vtlayout_of_tlayout, extended_layout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Tiling { subtile_layout, tile_inj }

inline_for_extraction noextract
let to_kuiper_strided_row_major
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (srm : strided_row_major (vtlayout_of_tlayout l))
  (#_ : squash (SZ.v srm.stride > 0))
  : Kuiper.Array2.Strided.strided_row_major l
= {
    offset = srm.offset;
    stride = (srm.stride <: szp);
    pf = (fun i j -> srm.pf i j);
  }

let lemma_writable_strided_row_major_stride_positive
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (srm : strided_row_major (vtlayout_of_tlayout l))
  (#_ : squash (rows > 1 /\ cols > 0))
  : Lemma (SZ.v srm.stride > 0)
= srm.pf 0 0;
  srm.pf 1 0;
  if SZ.v srm.stride = 0 then begin
    assert (l.imap.f (0, (0, ())) == l.imap.f (1, (0, ())));
    l.imap.is_inj (0, (0, ())) (1, (0, ()))
  end

let lemma_to_kuiper_aligned_strided_row_major
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (srm : strided_row_major (vtlayout_of_tlayout l))
  (#_ : squash (SZ.v srm.stride > 0))
  (n : pos)
  : Lemma
      (requires aligned_strided_row_major n srm)
      (ensures Kuiper.Array2.Strided.aligned_strided_row_major n
        (to_kuiper_strided_row_major l srm))
= ()

inline_for_extraction noextract
let to_kuiper_strided_col_major
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (scm : strided_col_major (vtlayout_of_tlayout l))
  (#_ : squash (SZ.v scm.stride > 0))
  : Kuiper.Array2.Strided.strided_col_major l
= {
    offset = scm.offset;
    stride = (scm.stride <: szp);
    pf = (fun i j -> scm.pf i j);
  }

let lemma_writable_strided_col_major_stride_positive
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (scm : strided_col_major (vtlayout_of_tlayout l))
  (#_ : squash (cols > 1 /\ rows > 0))
  : Lemma (SZ.v scm.stride > 0)
= scm.pf 0 0;
  scm.pf 0 1;
  if SZ.v scm.stride = 0 then begin
    assert (l.imap.f (0, (0, ())) == l.imap.f (0, (1, ())));
    l.imap.is_inj (0, (0, ())) (0, (1, ()))
  end

(* Instance for l2_row_major: cell_of_pos = i * cols + j *)
#push-options "--z3rlimit_factor 4"
inline_for_extraction noextract
instance strided_row_major_l2_row_major (#rows #cols : erased nat)
  (#_ : squash (cols > 0))
  {| d : concrete_sz cols |}
  : strided_row_major (vtlayout_of_tlayout (l2_row_major rows cols)) =
{
  offset = 0sz;
  stride = concr' d;
  pf = ez;
}

let lemma_aligned_strided_row_major_l2_row_major (#rows #cols : erased nat)
  (#_ : squash (cols > 0))
  {| d : concrete_sz cols |}
  (n : pos)
  : Lemma (requires n /?+ cols)
          (ensures aligned_strided_row_major n (strided_row_major_l2_row_major #rows #cols #_ #d))
  = ()

inline_for_extraction noextract
let mk_strided_row_major_bcast
  (#rows #cols : erased nat)
  (l : vlayout1 cols)
  (off : sz)
  (pf : squash (forall (i:natlt rows) (j:natlt cols).
                  vcell_of_pos (extended_layout l rows) i j == SZ.v off + j))
  : strided_row_major (extended_layout l rows) =
{
  offset = off;
  stride = 0sz;
  pf = (fun i j -> ());
}

inline_for_extraction noextract
instance strided_row_major_bcast_l1 (#rows #cols : erased nat)
  : strided_row_major (extended_layout (vtlayout_of_tlayout (l1_forward cols)) rows) =
{
  offset = 0sz;
  stride = 0sz;
  pf = (fun i j -> ());
}

let lemma_aligned_strided_row_major_bcast_l1 (#rows #cols : erased nat) (n : pos)
  : Lemma (ensures aligned_strided_row_major n (strided_row_major_bcast_l1 #rows #cols))
  = ()

let lemma_aligned_mk_strided_row_major_bcast
  (#rows #cols : erased nat)
  (l : vlayout1 cols) (off : sz) (pf : squash _) (n : pos)
  : Lemma (requires n /?+ SZ.v off)
          (ensures aligned_strided_row_major n
                     (mk_strided_row_major_bcast #rows #cols l off pf))
  = ()

(* Instance for l2_col_major: cell_of_pos = j * rows + i *)
inline_for_extraction noextract
instance strided_col_major_l2_col_major (#rows #cols : erased nat)
  (#_ : squash (rows > 0))
  {| d : concrete_sz rows |}
  : strided_col_major (vtlayout_of_tlayout (l2_col_major rows cols)) =
{
  offset = 0sz;
  stride = concr' d;
  pf = ez;
}
#pop-options

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
inline_for_extraction noextract
let strided_row_major_subtile_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (trows : sz {trows > 0 /\ trows /?+ rows})
  (tcols : sz {tcols > 0 /\ tcols /?+ cols})
  (tr    : szlt (rows / trows))
  (tc    : szlt (cols / tcols))
  : res : sz { SZ.v res == sub.offset + sub.stride * (tr * trows) + tc * tcols }
  = sub.pf (tr * trows) (tc * tcols);
    assert (cell_of_pos l (tr * trows) (tc * tcols) == sub.offset + sub.stride * (tr * trows) + tc * tcols);
    sub.offset +^ sub.stride *^ (tr *^ trows) +^ tc *^ tcols
#pop-options

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
let strided_row_major_subtile_proof
  (#rows #cols : nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (trows : nat {trows > 0 /\ trows /?+ rows})
  (tcols : nat {tcols > 0 /\ tcols /?+ cols})
  (tr    : natlt (rows / trows))
  (tc    : natlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  : Lemma (
      cell_of_pos (subtile_layout l trows tcols tr tc) i j ==
      sub.offset + sub.stride * (tr * trows) + tc * tcols
        + sub.stride * i + j
    )
=
  calc (==) {
    cell_of_pos (subtile_layout l trows tcols tr tc) i j <: int;
    == {}
    cell_of_pos l (tr * trows + i) (tc * tcols + j);
    == { sub.pf (tr * trows + i) (tc * tcols + j) }
    sub.offset + sub.stride * (tr * trows + i) + tc * tcols + j;
    == { FStar.Math.Lemmas.distributivity_add_right sub.stride (tr * trows) i }
    sub.offset + sub.stride * (tr * trows) + sub.stride * i + tc * tcols + j;
    == {}
      sub.offset + sub.stride * (tr * trows) + tc * tcols
        + sub.stride * i + j;
  };
  ()
#pop-options

inline_for_extraction noextract
instance strided_row_major_subtile (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_row_major (vtlayout_of_tlayout (subtile_layout l trows tcols tr tc)) =
{
  offset = strided_row_major_subtile_offset l (concr' c_trows) (concr' c_tcols) (concr' c_tr) (concr' c_tc);
  stride = sub.stride;
  pf = (fun i j -> strided_row_major_subtile_proof #rows #cols l trows tcols tr tc i j);
}

let lemma_subtile_strided_row_major_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            SZ.v (strided_row_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tr * trows) + tc * tcols)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).offset]
  = ()

let lemma_subtile_strided_row_major_stride
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_row_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            (strided_row_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_row_major_subtile l trows tcols tr tc).stride]
  = ()

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
inline_for_extraction noextract
let strided_col_major_subtile_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
  (trows : sz {trows > 0 /\ trows /?+ rows})
  (tcols : sz {tcols > 0 /\ tcols /?+ cols})
  (tr    : szlt (rows / trows))
  (tc    : szlt (cols / tcols))
  : res : sz { SZ.v res == sub.offset + sub.stride * (tc * tcols) + tr * trows }
  = sub.pf (tr * trows) (tc * tcols);
    assert (cell_of_pos l (tr * trows) (tc * tcols) == sub.offset + sub.stride * (tc * tcols) + tr * trows);
    sub.offset +^ sub.stride *^ (tc *^ tcols) +^ tr *^ trows
#pop-options

#push-options "--z3rlimit_factor 4 --fuel 0 --ifuel 0"
let strided_col_major_subtile_proof
  (#rows #cols : nat)
  (l : layout2 rows cols)
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
  (trows : nat {trows > 0 /\ trows /?+ rows})
  (tcols : nat {tcols > 0 /\ tcols /?+ cols})
  (tr    : natlt (rows / trows))
  (tc    : natlt (cols / tcols))
  (i : natlt trows)
  (j : natlt tcols)
  : Lemma (
      cell_of_pos (subtile_layout l trows tcols tr tc) i j ==
      sub.offset + sub.stride * (tc * tcols) + tr * trows
        + sub.stride * j + i
    )
=
  calc (==) {
    cell_of_pos (subtile_layout l trows tcols tr tc) i j <: int;
    == {}
    cell_of_pos l (tr * trows + i) (tc * tcols + j);
    == { sub.pf (tr * trows + i) (tc * tcols + j) }
    sub.offset + sub.stride * (tc * tcols + j) + (tr * trows + i);
    == { FStar.Math.Lemmas.distributivity_add_right sub.stride (tc * tcols) j }
    sub.offset + sub.stride * (tc * tcols) + sub.stride * j + (tr * trows + i);
    == {}
      sub.offset + sub.stride * (tc * tcols) + tr * trows
        + sub.stride * j + i;
  };
  ()
#pop-options

inline_for_extraction noextract
instance strided_col_major_subtile (#rows #cols : erased nat)
  (l : layout2 rows cols)
  (#_ : squash (SZ.fits (l.ulen)))
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : strided_col_major (vtlayout_of_tlayout (subtile_layout l trows tcols tr tc)) =
{
  offset = strided_col_major_subtile_offset l (concr' c_trows) (concr' c_tcols) (concr' c_tr) (concr' c_tc);
  stride = sub.stride;
  pf = (fun i j -> strided_col_major_subtile_proof #rows #cols l trows tcols tr tc i j);
}

let lemma_subtile_strided_col_major_offset
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            SZ.v (strided_col_major_subtile l trows tcols tr tc).offset
            ==
            sub.offset + sub.stride * (tc * tcols) + tr * trows)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).offset]
  = ()

let lemma_subtile_strided_col_major_stride
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| sub : strided_col_major (vtlayout_of_tlayout l) |}
  (trows : erased int {0 < trows /\ trows /?+ rows})
  (tcols : erased int {0 < tcols /\ tcols /?+ cols})
  (tr    : erased int {0 <= tr /\ tr < rows / trows})
  (tc    : erased int {0 <= tc /\ tc < cols / tcols})
  {| c_trows : concrete_sz trows,
     c_tcols : concrete_sz tcols,
     c_tr    : concrete_sz tr,
     c_tc    : concrete_sz tc,
  |}
  : Lemma (requires SZ.fits (l.ulen))
          (ensures
            (strided_col_major_subtile l trows tcols tr tc).stride
            ==
            sub.stride)
          [SMTPat (strided_col_major_subtile l trows tcols tr tc).stride]
  = ()
