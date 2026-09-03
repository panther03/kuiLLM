module Kuiops.Array2.Vectorized.Pipelined

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Chest
open Kuiops.PipelineCopy
open Pulse.Lib.Pledge

open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiops.Array2.Strided
open Kuiops.Array2.Vectorized { row_cells, row_cells_to_slice, row_slice_to_cells }
open Kuiper.TensorRO { vtlayout_of_tlayout }

module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn array2_vec_cpy_pipelined
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#drows #dcols : erased nat)
  (#dl : layout2 drows dcols)
  {| T.ctlayout dl, dstrided : strided_row_major (vtlayout_of_tlayout dl) |}
  (dm : array2 et dl)
  (di : szlt drows)
  (dj : szlt (dcols - chunk et + 1))
  (#srows #scols : erased nat)
  (#sl : layout2 srows scols)
  {| T.ctlayout sl, sstrided : strided_row_major (vtlayout_of_tlayout sl) |}
  (sm : array2 et sl)
  (si : szlt srows)
  (sj : szlt (scols - chunk et + 1))
  (#f : perm)
  (#dold #sv : erased (seq et))
  (#_ : squash (Seq.length dold == chunk et /\ Seq.length sv == chunk et))
  (#b : pipeline_batch_t)
  ()
  preserves gpu
  preserves batch_live b
  requires  row_cells dm 1.0R (SZ.v di) (SZ.v dj) (chunk et) dold
  requires  row_cells sm f (SZ.v si) (SZ.v sj) (chunk et) sv
  requires  pure (aligned' 16 (T.core dm) (cell_of_pos dl (SZ.v di) (SZ.v dj)))
  requires  pure (aligned' 16 (T.core sm) (cell_of_pos sl (SZ.v si) (SZ.v sj)))
  ensures   pledge0 (batch_done b)
              (row_cells dm 1.0R (SZ.v di) (SZ.v dj) (chunk et) sv **
               row_cells sm f (SZ.v si) (SZ.v sj) (chunk et) sv)
{
  row_cells_to_slice dm (SZ.v di) (SZ.v dj) (chunk et) dold;
  row_cells_to_slice sm (SZ.v si) (SZ.v sj) (chunk et) sv;

  dstrided.pf (SZ.v di) (SZ.v dj);
  sstrided.pf (SZ.v si) (SZ.v sj);

  let doff = dstrided.offset +^ dstrided.stride *^ di +^ dj;
  let soff = sstrided.offset +^ sstrided.stride *^ si +^ sj;

  array_vec_cpy_pipelined (T.core dm) doff (T.core sm) soff;

  with s'. assert
    pledge0 (batch_done b)
      ((pts_to_slice (T.core dm)
          (cell_of_pos dl (SZ.v di) (SZ.v dj))
          (cell_of_pos dl (SZ.v di) (SZ.v dj) + chunk et) s') **
       (pts_to_slice (T.core sm) #f
          (cell_of_pos sl (SZ.v si) (SZ.v sj))
          (cell_of_pos sl (SZ.v si) (SZ.v sj) + chunk et) sv));

  assert pure (Seq.equal s' sv);

  rewrite_pledge
    ((pts_to_slice (T.core dm)
        (cell_of_pos dl (SZ.v di) (SZ.v dj))
        (cell_of_pos dl (SZ.v di) (SZ.v dj) + chunk et) s') **
     (pts_to_slice (T.core sm) #f
        (cell_of_pos sl (SZ.v si) (SZ.v sj))
        (cell_of_pos sl (SZ.v si) (SZ.v sj) + chunk et) sv))
    (row_cells dm 1.0R (SZ.v di) (SZ.v dj) (chunk et) sv **
     row_cells sm f (SZ.v si) (SZ.v sj) (chunk et) sv)
    #emp_inames
    fn _ {
      row_slice_to_cells dm (SZ.v di) (SZ.v dj) (chunk et) sv;
      row_slice_to_cells sm (SZ.v si) (SZ.v sj) (chunk et) sv;
      ();
    };

  ();
}
#pop-options
