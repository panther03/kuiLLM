module Kuiops.Scatter

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiops.Common

module SZ = Kuiper.SizeT

(*

LATER: it would be great to have this, however I realized it will not work when going to 
the concrete layout, because the concrete layout requires reading from the gIdx tensor,
and that is an impure operation which we currently do not support in layouts.
We would need to carry around the permission to read gIdx in the layout, which is quite an
invasive change.

// A tensor layout for mapping indices through a lookup table.
// For now, the lookup table only stores 1 dimension of indices,
// but it could store tuples so that it maps N-D. 
// The LUT can be smaller than the mapped tensor. The resulting tensor 
// will then have the same shape as the LUT. 
let tlayout_lut
  (#r : erased nat) (#d1 #d2: shape r { shape_le d1 d2 })
  (dim: szlt r)
  (eIdx: chest d1 (szlt (d2 @! (SZ.v dim))))
  (eIdxInj: (i: (abs d1) -> (j: abs d1 {acc eIdx i == acc eIdx j}) -> squash (i == j)))
  (l: tlayout d2)
  : tlayout d1 = {
    ulen = sizeof d1;
    imap = {
      f = (fun i -> 
        let i' = abs_set_at2 d1 d2 dim (acc eIdx i) i in
        l.imap.f i');
      is_inj = (fun i j -> 
        l.imap.is_inj (abs_set_at2 d1 d2 dim (acc eIdx i) i) (abs_set_at2 d1 d2 dim (acc eIdx j) j);
          .... // TODO
        eIdxInj);
    }
  }
*)

inline_for_extraction noextract
fn scatter_gpu
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di) (cdo: cshape do)
  (dim: szlt r)
  (#lInp #lIdx: tlayout di) (#lOut: tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (n : sz{SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp: chest di et)
  (eInpInj: (i: (abs di) -> (j: abs di {acc eInp i == acc eInp j}) -> squash (i == j)))
  (eIdx: chest di (szlt (do @! (SZ.v dim))))
  (#fInp #fIdx: perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx eIdx)
  requires on gpu_loc (live gOut)
  ensures 
    exists* eOut. 
      on gpu_loc (gOut |-> eOut) **
      pure (vscatter_chest di do dim eInp eIdx eOut) {
  admit ()
}