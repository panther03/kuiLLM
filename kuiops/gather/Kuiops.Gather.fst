module Kuiops.Gather

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Shareable

open Kuiper.Kernel.TMap

module SZ = Kuiper.SizeT

let vfgather (#et: Type0) (#r : erased nat) (d : shape r) (dim : natlt r)
  (eInp : chest d et) (eIdx : chest d (szlt (d @! dim)))
  (x : abs d) (_: et) (o: et)
  : prop 
  = o == acc eInp (set_at dim (acc eIdx x) x)

unfold
let gather_frame (#et : Type0) (#r : erased nat) (d : shape r) (cd: cshape d)
  (dim: szlt r)
  (#lInp #lIdx: tlayout d)  {| ctlayout lInp, ctlayout lIdx |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (d @! (SZ.v dim))) lIdx {is_global gIdx})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (fInp fIdx: perm)
  (fr: perm): slprop =
    (tensor_pts_to gInp #((fInp /. (fInp +. fIdx)) *. fr) eInp) **
    (tensor_pts_to gIdx #((fIdx /. (fInp +. fIdx)) *. fr) eIdx) 

let gather_frame_shareable
  (#et : Type0) (#r : erased nat) (d : shape r) (cd: cshape d)
  (dim: szlt r)
  (#lInp #lIdx: tlayout d)  {| ctlayout lInp, ctlayout lIdx|}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (d @! (SZ.v dim))) lIdx {is_global gIdx})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (fInp fIdx: perm):
  GTot (shareable (gather_frame d cd dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx)) = 
  double_shareable 
  (fun fr -> tensor_pts_to gInp #fr eInp) 
  (fun fr -> tensor_pts_to gIdx #fr eIdx)
  (fInp /. (fInp +. fIdx)) (fIdx /. (fInp +. fIdx))

inline_for_extraction noextract
let rec set_at_conc (#r : erased nat) (#d : shape r) (dim : szlt r) (idx : szlt (d @! dim)) (x : conc d)
  : Tot (c : conc d {up c == set_at dim idx (up x)})
         (decreases (SZ.v dim))
  = assert r > 0;
    let (i1,i2) = x <: szlt (d @! 0) & conc (tail d) in
    if dim = 0sz then
      (idx, i2)
    else (
      let x1 = set_at_conc #_ #(tail d) (dim -^ 1sz) idx i2 in
      let x2: conc d = (i1, (x1 <: (conc (tail d)))) in
      x2
    )

fn fgather (#et : Type0) (#r : erased nat) (d : shape r) (cd: cshape d)
  (dim: szlt r)
  (#lInp #lIdx #lOut: tlayout d)  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (d @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (#fInp #fIdx: perm)
  (#fr: perm) (i: conc d) (x : et) 
preserves
  (gather_frame d cd dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr)
returns 
  r: et
ensures
  pure (vfgather d dim eInp eIdx (up i) x r)
{
  let idx = tensor_read gIdx i;
  let idx' = set_at_conc dim idx i;
  let inp = tensor_read gInp idx';

  return inp;
}

inline_for_extraction noextract
fn gather_gpu
  (#et : Type0) (#r : erased nat) (d : shape r) (cd: cshape d)
  (dim: szlt r)
  (#lInp #lIdx #lOut: tlayout d)  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (d @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (#fInp #fIdx: perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx eIdx)
  requires on gpu_loc (live gOut)
  ensures 
      on gpu_loc (gOut |-> gather_chest d eInp dim eIdx) {
  (*launch_sync 
    (kmap cd 
      (gather_frame d cd dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr)
      #(gather_frame_shareable ) (vf_equal f) (ff_from_pure f) n a #s #_ #1.0R);*)
  admit ();
}