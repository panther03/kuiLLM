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

val gather_frame (#et : Type0) (#r : erased nat) (d : shape r) (cd: cshape d)
  (dim: szlt r)
  (#lInp #lIdx #lOut: tlayout d)  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (d @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (fr: perm): slprop // TODO

fn fgather (#et : Type0) (#r : erased nat) (d : shape r) (cd: cshape d)
  (dim: szlt r)
  (#lInp #lIdx #lOut: tlayout d)  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (d @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (#fInp #fIdx: perm)
{
  admit ();
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
  
  admit ();
}