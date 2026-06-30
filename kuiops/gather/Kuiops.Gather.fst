module Kuiops.Gather

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Shareable
open Kuiops.Common

open Kuiper.Kernel.TMap

module SZ = Kuiper.SizeT

let vfgather (#et: Type0) (#r : erased nat) (d : shape r) (dim : natlt r)
  (eInp : chest d et) (eIdx : chest d (szlt (d @! dim)))
  (x : abs d) (_: et) (o: et)
  : prop 
  = o == acc eInp (abs_set_at dim (acc eIdx x) x)

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

instance gather_frame_shareable
  (#et : Type0) (#r : erased nat) (d : shape r) (cd: cshape d)
  (dim: szlt r)
  (#lInp #lIdx: tlayout d)  {| ctlayout lInp, ctlayout lIdx|}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (d @! (SZ.v dim))) lIdx {is_global gIdx})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (fInp fIdx: perm):
  shareable (gather_frame d cd dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx) = magic ()
(*
  WHY IS CALLING double_shareable A GHOST EFFECT??
  
  double_shareable 
  (fun fr -> tensor_pts_to gInp #fr eInp) 
  (fun fr -> tensor_pts_to gIdx #fr eIdx)
  (fInp /. (fInp +. fIdx)) (fIdx /. (fInp +. fIdx))
*)
  
inline_for_extraction noextract
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
norewrite
preserves
  (gather_frame d cd dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx fr)
returns 
  r: et
ensures
  pure (vfgather d dim eInp eIdx (up i) x r)
{
  let idx = tensor_read gIdx i;
  let idx' = conc_set_at dim idx i;
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
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp: chest d et)
  (eIdx: chest d (szlt (d @! (SZ.v dim))))
  (#fInp #fIdx: perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx eIdx)
  requires on gpu_loc (live gOut)
  ensures 
      on gpu_loc (gOut |-> gather_chest d eInp dim eIdx) {
  with eOut. assert on gpu_loc (gOut |-> eOut);
  let kfun = (kmap cd
      (gather_frame d cd dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx)
      #(gather_frame_shareable d cd dim #lInp #lIdx gInp gIdx eInp eIdx fInp fIdx)
      (vfgather d dim eInp eIdx) 
      (fgather d cd dim #lInp #lIdx #lOut gInp gIdx gOut eInp eIdx #fInp #fIdx)
      n gOut #eOut #_ #(fInp +. fIdx));
  launch_sync 
    kfun;
  with eOut'. assert on gpu_loc (gOut |-> eOut');
  assert pure (Kuiper.Chest.equal eOut' (gather_chest d eInp dim eIdx)); 
}