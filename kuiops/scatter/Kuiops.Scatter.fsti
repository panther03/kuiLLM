module Kuiops.Scatter

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiops.Common

module SZ = Kuiper.SizeT

let vscatter (#et: Type0) (#r : erased nat) 
  (di do: shape r {shape_le di do}) (dim : natlt r)
  (eInp : chest di et) (eIdx : chest di (szlt (do @! dim))) (eOut : chest do et)
  (i : abs di)
  : prop 
  = acc eOut (abs_set_at2 di do dim (acc eIdx i) i) == acc eInp i

let vscatter_chest
  (#et : Type0)
  (#r : erased nat)
  (di do: shape r {shape_le di do})
  (dim : natlt r)
  (eInp : chest di et) 
  (eIdx : chest di (szlt (do @! dim)))
  (eOut : chest do et): prop
  = chest_foralli (fun i _ -> vscatter di do dim eInp eIdx eOut i) eInp

// TODO: review this is correct against torch specification
inline_for_extraction noextract
fn scatter_gpu
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do }) (cdi: cshape di) (cdo: cshape do)
  (dim: szlt r)
  (#lInp #lIdx: tlayout di) (#lOut: tlayout do) {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp: tensor et lInp {is_global gInp})
  (gIdx: tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut: tensor et lOut {is_global gOut})
  // should be di or do?
  (n : sz{SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp: chest di et)
  (eIdx: chest di (szlt (do @! (SZ.v dim))))
  (#fInp #fIdx: perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx eIdx)
  requires on gpu_loc (live gOut)
  ensures 
    exists* eOut. 
      on gpu_loc (gOut |-> eOut) **
      pure (vscatter_chest di do dim eInp eIdx eOut)