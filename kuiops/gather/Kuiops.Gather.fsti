module Kuiops.Gather

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiops.Common

module SZ = Kuiper.SizeT

let gather_chest 
  (#et : Type0)
  (#r : erased nat)
  (d : shape r)
  (eInp : chest d et)
  (dim : natlt r)
  (eIdx : chest d (szlt (d @! dim))): chest d et
  = Kuiper.Chest.mk d
      (fun (x : abs d) ->
         let idx = acc eIdx x in
         let x' = abs_set_at dim idx x in
         acc eInp x')

(* LATER: technically, idx and out can be smaller than inp. PyTorch says:

'''
For a 3-D tensor the output is specified by:

out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

input and index must have the same number of dimensions. 
It is also required that index.size(d) <= input.size(d) for all dimensions d != dim. 
out will have the same shape as index. Note that input and index do not broadcast against each other. 
When index is empty, we always return an empty output with the same shape without further error checking.
'''

I read this to mean the i,j,k in the above example has the ranges of the dimensions of index, NOT of input. 
And `out` inherits the same shape as index. *)

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
      on gpu_loc (gOut |-> gather_chest d eInp dim eIdx)