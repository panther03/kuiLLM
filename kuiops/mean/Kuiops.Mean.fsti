module Kuiops.Mean

(* torch.mean(input, dim, keepdim=True): reduce a single axis `dim` of an
   n-dimensional tensor by summing over it and dividing by its length, keeping
   that axis with size 1. Modelled as a pointwise map over the (size-1-on-`dim`)
   output shape `dOut`; each output cell sums the `dInp @! dim` input cells that
   share its other coordinates and divides by `divisor`.

   The reduction and division are carried out with the *exact* element-type
   operations (no real-number approximation), so the spec is the same fold the
   kernel computes; agreement with PyTorch's mean (including that `divisor` is
   the float of the reduced length) is a property of the instantiation, checked
   empirically, not of this proof. *)

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiops.Common

module SZ = Kuiper.SizeT

(* Exact element-type sum of the first `k` cells along `dim`, for the input
   coordinate that agrees with output coordinate `x` off `dim`. Total in `k`
   (cells at or beyond the reduced length are skipped), so callers need not
   carry a bound to keep it well-typed. *)
let rec mean_sum (#et : Type0) {| scalar et |} (#r : nat)
  (dOut dInp : shape r { shape_le dOut dInp })
  (dim : natlt r) (eInp : chest dInp et) (x : abs dOut) (k : nat)
  : GTot et (decreases k)
  = if k = 0 then zero
    else
      let prev = mean_sum dOut dInp dim eInp x (k - 1) in
      if k - 1 < dInp @! dim
      then add prev (acc eInp (abs_set_at2 dOut dInp dim (k - 1) x))
      else prev

let mean_val (#et : Type0) {| floating et |} (#r : nat)
  (dOut dInp : shape r { shape_le dOut dInp })
  (dim : natlt r) (eInp : chest dInp et) (divisor : et) (x : abs dOut) : GTot et
  = div (mean_sum dOut dInp dim eInp x (dInp @! dim)) divisor

let mean_chest (#et : Type0) {| floating et |} (#r : nat)
  (dOut dInp : shape r { shape_le dOut dInp })
  (dim : natlt r) (divisor : et) (eInp : chest dInp et) : chest dOut et
  = Kuiper.Chest.mk dOut (mean_val dOut dInp dim eInp divisor)

inline_for_extraction noextract
fn mean_gpu
  (#et : Type0) {| floating et |} (#r : nat)
  (dOut dInp : shape r { shape_le dOut dInp })
  (cdOut : cshape dOut) (cdInp : cshape dInp)
  (dim : szlt r)
  (len : sz { SZ.v len == dInp @! dim })
  (divisor : et)
  (#lInp : tlayout dInp) (#lOut : tlayout dOut)
  {| ctlayout lInp, ctlayout lOut |}
  (gInp : tensor et lInp { is_global gInp })
  (gOut : tensor et lOut { is_global gOut })
  (n : sz { SZ.v n == sizeof dOut /\ n <= max_blocks * max_threads /\ n > 0 })
  (eInp : chest dInp et)
  (#fInp : perm)
  preserves cpu ** on gpu_loc (gInp |-> Frac fInp eInp)
  requires on gpu_loc (live gOut)
  ensures
    on gpu_loc (gOut |-> mean_chest dOut dInp dim divisor eInp)
