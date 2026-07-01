module Kuiops.Mean

(* Implementation of the verified torch.mean(.., dim, keepdim=True) kernel; see
   Kuiops.Mean.fsti for the public spec (mean_sum / mean_val / mean_chest) and
   the mean_gpu signature. The per-element function folds the reduced axis with
   the scalar `add`, then divides by `divisor`; the kernel maps it over the
   output via kmap, exactly like Kuiops.Cat maps its copy. *)

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Shareable
open Kuiops.Common

open Kuiper.Kernel.TMap

module SZ = Kuiper.SizeT

(* The per-element value relation linking mean_val to the map's contract. *)
let vmean (#et : Type0) {| floating et |} (#r : nat)
  (dOut dInp : shape r { shape_le dOut dInp })
  (dim : natlt r) (eInp : chest dInp et) (divisor : et)
  (x : abs dOut) (_old : et) (o : et) : prop
  = o == mean_val dOut dInp dim eInp divisor x

(* Per-element function: sum the reduced axis for one output coordinate and
   divide by `divisor`. The single read-only input is held by the frame at a
   fraction `fr`; each `tensor_read` only borrows it. *)
inline_for_extraction noextract
fn fmean
  (#et : Type0) {| floating et |} (#r : nat)
  (dOut dInp : shape r { shape_le dOut dInp })
  (cdOut : cshape dOut) (cdInp : cshape dInp)
  (dim : szlt r)
  (len : sz { SZ.v len == dInp @! dim })
  (divisor : et)
  (#lInp : tlayout dInp) {| ctlayout lInp |}
  (gInp : tensor et lInp { is_global gInp })
  (eInp : chest dInp et)
  (#fr : perm)
  (i : conc dOut) (x : et)
norewrite
preserves
  (tensor_pts_to gInp #fr eInp)
returns
  res : et
ensures
  pure (vmean dOut dInp (SZ.v dim) eInp divisor (up i) x res)
{
  let mut sumv : et = zero;
  let mut k : sz = 0sz;
  while (!k <^ len)
    invariant
      live sumv ** live k **
      pure (SZ.v !k <= SZ.v len /\
            !sumv == mean_sum dOut dInp (SZ.v dim) eInp (up i) (SZ.v !k))
    decreases (SZ.v len - SZ.v !k)
  {
    let kc = !k;
    let kidx : szlt (dInp @! SZ.v dim) = kc;
    let iidx = conc_set_at2 cdOut cdInp dim kidx i;
    let v = tensor_read gInp iidx;
    assert (pure (up iidx == abs_set_at2 dOut dInp (SZ.v dim) (SZ.v kc) (up i)));
    assert (pure (v == acc eInp (abs_set_at2 dOut dInp (SZ.v dim) (SZ.v kc) (up i))));
    sumv := !sumv `add` v;
    k := kc +^ 1sz;
  };
  let s = !sumv;
  div s divisor
}

(* The single read-only input frame; shareable across threads via the library
   instance for tensor points-to. *)
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
{
  with eOut. assert on gpu_loc (gOut |-> eOut);
  launch_sync
    (kmap cdOut
      (fun fr -> tensor_pts_to gInp #fr eInp)
      #(tensor_pts_to_shareable gInp eInp)
      (vmean dOut dInp (SZ.v dim) eInp divisor)
      (fmean dOut dInp cdOut cdInp dim len divisor #lInp gInp eInp)
      n gOut #eOut #_ #fInp);
  with eOut'. assert on gpu_loc (gOut |-> eOut');
  assert pure (Kuiper.Chest.equal eOut'
                 (mean_chest dOut dInp dim divisor eInp));
}
