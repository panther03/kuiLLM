module Kuiops.Cat

(* Asynchronous entry point for the verified Kuiper cat kernel: launches on the
   caller's stream without syncing, so it is legal under CUDA graph capture.
   Ownership of every operand moves into the pledge redeemed at epoch end. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Kernel.Cat

module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn cat_async
  (#et : Type0) (#r : nat)
  (dA dB dout : shape r)
  (cdA : cshape dA) (cdB : cshape dB) (cdout : cshape dout)
  (dim : natlt r) (dimsz : szlt r { SZ.v dimsz == dim })
  (na : sz { SZ.v na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (#lA : tlayout dA) (#lB : tlayout dB) (#lOut : tlayout dout)
  {| ctlayout lA, ctlayout lB, ctlayout lOut |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (gOut : tensor et lOut {is_global gOut})
  (n : sz {SZ.v n == sizeof dout /\ n <= max_blocks * max_threads /\ n > 0})
  (eA : chest dA et)
  (eB : chest dB et)
  (s : stream_t)
  (#fA #fB : perm)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB) **
           on gpu_loc (live gOut)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
                   (gOut |-> cat_chest dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB)))
{
  with eOut. assert on gpu_loc (gOut |-> eOut);
  Kuiops.Kernel.launch (cat_kd dA dB dout cdA cdB cdout dim dimsz na pf_sz pfA pfB
            gA gB gOut n eA eB #eOut #fA #fB) s;
}
