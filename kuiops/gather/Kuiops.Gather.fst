module Kuiops.Gather

(* Asynchronous entry point for the verified Kuiper gather kernel. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Kernel.Gather

module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn gather_async
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do })
  (cdi : cshape di) (cdo : cshape do)
  (dim : szlt r)
  (#lInp : tlayout do) (#lIdx #lOut : tlayout di)
  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp : tensor et lInp {is_global gInp})
  (gIdx : tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut : tensor et lOut {is_global gOut})
  (n : sz {SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp : chest do et)
  (eIdx : chest di (szlt (do @! (SZ.v dim))))
  (s : stream_t)
  (#fInp #fIdx : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires on gpu_loc (gInp |-> Frac fInp eInp) ** on gpu_loc (gIdx |-> Frac fIdx eIdx) **
           on gpu_loc (live gOut)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gInp |-> Frac fInp eInp) ** (gIdx |-> Frac fIdx eIdx) **
                   (gOut |-> gather_chest di do eInp dim eIdx)))
{
  with eOut. assert on gpu_loc (gOut |-> eOut);
  launch (gather_kd di do cdi cdo dim gInp gIdx gOut n eInp eIdx #eOut #fInp #fIdx) s;
}
