module Kuiops.Scatter

(* Asynchronous entry point for the verified Kuiper scatter kernel. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Kernel.Scatter

module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn scatter_async
  (#et : Type0) (#r : erased nat) (di do : shape r { shape_le di do })
  (cdi : cshape di) (cdo : cshape do)
  (dim : szlt r)
  (#lInp #lIdx : tlayout di) (#lOut : tlayout do)
  {| ctlayout lInp, ctlayout lIdx, ctlayout lOut |}
  (gInp : tensor et lInp {is_global gInp})
  (gIdx : tensor (szlt (do @! (SZ.v dim))) lIdx {is_global gIdx})
  (gOut : tensor et lOut {is_global gOut})
  (n : sz {SZ.v n == sizeof di /\ n <= max_blocks * max_threads /\ n > 0})
  (eInp : chest di et)
  (eIdx : chest di (szlt (do @! (SZ.v dim))) { chest_inj di do (SZ.v dim) eIdx })
  (s : stream_t)
  (#fInp #fIdx : perm)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires on gpu_loc (gInp |-> Frac fInp eInp) **
           on gpu_loc (gIdx |-> Frac fIdx (eIdx <: chest di (szlt (do @! (SZ.v dim))))) **
           on gpu_loc (live gOut)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc ((gInp |-> Frac fInp eInp) **
                   (gIdx |-> Frac fIdx (eIdx <: chest di (szlt (do @! (SZ.v dim))))) **
                   (exists* eOut. (gOut |-> eOut) **
                      pure (vscatter_chest di do (SZ.v dim) eInp eIdx eOut))))
{
  with eOut. assert on gpu_loc (gOut |-> eOut);
  Kuiops.Kernel.launch (scatter_kd di do cdi cdo dim gInp gIdx gOut n eInp eIdx #eOut #fInp #fIdx) s;
}
