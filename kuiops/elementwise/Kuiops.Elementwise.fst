module Kuiops.Elementwise

(* Asynchronous entry points for the verified Kuiper pointwise-map kernels.

   The blocking [Kuiper.Kernel.Map.map_gpu*] wrappers create their own stream and
   sync on it, which is illegal under CUDA graph capture. These take the caller's
   stream instead and only [launch], so the result is owned by a pledge redeemed
   when that stream's epoch completes -- exactly the contract the C boundary
   needs, since Torch orders the consumer on the same stream. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

inline_for_extraction noextract
fn map_async
  (#et : Type0)
  (f : et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (s : stream_t)
  (#sa : chest1 et lena)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires on gpu_loc (a |-> sa)
  ensures  Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e)) (on gpu_loc (a |-> chest_map f sa))
{
  Kuiops.Kernel.launch (map_kd f lena a #sa) s;
}

inline_for_extraction noextract
fn map_to_async
  (#it #ot : Type0)
  (f : it -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#li : layout1 lena) {| ctlayout li |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (input : array1 it li { is_global input })
  (output : array1 ot lo { is_global output })
  (s : stream_t)
  (#si : chest1 it lena)
  (#so : chest1 ot lena)
  (#fi : perm)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires on gpu_loc (input |-> Frac fi si) ** on gpu_loc (output |-> so)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc ((input |-> Frac fi si) **
                   (output |-> mk1 (fun i -> f (acc1 si i)))))
{
  Kuiops.Kernel.launch (map_to_kd f lena input output #si #so #fi) s;
}

inline_for_extraction noextract
fn map2_async
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la { is_global a })
  (b : array1 et lb { is_global b })
  (s : stream_t)
  (#sa #sb : chest1 et lena)
  (#fb : perm)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires on gpu_loc (b |-> Frac fb sb) ** on gpu_loc (a |-> sa)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc ((b |-> Frac fb sb) ** (a |-> chest1_map2 f sa sb)))
{
  Kuiops.Kernel.launch (map2_kd f lena a b #sa #sb #fb) s;
}

inline_for_extraction noextract
fn map3_to_async
  (#at #bt #ct #ot : Type0)
  (f : at -> bt -> ct -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (#lc : layout1 lena) {| ctlayout lc |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (a : array1 at la { is_global a })
  (b : array1 bt lb { is_global b })
  (c : array1 ct lc { is_global c })
  (output : array1 ot lo { is_global output })
  (s : stream_t)
  (#sa : chest1 at lena) (#sb : chest1 bt lena) (#sc : chest1 ct lena)
  (#so : chest1 ot lena)
  (#fa #fb #fc : perm)
  (#e : Kuiops.Epoch.epoch_t)
  preserves cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires
    on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb) **
    on gpu_loc (c |-> Frac fc sc) ** on gpu_loc (output |-> so)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc ((a |-> Frac fa sa) ** (b |-> Frac fb sb) ** (c |-> Frac fc sc) **
                   (output |-> mk1 (fun i -> f (acc1 sa i) (acc1 sb i) (acc1 sc i)))))
{
  Kuiops.Kernel.launch (map3_to_kd f lena a b c output #sa #sb #sc #so #fa #fb #fc) s;
}
