module Kuiops.Kernel
#lang-pulse

open Pulse
open Kuiper.Base
open Kuiper.Kernel.Desc
open Kuiper.Kernel.Stream
open Pulse.Lib.Pledge
open Kuiops.Epoch

(* Trusted launch boundary for KuiLLM's stream-ordered epoch model.  The
   implementation is the same CUDA kernel launch emitted for Kuiper's
   [launch_kernel_full]; only the ownership protocol is project-specific. *)
inline_for_extraction noextract
fn launch
  (#pre #post : slprop)
  (k : kernel_desc pre post)
  (s : stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    on gpu_loc pre
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_flushed s (epoch_next e)) (on gpu_loc post)

(* As above, but consumes the result of earlier work on the same stream. *)
inline_for_extraction noextract
fn launch_pledged
  (#pre #post : slprop)
  (k : kernel_desc pre post)
  (s : stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    pledge0 (epoch_flushed s e) (on gpu_loc pre)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_flushed s (epoch_next e)) (on gpu_loc post)
