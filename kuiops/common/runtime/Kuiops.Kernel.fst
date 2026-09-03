module Kuiops.Kernel
#lang-pulse

open Pulse
open Kuiper.Base
open Kuiper.Kernel.Desc
open Kuiper.Kernel.Stream
open Pulse.Lib.Pledge
open Kuiops.Epoch

(* The ghost bridges account for kuiLLM's stronger queue-position model. The
   only computational operation is Kuiper's packaged launch primitive, whose
   extraction plugin emits the CUDA launch. *)
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
{
  prepare_launch s e;
  Kuiper.Kernel.Base.launch_kernel_full k s;
  finish_launch s e;
}

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
{
  prepare_pledged_launch s e;
  Kuiper.Kernel.Base.launch_kernel_full k s;
  finish_launch s e;
}
