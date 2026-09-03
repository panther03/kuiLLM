module Kuiops.Epoch
#lang-pulse

open Pulse
open Kuiper.Base
open Kuiper.Kernel.Stream
open Pulse.Lib.Pledge

module KE = Kuiper.Epoch

(* A position in a stream's work queue.  This stronger, stream-ordered epoch
   model is specific to KuiLLM for now; keeping it here makes that assumption
   explicit instead of silently replacing the binary Kuiper package's API. *)
type epoch_t = erased nat

unfold
let epoch_next (e : epoch_t) : epoch_t = hide (reveal e + 1)

(* Exclusive ownership of the current queue position. *)
val epoch_live (s : stream_t) (e : epoch_t) : slprop

(* Duplicable evidence that work before [e] has arrived in stream order. *)
val epoch_flushed (s : stream_t) (e : epoch_t) : slprop

(* Trusted bridges between kuiLLM's stream-ordered epoch model and Kuiper's
   launch primitive. They are ghost-only: extraction erases them, leaving the
   packaged [launch_kernel_full] call for Kuiper's extraction plugin. *)
ghost
fn prepare_launch
  (#p : slprop)
  (s : stream_t)
  (e : epoch_t)
  preserves cpu ** stream_live s ** on gpu_loc p
  requires epoch_live s e
  ensures KE.epoch_live s e

ghost
fn prepare_pledged_launch
  (#p : slprop)
  (s : stream_t)
  (e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    pledge0 (epoch_flushed s e) (on gpu_loc p)
  ensures
    KE.epoch_live s e **
    on gpu_loc p

ghost
fn finish_launch
  (#p : slprop)
  (s : stream_t)
  (e : epoch_t)
  preserves cpu ** stream_live s
  requires
    KE.epoch_live s e **
    pledge0 (KE.epoch_done s e) (on gpu_loc p)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_flushed s (epoch_next e)) (on gpu_loc p)
