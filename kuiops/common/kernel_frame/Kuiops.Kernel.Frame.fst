module Kuiops.Kernel.Frame

(* Framing a kernel descriptor.

   [launch_kernel_full] consumes a pledge for the whole of the next kernel's
   [full_pre], so chaining two launches requires the second descriptor's
   [full_pre] to name everything the first one's pledge carries -- including the
   operands the second kernel does not touch at all.  [desc_frame] adds such a
   passenger to a descriptor without disturbing it: [fr] goes straight into the
   descriptor's host-retained [frame], so it is never sent to the device and
   never appears in a per-thread precondition, and comes back out in
   [full_post].

   TODO: upstream to Kuiper. *)

#lang-pulse

open Kuiper
open Kuiper.ForEvery
open Kuiper.Kernel.Desc { kernel_desc }

ghost
fn framed_setup
  (#pre #post : slprop)
  (fr : slprop)
  (k : kernel_desc pre post)
  ()
  requires fr ** pre
  ensures (forall+ (bid : natlt k.nblk). k.block_pre bid) ** (fr ** k.frame)
{
  let st = k.setup;
  st ();
}

ghost
fn framed_teardown
  (#pre #post : slprop)
  (fr : slprop)
  (k : kernel_desc pre post)
  ()
  requires (forall+ (bid : natlt k.nblk). k.block_post bid) ** (fr ** k.frame)
  ensures fr ** post
{
  let td = k.teardown;
  td ();
}

inline_for_extraction noextract
let desc_frame
  (#pre #post : slprop)
  (fr : slprop)
  (k : kernel_desc pre post)
  : kernel_desc (fr ** pre) (fr ** post)
= { k with
    frame = fr ** k.frame;
    setup = framed_setup fr k;
    teardown = framed_teardown fr k;
  }
