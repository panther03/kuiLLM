module Kuiops.Sdpa

#lang-pulse
open Kuiper
open Kuiper.Tensor.Layout.Alg
open Kuiper.Kernel.SDPA.LSE
module SZ = Kuiper.SizeT

(* Polymorphic, row-major instantiation of the verified SDPA (efficient
   attention) kernel. Layouts are fixed to batched-row-major (PyTorch's default
   contiguous (N, H, L, E) layout); the element type stays polymorphic. *)
inline_for_extraction noextract
let sdpa_lse_rm
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (n h : szp)
  (l s : szp)
  (e ev : szp { SZ.fits (n * h * l * e) /\ SZ.fits (n * h * s * e) /\
                SZ.fits (n * h * s * ev) /\ SZ.fits (n * h * l * s) }) =
  sdpa_lse_naive #et n h l s e ev
  #(l4_batched_row_major n h l e)
  #(l4_batched_row_major n h s e)
  #(l4_batched_row_major n h s ev)
  #(l4_batched_row_major n h l s)
  #(c_l4_batched_row_major _ h l e)
  #(c_l4_batched_row_major _ h s e)
  #(c_l4_batched_row_major _ h s ev)
  #(c_l4_batched_row_major _ h l s)
