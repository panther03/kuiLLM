module Kuiops.Bmm

#lang-pulse
open Kuiper
module K = Kuiper.Kernel.BatchedGEMM
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

(* Row-major batched matmul. [comb2 x y = y] discards the accumulator, so the
   passed-in output buffer is fully overwritten with the batched product. *)
inline_for_extraction noextract
let bmm_rm
  (#et : Type0) {| scalar et |}
  (batch rows shared cols : szp {
    SZ.fits (batch * rows * shared) /\
    SZ.fits (batch * shared * cols) /\
    SZ.fits (batch * rows * cols) })
  =
    K.bmmcomb_gpu_exact #et #_ (MS.comb2) batch rows shared cols
    #(l3_batched_row_major batch rows shared)
    #(l3_batched_row_major batch shared cols)
    #(l3_batched_row_major batch rows cols)
