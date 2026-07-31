module Kuiops.Sdpa.Flash.Inst

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout.Alg
open Kuiops.Sdpa.Flash

module SZ = Kuiper.SizeT
module FD = Kuiops.Sdpa.Flash.KernelDesc

let sdpa_flash_bf16_f32
  (nblk : szp { SZ.v nblk <= max_blocks })
  (nw nthr : szp {
    SZ.v nthr == FD.block_threads nw /\
    SZ.v nthr <= max_threads })
  (b hq hkv group sq rows tiles sk : szp {
    SZ.v nblk == SZ.v b * SZ.v hkv * SZ.v tiles /\
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq /\
    SZ.fits (SZ.v rows + 15) /\
    SZ.v tiles == (SZ.v rows + 15) / 16 /\
    SZ.v rows <= SZ.v tiles * 16 /\
    SZ.fits (SZ.v tiles * 16) /\
    SZ.fits (SZ.v hkv * SZ.v group + SZ.v rows) })
  (d : szp { 16 /?+ SZ.v d })
  (#_ : squash (SZ.fits (SZ.v sq * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v hq * (SZ.v sq * SZ.v d))))
  (#_ : squash (SZ.fits (SZ.v b * (SZ.v hq * (SZ.v sq * SZ.v d)))))
  (#_ : squash (SZ.fits (SZ.v sk * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v hkv * (SZ.v sk * SZ.v d))))
  (#_ : squash (SZ.fits (SZ.v b * (SZ.v hkv * (SZ.v sk * SZ.v d)))))
  (#_ : squash (SZ.fits (SZ.v sq * SZ.v sk)))
  (#_ : squash (SZ.fits (SZ.v hq * (SZ.v sq * SZ.v sk))))
  (#_ : squash (SZ.fits (SZ.v b * (SZ.v hq * (SZ.v sq * SZ.v sk))))) =
  sdpa_flash_async
    #Kuiper.BFloat16.t #Kuiper.Float32.t
    nblk nw nthr b hq hkv group sq rows tiles sk d
    #(l4_batched_row_major b hq sq d)
    #(l4_batched_row_major b hkv sk d)
    #(l4_batched_row_major b hkv sk d)
    #(l4_batched_row_major b hq sq sk)
    #(l4_batched_row_major b hq sq d)
    #(c_l4_batched_row_major _ hq sq d)
    #(c_l4_batched_row_major _ hkv sk d)
    #(c_l4_batched_row_major _ hkv sk d)
    #(c_l4_batched_row_major _ hq sq sk)
    #(c_l4_batched_row_major _ hq sq d)
