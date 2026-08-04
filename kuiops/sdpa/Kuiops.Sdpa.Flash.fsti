module Kuiops.Sdpa.Flash

#lang-pulse

open Kuiper
open Kuiper.Floating
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.TensorCore
open Pulse.Lib.Pledge
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts

inline_for_extraction noextract
fn sdpa_flash_async
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (nblk : szp { SZ.v nblk <= max_blocks })
  (nw nthr : szp {
    SZ.v nthr == block_threads nw /\
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
  (#lgQ : layout4 b hq sq d)
  (#lgK #lgV : layout4 b hkv sk d)
  (#lgmask : layout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| ctlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : array4 et_ab lgmask { Kuiper.Tensor.is_global gmask })
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (scale : et_acc)
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v nw * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * 16)))
  (#_ : squash (SZ.fits (SZ.v nw * 16 * SZ.v d)))
  (#_ : squash (SZ.fits (16 * SZ.v d + SZ.v nthr)))
  (#_ : squash (SZ.fits (16 * SZ.v d + BW.warp_size)))
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (#_ : squash (SZ.fits (SZ.v sk + 32)))
  (#_ : squash (SZ.fits (SZ.v sk + 32 + SZ.v nw)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  (#fQ #fK #fV #fmask : perm)
  (#eQ : chest (b @| hq @| sq @| d @| INil) et_ab)
  (#eK #eV : chest (b @| hkv @| sk @| d @| INil) et_ab)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (s : stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (
      (gQ |-> Frac fQ eQ) **
      (gK |-> Frac fK eK) **
      (gV |-> Frac fV eV) **
      (gmask |-> Frac fmask emask) **
      live gout)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (
        (gQ |-> Frac fQ eQ) **
        (gK |-> Frac fK eK) **
        (gV |-> Frac fV eV) **
        (gmask |-> Frac fmask emask) **
        live gout))
