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
module TRO = Kuiper.TensorRO
module BW = Kuiper.Barrier.Warp
module FC = Kuiper.Float.Casts
module FSp = Kuiops.Sdpa.Flash.Split
module FSpec = Kuiops.Sdpa.Flash.Spec

inline_for_extraction noextract
fn sdpa_flash_async
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |} {| floating_real_like et_acc |}
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
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lout : layout4 b hq sq d)
  {| ctlayout lgQ |} {| ctlayout lgK |} {| ctlayout lgV |}
  {| TRO.cvtlayout lgmask |} {| ctlayout lout |}
  (gQ : array4 et_ab lgQ { Kuiper.Tensor.is_global gQ })
  (gK : array4 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array4 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : TRO.roarray4 et_ab lgmask { TRO.is_global gmask })
  (gout : array4 et_ab lout { Kuiper.Tensor.is_global gout })
  (causal : bool) (has_mask : bool) (scale : et_acc)
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
  preserves cpu ** stream_live s
  requires epoch_live s e
  requires
    on gpu_loc (
      (gQ |-> Frac fQ eQ) **
      (gK |-> Frac fK eK) **
      (gV |-> Frac fV eV) **
      (gmask |-> Frac fmask emask) **
      live gout) **
    pure (FSpec.sdpa_flash_finite (SZ.v group) (SZ.v rows)
            eQ eK emask has_mask causal scale)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_flushed s (epoch_next e))
      (on gpu_loc (
        (gQ |-> Frac fQ eQ) **
        (gK |-> Frac fK eK) **
        (gV |-> Frac fV eV) **
        (gmask |-> Frac fmask emask) **
        (gout |-> Frac 1.0R
           (FSp.flash_out_chest b hq hkv group sq rows d
              (flash_out_vfun nw d b hq hkv group sq rows sk eQ eK eV emask has_mask causal scale))))) **
    pure (FSp.flash_out_chest b hq hkv group sq rows d
            (flash_out_vfun nw d b hq hkv group sq rows sk eQ eK eV emask has_mask causal scale)
          %~ FSpec.sdpa_flash_real (SZ.v group)
               (to_real_chest eQ) (to_real_chest eK) (to_real_chest eV)
               (to_real_chest emask) (to_real scale) causal has_mask)
