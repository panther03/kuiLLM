module Kuiops.Sdpa.Flash.Vals

(* The per-warp and block-wide accumulator values the flash-attention kernel
   publishes through its block barriers, as functions of the tile parameters
   the thread body works with.

   These are deliberately abstract: their bodies mention the whole online
   softmax specification tower, and letting the unifier delta-reduce them at
   slprop-matching time makes downstream Pulse checking intractable.  Clients
   see them only through the [*_def] lemmas below. *)

open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Kernel.FlashAttention.KernelDesc

module SZ = Kuiper.SizeT
module SF = Kuiops.Sdpa.Flash.Spec.Float
module SS = Kuiops.Sdpa.Flash.Spec.Step
module FC = Kuiper.Float.Casts

let flash_eM_at
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  : GTot (chest2 et_acc (SZ.v nw) 16)
= SS.block_m emask has_mask causal bi kvh group (SZ.v rows) r0 scale
    (SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0) eKg (SZ.v nw)
    (SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal)


let flash_eL_at
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  : GTot (chest2 et_acc (SZ.v nw) 16)
= SS.block_l emask has_mask causal bi kvh group (SZ.v rows) r0 scale
    (SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0) eKg (SZ.v nw)
    (SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal)

let flash_eM_at_def
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  : Lemma
      (flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal scale
         bi kvh group r0
       == SS.block_m emask has_mask causal bi kvh group (SZ.v rows) r0 scale
            (SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0) eKg (SZ.v nw)
            (SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal))
  = ()

let flash_eL_at_def
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  : Lemma
      (flash_eL_at nw d b hq sq rows sk eQ eKg emask has_mask causal scale
         bi kvh group r0
       == SS.block_l emask has_mask causal bi kvh group (SZ.v rows) r0 scale
            (SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0) eKg (SZ.v nw)
            (SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal))
  = ()

(* The rescaling factors and block-wide denominators warp 0 publishes through
   block barrier 2, and the per-warp output tiles that go with them. *)

let flash_escale_at
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  : GTot (chest2 et_acc (SZ.v nw) 16)
= ematrix_stride_from_tiles 1 16
    (fun (_tr : natlt 1) (tc : natlt 16) ->
      SF.gscale_col
        (flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal scale
           bi kvh group r0) tc)


let flash_egl_at
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  : GTot (chest1 et_acc 16)
= mk1 (fun i ->
    SF.gsum
      (flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal scale
         bi kvh group r0)
      (flash_eL_at nw d b hq sq rows sk eQ eKg emask has_mask causal scale
         bi kvh group r0)
      (SF.gmax
        (flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal scale
           bi kvh group r0) i (SZ.v nw))
      i (SZ.v nw))


let flash_eO_at
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  : GTot (chest2 et_acc (SZ.v nw * 16) (SZ.v d))
= ematrix_from_tiles 16 (SZ.v d <: pos)
    (fun (w : natlt (SZ.v nw)) (_tc : natlt 1) ->
      SS.block_O emask has_mask causal bi kvh group (SZ.v rows) r0 scale
        (SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0) eKg eVg (SZ.v nw)
        (SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal) w)

(* [ematrix_stride_from_tiles] recovers its tiles, the stride analogue of
   [tiles_from_subtiles_id]. *)
let stride_idx_split (s : pos) (i : nat) (t : natlt s)
  : Lemma ((i * s + t) % s == t /\ (i * s + t) / s == i)
= FStar.Math.Lemmas.lemma_mod_plus t i s;
  FStar.Math.Lemmas.small_mod t s;
  FStar.Math.Lemmas.lemma_div_plus t i s;
  FStar.Math.Lemmas.small_div t s

(* Preserve the [mk2] head for the SMT pattern after the packaged stride
   helpers unfold through two nested matrix constructors. *)
let stride_acc2_mk2 (#et : Type) (#d0 #d1 : nat)
  (g : natlt d0 -> natlt d1 -> GTot et) (i : natlt d0) (j : natlt d1)
  : Lemma (acc2 (mk2 g) i j == g i j)
      [SMTPat (acc2 (mk2 g) i j)]
= ()

#push-options "--fuel 1 --ifuel 2 --z3rlimit 40"
let stride_from_tiles_subtile
  (#et : Type0) (#rows #cols : nat)
  (srows : pos { srows /? rows }) (scols : pos { scols /? cols })
  (f : natlt srows -> natlt scols -> chest2 et (rows / srows) (cols / scols))
  (tr : natlt srows) (tc : natlt scols)
  : Lemma
      (ematrix_stride_subtile
        (ematrix_stride_from_tiles srows scols f) srows scols tr tc == f tr tc)
      [SMTPat (ematrix_stride_subtile
                (ematrix_stride_from_tiles srows scols f) srows scols tr tc)]
= introduce forall (i : natlt (rows / srows)) (j : natlt (cols / scols)).
    acc2 (ematrix_stride_subtile
            (ematrix_stride_from_tiles srows scols f) srows scols tr tc) i j
      == acc2 (f tr tc) i j
  with (stride_idx_split srows i tr; stride_idx_split scols j tc);
  assert (equal
    (ematrix_stride_subtile
      (ematrix_stride_from_tiles srows scols f) srows scols tr tc) (f tr tc))
#pop-options

#push-options "--fuel 1 --ifuel 2"
(* Column [i] of the block-wide rescaling matrix and denominator vector. *)
let flash_escale_egl_at_def
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  (i : natlt 16)
  : Lemma
      (ematrix_stride_subtile
        (flash_escale_at nw d b hq sq rows sk eQ eKg emask has_mask causal
           scale bi kvh group r0) 1 16 0 i
       == SF.gscale_col
            (flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal
               scale bi kvh group r0) i /\
       acc1
        (flash_egl_at nw d b hq sq rows sk eQ eKg emask has_mask causal
           scale bi kvh group r0) i
       == SF.gsum
            (flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal
               scale bi kvh group r0)
            (flash_eL_at nw d b hq sq rows sk eQ eKg emask has_mask causal
               scale bi kvh group r0)
            (SF.gmax
              (flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal
                 scale bi kvh group r0) i (SZ.v nw))
            i (SZ.v nw))
= ()

#pop-options

(* Warp [w]'s tile of the block-wide output matrix. *)
let flash_eO_at_def
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  (w : natlt (SZ.v nw))
  : Lemma
      (ematrix_subtile
        (flash_eO_at nw d b hq sq rows sk eQ eKg eVg emask has_mask causal
           scale bi kvh group r0) 16 (SZ.v d <: pos) w 0
       == SS.block_O emask has_mask causal bi kvh group (SZ.v rows) r0 scale
            (SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0) eKg eVg (SZ.v nw)
            (SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal) w)
= ()
