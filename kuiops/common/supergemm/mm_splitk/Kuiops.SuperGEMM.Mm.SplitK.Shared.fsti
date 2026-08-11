module Kuiops.SuperGEMM.Mm.SplitK.Shared

(* Shared-memory layout of the split-K tensor-core GEMM (partials -> fp32 ws).

   FOUR [SHArray] entries -- A buffer 0/1 and B buffer 0/1.  The non-split
   sibling ([Mm.Shared]) carries a fifth, the fp32 epilogue scratch; split-K
   has no epilogue (accumulator and workspace are both fp32, so fragments go
   straight to global with [mma_store]) and therefore no scratch, which is
   exactly the shared-memory saving the reference CUDA advertises.

   The pipeline machinery ([Mm.Barrier], [Mm.Stage], [Mm.KLoop], [Mm.Params])
   is reused verbatim.  Only the descriptor and the per-thread global-side
   capability differ: the output capability here is a warp-tile [1/warp_size]
   share of the workspace ([SplitK.Output.ws_warp_live]) rather than the
   epilogue's per-lane cell partition.

   The workspace is viewed as a single [(mws, n)] matrix with [mws = splits*m];
   because it is contiguous, split [z]'s [(m, n)] plane is rows
   [z*m .. z*m+m), so the ordinary row-major block decode of a flattened
   [bid = z*nblk_z + block] lands on exactly the right tile and no 3-D grid is
   needed.  [splits] therefore does not appear in this module at all.

   Step 1: memory safety only. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling

module SZ = Kuiper.SizeT

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.Array2.Layout.Skewed { skew_residual }
open Kuiops.SuperGEMM.Mm.SplitK.Output { ws_warp_live, ws_warp_approximates }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_live, pipe_q }
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params

(* Divisibility side conditions of the workspace warp-tile partition, in the
   [/?+] form [ws_warp_live] wants.  All of them follow from the ordinary
   [wm /?+ bm /\ wn /?+ bn /\ frag /?+ wm /\ frag /?+ wn] tiling facts (see
   [SplitK.Kernel.out_tiling_facts]); naming them keeps that derivation in one
   place instead of at every use site. *)
unfold
let out_ok (bm bn : pos) (wm wn : szp) (mws n : nat) : prop =
  mfrag wm * frag == SZ.v wm /\ nfrag wn * frag == SZ.v wn /\
  mws % bm == 0 /\ n % bn == 0 /\
  bm % (mfrag wm * frag) == 0 /\ bn % (nfrag wn * frag) == 0 /\
  bm % frag == 0 /\ bn % frag == 0

(* ---- shared-array descriptor (four entries: no epilogue scratch) ---- *)

inline_for_extraction noextract
let shmems_desc
  (et_ab et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  : list shmem_desc
=
  [ SHArray et_ab (abuf_sz bm bk skew);
    SHArray et_ab (abuf_sz bm bk skew);
    SHArray et_ab (abuf_sz bn bk skew);
    SHArray et_ab (abuf_sz bn bk skew) ]

inline_for_extraction noextract
let sar_a0
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  : larray et_ab (SZ.v bm * ldt bk skew)
= fst sh

inline_for_extraction noextract
let sar_a1
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  : larray et_ab (SZ.v bm * ldt bk skew)
= fst (snd sh)

inline_for_extraction noextract
let sar_b0
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  : larray et_ab (SZ.v bn * ldt bk skew)
= fst (snd (snd sh))

inline_for_extraction noextract
let sar_b1
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  : larray et_ab (SZ.v bn * ldt bk skew)
= fst (snd (snd (snd sh)))

(* ---- per-thread shared ownership ---- *)

let shared_thread_live
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (tid : natlt (SZ.v nthr))
  : slprop
= pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
  pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
  pipe_live (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
  pipe_live (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid

let shared_thread_final
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (ktiles : pos)
  (tid : natlt (SZ.v nthr))
  : slprop
= pipe_q bm bn bk skew
    (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
    (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
    (SZ.v nthr) ktiles (ktiles - 1) tid

(* ---- global-side per-thread pre/post (sh-free) ----
   [gW] is the whole [(mws, n)] workspace; the output capability is the warp's
   [1/warp_size] share of its [(wm, wn)] tile, which is what [mma_store]
   consumes.  [kpre1] leaves the contents unspecified; [kpost1] pins them to
   the corresponding tile of [rW]. *)
unfold
let kpre1
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_acc : Type0) {| scalar et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn wm wn : szp)
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= gA |-> Frac (fA /. (nblk * nthr)) eA **
  gB |-> Frac (fB /. (nblk * nthr)) eB **
  ws_warp_live gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn) bid tid **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (aligned 16 (core gW))

(* The functional counterpart of [kpre1]: the same capabilities, but the
   workspace share now approximates [rW]. *)
unfold
let kpost1
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_acc : Type0) {| scalar et_acc |} {| real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn wm wn : szp)
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= gA |-> Frac (fA /. (nblk * nthr)) eA **
  gB |-> Frac (fB /. (nblk * nthr)) eB **
  ws_warp_approximates gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn)
    bid tid rW **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (aligned 16 (core gW))

unfold
let kpre
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpre1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid tid **
  shared_thread_live bm bn bk wm wn skew sh nthr tid

unfold
let kpost
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  {| real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (ktiles : pos)
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpost1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid tid **
  shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid

(* ---- block-level pre/post (sh-free) and frame ---- *)

unfold
let block_pre
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_acc : Type0) {| scalar et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn wm wn : szp)
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpre1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid tid

(* The functional block-level postcondition. *)
unfold
let block_post
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_acc : Type0) {| scalar et_acc |} {| real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn wm wn : szp)
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpost1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid tid

let block_frame
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  : slprop
= skew_residual (sar_a0 bm bn bk wm wn skew sh) (SZ.v bm) (SZ.v bk) (SZ.v skew) **
  skew_residual (sar_a1 bm bn bk wm wn skew sh) (SZ.v bm) (SZ.v bk) (SZ.v skew) **
  skew_residual (sar_b0 bm bn bk wm wn skew sh) (SZ.v bn) (SZ.v bk) (SZ.v skew) **
  skew_residual (sar_b1 bm bn bk wm wn skew sh) (SZ.v bn) (SZ.v bk) (SZ.v skew)

(* ---- block setup / teardown (ghost) ---- *)

ghost
fn block_setup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk)
  ()
  requires
    live_c_shmems sh **
    block_pre gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid
  ensures
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gW bm bn bk wm wn skew fA fB nblk nthr sh bid tid) **
    block_frame bm bn bk wm wn skew sh

ghost
fn block_teardown
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  {| real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (ktiles : pos)
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (bid : natlt nblk)
  ()
  requires
    (forall+ (tid : natlt nthr).
      kpost gA eA gB eB gW bm bn bk wm wn skew fA fB nblk nthr ktiles sh rW bid tid) **
    block_frame bm bn bk wm wn skew sh
  ensures
    live_c_shmems sh **
    block_post gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid

(* ---- block-sendability of the per-thread pre/post ---- *)

val kpre_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW { is_global gW })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpre gA eA gB eB gW bm bn bk wm wn skew fA fB nblk nthr sh bid tid)

val kpost_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  {| real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW { is_global gW })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (ktiles : pos)
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpost gA eA gB eB gW bm bn bk wm wn skew fA fB nblk nthr ktiles sh rW bid tid)

(* ---- shared-buffer 16-byte alignment (for cp.async in [kloop]) ---- *)
val shared_buffers_aligned16
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  ()
  : Lemma
      (aligned 16 (core (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh))) /\
       aligned 16 (core (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh))) /\
       aligned 16 (core (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh))) /\
       aligned 16 (core (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh))))
