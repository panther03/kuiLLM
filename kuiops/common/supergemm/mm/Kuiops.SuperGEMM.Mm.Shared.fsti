module Kuiops.SuperGEMM.Mm.Shared

(* Shared-memory layout of the software-pipelined tensor-core GEMM
   (D = A @ B^T).

   Five [SHArray] entries (order matters for 16-byte alignment):
     A buffer 0, A buffer 1  : [SHArray et_ab (bm * ldt)]
     B buffer 0, B buffer 1  : [SHArray et_ab (bn * ldt)]  (B staged (bn,bk) row-major)
     epilogue scratch        : [SHArray et_acc (warps * 16 * lde)]  (fp32, per-warp)

   Each pipeline buffer is viewed as an [l2_skewed_row_major] tile; [block_setup]
   uses [skew_split] to expose that view and parks the (dead) skew pad cells in
   [block_frame] as [skew_residual].  The whole-tile read shares carried per
   thread ([pipe_sharing]) are what is block-sendable; the barrier (module 3)
   flip-flops them to/from the writable [own_strided_chunks] partition.

   The epilogue scratch is per-warp: warp [w] owns a [Frac (1/warp_size)] read
   share of the [(16, wn)] sub-tile at offset [w * 16 * lde], cooperatively with
   its 32 lanes.

   The scratch is a SEPARATE, non-aliased allocation (Kuiper cannot express
   lifetime retyping of one shared allocation onto the pipeline buffers); the
   measured cost of not aliasing is accepted, per DESIGN.md.

   Step 1: memory safety only.  Staged contents are existentially quantified;
   the output tile is [output_lane_live] (contents unspecified), identical in
   shape to the synchronous TN GEMM so a functional postcondition can be layered
   on later. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling

module SZ = Kuiper.SizeT

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live' }
open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major, skew_residual }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_sharing, pipe_live, pipe_q }
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params
module FB = Kuiops.GEMM.T.FlipFlopBarrier2

(* ---- shared-array descriptor ---- *)

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
    SHArray et_ab (abuf_sz bn bk skew);
    SHArray et_acc (scratch_sz et_acc bm bn wm wn) ]

(* ---- accessors ---- *)

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

inline_for_extraction noextract
let sar_scratch
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
= fst (snd (snd (snd (snd sh))))

(* ---- epilogue scratch: skewed (warps*16, wn) row-major view ---- *)

(* Return types are intentionally inferred: annotating them with an
   [eskew]/[lde]-mentioning layout trips typeclass resolution on a
   tuple-projection body (see module notes). *)

inline_for_extraction noextract
let scratch_matrix
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
= from_array
    (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
    (sar_scratch bm bn bk wm wn skew sh)

inline_for_extraction noextract
let scratch_tile
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (wid : natlt (warps bm bn wm wn))
= array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid 0

let scratch_tile_live
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (tid : natlt (SZ.v nthr))
  : slprop
= exists* (eAcc : chest2 et_acc frag (SZ.v wn)).
    scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size)
      |-> Frac (1.0R /. warp_size) eAcc

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
  pipe_live (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
  scratch_tile_live bm bn bk wm wn skew sh nthr tid

(* ---- per-thread shared ownership on EXIT of [kloop] ----
   [kloop] returns [pipe_q ... (ktiles-1)]: read shares of the last-used
   buffer pair and live chunks of the other pair.  This is deliberately
   DIFFERENT from [shared_thread_live] (the entry ownership); [kpre] and
   [kpost] are independent [kernel_desc] fields. *)
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
    (SZ.v nthr) ktiles (ktiles - 1) tid **
  scratch_tile_live bm bn bk wm wn skew sh nthr tid

(* ---- global-side per-thread precondition (sh-free) ----
   D = A @ B^T; no C input.  The output tiling is hardcoded to the band form
   [(tm, tn, wmf, wnf) = (frag, wn, mfrag wm, 1)] the Epilogue commits to; the
   element warp dims [wm x wn] are the szp parameters.  D is layout-generic
   ([lD] + [strided_row_major]) and carries its own element type [et_d]. *)
unfold
let kpre1
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= gA |-> Frac (fA /. (nblk * nthr)) eA **
  gB |-> Frac (fB /. (nblk * nthr)) eB **
  output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (aligned 16 (core gD))

(* On exit the A/B global read shares come back at an UNSPECIFIED chest: the
   pipelined staging path ([kloop]) re-materialises each fractional share
   through the [cp.async] batch pledge, which (in step 1, memory safety only)
   does not track that a global read preserves content.  Pinning the chest to
   [eA]/[eB] is deferred to the functional-spec step; hence the existentials.
   [output_lane_live'] is likewise contents-unspecified. *)
unfold
let kpost1
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= (exists* (eA' : chest2 et_ab (SZ.v m) (SZ.v k)). gA |-> Frac (fA /. (nblk * nthr)) eA') **
  (exists* (eB' : chest2 et_ab (SZ.v n) (SZ.v k)). gB |-> Frac (fB /. (nblk * nthr)) eB') **
  output_lane_live' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (aligned 16 (core gD))

(* ---- full per-thread pre/post: global side + shared ownership ---- *)
unfold
let kpre
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid **
  shared_thread_live bm bn bk wm wn skew sh nthr tid

unfold
let kpost
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpost1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid **
  shared_thread_final bm bn bk wm wn skew sh nthr (SZ.v k / SZ.v bk) tid

(* ---- block-level pre/post (sh-free) and frame ---- *)

unfold
let block_pre
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid

unfold
let block_post
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpost1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid

(* Dead skew-pad cells of the four A/B pipeline buffers and the scratch. *)
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
  skew_residual (sar_b1 bm bn bk wm wn skew sh) (SZ.v bn) (SZ.v bk) (SZ.v skew) **
  skew_residual (sar_scratch bm bn bk wm wn skew sh)
    (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)

(* ---- block setup / teardown (ghost) ----
   [block_setup] shares the four skewed pipeline buffers into whole-tile read
   shares carried per thread ([pipe_sharing]) and the skewed scratch into
   per-warp/per-lane subtile shares ([scratch_tile_live]); the dead skew-pad
   cells of all five allocations are parked in [block_frame].  [block_teardown]
   is the reverse. *)

ghost
fn block_setup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk)
  ()
  requires
    live_c_shmems sh **
    block_pre gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid
  ensures
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gD bm bn bk wm wn skew
        fA fB nblk nthr sh bid tid) **
    block_frame bm bn bk wm wn skew sh

ghost
fn block_teardown
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk)
  ()
  requires
    (forall+ (tid : natlt nthr).
      kpost gA eA gB eB gD bm bn bk wm wn skew
        fA fB nblk nthr sh bid tid) **
    block_frame bm bn bk wm wn skew sh
  ensures
    live_c_shmems sh **
    block_post gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid

(* ---- block-sendability of the per-thread pre/post ----
   Needed to populate the [kpre_sendable] / [kpost_sendable] fields of
   [kernel_desc].  The global A/B fractions and the output lane come from the
   TensorCore output-tiling sendability; the shared read-shares are sendable
   because each backing shared allocation is a block array. *)

val kpre_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD { is_global gD })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpre gA eA gB eB gD bm bn bk wm wn skew
        fA fB nblk nthr sh bid tid)

val kpost_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_d lD { is_global gD })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpost gA eA gB eB gD bm bn bk wm wn skew
        fA fB nblk nthr sh bid tid)

(* ---- shared-buffer 16-byte alignment (for cp.async in [kloop]) ----
   Each of the four pipeline buffers is a block array (by [c_shmems_inv]),
   hence 16-byte aligned ([Kuiops.SHMem.Aligned.shmem_aligned16]).  The kernel
   body ([Kuiops.SuperGEMM.Mm.Kernel]) needs exactly these four facts to
   discharge [kloop]'s [sq_al].  The [c_shmems_inv] witness is available inside
   the [f] field (its [sh] binder is refined), so this is a real derivation,
   not an axiom. *)
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
