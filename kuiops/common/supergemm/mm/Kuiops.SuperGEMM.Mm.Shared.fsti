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
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc { output_lane_live }
open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major, skew_residual }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_sharing }
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
= pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
  pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
  pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
  pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
  scratch_tile_live bm bn bk wm wn skew sh nthr tid

(* ---- global-side per-thread precondition (sh-free) ---- *)

(* D = A @ B^T; no C input.  [tm tn wm wn] are the tensor-core output tiling
   (the Kernel instantiates [tm = tn = frag], [wm = mfrag], [wn = nfrag]);
   they stay generic here. *)
unfold
let kpre1
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn : szp) (tm tn wm wn : pos)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                wm * tm /?+ SZ.v bm /\ wn * tn /?+ SZ.v bn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (wm * tm) * (SZ.v bn / (wn * tn)) * warp_size})
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= gA |-> Frac (fA /. (nblk * nthr)) eA **
  gB |-> Frac (fB /. (nblk * nthr)) eB **
  output_lane_live gD (SZ.v bm) (SZ.v bn) tm tn wm wn bid tid **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (aligned 16 (core gD))

(* For step 1 the output tile is just "live" both before and after, so
   [kpost1] and [kpre1] have the same body (distinct names so the Kernel can
   later refine [kpost1] to a functional postcondition). *)
unfold
let kpost1
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn : szp) (tm tn wm wn : pos)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                wm * tm /?+ SZ.v bm /\ wn * tn /?+ SZ.v bn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (wm * tm) * (SZ.v bn / (wn * tn)) * warp_size})
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= gA |-> Frac (fA /. (nblk * nthr)) eA **
  gB |-> Frac (fB /. (nblk * nthr)) eB **
  output_lane_live gD (SZ.v bm) (SZ.v bn) tm tn wm wn bid tid **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (aligned 16 (core gD))

(* ---- full per-thread pre/post: global side + shared ownership ----
   The reconciliation squash ties the output tiling [tm tn wm wn] to the
   shared-tile warp dims: the element warp height/width is [wm*tm]/[wn*tn].
   The Kernel discharges it with [wm := mfrag], [tm := frag] (so
   [mfrag*frag == wm_element]). *)
unfold
let kpre
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (tm tn wmf wnf : pos)
  (#_ : squash (wmf * tm == SZ.v wm /\ wnf * tn == SZ.v wn))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpre1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid **
  shared_thread_live bm bn bk wm wn skew sh nthr tid

unfold
let kpost
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (tm tn wmf wnf : pos)
  (#_ : squash (wmf * tm == SZ.v wm /\ wnf * tn == SZ.v wn))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpost1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid **
  shared_thread_live bm bn bk wm wn skew sh nthr tid

(* ---- block-level pre/post (sh-free) and frame ---- *)

unfold
let block_pre
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn : szp) (tm tn wm wn : pos)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                wm * tm /?+ SZ.v bm /\ wn * tn /?+ SZ.v bn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (wm * tm) * (SZ.v bn / (wn * tn)) * warp_size})
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpre1 gA eA gB eB gD bm bn tm tn wm wn fA fB nblk nthr bid tid

unfold
let block_post
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn : szp) (tm tn wm wn : pos)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                wm * tm /?+ SZ.v bm /\ wn * tn /?+ SZ.v bn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (wm * tm) * (SZ.v bn / (wn * tn)) * warp_size})
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpost1 gA eA gB eB gD bm bn tm tn wm wn fA fB nblk nthr bid tid

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
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (tm tn wmf wnf : pos)
  (#_ : squash (wmf * tm == SZ.v wm /\ wnf * tn == SZ.v wn))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk)
  ()
  requires
    live_c_shmems sh **
    block_pre gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid
  ensures
    (forall+ (tid : natlt nthr).
      kpre gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid) **
    block_frame bm bn bk wm wn skew sh

ghost
fn block_teardown
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)))
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (tm tn wmf wnf : pos)
  (#_ : squash (wmf * tm == SZ.v wm /\ wnf * tn == SZ.v wn))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk)
  ()
  requires
    (forall+ (tid : natlt nthr).
      kpost gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid) **
    block_frame bm bn bk wm wn skew sh
  ensures
    live_c_shmems sh **
    block_post gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid

(* ---- block-sendability of the per-thread pre/post ----
   Needed to populate the [kpre_sendable] / [kpost_sendable] fields of
   [kernel_desc].  The global A/B fractions and the output lane come from the
   TensorCore output-tiling sendability; the shared read-shares are sendable
   because each backing shared allocation is a block array. *)

val kpre_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)) { is_global gD })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (tm tn wmf wnf : pos)
  (#_ : squash (wmf * tm == SZ.v wm /\ wnf * tn == SZ.v wn))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpre gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid)

val kpost_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)) { is_global gD })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (tm tn wmf wnf : pos)
  (#_ : squash (wmf * tm == SZ.v wm /\ wnf * tn == SZ.v wn))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpost gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid)
