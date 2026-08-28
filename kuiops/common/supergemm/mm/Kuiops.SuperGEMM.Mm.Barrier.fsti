module Kuiops.SuperGEMM.Mm.Barrier

(* The pipelined one-barrier-per-k-tile flip-flop contract of SuperGEMM.

   A double-buffered software pipeline stages k-tile [it+1] while the math for
   k-tile [it] runs, so a single [__syncthreads()] per k-tile must do two things
   at once, to two DIFFERENT buffers:

     - publish the buffer just staged ([b = it & 1]): its per-thread partitioned
       ownership becomes a whole-buffer read share (the ODD transform of
       [Kuiops.GEMM.T.FlipFlopBarrier2]);
     - reclaim the buffer just consumed ([o = 1 - b]): its whole-buffer read
       share becomes fresh per-thread partitioned ownership, ready to be staged
       again (the EVEN transform).

   At the first k-tile ([it = 0]) there is nothing consumed yet, so the [o]
   buffer simply passes through as live per-thread ownership.

   This is memory-safety only (step 1): the staged contents are existentially
   quantified everywhere.  A and B are BOTH staged (rows x bk) row-major-skewed,
   so both sides use the plain row-major [own_strided_chunks] partition -- the
   column-major view of B is only taken at fragment-load time (module 5).

   The transform is an instantiation of the two FlipFlopBarrier2 halves at the
   skewed [array2] views, using its exported array2-level chunk (de)composition
   primitives rather than reproving them. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiops.Array2.Strided

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module FB = Kuiops.GEMM.T.FlipFlopBarrier2

open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major }
open Kuiops.SuperGEMM.Mm.Params { ldt }
open Kuiper.EMatrix.Tiling { ematrix_subtile }

(* Skewed (rows x bk, pad skew) row-major view of one pipeline buffer. *)
inline_for_extraction noextract
let skewed_view
  (#et : Type0)
  (rows bk skew : szp)
  (sar : larray et (SZ.v rows * ldt bk skew))
  : array2 et (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew))
= from_array (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew)) sar

(* Per-thread partitioned ownership of a buffer, contents unspecified. *)
let pipe_live
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (nthr : pos)
  (tid : natlt nthr)
  : slprop
= FB.live_strided_chunks m nthr tid

(* Whole-buffer read share, contents unspecified. *)
let pipe_sharing
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (nthr : pos)
  : slprop
= exists* em. FB.bp_sharing m em nthr

let pipe_p
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (ktiles : nat)
  : B.barrier_side nthr
= fun it tid ->
    if it >= ktiles then emp
    else begin
      let mAb = skewed_view bm bk skew (if it % 2 = 0 then sarA0 else sarA1) in
      let mAo = skewed_view bm bk skew (if it % 2 = 0 then sarA1 else sarA0) in
      let mBb = skewed_view bn bk skew (if it % 2 = 0 then sarB0 else sarB1) in
      let mBo = skewed_view bn bk skew (if it % 2 = 0 then sarB1 else sarB0) in
      pipe_live mAb nthr tid **
      pipe_live mBb nthr tid **
      (if it = 0 then
         pipe_live mAo nthr tid ** pipe_live mBo nthr tid
       else
         pipe_sharing mAo nthr ** pipe_sharing mBo nthr)
    end

let pipe_q
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (ktiles : nat)
  : B.barrier_side nthr
= fun it tid ->
    if it >= ktiles then emp
    else begin
      let mAb = skewed_view bm bk skew (if it % 2 = 0 then sarA0 else sarA1) in
      let mAo = skewed_view bm bk skew (if it % 2 = 0 then sarA1 else sarA0) in
      let mBb = skewed_view bn bk skew (if it % 2 = 0 then sarB0 else sarB1) in
      let mBo = skewed_view bn bk skew (if it % 2 = 0 then sarB1 else sarB0) in
      pipe_sharing mAb nthr ** pipe_sharing mBb nthr **
      pipe_live mAo nthr tid ** pipe_live mBo nthr tid
    end

let pipe_contract
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (ktiles : nat)
  : B.contract nthr
= {
    B.rin  = pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles;
    B.rout = pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles;
  }

let pipe_tok
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (ktiles : nat)
  : slprop
= B.barrier_tok (pipe_contract bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles)

(* One barrier iteration's transform: [rin it] -> [rout it] for every thread.
   The fits/divisibility side conditions are the SuperGEMM [constraints]
   consequences (see [Kuiops.SuperGEMM.Mm.Params]). *)
ghost
fn pipe_p_to_q_transform
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (ktiles : nat)
  (#_ : squash (SZ.fits (SZ.v bm * ldt bk skew)))
  (#_ : squash (SZ.fits (SZ.v bn * ldt bk skew)))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
  ensures
    forall+ (tid : natlt nthr).
      pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid

(* Openers for [pipe_q]: at a live iteration ([it < ktiles]) of a given parity,
   convert every thread's [pipe_q] product into the concrete read-share /
   live-chunk pieces, naming the physical buffers.  The parity is passed as a
   squash so the [if it%2=0] selection inside [pipe_q] reduces.  [block_teardown]
   (module 3) uses these to gather the shared buffers back. *)
ghost
fn unfold_pipe_q_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (ktiles : nat)
  (it : nat)
  (#_ : squash (it < ktiles /\ it % 2 = 0))
  requires
    forall+ (tid : natlt nthr).
      pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
  ensures
    forall+ (tid : natlt nthr).
      (pipe_sharing (skewed_view bm bk skew sarA0) nthr **
       pipe_sharing (skewed_view bn bk skew sarB0) nthr **
       pipe_live (skewed_view bm bk skew sarA1) nthr tid **
       pipe_live (skewed_view bn bk skew sarB1) nthr tid)

ghost
fn unfold_pipe_q_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (ktiles : nat)
  (it : nat)
  (#_ : squash (it < ktiles /\ it % 2 = 1))
  requires
    forall+ (tid : natlt nthr).
      pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid
  ensures
    forall+ (tid : natlt nthr).
      (pipe_sharing (skewed_view bm bk skew sarA1) nthr **
       pipe_sharing (skewed_view bn bk skew sarB1) nthr **
       pipe_live (skewed_view bm bk skew sarA0) nthr tid **
       pipe_live (skewed_view bn bk skew sarB0) nthr tid)

(* ---------------------------------------------------------------------------
   Content-carrying (_c) siblings.

   The pipeline stages one k-tile per barrier iteration; at barrier iteration
   [it] the PUBLISHED buffer holds k-tile [it] exactly, whose contents are a
   subtile of the global operands: for A it is
   [ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row it] and for B it is
   [ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col it].  The RECLAIMED slot
   stays content-FREE (existential), exactly as in the content-free versions.
   --------------------------------------------------------------------------- *)

(* content-carrying per-thread ownership / whole-buffer read share *)
let pipe_live_c
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l) (em : chest2 et rows cols)
  (nthr : pos) (tid : natlt nthr)
  : slprop
= FB.own_strided_chunks m em nthr tid

let pipe_sharing_c
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l) (em : chest2 et rows cols)
  (nthr : pos)
  : slprop
= FB.bp_sharing m em nthr

let pipe_p_c
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  : B.barrier_side nthr
= fun it tid ->
    if it >= SZ.v k / SZ.v bk then emp
    else begin
      let mAb = skewed_view bm bk skew (if it % 2 = 0 then sarA0 else sarA1) in
      let mAo = skewed_view bm bk skew (if it % 2 = 0 then sarA1 else sarA0) in
      let mBb = skewed_view bn bk skew (if it % 2 = 0 then sarB0 else sarB1) in
      let mBo = skewed_view bn bk skew (if it % 2 = 0 then sarB1 else sarB0) in
      pipe_live_c mAb (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row it) nthr tid **
      pipe_live_c mBb (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col it) nthr tid **
      (if it = 0 then
         pipe_live mAo nthr tid ** pipe_live mBo nthr tid
       else
         pipe_sharing mAo nthr ** pipe_sharing mBo nthr)
    end

let pipe_q_c
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  : B.barrier_side nthr
= fun it tid ->
    if it >= SZ.v k / SZ.v bk then emp
    else begin
      let mAb = skewed_view bm bk skew (if it % 2 = 0 then sarA0 else sarA1) in
      let mAo = skewed_view bm bk skew (if it % 2 = 0 then sarA1 else sarA0) in
      let mBb = skewed_view bn bk skew (if it % 2 = 0 then sarB0 else sarB1) in
      let mBo = skewed_view bn bk skew (if it % 2 = 0 then sarB1 else sarB0) in
      pipe_sharing_c mAb (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row it) nthr **
      pipe_sharing_c mBb (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col it) nthr **
      pipe_live mAo nthr tid ** pipe_live mBo nthr tid
    end

let pipe_contract_c
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  : B.contract nthr
= {
    B.rin  = pipe_p_c m n k bm bn bk skew eA eB block_row block_col sarA0 sarA1 sarB0 sarB1 nthr sq;
    B.rout = pipe_q_c m n k bm bn bk skew eA eB block_row block_col sarA0 sarA1 sarB0 sarB1 nthr sq;
  }

let pipe_tok_c
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  : slprop
= B.barrier_tok (pipe_contract_c m n k bm bn bk skew eA eB block_row block_col
                                 sarA0 sarA1 sarB0 sarB1 nthr sq)

(* One barrier iteration's transform: [rin it] -> [rout it] for every thread,
   content-carrying version. *)
ghost
fn pipe_p_to_q_transform_c
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (#_ : squash (SZ.fits (SZ.v bm * ldt bk skew)))
  (#_ : squash (SZ.fits (SZ.v bn * ldt bk skew)))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      pipe_p_c m n k bm bn bk skew eA eB block_row block_col
               sarA0 sarA1 sarB0 sarB1 nthr sq it tid
  ensures
    forall+ (tid : natlt nthr).
      pipe_q_c m n k bm bn bk skew eA eB block_row block_col
               sarA0 sarA1 sarB0 sarB1 nthr sq it tid

(* Forget published content back to the content-free [pipe_q] so the Shared
   boundary is untouched. *)
ghost
fn pipe_q_c_forget
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (it : nat)
  requires
    forall+ (tid : natlt nthr).
      pipe_q_c m n k bm bn bk skew eA eB block_row block_col
               sarA0 sarA1 sarB0 sarB1 nthr sq it tid
  ensures
    forall+ (tid : natlt nthr).
      pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr (SZ.v k / SZ.v bk) it tid

(* Parity openers for [pipe_q_c]: the PUBLISHED sharings carry content. *)
ghost
fn unfold_pipe_q_c_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (it : nat)
  (#_ : squash (it < SZ.v k / SZ.v bk /\ it % 2 = 0))
  requires
    forall+ (tid : natlt nthr).
      pipe_q_c m n k bm bn bk skew eA eB block_row block_col
               sarA0 sarA1 sarB0 sarB1 nthr sq it tid
  ensures
    forall+ (tid : natlt nthr).
      (pipe_sharing_c (skewed_view bm bk skew sarA0)
         (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row it) nthr **
       pipe_sharing_c (skewed_view bn bk skew sarB0)
         (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col it) nthr **
       pipe_live (skewed_view bm bk skew sarA1) nthr tid **
       pipe_live (skewed_view bn bk skew sarB1) nthr tid)

ghost
fn unfold_pipe_q_c_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (it : nat)
  (#_ : squash (it < SZ.v k / SZ.v bk /\ it % 2 = 1))
  requires
    forall+ (tid : natlt nthr).
      pipe_q_c m n k bm bn bk skew eA eB block_row block_col
               sarA0 sarA1 sarB0 sarB1 nthr sq it tid
  ensures
    forall+ (tid : natlt nthr).
      (pipe_sharing_c (skewed_view bm bk skew sarA1)
         (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row it) nthr **
       pipe_sharing_c (skewed_view bn bk skew sarB1)
         (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col it) nthr **
       pipe_live (skewed_view bm bk skew sarA0) nthr tid **
       pipe_live (skewed_view bn bk skew sarB0) nthr tid)

(* Inverses used to BUILD [pipe_p_c] from the concrete pieces. *)
ghost
fn fold_pipe_p_c_even
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (it : nat)
  (#_ : squash (it < SZ.v k / SZ.v bk /\ it % 2 = 0))
  requires
    forall+ (tid : natlt nthr).
      (pipe_live_c (skewed_view bm bk skew sarA0)
         (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row it) nthr tid **
       pipe_live_c (skewed_view bn bk skew sarB0)
         (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col it) nthr tid **
       (if it = 0 then
          pipe_live (skewed_view bm bk skew sarA1) nthr tid **
          pipe_live (skewed_view bn bk skew sarB1) nthr tid
        else
          pipe_sharing (skewed_view bm bk skew sarA1) nthr **
          pipe_sharing (skewed_view bn bk skew sarB1) nthr))
  ensures
    forall+ (tid : natlt nthr).
      pipe_p_c m n k bm bn bk skew eA eB block_row block_col
               sarA0 sarA1 sarB0 sarB1 nthr sq it tid

ghost
fn fold_pipe_p_c_odd
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (m n k : szp)
  (bm bn bk skew : szp)
  (eA : chest2 et (SZ.v m) (SZ.v k))
  (eB : chest2 et (SZ.v n) (SZ.v k))
  (block_row : nat { block_row < SZ.v m / SZ.v bm })
  (block_col : nat { block_col < SZ.v n / SZ.v bn })
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (nthr : pos)
  (sq : squash (SZ.v bm > 0 /\ SZ.v bm /? SZ.v m /\ SZ.v bn > 0 /\ SZ.v bn /? SZ.v n /\
                SZ.v bk > 0 /\ SZ.v bk /? SZ.v k))
  (it : nat)
  (#_ : squash (it < SZ.v k / SZ.v bk /\ it % 2 = 1))
  requires
    forall+ (tid : natlt nthr).
      (pipe_live_c (skewed_view bm bk skew sarA1)
         (ematrix_subtile eA (SZ.v bm) (SZ.v bk) block_row it) nthr tid **
       pipe_live_c (skewed_view bn bk skew sarB1)
         (ematrix_subtile eB (SZ.v bn) (SZ.v bk) block_col it) nthr tid **
       (pipe_sharing (skewed_view bm bk skew sarA0) nthr **
        pipe_sharing (skewed_view bn bk skew sarB0) nthr))
  ensures
    forall+ (tid : natlt nthr).
      pipe_p_c m n k bm bn bk skew eA eB block_row block_col
               sarA0 sarA1 sarB0 sarB1 nthr sq it tid
