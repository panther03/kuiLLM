module Kuiops.SuperGEMM.Mm.Epi.SharedGather

(* Target-agnostic shared-memory gather for the epilogue variant.

   [Mm.Shared.block_teardown] pins the per-thread post to [Mm.Shared.kpost],
   whose real target is a [chest_map] of the warp's product; the epilogue
   variant's target is a [chest_comb] against C, so [block_teardown] cannot be
   reused directly.  Only its GLOBAL side differs -- the shared-memory gather
   below is identical.

   TODO(upstream): [gather_pipe_buffer], [gather_pipe_buffer_live],
   [gather_scratch_from_threads] and [gather_last_pipe_buffers] are copied
   verbatim from [Kuiops.SuperGEMM.Mm.Shared] (they are private to that
   module); [shared_gather] is the target-agnostic half of
   [Mm.Shared.block_teardown] and belongs there. *)

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major, skew_residual, skew_join }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile_idx_rows, block_tile_idx_cols, warp_tile_idx_rows, warp_tile_idx_cols }
open Kuiops.SuperGEMM.Mm.Output { output_fragment', output_lane_approximates' }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Shared
open Kuiops.SHMem.CellSend { live_strided_chunks_block_sendable }
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_sharing, pipe_live, pipe_q,
  unfold_pipe_q_even, unfold_pipe_q_odd }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params
module FB = Kuiops.GEMM.T.FlipFlopBarrier2

let scratch_fits
  (et_ab et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  : squash (SZ.fits (warps bm bn wm wn * frag * lde et_acc wn))
= ()

ghost
fn gather_pipe_buffer
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (rows bk skew : szp)
  (nthr : pos)
  (sar : larray et (SZ.v rows * ldt bk skew))
  (#_ : squash (SZ.fits (SZ.v rows * ldt bk skew)))
  requires
    (forall+ (_ : natlt nthr). pipe_sharing (skewed_view rows bk skew sar) nthr) **
    skew_residual sar (SZ.v rows) (SZ.v bk) (SZ.v skew)
  ensures (exists* (v : Seq.seq et). pts_to sar v)
{
  forevery_map
    #(natlt nthr)
    (fun _ -> pipe_sharing (skewed_view rows bk skew sar) nthr)
    (fun _ -> exists* em. skewed_view rows bk skew sar |-> Frac (1.0R /. nthr) em)
    fn _ {
      unfold (pipe_sharing (skewed_view rows bk skew sar) nthr);
      with em. assert FB.bp_sharing (skewed_view rows bk skew sar) em nthr;
      unfold (FB.bp_sharing (skewed_view rows bk skew sar) em nthr);
    };
  tensor_gather_n_underspec (skewed_view rows bk skew sar) nthr;
  with em. assert skewed_view rows bk skew sar |-> em;
  rewrite each
    skewed_view rows bk skew sar
    as from_array (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew)) sar;
  skew_join (SZ.v rows) (SZ.v bk) (SZ.v skew) sar;
}

ghost
fn gather_pipe_buffer_live
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (rows bk skew : szp)
  (nthr : pos)
  (sar : larray et (SZ.v rows * ldt bk skew))
  (#_ : squash (SZ.fits (SZ.v rows * ldt bk skew)))
  requires
    (forall+ (tid : natlt nthr). pipe_live (skewed_view rows bk skew sar) nthr tid) **
    skew_residual sar (SZ.v rows) (SZ.v bk) (SZ.v skew)
  ensures (exists* (v : Seq.seq et). pts_to sar v)
{
  forevery_map
    #(natlt nthr)
    (fun tid -> pipe_live (skewed_view rows bk skew sar) nthr tid)
    (fun tid -> FB.live_strided_chunks (skewed_view rows bk skew sar) nthr tid)
    fn tid {
      unfold (pipe_live (skewed_view rows bk skew sar) nthr tid);
    };
  FB.join_array2_from_strided_chunks_underspec (skewed_view rows bk skew sar) nthr;
  with em. assert skewed_view rows bk skew sar |-> em;
  rewrite each
    skewed_view rows bk skew sar
    as from_array (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew)) sar;
  skew_join (SZ.v rows) (SZ.v bk) (SZ.v skew) sar;
}

(* ------------------------------------------------------------------ *)
(* Split the skewed epilogue scratch into per-warp / per-lane shares.  *)

#push-options "--split_queries no"
ghost
fn gather_scratch_from_threads
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  requires
    (forall+ (tid : natlt (SZ.v nthr)).
      scratch_tile_live bm bn bk wm wn skew sh nthr tid) **
    skew_residual (sar_scratch bm bn bk wm wn skew sh)
      (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)
  ensures (exists* (v : Seq.seq et_acc). pts_to (sar_scratch bm bn bk wm wn skew sh) v)
{
  P.warp_divides_nthr bm bn wm wn;
  let scratch_fits_pf = scratch_fits et_ab et_acc bm bn bk wm wn skew;
  forevery_map
    #(natlt (SZ.v nthr))
    (fun tid -> scratch_tile_live bm bn bk wm wn skew sh nthr tid)
    (fun tid ->
      exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
        scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size)
          |-> Frac (1.0R /. warp_size) eAcc)
    fn tid {
      unfold (scratch_tile_live bm bn bk wm wn skew sh nthr tid);
    };
  forevery_factor' (SZ.v nthr) (warps bm bn wm wn) (SZ.v warp_size)
    (fun wid _lane ->
      exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
        scratch_tile bm bn bk wm wn skew sh wid
          |-> Frac (1.0R /. warp_size) eAcc);
  forevery_map
    (fun (wid : natlt (warps bm bn wm wn)) ->
      forall+ (_lane : natlt (SZ.v warp_size)).
        exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
          scratch_tile bm bn bk wm wn skew sh wid
            |-> Frac (1.0R /. warp_size) eAcc)
    (fun (wid : natlt (warps bm bn wm wn)) ->
      exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
        scratch_tile bm bn bk wm wn skew sh wid |-> eAcc)
    fn wid {
      tensor_gather_n_underspec (scratch_tile bm bn bk wm wn skew sh wid) (SZ.v warp_size) #1.0R;
    };
  forevery_map
    (fun (wid : natlt (warps bm bn wm wn)) ->
      exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
        scratch_tile bm bn bk wm wn skew sh wid |-> eAcc)
    (fun (wid : natlt (warps bm bn wm wn)) ->
      forall+ (tc : natlt 1).
        exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
          array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid tc
            |-> eAcc)
    fn wid {
      rewrite each
        scratch_tile bm bn bk wm wn skew sh wid
        as array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid 0;
      forevery_singleton_intro #(natlt 1)
        (fun tc ->
          exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
            array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid tc
              |-> eAcc);
    };
  forevery_rw_size2
    (warps bm bn wm wn) ((warps bm bn wm wn * frag) / frag)
    1 (SZ.v wn / SZ.v wn)
    #(fun (wid : natlt (warps bm bn wm wn)) (tc : natlt 1) ->
      exists* (eAcc : chest (frag @| SZ.v wn @| INil) et_acc).
        array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid tc
          |-> eAcc);
  array2_untile_underspec (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn);
  rewrite each
    scratch_matrix bm bn bk wm wn skew sh
    as from_array
      (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
      (sar_scratch bm bn bk wm wn skew sh);
  skew_join (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)
    (sar_scratch bm bn bk wm wn skew sh);
}
#pop-options

#push-options "--split_queries no"
(* Resolve the parity of the last-used k-tile [it0] and gather the four
   pipeline buffers back to full ownership.  Factored into its own [fn] with an
   explicit postcondition so the two-way parity [if] does not force Pulse to
   infer a parity-indexed join type over the mixed scratch/buffer element
   types. *)
ghost
fn gather_last_pipe_buffers
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (ktiles it0 : nat)
  (#_ : squash (it0 < ktiles))
  requires
    (forall+ (tid : natlt (SZ.v nthr)).
      pipe_q bm bn bk skew
        (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
        (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
        (SZ.v nthr) ktiles it0 tid) **
    skew_residual (sar_a0 bm bn bk wm wn skew sh) (SZ.v bm) (SZ.v bk) (SZ.v skew) **
    skew_residual (sar_a1 bm bn bk wm wn skew sh) (SZ.v bm) (SZ.v bk) (SZ.v skew) **
    skew_residual (sar_b0 bm bn bk wm wn skew sh) (SZ.v bn) (SZ.v bk) (SZ.v skew) **
    skew_residual (sar_b1 bm bn bk wm wn skew sh) (SZ.v bn) (SZ.v bk) (SZ.v skew)
  ensures
    live_c_shmem (fst sh) **
    live_c_shmem (fst (snd sh)) **
    live_c_shmem (fst (snd (snd sh))) **
    live_c_shmem (fst (snd (snd (snd sh))))
{
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  if (it0 % 2 = 0) {
    unfold_pipe_q_even bm bn bk skew
      (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
      (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
      (SZ.v nthr) ktiles it0 #();
    forevery_unzip #(natlt (SZ.v nthr))
      (fun _ ->
        pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr))
      (fun tid ->
        pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
        pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
        pipe_live (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
    forevery_unzip #(natlt (SZ.v nthr))
      (fun _ ->
        pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr))
      (fun tid ->
        pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
        pipe_live (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
    forevery_unzip #(natlt (SZ.v nthr))
      (fun tid ->
        pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid)
      (fun tid ->
        pipe_live (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
    gather_pipe_buffer bm bk skew (SZ.v nthr) (sar_a0 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst sh);
    gather_pipe_buffer bn bk skew (SZ.v nthr) (sar_b0 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst (snd (snd sh)));
    gather_pipe_buffer_live bm bk skew (SZ.v nthr) (sar_a1 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst (snd sh));
    gather_pipe_buffer_live bn bk skew (SZ.v nthr) (sar_b1 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst (snd (snd (snd sh))));
  } else {
    unfold_pipe_q_odd bm bn bk skew
      (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
      (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
      (SZ.v nthr) ktiles it0 #();
    forevery_unzip #(natlt (SZ.v nthr))
      (fun _ ->
        pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr))
      (fun tid ->
        pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
        pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
        pipe_live (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
    forevery_unzip #(natlt (SZ.v nthr))
      (fun _ ->
        pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr))
      (fun tid ->
        pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
        pipe_live (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
    forevery_unzip #(natlt (SZ.v nthr))
      (fun tid ->
        pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid)
      (fun tid ->
        pipe_live (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
    gather_pipe_buffer bm bk skew (SZ.v nthr) (sar_a1 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst (snd sh));
    gather_pipe_buffer bn bk skew (SZ.v nthr) (sar_b1 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst (snd (snd (snd sh))));
    gather_pipe_buffer_live bm bk skew (SZ.v nthr) (sar_a0 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst sh);
    gather_pipe_buffer_live bn bk skew (SZ.v nthr) (sar_b0 bm bn bk wm wn skew sh);
    fold_live_c_shmem (fst (snd (snd sh)));
  }
}
#pop-options


(* The shared-memory half of [Mm.Shared.block_teardown]: gather the per-thread
   final shared state back to whole-allocation ownership.  No global-side
   pre/post, hence usable by any epilogue. *)
#push-options "--split_queries no"
ghost
fn shared_gather
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (ktiles : pos)
  ()
  requires
    (forall+ (tid : natlt nthr).
      shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid) **
    block_frame bm bn bk wm wn skew sh
  ensures
    live_c_shmems sh
{
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  unfold (block_frame bm bn bk wm wn skew sh);

  forevery_map #(natlt (SZ.v nthr))
    (fun tid -> shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid)
    (fun tid ->
      pipe_q bm bn bk skew
        (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
        (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
        (SZ.v nthr) ktiles (ktiles - 1) tid **
      scratch_tile_live bm bn bk wm wn skew sh nthr tid)
    fn tid {
      unfold (shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid);
    };
  forevery_unzip #(natlt (SZ.v nthr))
    (fun tid ->
      pipe_q bm bn bk skew
        (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
        (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
        (SZ.v nthr) ktiles (ktiles - 1) tid)
    (fun tid -> scratch_tile_live bm bn bk wm wn skew sh nthr tid);

  gather_last_pipe_buffers bm bn bk wm wn skew nthr sh ktiles (ktiles - 1) #();

  gather_scratch_from_threads bm bn bk wm wn skew nthr sh;
  fold_live_c_shmem (fst (snd (snd (snd (snd sh)))));

  fold_live_c_shmems_nil (snd (snd (snd (snd (snd sh))))) #1.0R;
  fold_live_c_shmems_cons (snd (snd (snd (snd sh)))) #1.0R;
  fold_live_c_shmems_cons (snd (snd (snd sh))) #1.0R;
  fold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  fold_live_c_shmems_cons (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;
}
#pop-options

(* TODO(upstream): copied verbatim from [Kuiops.SuperGEMM.Mm.Shared] (private
   there). *)
#push-options "--split_queries always"
let output_lane_approximates_sendable'
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et lD { is_global gD })
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ SZ.v m /\ bn /?+ SZ.v n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (nblk : szp{SZ.v nblk == SZ.v m / bm * (SZ.v n / bn)})
  (nthr : szp{SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (bid : natlt nblk)
  (tid : natlt nthr)
  (rD : chest2 real (wm * tm) (wn * tn))
  : is_send_across block_of (output_lane_approximates' gD bm bn tm tn wm wn bid tid rD)
=
  FStar.Math.Lemmas.cancel_mul_div wm tm;
  FStar.Math.Lemmas.cancel_mul_div wn tn;
  let wid : natlt (bm / (wm * tm) * (bn / (wn * tn))) = tid / warp_size in
  let lane : natlt warp_size = tid % warp_size in
  assert_norm (
    reveal (block_tile_idx_rows (SZ.v m) (SZ.v n) bm bn bid)
      == bid / (SZ.v n / bn));
  assert_norm (
    reveal (block_tile_idx_cols (SZ.v m) (SZ.v n) bm bn bid)
      == bid % (SZ.v n / bn));
  assert_norm (
    reveal (warp_tile_idx_rows bm bn (wm * tm) (wn * tn) wid)
      == wid / (bn / (wn * tn)));
  assert_norm (
    reveal (warp_tile_idx_cols bm bn (wm * tm) (wn * tn) wid)
      == wid % (bn / (wn * tn)));
  assert (forall (mi : natlt wm) (nj : natlt wn).
    is_global (output_fragment' gD bm bn tm tn wm wn bid wid mi nj));
  solve
#pop-options


(* TODO(upstream): copied verbatim from [Kuiops.SuperGEMM.Mm.Shared]
   (private there). *)

let pipe_sharing_sendable
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (rows bk skew : szp)
  (sar : larray et (SZ.v rows * ldt bk skew))
  (#_ : squash (is_block_array sar))
  (nthr : pos)
  : is_send_across block_of (pipe_sharing (skewed_view rows bk skew sar) nthr)
=
  let m = skewed_view rows bk skew sar in
  let ff (em : chest2 et (SZ.v rows) (SZ.v bk))
    : is_send_across block_of (FB.bp_sharing m em nthr) =
    is_send_across_tensor m block_of #_ #(1.0R /. nthr) em in
  is_send_across_exists (fun em -> FB.bp_sharing m em nthr) #ff

(* [pipe_live] = [live_strided_chunks] of a block array is block-sendable
   (Part B: [Kuiops.SHMem.CellSend]). *)
let pipe_live_sendable
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (rows bk skew : szp)
  (sar : larray et (SZ.v rows * ldt bk skew))
  (#_ : squash (is_block_array sar))
  (nthr : pos)
  (tid : natlt nthr)
  : is_send_across block_of (pipe_live (skewed_view rows bk skew sar) nthr tid)
=
  live_strided_chunks_block_sendable (skewed_view rows bk skew sar) #_ nthr tid

let scratch_tile_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (is_block_array (sar_scratch bm bn bk wm wn skew sh)))
  (tid : natlt (SZ.v nthr))
  (eAcc : chest2 et_acc frag (SZ.v wn))
  : is_send_across block_of (
      scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size)
        |-> Frac (1.0R /. warp_size) eAcc)
= is_send_across_tensor
    (scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size))
    block_of #_ #(1.0R /. warp_size) eAcc

let scratch_tile_live_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (is_block_array (sar_scratch bm bn bk wm wn skew sh)))
  (tid : natlt (SZ.v nthr))
  : is_send_across block_of (scratch_tile_live bm bn bk wm wn skew sh nthr tid)
=
  let ff (eAcc : chest2 et_acc frag (SZ.v wn))
    : is_send_across block_of (
        scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size)
          |-> Frac (1.0R /. warp_size) eAcc) =
    scratch_tile_sendable bm bn bk wm wn skew nthr sh #_ tid eAcc in
  is_send_across_exists
    (fun eAcc ->
      scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size)
        |-> Frac (1.0R /. warp_size) eAcc)
    #ff

let shmem_inv_components
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  : squash (
      is_block_array (sar_a0 bm bn bk wm wn skew sh) /\
      is_block_array (sar_a1 bm bn bk wm wn skew sh) /\
      is_block_array (sar_b0 bm bn bk wm wn skew sh) /\
      is_block_array (sar_b1 bm bn bk wm wn skew sh) /\
      is_block_array (sar_scratch bm bn bk wm wn skew sh))
=
  assert (c_shmem_inv (fst sh));
  assert (c_shmems_inv (snd sh));
  assert (c_shmem_inv (fst (snd sh)));
  assert (c_shmems_inv (snd (snd sh)));
  assert (c_shmem_inv (fst (snd (snd sh))));
  assert (c_shmems_inv (snd (snd (snd sh))));
  assert (c_shmem_inv (fst (snd (snd (snd sh)))));
  assert (c_shmems_inv (snd (snd (snd (snd sh)))));
  assert (c_shmem_inv (fst (snd (snd (snd (snd sh))))));
  ()

let shared_thread_live_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (tid : natlt (SZ.v nthr))
  : is_send_across block_of (shared_thread_live bm bn bk wm wn skew sh nthr tid)
=
  let invs = shmem_inv_components bm bn bk wm wn skew sh #_ in
  let sendA0 =
    pipe_live_sendable bm bk skew (sar_a0 bm bn bk wm wn skew sh) #_ (SZ.v nthr) tid in
  let sendA1 =
    pipe_live_sendable bm bk skew (sar_a1 bm bn bk wm wn skew sh) #_ (SZ.v nthr) tid in
  let sendB0 =
    pipe_live_sendable bn bk skew (sar_b0 bm bn bk wm wn skew sh) #_ (SZ.v nthr) tid in
  let sendB1 =
    pipe_live_sendable bn bk skew (sar_b1 bm bn bk wm wn skew sh) #_ (SZ.v nthr) tid in
  let sendScratch =
    scratch_tile_live_sendable bm bn bk wm wn skew nthr sh #_ tid in
  let sB1S = is_send_across_star _ _ #sendB1 #sendScratch in
  let sB0 = is_send_across_star _ _ #sendB0 #sB1S in
  let sA1 = is_send_across_star _ _ #sendA1 #sB0 in
  is_send_across_star _ _ #sendA0 #sA1

(* Block-sendability of one [pipe_q] product.  The [if it >= ktiles] guard is
   discharged at the term level, so no separate parity reasoning is needed:
   both pieces of each pair are block arrays, so both the [pipe_sharing] and
   [pipe_live] halves are block-sendable regardless of which is which. *)
let pipe_q_sendable
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (bm bn bk skew : szp)
  (sarA0 sarA1 : larray et (SZ.v bm * ldt bk skew))
  (sarB0 sarB1 : larray et (SZ.v bn * ldt bk skew))
  (#_ : squash (is_block_array sarA0 /\ is_block_array sarA1 /\
                is_block_array sarB0 /\ is_block_array sarB1))
  (nthr : pos)
  (ktiles : nat)
  (it : nat)
  (tid : natlt nthr)
  : is_send_across block_of
      (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
=
  if it >= ktiles then
    (is_send_across_placeless #_ #block_of emp
      <: is_send_across block_of
           (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid))
  else begin
    let mAb = skewed_view bm bk skew (if it % 2 = 0 then sarA0 else sarA1) in
    let mAo = skewed_view bm bk skew (if it % 2 = 0 then sarA1 else sarA0) in
    let mBb = skewed_view bn bk skew (if it % 2 = 0 then sarB0 else sarB1) in
    let mBo = skewed_view bn bk skew (if it % 2 = 0 then sarB1 else sarB0) in
    let sAb : is_send_across block_of (pipe_sharing mAb nthr) =
      pipe_sharing_sendable bm bk skew (if it % 2 = 0 then sarA0 else sarA1) #_ nthr in
    let sBb : is_send_across block_of (pipe_sharing mBb nthr) =
      pipe_sharing_sendable bn bk skew (if it % 2 = 0 then sarB0 else sarB1) #_ nthr in
    let sAo : is_send_across block_of (pipe_live mAo nthr tid) =
      pipe_live_sendable bm bk skew (if it % 2 = 0 then sarA1 else sarA0) #_ nthr tid in
    let sBo : is_send_across block_of (pipe_live mBo nthr tid) =
      pipe_live_sendable bn bk skew (if it % 2 = 0 then sarB1 else sarB0) #_ nthr tid in
    let sAoBo = is_send_across_star _ _ #sAo #sBo in
    let sBbAoBo = is_send_across_star _ _ #sBb #sAoBo in
    (is_send_across_star _ _ #sAb #sBbAoBo
      <: is_send_across block_of
           (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid))
  end

let shared_thread_final_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (ktiles : pos)
  (tid : natlt (SZ.v nthr))
  : is_send_across block_of (shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid)
=
  let invs = shmem_inv_components bm bn bk wm wn skew sh #_ in
  let sendQ =
    pipe_q_sendable bm bn bk skew
      (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
      (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
      #_ (SZ.v nthr) ktiles (ktiles - 1) tid in
  let sendScratch =
    scratch_tile_live_sendable bm bn bk wm wn skew nthr sh #_ tid in
  is_send_across_star _ _ #sendQ #sendScratch
