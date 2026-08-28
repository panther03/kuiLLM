module Kuiops.SuperGEMM.Mm.SplitK.Shared

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiops.Array2.Strided
open Kuiper.Tensor.Tiling

module SZ = Kuiper.SizeT

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiops.Kernel.GEMM.TensorCore2D.KernelDesc { warp_tile_pts_to, precip }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile, warp_tile }
open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major, skew_residual, skew_split, skew_join }
open Kuiops.SuperGEMM.Mm.SplitK.Output { ws_warp_live, ws_warp_approximates }
open Kuiops.Kernel.GEMM.TensorCore2D.To.EpilogueState { epilogue_warp_input }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_sharing, pipe_live, pipe_q,
  unfold_pipe_q_even, unfold_pipe_q_odd }
open Kuiops.SHMem.CellSend { live_strided_chunks_block_sendable }
module T = Kuiper.Tensor
module P = Kuiops.SuperGEMM.Mm.Params
module FB = Kuiops.GEMM.T.FlipFlopBarrier2
module Aligned = Kuiops.SHMem.Aligned

(* ------------------------------------------------------------------ *)
(* One skewed pipeline buffer: read shares / disjoint writable chunks. *)
(* ------------------------------------------------------------------ *)

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
fn split_pipe_buffer_live
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (rows bk skew : szp)
  (nthr : pos)
  (sar : larray et (SZ.v rows * ldt bk skew))
  (#_ : squash (SZ.fits (SZ.v rows * ldt bk skew)))
  requires (exists* (v : Seq.seq et). pts_to sar v)
  ensures
    (forall+ (tid : natlt nthr). pipe_live (skewed_view rows bk skew sar) nthr tid) **
    skew_residual sar (SZ.v rows) (SZ.v bk) (SZ.v skew)
{
  skew_split (SZ.v rows) (SZ.v bk) (SZ.v skew) sar;
  with em. assert
    from_array (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew)) sar |-> em;
  rewrite each
    from_array (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew)) sar
    as skewed_view rows bk skew sar;
  FB.split_array2_into_strided_chunks (skewed_view rows bk skew sar) nthr;
  forevery_map
    #(natlt nthr)
    (fun tid -> FB.own_strided_chunks (skewed_view rows bk skew sar) em nthr tid)
    (fun tid -> pipe_live (skewed_view rows bk skew sar) nthr tid)
    fn tid {
      fold (FB.live_strided_chunks (skewed_view rows bk skew sar) nthr tid);
      fold (pipe_live (skewed_view rows bk skew sar) nthr tid);
    };
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
(* block_setup / block_teardown                                        *)
(* ------------------------------------------------------------------ *)

#push-options ""
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
{
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  unfold_live_c_shmems_cons sh #1.0R;
  unfold_live_c_shmems_cons (snd sh) #1.0R;
  unfold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  unfold_live_c_shmems_cons (snd (snd (snd sh))) #1.0R;
  unfold_live_c_shmems_nil (snd (snd (snd (snd sh)))) #1.0R;

  unfold_live_c_shmem (sar_a0 bm bn bk wm wn skew sh);
  split_pipe_buffer_live bm bk skew (SZ.v nthr) (sar_a0 bm bn bk wm wn skew sh);
  unfold_live_c_shmem (sar_a1 bm bn bk wm wn skew sh);
  split_pipe_buffer_live bm bk skew (SZ.v nthr) (sar_a1 bm bn bk wm wn skew sh);
  unfold_live_c_shmem (sar_b0 bm bn bk wm wn skew sh);
  split_pipe_buffer_live bn bk skew (SZ.v nthr) (sar_b0 bm bn bk wm wn skew sh);
  unfold_live_c_shmem (sar_b1 bm bn bk wm wn skew sh);
  split_pipe_buffer_live bn bk skew (SZ.v nthr) (sar_b1 bm bn bk wm wn skew sh);

  forevery_zip #(natlt (SZ.v nthr))
    (fun tid ->
      pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid)
    (fun tid ->
      pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
  forevery_zip #(natlt (SZ.v nthr))
    (fun tid ->
      pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
      pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid)
    (fun tid ->
      pipe_live (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
  forevery_zip #(natlt (SZ.v nthr))
    (fun tid ->
      (pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
       pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid) **
      pipe_live (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid)
    (fun tid ->
      pipe_live (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid);
  forevery_map #(natlt (SZ.v nthr))
    (fun tid ->
      ((pipe_live (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid **
        pipe_live (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid) **
       pipe_live (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) tid) **
      pipe_live (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) tid)
    (fun tid -> shared_thread_live bm bn bk wm wn skew sh nthr tid)
    fn tid {
      fold (shared_thread_live bm bn bk wm wn skew sh nthr tid);
    };
  forevery_zip #(natlt (SZ.v nthr))
    (fun tid ->
      kpre1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid tid)
    (fun tid -> shared_thread_live bm bn bk wm wn skew sh nthr tid);
  fold (block_frame bm bn bk wm wn skew sh);
}
#pop-options

#push-options ""
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
    block_frame bm bn bk wm wn skew sh
  ensures
    live_c_shmem (fst sh) **
    live_c_shmem (fst (snd sh)) **
    live_c_shmem (fst (snd (snd sh))) **
    live_c_shmem (fst (snd (snd (snd sh))))
{
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  unfold (block_frame bm bn bk wm wn skew sh);
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

#push-options ""
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
{
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;

  forevery_unzip #(natlt (SZ.v nthr))
    (fun tid -> kpost1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid tid)
    (fun tid -> shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid);

  forevery_map #(natlt (SZ.v nthr))
    (fun tid -> shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid)
    (fun tid ->
      pipe_q bm bn bk skew
        (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
        (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
        (SZ.v nthr) ktiles (ktiles - 1) tid)
    fn tid {
      unfold (shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid);
    };

  gather_last_pipe_buffers bm bn bk wm wn skew nthr sh ktiles (ktiles - 1) #();

  fold_live_c_shmems_nil (snd (snd (snd (snd sh)))) #1.0R;
  fold_live_c_shmems_cons (snd (snd (snd sh))) #1.0R;
  fold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  fold_live_c_shmems_cons (snd sh) #1.0R;
  fold_live_c_shmems_cons sh #1.0R;
}
#pop-options

(* ------------------------------------------------------------------ *)
(* Block-sendability                                                   *)
(* ------------------------------------------------------------------ *)

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
      is_block_array (sar_b1 bm bn bk wm wn skew sh))
=
  assert (c_shmem_inv (fst sh));
  assert (c_shmems_block_inv (snd sh));
  assert (c_shmem_inv (fst (snd sh)));
  assert (c_shmems_block_inv (snd (snd sh)));
  assert (c_shmem_inv (fst (snd (snd sh))));
  assert (c_shmems_block_inv (snd (snd (snd sh))));
  assert (c_shmem_inv (fst (snd (snd (snd sh)))));
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
  let sB0 = is_send_across_star _ _ #sendB0 #sendB1 in
  let sA1 = is_send_across_star _ _ #sendA1 #sB0 in
  is_send_across_star _ _ #sendA0 #sA1

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
  pipe_q_sendable bm bn bk skew
    (sar_a0 bm bn bk wm wn skew sh) (sar_a1 bm bn bk wm wn skew sh)
    (sar_b0 bm bn bk wm wn skew sh) (sar_b1 bm bn bk wm wn skew sh)
    #_ (SZ.v nthr) ktiles (ktiles - 1) tid

(* The workspace warp tile is a subtile of a global array, hence global, hence
   block-sendable; the contents are existential, so [is_send_across_exists]
   closes it. *)
let ws_warp_live_sendable
  (#et : Type0) {| scalar et |}
  (#mws #n : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gW : array2 et lW { is_global gW })
  (bm bn : pos) (wm wn : szp)
  (#_ : squash (out_ok bm bn wm wn (SZ.v mws) (SZ.v n)))
  (nblk : szp{SZ.v nblk == SZ.v mws / bm * (SZ.v n / bn)})
  (nthr : szp{SZ.v nthr == bm / (mfrag wm * frag) * (bn / (nfrag wn * frag)) * warp_size})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of
      (ws_warp_live gW bm bn frag frag (mfrag wm) (nfrag wn) bid tid)
=
  let wid : natlt (bm / (mfrag wm * frag) * (bn / (nfrag wn * frag))) = tid / warp_size in
  let tl = warp_tile (block_tile gW bm bn bid) (mfrag wm * frag) (nfrag wn * frag) wid in
  let ff (em : chest2 et (mfrag wm * frag) (nfrag wn * frag))
    : is_send_across block_of
        (warp_tile_pts_to gW bm bn frag frag (mfrag wm) (nfrag wn) bid wid em) =
    send_across_if_send_across_gpu
      (warp_tile_pts_to gW bm bn frag frag (mfrag wm) (nfrag wn) bid wid em)
      (is_send_across_global_tensor tl #(precip warp_size) em) in
  is_send_across_exists
    (fun em -> warp_tile_pts_to gW bm bn frag frag (mfrag wm) (nfrag wn) bid wid em)
    #ff

let kpre1_sendable
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_acc : Type0) {| scalar et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW { is_global gW })
  (bm bn wm wn : szp)
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of (
      kpre1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid tid)
=
  let output_send =
    ws_warp_live_sendable gW (SZ.v bm) (SZ.v bn) wm wn #_ nblk nthr bid tid in
  let pA = gA |-> Frac (fA /. (nblk * nthr)) eA in
  let pB = gB |-> Frac (fB /. (nblk * nthr)) eB in
  let pOutput = ws_warp_live gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn) bid tid in
  let pAlignedA = pure (aligned 16 (core gA)) in
  let pAlignedB = pure (aligned 16 (core gB)) in
  let pAlignedW = pure (aligned 16 (core gW)) in
  let pPure = pAlignedA ** pAlignedB ** pAlignedW in
  let sendA : is_send_across block_of pA =
    send_across_if_send_across_gpu pA
      (is_send_across_global_tensor gA #(fA /. (nblk * nthr)) eA) in
  let sendB : is_send_across block_of pB =
    send_across_if_send_across_gpu pB
      (is_send_across_global_tensor gB #(fB /. (nblk * nthr)) eB) in
  let sendAlignedA : is_send_across block_of pAlignedA =
    is_send_across_placeless pAlignedA
      #(placeless_pure (aligned 16 (core gA))) in
  let sendAlignedB : is_send_across block_of pAlignedB =
    is_send_across_placeless pAlignedB
      #(placeless_pure (aligned 16 (core gB))) in
  let sendAlignedW : is_send_across block_of pAlignedW =
    is_send_across_placeless pAlignedW
      #(placeless_pure (aligned 16 (core gW))) in
  let sendPure =
    is_send_across_star pAlignedA (pAlignedB ** pAlignedW)
      #sendAlignedA
      #(is_send_across_star pAlignedB pAlignedW #sendAlignedB #sendAlignedW) in
  let sendOutputPure =
    is_send_across_star pOutput pPure #output_send #sendPure in
  let sendBOutput =
    is_send_across_star pB (pOutput ** pPure) #sendB #sendOutputPure in
  is_send_across_star pA (pB ** pOutput ** pPure) #sendA #sendBOutput

let ws_warp_approximates_sendable
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#mws #n : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gW : array2 et lW { is_global gW })
  (bm bn : pos) (wm wn : szp)
  (#_ : squash (out_ok bm bn wm wn (SZ.v mws) (SZ.v n)))
  (nblk : szp{SZ.v nblk == SZ.v mws / bm * (SZ.v n / bn)})
  (nthr : szp{SZ.v nthr == bm / (mfrag wm * frag) * (bn / (nfrag wn * frag)) * warp_size})
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of
      (ws_warp_approximates gW bm bn frag frag (mfrag wm) (nfrag wn) bid tid rW)
=
  let wid : natlt (bm / (mfrag wm * frag) * (bn / (nfrag wn * frag))) = tid / warp_size in
  let rm = epilogue_warp_input rW bm bn frag frag (mfrag wm) (nfrag wn) bid tid in
  let tl = warp_tile (block_tile gW bm bn bid) (mfrag wm * frag) (nfrag wn * frag) wid in
  let ff (em : chest2 et (mfrag wm * frag) (nfrag wn * frag))
    : is_send_across block_of
        (warp_tile_pts_to gW bm bn frag frag (mfrag wm) (nfrag wn) bid wid em **
         pure (em %~ rm)) =
    is_send_across_star
      (warp_tile_pts_to gW bm bn frag frag (mfrag wm) (nfrag wn) bid wid em)
      (pure (em %~ rm))
      #(send_across_if_send_across_gpu
          (warp_tile_pts_to gW bm bn frag frag (mfrag wm) (nfrag wn) bid wid em)
          (is_send_across_global_tensor tl #(precip warp_size) em))
      #(is_send_across_placeless (pure (em %~ rm)) #(placeless_pure (em %~ rm))) in
  is_send_across_exists
    (fun em ->
      warp_tile_pts_to gW bm bn frag frag (mfrag wm) (nfrag wn) bid wid em **
      pure (em %~ rm))
    #ff

let kpost1_sendable
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_acc : Type0) {| scalar et_acc |} {| real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW { is_global gW })
  (bm bn wm wn : szp)
  (#_ : squash (out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of (
      kpost1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid tid)
=
  let output_send =
    ws_warp_approximates_sendable gW (SZ.v bm) (SZ.v bn) wm wn #_ nblk nthr rW bid tid in
  let pA = gA |-> Frac (fA /. (nblk * nthr)) eA in
  let pB = gB |-> Frac (fB /. (nblk * nthr)) eB in
  let pOutput =
    ws_warp_approximates gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn)
      bid tid rW in
  let pAlignedA = pure (aligned 16 (core gA)) in
  let pAlignedB = pure (aligned 16 (core gB)) in
  let pAlignedW = pure (aligned 16 (core gW)) in
  let pPure = pAlignedA ** pAlignedB ** pAlignedW in
  let sendA : is_send_across block_of pA =
    send_across_if_send_across_gpu pA
      (is_send_across_global_tensor gA #(fA /. (nblk * nthr)) eA) in
  let sendB : is_send_across block_of pB =
    send_across_if_send_across_gpu pB
      (is_send_across_global_tensor gB #(fB /. (nblk * nthr)) eB) in
  let sendAlignedA : is_send_across block_of pAlignedA =
    is_send_across_placeless pAlignedA
      #(placeless_pure (aligned 16 (core gA))) in
  let sendAlignedB : is_send_across block_of pAlignedB =
    is_send_across_placeless pAlignedB
      #(placeless_pure (aligned 16 (core gB))) in
  let sendAlignedW : is_send_across block_of pAlignedW =
    is_send_across_placeless pAlignedW
      #(placeless_pure (aligned 16 (core gW))) in
  let sendPure =
    is_send_across_star pAlignedA (pAlignedB ** pAlignedW)
      #sendAlignedA
      #(is_send_across_star pAlignedB pAlignedW #sendAlignedB #sendAlignedW) in
  let sendOutputPure =
    is_send_across_star pOutput pPure #output_send #sendPure in
  let sendBOutput =
    is_send_across_star pB (pOutput ** pPure) #sendB #sendOutputPure in
  is_send_across_star pA (pB ** pOutput ** pPure) #sendA #sendBOutput

let kpre_sendable
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
=
  let base_send =
    kpre1_sendable gA eA gB eB gW bm bn wm wn #_ fA fB nblk nthr bid tid in
  let shared_send =
    shared_thread_live_sendable bm bn bk wm wn skew nthr sh #_ tid in
  is_send_across_star
    (kpre1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid tid)
    (shared_thread_live bm bn bk wm wn skew sh nthr tid)
    #base_send #shared_send

let kpost_sendable
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
=
  let base_send =
    kpost1_sendable gA eA gB eB gW bm bn wm wn #_ fA fB nblk nthr rW bid tid in
  let shared_send =
    shared_thread_final_sendable bm bn bk wm wn skew nthr sh #_ ktiles tid in
  is_send_across_star
    (kpost1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid tid)
    (shared_thread_final bm bn bk wm wn skew sh nthr ktiles tid)
    #base_send #shared_send

(* ------------------------------------------------------------------ *)
(* Shared-buffer 16-byte alignment                                     *)
(* ------------------------------------------------------------------ *)

let shared_buffers_aligned16
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
=
  shmem_inv_components bm bn bk wm wn skew sh;
  Aligned.shmem_aligned16 (sar_a0 bm bn bk wm wn skew sh);
  Aligned.shmem_aligned16 (sar_a1 bm bn bk wm wn skew sh);
  Aligned.shmem_aligned16 (sar_b0 bm bn bk wm wn skew sh);
  Aligned.shmem_aligned16 (sar_b1 bm bn bk wm wn skew sh);
  assert (core (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh))
            == sar_a0 bm bn bk wm wn skew sh);
  assert (core (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh))
            == sar_a1 bm bn bk wm wn skew sh);
  assert (core (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh))
            == sar_b0 bm bn bk wm wn skew sh);
  assert (core (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh))
            == sar_b1 bm bn bk wm wn skew sh)
