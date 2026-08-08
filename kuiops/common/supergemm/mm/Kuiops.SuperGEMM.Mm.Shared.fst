module Kuiops.SuperGEMM.Mm.Shared

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc { output_lane_live, output_fragment, live_lane_cells }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile_idx_rows, block_tile_idx_cols, warp_tile_idx_rows, warp_tile_idx_cols }
open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major, skew_residual, skew_split, skew_join }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Barrier { skewed_view, pipe_sharing }
module P = Kuiops.SuperGEMM.Mm.Params
module FB = Kuiops.GEMM.T.FlipFlopBarrier2

(* ------------------------------------------------------------------ *)
(* Extract the fits fact for the skewed scratch from [constraints].   *)
(* ------------------------------------------------------------------ *)

let scratch_fits
  (et_ab et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  : squash (SZ.fits (warps bm bn wm wn * frag * lde et_acc wn))
= ()

(* ------------------------------------------------------------------ *)
(* Split one skewed pipeline buffer into per-thread read shares.       *)
(* ------------------------------------------------------------------ *)

ghost
fn split_pipe_buffer
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (rows bk skew : szp)
  (nthr : pos)
  (sar : larray et (SZ.v rows * ldt bk skew))
  (#_ : squash (SZ.fits (SZ.v rows * ldt bk skew)))
  requires (exists* (v : Seq.seq et). pts_to sar v)
  ensures
    (forall+ (_ : natlt nthr). pipe_sharing (skewed_view rows bk skew sar) nthr) **
    skew_residual sar (SZ.v rows) (SZ.v bk) (SZ.v skew)
{
  skew_split (SZ.v rows) (SZ.v bk) (SZ.v skew) sar;
  with em. assert
    from_array (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew)) sar |-> em;
  rewrite each
    from_array (l2_skewed_row_major (SZ.v rows) (SZ.v bk) (SZ.v skew)) sar
    as skewed_view rows bk skew sar;
  tensor_share_n (skewed_view rows bk skew sar) nthr;
  forevery_map
    #(natlt nthr)
    (fun _ -> skewed_view rows bk skew sar |-> Frac (1.0R /. nthr) em)
    (fun _ -> pipe_sharing (skewed_view rows bk skew sar) nthr)
    fn _ {
      fold (FB.bp_sharing (skewed_view rows bk skew sar) em nthr);
      fold (pipe_sharing (skewed_view rows bk skew sar) nthr);
    };
}

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

(* ------------------------------------------------------------------ *)
(* Split the skewed epilogue scratch into per-warp / per-lane shares.  *)
(* ------------------------------------------------------------------ *)

#push-options "--split_queries no"
ghost
fn split_scratch_to_threads
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  requires (exists* (v : Seq.seq et_acc). pts_to (sar_scratch bm bn bk wm wn skew sh) v)
  ensures
    (forall+ (tid : natlt (SZ.v nthr)).
      scratch_tile_live bm bn bk wm wn skew sh nthr tid) **
    skew_residual (sar_scratch bm bn bk wm wn skew sh)
      (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)
{
  P.warp_divides_nthr bm bn wm wn;
  let scratch_fits_pf = scratch_fits et_ab et_acc bm bn bk wm wn skew;
  skew_split (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc)
    (sar_scratch bm bn bk wm wn skew sh);
  with em. assert
    from_array
      (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
      (sar_scratch bm bn bk wm wn skew sh) |-> em;
  rewrite each
    from_array
      (l2_skewed_row_major (warps bm bn wm wn * frag) (SZ.v wn) (eskew et_acc))
      (sar_scratch bm bn bk wm wn skew sh)
    as scratch_matrix bm bn bk wm wn skew sh;
  array2_tile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) #em;
  forevery_rw_size2
    ((warps bm bn wm wn * frag) / frag) (warps bm bn wm wn)
    (SZ.v wn / SZ.v wn) 1
    #(fun (wid : natlt ((warps bm bn wm wn * frag) / frag))
          (tc : natlt (SZ.v wn / SZ.v wn)) ->
      array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid tc
        |-> ematrix_subtile em frag (SZ.v wn) wid tc);
  forevery_map
    (fun (wid : natlt (warps bm bn wm wn)) ->
      forall+ (tc : natlt 1).
        array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid tc
          |-> ematrix_subtile em frag (SZ.v wn) wid tc)
    (fun (wid : natlt (warps bm bn wm wn)) ->
      array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid 0
        |-> ematrix_subtile em frag (SZ.v wn) wid 0)
    fn wid {
      forevery_singleton_elim #(natlt 1)
        (fun (tc : natlt 1) ->
          array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid tc
            |-> ematrix_subtile em frag (SZ.v wn) wid tc);
    };

  forevery_map
    (fun (wid : natlt (warps bm bn wm wn)) ->
      array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid 0
        |-> ematrix_subtile em frag (SZ.v wn) wid 0)
    (fun (wid : natlt (warps bm bn wm wn)) ->
      forall+ (lane : natlt (SZ.v warp_size)).
        scratch_tile_live bm bn bk wm wn skew sh nthr
          (wid * SZ.v warp_size + lane))
    fn wid {
      rewrite each
        array2_subtile (scratch_matrix bm bn bk wm wn skew sh) frag (SZ.v wn) wid 0
        as scratch_tile bm bn bk wm wn skew sh wid;
      tensor_share_n (scratch_tile bm bn bk wm wn skew sh wid) (SZ.v warp_size);
      forevery_map
        #(natlt (SZ.v warp_size))
        (fun lane ->
          scratch_tile bm bn bk wm wn skew sh wid
            |-> Frac (1.0R /. warp_size) (ematrix_subtile em frag (SZ.v wn) wid 0))
        (fun lane ->
          scratch_tile_live bm bn bk wm wn skew sh nthr
            (wid * SZ.v warp_size + lane))
        fn lane {
          assert pure ((wid * SZ.v warp_size + lane) / SZ.v warp_size == wid);
          rewrite each
            scratch_tile bm bn bk wm wn skew sh wid
            as scratch_tile bm bn bk wm wn skew sh
              ((wid * SZ.v warp_size + lane) / SZ.v warp_size);
          fold (scratch_tile_live bm bn bk wm wn skew sh nthr
            (wid * SZ.v warp_size + lane));
        };
    };
  forevery_unfactor' (SZ.v nthr) (warps bm bn wm wn) (SZ.v warp_size) _;
  forevery_map
    #(natlt (SZ.v nthr))
    (fun tid ->
      scratch_tile_live bm bn bk wm wn skew sh nthr
        ((tid / SZ.v warp_size) * SZ.v warp_size + tid % SZ.v warp_size))
    (fun tid -> scratch_tile_live bm bn bk wm wn skew sh nthr tid)
    fn tid {
      unfold (scratch_tile_live bm bn bk wm wn skew sh nthr
        ((tid / SZ.v warp_size) * SZ.v warp_size + tid % SZ.v warp_size));
      assert pure (
        (tid / SZ.v warp_size) * SZ.v warp_size + tid % SZ.v warp_size == tid);
      rewrite each
        scratch_tile bm bn bk wm wn skew sh
          (((tid / SZ.v warp_size) * SZ.v warp_size + tid % SZ.v warp_size)
            / SZ.v warp_size)
        as scratch_tile bm bn bk wm wn skew sh (tid / SZ.v warp_size);
      fold (scratch_tile_live bm bn bk wm wn skew sh nthr tid);
    };
}
#pop-options

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

(* ------------------------------------------------------------------ *)
(* block_setup / block_teardown                                        *)
(* ------------------------------------------------------------------ *)

#push-options "--split_queries no"
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
{
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  unfold_live_c_shmems_cons sh #1.0R;
  unfold_live_c_shmems_cons (snd sh) #1.0R;
  unfold_live_c_shmems_cons (snd (snd sh)) #1.0R;
  unfold_live_c_shmems_cons (snd (snd (snd sh))) #1.0R;
  unfold_live_c_shmems_cons (snd (snd (snd (snd sh)))) #1.0R;
  unfold_live_c_shmems_nil (snd (snd (snd (snd (snd sh))))) #1.0R;

  unfold_live_c_shmem (sar_a0 bm bn bk wm wn skew sh);
  split_pipe_buffer bm bk skew (SZ.v nthr) (sar_a0 bm bn bk wm wn skew sh);
  unfold_live_c_shmem (sar_a1 bm bn bk wm wn skew sh);
  split_pipe_buffer bm bk skew (SZ.v nthr) (sar_a1 bm bn bk wm wn skew sh);
  unfold_live_c_shmem (sar_b0 bm bn bk wm wn skew sh);
  split_pipe_buffer bn bk skew (SZ.v nthr) (sar_b0 bm bn bk wm wn skew sh);
  unfold_live_c_shmem (sar_b1 bm bn bk wm wn skew sh);
  split_pipe_buffer bn bk skew (SZ.v nthr) (sar_b1 bm bn bk wm wn skew sh);
  unfold_live_c_shmem (sar_scratch bm bn bk wm wn skew sh);
  split_scratch_to_threads bm bn bk wm wn skew nthr sh;

  forevery_zip #(natlt (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr));
  forevery_zip #(natlt (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr));
  forevery_zip #(natlt (SZ.v nthr))
    (fun _ ->
      (pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
       pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr)) **
      pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr));
  forevery_zip #(natlt (SZ.v nthr))
    (fun _ ->
      ((pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
        pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr)) **
       pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr)) **
      pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun tid -> scratch_tile_live bm bn bk wm wn skew sh nthr tid);
  forevery_map #(natlt (SZ.v nthr))
    (fun tid ->
      (((pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
         pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr)) **
        pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr)) **
       pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr)) **
      scratch_tile_live bm bn bk wm wn skew sh nthr tid)
    (fun tid -> shared_thread_live bm bn bk wm wn skew sh nthr tid)
    fn tid {
      fold (shared_thread_live bm bn bk wm wn skew sh nthr tid);
    };
  forevery_zip #(natlt (SZ.v nthr))
    (fun tid ->
      kpre1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid)
    (fun tid -> shared_thread_live bm bn bk wm wn skew sh nthr tid);
  forevery_map #(natlt (SZ.v nthr))
    (fun tid ->
      kpre1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid **
      shared_thread_live bm bn bk wm wn skew sh nthr tid)
    (fun tid ->
      kpre gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid)
    fn tid {
      fold (kpre gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid);
    };
  fold (block_frame bm bn bk wm wn skew sh);
}
#pop-options

#push-options "--split_queries no"
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
{
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  unfold (block_frame bm bn bk wm wn skew sh);

  forevery_map #(natlt (SZ.v nthr))
    (fun tid ->
      kpost gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid)
    (fun tid ->
      kpost1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid **
      shared_thread_live bm bn bk wm wn skew sh nthr tid)
    fn tid {
      unfold (kpost gA eA gB eB gD bm bn bk wm wn skew tm tn wmf wnf
        fA fB nblk nthr sh bid tid);
    };
  forevery_unzip #(natlt (SZ.v nthr))
    (fun tid ->
      kpost1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid)
    (fun tid -> shared_thread_live bm bn bk wm wn skew sh nthr tid);

  forevery_map #(natlt (SZ.v nthr))
    (fun tid -> shared_thread_live bm bn bk wm wn skew sh nthr tid)
    (fun tid ->
      pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      scratch_tile_live bm bn bk wm wn skew sh nthr tid)
    fn tid {
      unfold (shared_thread_live bm bn bk wm wn skew sh nthr tid);
    };
  forevery_unzip #(natlt (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bm bk skew (sar_a0 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun tid ->
      pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      scratch_tile_live bm bn bk wm wn skew sh nthr tid);
  forevery_unzip #(natlt (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bm bk skew (sar_a1 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun tid ->
      pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      scratch_tile_live bm bn bk wm wn skew sh nthr tid);
  forevery_unzip #(natlt (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bn bk skew (sar_b0 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun tid ->
      pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr) **
      scratch_tile_live bm bn bk wm wn skew sh nthr tid);
  forevery_unzip #(natlt (SZ.v nthr))
    (fun _ ->
      pipe_sharing (skewed_view bn bk skew (sar_b1 bm bn bk wm wn skew sh)) (SZ.v nthr))
    (fun tid -> scratch_tile_live bm bn bk wm wn skew sh nthr tid);

  gather_pipe_buffer bm bk skew (SZ.v nthr) (sar_a0 bm bn bk wm wn skew sh);
  fold_live_c_shmem (fst sh);
  gather_pipe_buffer bm bk skew (SZ.v nthr) (sar_a1 bm bn bk wm wn skew sh);
  fold_live_c_shmem (fst (snd sh));
  gather_pipe_buffer bn bk skew (SZ.v nthr) (sar_b0 bm bn bk wm wn skew sh);
  fold_live_c_shmem (fst (snd (snd sh)));
  gather_pipe_buffer bn bk skew (SZ.v nthr) (sar_b1 bm bn bk wm wn skew sh);
  fold_live_c_shmem (fst (snd (snd (snd sh))));
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
    pipe_sharing_sendable bm bk skew (sar_a0 bm bn bk wm wn skew sh) #_ (SZ.v nthr) in
  let sendA1 =
    pipe_sharing_sendable bm bk skew (sar_a1 bm bn bk wm wn skew sh) #_ (SZ.v nthr) in
  let sendB0 =
    pipe_sharing_sendable bn bk skew (sar_b0 bm bn bk wm wn skew sh) #_ (SZ.v nthr) in
  let sendB1 =
    pipe_sharing_sendable bn bk skew (sar_b1 bm bn bk wm wn skew sh) #_ (SZ.v nthr) in
  let sendScratch =
    scratch_tile_live_sendable bm bn bk wm wn skew nthr sh #_ tid in
  let sB1S = is_send_across_star _ _ #sendB1 #sendScratch in
  let sB0 = is_send_across_star _ _ #sendB0 #sB1S in
  let sA1 = is_send_across_star _ _ #sendA1 #sB0 in
  is_send_across_star _ _ #sendA0 #sA1

let output_lane_live_sendable
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (gD : array2 et (rm (SZ.v m) (SZ.v n)) { is_global gD })
  (bm bn tm tn wm wn : pos)
  (#_ : squash (bm /?+ SZ.v m /\ bn /?+ SZ.v n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (nblk : szp{SZ.v nblk == SZ.v m / bm * (SZ.v n / bn)})
  (nthr : szp{SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : is_send_across block_of (output_lane_live gD bm bn tm tn wm wn bid tid)
=
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
    is_global (output_fragment gD bm bn tm tn wm wn bid wid mi nj));
  solve

let kpre1_sendable
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gD : array2 et_acc (rm (SZ.v m) (SZ.v n)) { is_global gD })
  (bm bn : szp) (tm tn wm wn : pos)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                wm * tm /?+ SZ.v bm /\ wn * tn /?+ SZ.v bn))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (wm * tm) * (SZ.v bn / (wn * tn)) * warp_size})
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of (
      kpre1 gA eA gB eB gD bm bn tm tn wm wn fA fB nblk nthr bid tid)
=
  let output_send =
    output_lane_live_sendable gD (SZ.v bm) (SZ.v bn) tm tn wm wn #_ nblk nthr bid tid in
  let pA = gA |-> Frac (fA /. (nblk * nthr)) eA in
  let pB = gB |-> Frac (fB /. (nblk * nthr)) eB in
  let pOutput = output_lane_live gD (SZ.v bm) (SZ.v bn) tm tn wm wn bid tid in
  let pAlignedA = pure (aligned 16 (core gA)) in
  let pAlignedB = pure (aligned 16 (core gB)) in
  let pAlignedD = pure (aligned 16 (core gD)) in
  let pPure = pAlignedA ** pAlignedB ** pAlignedD in
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
  let sendAlignedD : is_send_across block_of pAlignedD =
    is_send_across_placeless pAlignedD
      #(placeless_pure (aligned 16 (core gD))) in
  let sendPure =
    is_send_across_star pAlignedA (pAlignedB ** pAlignedD)
      #sendAlignedA
      #(is_send_across_star pAlignedB pAlignedD #sendAlignedB #sendAlignedD) in
  let sendOutputPure =
    is_send_across_star pOutput pPure #output_send #sendPure in
  let sendBOutput =
    is_send_across_star pB (pOutput ** pPure) #sendB #sendOutputPure in
  is_send_across_star pA (pB ** pOutput ** pPure) #sendA #sendBOutput

let kpre_sendable
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
=
  let base_send =
    kpre1_sendable gA eA gB eB gD bm bn tm tn wmf wnf #_
      fA fB nblk nthr bid tid in
  let shared_send =
    shared_thread_live_sendable bm bn bk wm wn skew nthr sh #_ tid in
  is_send_across_star
    (kpre1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid)
    (shared_thread_live bm bn bk wm wn skew sh nthr tid)
    #base_send #shared_send

let kpost_sendable
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
=
  let base_send =
    kpre1_sendable gA eA gB eB gD bm bn tm tn wmf wnf #_
      fA fB nblk nthr bid tid in
  let shared_send =
    shared_thread_live_sendable bm bn bk wm wn skew nthr sh #_ tid in
  is_send_across_star
    (kpost1 gA eA gB eB gD bm bn tm tn wmf wnf fA fB nblk nthr bid tid)
    (shared_thread_live bm bn bk wm wn skew sh nthr tid)
    #base_send #shared_send
