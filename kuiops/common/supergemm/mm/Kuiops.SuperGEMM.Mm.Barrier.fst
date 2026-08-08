module Kuiops.SuperGEMM.Mm.Barrier

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module FB = Kuiops.GEMM.T.FlipFlopBarrier2

open Kuiops.Array2.Layout.Skewed { l2_skewed_row_major }
open Kuiops.SuperGEMM.Mm.Params { ldt }

#set-options "--fuel 1 --ifuel 1 --z3rlimit 15"

(* ---- Array2-level buffer transforms (existential content) ----
   [own_to_sharing_ex] is the ODD FlipFlopBarrier2 transform and
   [sharing_ex_to_live] the EVEN one, both instantiated at an arbitrary
   [array2] view using the exported chunk (de)composition primitives. *)

ghost
fn own_to_sharing_ex
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (nthr : pos)
  (#_ : squash (SZ.fits (l.ulen)))
  requires
    forall+ (tid : natlt nthr). pipe_live m nthr tid
  ensures
    forall+ (tid : natlt nthr). pipe_sharing m nthr
{
  forevery_map
    (fun (tid : natlt nthr) -> pipe_live m nthr tid)
    (fun (tid : natlt nthr) -> FB.live_strided_chunks m nthr tid)
    fn tid { unfold pipe_live m nthr tid };
  FB.join_array2_from_strided_chunks_underspec m nthr;
  with s. assert m |-> s;
  tensor_share_n m nthr;
  forevery_map
    (fun (tid : natlt nthr) -> m |-> Frac (1.0R /. nthr) s)
    (fun (tid : natlt nthr) -> pipe_sharing m nthr)
    fn tid {
      fold FB.bp_sharing m s nthr;
      fold pipe_sharing m nthr;
    };
}

ghost
fn sharing_ex_to_live
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (nthr : pos)
  (#_ : squash (SZ.fits (l.ulen)))
  requires
    forall+ (tid : natlt nthr). pipe_sharing m nthr
  ensures
    forall+ (tid : natlt nthr). pipe_live m nthr tid
{
  forevery_map
    (fun (tid : natlt nthr) -> pipe_sharing m nthr)
    (fun (tid : natlt nthr) -> exists* s. tensor_pts_to m #(1.0R /. nthr) s)
    fn tid {
      unfold pipe_sharing m nthr;
      with em. assert FB.bp_sharing m em nthr;
      unfold FB.bp_sharing m em nthr;
    };
  tensor_gather_n_underspec m nthr;
  with s. assert m |-> s;
  FB.split_array2_into_strided_chunks m nthr;
  forevery_map
    (fun (tid : natlt nthr) -> FB.own_strided_chunks m s nthr tid)
    (fun (tid : natlt nthr) -> pipe_live m nthr tid)
    fn tid {
      fold FB.live_strided_chunks m nthr tid;
      fold pipe_live m nthr tid;
    };
}

#push-options "--split_queries no"
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
{
  if (it >= ktiles) {
    forevery_map
      (fun (tid : natlt nthr) ->
        pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
      (fun (tid : natlt nthr) ->
        pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
      fn tid {
        rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid) as emp;
        rewrite emp as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid);
      };
  } else {
    let mAb = skewed_view bm bk skew (if it % 2 = 0 then sarA0 else sarA1);
    let mAo = skewed_view bm bk skew (if it % 2 = 0 then sarA1 else sarA0);
    let mBb = skewed_view bn bk skew (if it % 2 = 0 then sarB0 else sarB1);
    let mBo = skewed_view bn bk skew (if it % 2 = 0 then sarB1 else sarB0);
    if (it = 0) {
      forevery_map
        (fun (tid : natlt nthr) ->
          pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
        (fun (tid : natlt nthr) ->
          pipe_live mAb nthr tid ** pipe_live mBb nthr tid **
          (pipe_live mAo nthr tid ** pipe_live mBo nthr tid))
        fn tid {
          rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
            as (pipe_live mAb nthr tid ** pipe_live mBb nthr tid **
                (pipe_live mAo nthr tid ** pipe_live mBo nthr tid));
        };
      forevery_unzip
        (fun (tid : natlt nthr) -> pipe_live mAb nthr tid)
        (fun (tid : natlt nthr) ->
          pipe_live mBb nthr tid ** (pipe_live mAo nthr tid ** pipe_live mBo nthr tid));
      forevery_unzip
        (fun (tid : natlt nthr) -> pipe_live mBb nthr tid)
        (fun (tid : natlt nthr) -> pipe_live mAo nthr tid ** pipe_live mBo nthr tid);
      forevery_unzip
        (fun (tid : natlt nthr) -> pipe_live mAo nthr tid)
        (fun (tid : natlt nthr) -> pipe_live mBo nthr tid);
      own_to_sharing_ex mAb nthr;
      own_to_sharing_ex mBb nthr;
      forevery_zip
        (fun (tid : natlt nthr) -> pipe_live mAo nthr tid)
        (fun (tid : natlt nthr) -> pipe_live mBo nthr tid);
      forevery_zip
        (fun (tid : natlt nthr) -> pipe_sharing mBb nthr)
        (fun (tid : natlt nthr) -> pipe_live mAo nthr tid ** pipe_live mBo nthr tid);
      forevery_zip
        (fun (tid : natlt nthr) -> pipe_sharing mAb nthr)
        (fun (tid : natlt nthr) ->
          pipe_sharing mBb nthr ** (pipe_live mAo nthr tid ** pipe_live mBo nthr tid));
      forevery_map
        (fun (tid : natlt nthr) ->
          pipe_sharing mAb nthr **
          (pipe_sharing mBb nthr ** (pipe_live mAo nthr tid ** pipe_live mBo nthr tid)))
        (fun (tid : natlt nthr) ->
          pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
        fn tid {
          rewrite (pipe_sharing mAb nthr **
                   (pipe_sharing mBb nthr ** (pipe_live mAo nthr tid ** pipe_live mBo nthr tid)))
            as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid);
        };
    } else {
      forevery_map
        (fun (tid : natlt nthr) ->
          pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
        (fun (tid : natlt nthr) ->
          pipe_live mAb nthr tid ** pipe_live mBb nthr tid **
          (pipe_sharing mAo nthr ** pipe_sharing mBo nthr))
        fn tid {
          rewrite (pipe_p bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
            as (pipe_live mAb nthr tid ** pipe_live mBb nthr tid **
                (pipe_sharing mAo nthr ** pipe_sharing mBo nthr));
        };
      forevery_unzip
        (fun (tid : natlt nthr) -> pipe_live mAb nthr tid)
        (fun (tid : natlt nthr) ->
          pipe_live mBb nthr tid ** (pipe_sharing mAo nthr ** pipe_sharing mBo nthr));
      forevery_unzip
        (fun (tid : natlt nthr) -> pipe_live mBb nthr tid)
        (fun (tid : natlt nthr) -> pipe_sharing mAo nthr ** pipe_sharing mBo nthr);
      forevery_unzip
        (fun (tid : natlt nthr) -> pipe_sharing mAo nthr)
        (fun (tid : natlt nthr) -> pipe_sharing mBo nthr);
      own_to_sharing_ex mAb nthr;
      own_to_sharing_ex mBb nthr;
      sharing_ex_to_live mAo nthr;
      sharing_ex_to_live mBo nthr;
      forevery_zip
        (fun (tid : natlt nthr) -> pipe_live mAo nthr tid)
        (fun (tid : natlt nthr) -> pipe_live mBo nthr tid);
      forevery_zip
        (fun (tid : natlt nthr) -> pipe_sharing mBb nthr)
        (fun (tid : natlt nthr) -> pipe_live mAo nthr tid ** pipe_live mBo nthr tid);
      forevery_zip
        (fun (tid : natlt nthr) -> pipe_sharing mAb nthr)
        (fun (tid : natlt nthr) ->
          pipe_sharing mBb nthr ** (pipe_live mAo nthr tid ** pipe_live mBo nthr tid));
      forevery_map
        (fun (tid : natlt nthr) ->
          pipe_sharing mAb nthr **
          (pipe_sharing mBb nthr ** (pipe_live mAo nthr tid ** pipe_live mBo nthr tid)))
        (fun (tid : natlt nthr) ->
          pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
        fn tid {
          rewrite (pipe_sharing mAb nthr **
                   (pipe_sharing mBb nthr ** (pipe_live mAo nthr tid ** pipe_live mBo nthr tid)))
            as (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid);
        };
    }
  }
}
#pop-options

#push-options "--fuel 2 --split_queries no"
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
{
  forevery_map #(natlt nthr)
    (fun tid -> pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
    (fun tid ->
      pipe_sharing (skewed_view bm bk skew sarA0) nthr **
      pipe_sharing (skewed_view bn bk skew sarB0) nthr **
      pipe_live (skewed_view bm bk skew sarA1) nthr tid **
      pipe_live (skewed_view bn bk skew sarB1) nthr tid)
    fn tid {
      rewrite
        (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
      as
        (pipe_sharing (skewed_view bm bk skew sarA0) nthr **
         pipe_sharing (skewed_view bn bk skew sarB0) nthr **
         pipe_live (skewed_view bm bk skew sarA1) nthr tid **
         pipe_live (skewed_view bn bk skew sarB1) nthr tid);
    };
}
#pop-options

#push-options "--fuel 2 --split_queries no"
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
{
  forevery_map #(natlt nthr)
    (fun tid -> pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
    (fun tid ->
      pipe_sharing (skewed_view bm bk skew sarA1) nthr **
      pipe_sharing (skewed_view bn bk skew sarB1) nthr **
      pipe_live (skewed_view bm bk skew sarA0) nthr tid **
      pipe_live (skewed_view bn bk skew sarB0) nthr tid)
    fn tid {
      rewrite
        (pipe_q bm bn bk skew sarA0 sarA1 sarB0 sarB1 nthr ktiles it tid)
      as
        (pipe_sharing (skewed_view bm bk skew sarA1) nthr **
         pipe_sharing (skewed_view bn bk skew sarB1) nthr **
         pipe_live (skewed_view bm bk skew sarA0) nthr tid **
         pipe_live (skewed_view bn bk skew sarB0) nthr tid);
    };
}
#pop-options
