module Kuiops.SHMem.CellSend

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.SHMem { is_block_array }

#set-options "--fuel 1 --ifuel 1 --z3rlimit 15"

module FB = Kuiops.GEMM.T.FlipFlopBarrier2

let is_send_across_tensor_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (vis : visibility)
  (#_ : squash (visibility_of (core a) == vis))
  (#f : perm)
  (i : abs d)
  (v : et)
  : is_send_across vis (tensor_pts_to_cell a #f i v)
=
  tensor_pts_to_cell_eq a i f v;
  let idx = l.imap.f i in
  (* pts_to_cell unfolds to pts_to_slice, which is block/anywhere sendable at
     the visibility of the backing array. *)
  let base
    : is_send_across (visibility_of (core a))
        (pts_to_cell (core a) #f idx v) = solve in
  coerce_eq () base

let own_strided_chunks_block_sendable
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#_ : squash (is_block_array (core m)))
  (em : chest2 et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : is_send_across block_of (FB.own_strided_chunks m em nthr tid)
=
  is_send_across_forevery
    #(ij : (natlt rows & natlt cols){
        Kuiper.Kernel.GEMM.Copy.Vec2.in_chunk (chunk et) rows cols nthr tid ij})
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    block_of
    #(fun ij ->
        is_send_across_tensor_cell m block_of #_
          (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))

let live_strided_chunks_block_sendable
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#_ : squash (is_block_array (core m)))
  (nthr : nat)
  (tid : natlt nthr)
  : is_send_across block_of (FB.live_strided_chunks m nthr tid)
=
  is_send_across_exists
    (fun em -> FB.own_strided_chunks m em nthr tid)
    #(fun em -> own_strided_chunks_block_sendable m #_ em nthr tid)
