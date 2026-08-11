module Kuiops.SHMem.CellSend

(* Block-array sendability of the strided-chunk ownership predicates.

   Upstream builds the *global* tensor-cell send instance
   ([Kuiper.Tensor.is_send_across_global_tensor_cell], at [gpu_of]) but not a
   general-visibility one, so shared-memory (block-array) tiles cannot be shown
   [is_send_across block_of] through the existing library.  This module supplies
   the missing piece:

     - [is_send_across_tensor_cell]: general-visibility cell sendability,
       proved from [tensor_pts_to_cell_eq] + [is_send_pts_to_slice];
     - [own_strided_chunks_block_sendable] / [live_strided_chunks_block_sendable]:
       the [Kuiops.GEMM.T.FlipFlopBarrier2] chunk-partition predicates are
       block-sendable when the underlying [core] is a block array.

   TODO(upstream): [is_send_across_tensor_cell] belongs next to
   [is_send_across_tensor] in [Kuiper.Tensor]; the chunk-partition wrappers next
   to [own_strided_chunks] in [Kuiops.GEMM.T.FlipFlopBarrier2]. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.SHMem { is_block_array }

module FB = Kuiops.GEMM.T.FlipFlopBarrier2

(* General-visibility sendability of a single tensor cell. *)
val is_send_across_tensor_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (vis : visibility)
  (#_ : squash (visibility_of (core a) == vis))
  (#f : perm)
  (i : abs d)
  (v : et)
  : is_send_across vis (tensor_pts_to_cell a #f i v)

(* The writable chunk-partition of a block-array tile is block-sendable. *)
val own_strided_chunks_block_sendable
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#_ : squash (is_block_array (core m)))
  (em : chest2 et rows cols)
  (nthr : nat)
  (tid : natlt nthr)
  : is_send_across block_of (FB.own_strided_chunks m em nthr tid)

val live_strided_chunks_block_sendable
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#_ : squash (is_block_array (core m)))
  (nthr : nat)
  (tid : natlt nthr)
  : is_send_across block_of (FB.live_strided_chunks m nthr tid)
