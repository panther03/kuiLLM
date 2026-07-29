module Kuiops.HReducePoly.Exact

(* Exact reduction over the innermost dimension of a tensor. Each outer
   (batch) index is reduced by one CUDA block. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Functions { is_associative }
open Kuiper.Tensor
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT

val snoc_shape (#r : nat) (d : shape r) (n : nat) : GTot (shape (r + 1))

val abs_snoc (#r : nat) (#d : shape r) (#n : nat)
  (i : abs d) (j : natlt n) : GTot (abs (snoc_shape d n))

val inner_seq (#et_i : Type0) (#r : nat) (#d : shape r) (#n : nat)
  (v : chest (snoc_shape d n) et_i) (i : abs d) : GTot (lseq et_i n)

let rfold1 (#et : Type0) (f : et -> et -> et)
  (s : seq et { Seq.length s > 0 }) : GTot et =
  seq_fold_left f (s @! 0) (Seq.slice s 1 (Seq.length s))

let reduced
  (#et_i #et #et_o : Type0)
  (#r : nat) (#d : shape r) (#n : pos)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (v : chest (snoc_shape d n) et_i)
  (i : abs d)
  : GTot et_o =
  post_map (rfold1 f (lseq_map pre_map (inner_seq v i)))

inline_for_extraction noextract
// `sized` needed because we allocate a shared memory array dynamically with element type `et`
type reduce_ty (et_i et et_o : Type0) {| sized et |} =
  fn (#r : erased nat)
     (#d : shape r)
     (cd : cshape d)
     (f : (et -> et -> et) { is_associative f })
     (pre_map : et_i -> et)
     (post_map : et -> et_o)
     (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
     (cols : szp)
     (nth : szp {
       nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
     (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
     (#lout : tlayout d) {| ctlayout lout |}
     (a : tensor et_i lin { is_global a })
     (out : tensor et_o lout { is_global out })
     (s : stream_t)
     (#va : chest (snoc_shape d cols) et_i)
     (#vout : chest d et_o)
     (#e : epoch_t)
  preserves
    cpu ** stream_live s ** epoch_live s e
  requires
    on gpu_loc (a |-> va) ** on gpu_loc (out |-> vout)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (
        (a |-> va) **
        (out |-> mk d (fun i -> reduced f pre_map post_map va i))))

inline_for_extraction noextract
val reduce (#et_i #et #et_o : Type0) {| sized et |} : reduce_ty et_i et et_o

inline_for_extraction noextract
fn reduce_indexed
  (#et_i #et #et_o : Type0) {| sized et |}
  (#r : erased nat)
  (#d : shape r)
  (cd : cshape d)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp {
    nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (index : conc d -> szlt cols -> conc (snoc_shape d cols))
  (index_up : (i:conc d -> j:szlt cols ->
    Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (input : tensor et_i lin { is_global input })
  (output : tensor et_o lout { is_global output })
  (s : stream_t)
  (#vin : chest (snoc_shape d cols) et_i)
  (#vout : chest d et_o)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires on gpu_loc (input |-> vin) ** on gpu_loc (output |-> vout)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (
        (input |-> vin) **
        (output |-> mk d (fun i ->
          reduced f pre_map post_map vin i))))
