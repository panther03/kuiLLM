module Kuiops.HReducePoly.Exact

(* Exact reduction polymorphic in the (associative) reduction operator.

   Exact analogue of [Kuiper.Kernel.HReduce.reduce]: instead of summing with the
   scalar [add] up to a real-valued approximation, it folds an arbitrary
   [reduce : et -> et -> et] required only to be associative, and returns the
   *exact* left-to-right reduction of the (pre-mapped) input. Suitable for
   operators such as [any], [all], [max], ... The input must be non-empty with at
   least as many elements as threads ([nth <= lena]) so no identity is needed. *)

#lang-pulse

open Kuiper
open Kuiper.Functions { is_associative }
open Kuiper.Tensor
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT

(* Non-empty left-to-right reduction ([foldl1]) of a sequence. *)
let rfold1 (#et:Type0) (f : et -> et -> et) (s : seq et { Seq.length s > 0 })
  : GTot et
  = seq_fold_left f (s @! 0) (Seq.slice s 1 (Seq.length s))

// TODO: should only need a fraction of a
// TODO: support different accumulator type (need to supply another function to convert initial element)
// This can be useful in the case of tracking both a sum and a max, for instance.
inline_for_extraction noextract
type reduce_ty (et : Type0) {| sized et |} =
  fn (f : (et -> et -> et) { is_associative f })
     (pre_map : et -> et)
     (nth : szp { nth <= max_threads })
     (lena : sz { SZ.fits (lena + nth) /\ SZ.v nth <= SZ.v lena })
     (#l : layout1 lena) {| ctlayout l |}
     (a : array1 et l { is_global a })
     (#va : chest1 et lena)
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    emp
  returns
    res : et
  ensures
    pure (res == rfold1 f (lseq_map pre_map (chest1_to_seq va)))

inline_for_extraction noextract
val reduce (#et:Type0) {| sized et |} : reduce_ty et
