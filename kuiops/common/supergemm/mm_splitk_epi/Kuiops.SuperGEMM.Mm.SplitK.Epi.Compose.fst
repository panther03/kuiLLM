module Kuiops.SuperGEMM.Mm.SplitK.Epi.Compose

(* Composing the two passes.  Reducing the workspace pass 1 wrote reproduces
   the full k reduction -- that is [Kuiops.SuperGEMM.Mm.SplitK.Compose], reused
   verbatim -- so combining the reduced workspace with C reproduces the
   epilogue applied ONCE to the complete product.  The split is therefore
   invisible in the top-level spec, and the [beta * C] term is added exactly
   once rather than once per split. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiops.SuperGEMM.Mm.SplitK.WsLemmas { ws_target }
open Kuiops.SuperGEMM.Mm.SplitK.Compose { reduce_ws_target }
open Kuiops.SuperGEMM.Mm.SplitK.Epi.ReduceLemmas { gran_target }

module MS = Kuiper.Spec.GEMM

#set-options "--fuel 1 --ifuel 1 --z3rlimit 10"

let gran_target_mmcomb
  (#rows #cols #shared : pos) (mws splits ks : pos)
  (rC : chest2 real rows cols)
  (rA : chest2 real rows shared) (rB : chest2 real cols shared)
  (comb_r : binop real)
  (sq : squash (shared == splits * ks /\ mws == splits * rows))
  : Lemma (gran_target rows splits rC (ws_target mws splits ks rA rB sq) comb_r
           == MS.mmcomb comb_r rC rA (mtranspose rB))
= reduce_ws_target mws splits ks rA rB sq
