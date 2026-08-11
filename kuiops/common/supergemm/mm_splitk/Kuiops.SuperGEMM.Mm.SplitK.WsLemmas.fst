module Kuiops.SuperGEMM.Mm.SplitK.WsLemmas

(* The real matrix the pass-1 kernel writes into the workspace, and the bridge
   from one warp's [warp_matmul] target to the corresponding tile of it.

   The workspace is [(splits*rows, cols)]: row slab [z] holds the partial
   product of split [z].  A block's flat id decodes to a row-block [brg] of the
   whole slab stack, which splits as [z * (rows/bm) + block_row]; the warp then
   sits at [(idx_m, idx_n)] inside that block.  Composing the three levels of
   [ematrix_subtile] turns the warp's target into the tile of [ws_target] that
   [Output.ws_warp_approximates] demands. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul, warp_matmul_is_subtile, mtranspose_subtile }
open Kuiops.SuperGEMM.Mm.KernelLemmas { subtile_subtile_compose, coerce_subtile_col }
open Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas { split_partial }

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module ML = FStar.Math.Lemmas
module SH = Kuiops.SuperGEMM.Mm.Shared

#set-options "--fuel 1 --ifuel 1 --z3rlimit 15"

(* The whole workspace, as a real matrix: slab [z] is split [z]'s partial.
   [mws] and [shared] are separate indices tied to [splits] by the squash, so
   callers holding a [chest2 real rows k] with [k == splits*ks] need no
   coercion. *)
let ws_target
  (#rows #cols #shared : pos) (mws splits ks : pos)
  (rA : chest2 real rows shared) (rB : chest2 real cols shared)
  (_ : squash (shared == splits * ks /\ mws == splits * rows))
  : chest2 real mws cols
= ML.cancel_mul_div splits rows;
  ML.cancel_mul_div splits ks;
  ematrix_from_tiles #real #mws #cols rows cols
    (fun (z : natlt (mws / rows)) (_ : natlt (cols / cols)) ->
      split_partial #rows #cols #shared ks rA (mtranspose rB) z)

(* Slab [z] of [ws_target] is split [z]'s partial product. *)
let ws_target_slab
  (#rows #cols #shared : pos) (mws splits ks : pos)
  (rA : chest2 real rows shared) (rB : chest2 real cols shared)
  (sq : squash (shared == splits * ks /\ mws == splits * rows))
  (z : natlt splits)
  : Lemma
      (requires True)
      (ensures (ML.cancel_mul_div splits rows;
                ML.cancel_mul_div splits ks;
                rows /? mws /\ ks /? shared /\
                ematrix_subtile (ws_target mws splits ks rA rB sq) rows cols z 0
                == split_partial #rows #cols #shared ks rA (mtranspose rB) z))
= ML.cancel_mul_div splits rows;
  ML.cancel_mul_div splits ks

(* Index algebra only: re-associating the (block, warp) tiling of an abstract
   [mws x cols] matrix as (slab, warp).  Kept free of [ws_target] so it never
   unfolds [ematrix_from_tiles]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 20 --split_queries always"
let subtile_reassoc
  (#mws #cols : pos)
  (rW : chest2 real mws cols)
  (rows bm bn wm wn : pos)
  (z : natlt (mws / rows))
  (block_row : natlt (rows / bm)) (bcol : natlt (cols / bn))
  (idx_m : natlt (bm / wm)) (idx_n : natlt (bn / wn))
  (_ : squash (rows /? mws /\ bm /? rows /\ bn /? cols /\ wm /? bm /\ wn /? bn))
  : Lemma
      (requires
        block_row * (bm / wm) + idx_m < rows / wm /\
        bcol * (bn / wn) + idx_n < cols / wn)
      (ensures
        z * (rows / bm) + block_row < mws / bm /\
        ematrix_subtile
          (ematrix_subtile rW bm bn (z * (rows / bm) + block_row) bcol)
          wm wn idx_m idx_n
        == ematrix_subtile
             (ematrix_subtile rW rows cols z 0)
             wm wn
             (block_row * (bm / wm) + idx_m)
             (bcol * (bn / wn) + idx_n))
= let brg : nat = z * (rows / bm) + block_row in
  let grow : nat = block_row * (bm / wm) + idx_m in
  let gcol : nat = bcol * (bn / wn) + idx_n in
  Kuiper.Divides.lemma_divides_trans wm bm rows;
  Kuiper.Divides.lemma_divides_trans wn bn cols;
  Kuiper.Divides.lemma_divides_trans bm rows mws;
  Kuiper.Divides.lemma_divides_trans wm rows mws;
  SH.grow_bound mws rows bm z block_row;
  SH.grow_bound mws rows wm z grow;
  SH.div_compose rows bm wm;
  ML.distributivity_add_left (z * (rows / bm)) block_row (bm / wm);
  ML.paren_mul_right z (rows / bm) (bm / wm);
  assert (brg * (bm / wm) + idx_m == z * (rows / wm) + grow);
  subtile_subtile_compose rW bm bn wm wn brg bcol idx_m idx_n ();
  subtile_subtile_compose rW rows cols wm wn z 0 grow gcol ()
#pop-options

(* The three-level composition: the (bm,bn) block [brg,bcol] of the workspace,
   further tiled at (wm,wn) by [idx_m,idx_n], is the warp's [warp_matmul] over
   split [z]'s k-slice. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 20 --split_queries always"
let ws_warp_target
  (#rows #cols #shared : pos) (mws splits ks : pos)
  (rA : chest2 real rows shared) (rB : chest2 real cols shared)
  (bm bn wm wn : pos)
  (z : natlt splits)
  (block_row : natlt (rows / bm)) (bcol : natlt (cols / bn))
  (idx_m : natlt (bm / wm)) (idx_n : natlt (bn / wn))
  (sq : squash (shared == splits * ks /\ mws == splits * rows))
  (_ : squash (bm /? rows /\ bn /? cols /\ wm /? bm /\ wn /? bn /\
               wm /? rows /\ wn /? cols /\
               bm /? mws /\ wm /? mws /\ ks /? shared))
  : Lemma
      (requires
        block_row * (bm / wm) + idx_m < rows / wm /\
        bcol * (bn / wn) + idx_n < cols / wn /\
        z * (rows / bm) + block_row < mws / bm)
      (ensures
        ematrix_subtile
          (ematrix_subtile (ws_target mws splits ks rA rB sq) bm bn
            (z * (rows / bm) + block_row) bcol)
          wm wn idx_m idx_n
        == warp_matmul
             (ematrix_subtile rA rows ks 0 z)
             (ematrix_subtile rB cols ks 0 z)
             wm wn
             (block_row * (bm / wm) + idx_m)
             (bcol * (bn / wn) + idx_n))
= ML.cancel_mul_div splits rows;
  ML.cancel_mul_div splits ks;
  let grow : natlt (rows / wm) = block_row * (bm / wm) + idx_m in
  let gcol : natlt (cols / wn) = bcol * (bn / wn) + idx_n in
  subtile_reassoc (ws_target mws splits ks rA rB sq) rows bm bn wm wn
    z block_row bcol idx_m idx_n ();
  ws_target_slab mws splits ks rA rB sq z;
  warp_matmul_is_subtile
    (ematrix_subtile rA rows ks 0 z) (ematrix_subtile rB cols ks 0 z)
    wm wn grow gcol
#pop-options
