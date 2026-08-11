module Kuiops.SuperGEMM.Mm.Spec

(* Specification-level glue for the software-pipelined GEMM.

   The kernel computes [D = post_map (A @ B^T)] with B stored row-major over
   [(cols, shared)] -- the (N, K) operand PyTorch hands to [aten.mm].  The
   top-level postcondition therefore names [MS.matmul rA (mtranspose rB)],
   whose second operand is indexed [(shared, cols)], while everything the
   kernel physically stages is indexed [(cols, shared)].

   [mtranspose_subtile] is the single bridge between those two views: a tile of
   a transpose is the transpose of the swapped tile.  With it, [warp_matmul]
   below can be stated entirely in B's PHYSICAL indexing, so the k-loop and the
   epilogue never mention [mtranspose rB]; only [warp_matmul_is_subtile]
   crosses over, once. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling

module MS = Kuiper.Spec.GEMM

(* A tile of a transpose is the transpose of the tile with its indices and
   extents swapped.  Both sides are [mk2] over the same cell function, so
   [chest2_ext] closes it. *)
val mtranspose_subtile
  (#et : Type)
  (#rows #cols : nat)
  (m : chest2 et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : Lemma (mtranspose (ematrix_subtile m trows tcols tr tc)
           == ematrix_subtile (mtranspose m) tcols trows tc tr)
          [SMTPat (mtranspose (ematrix_subtile m trows tcols tr tc))]

(* The (row, col) warp/block tile of [A @ B^T], stated in B's physical
   [(cols, shared)] indexing: a full-width slice of A against a full-width
   slice of B, the latter transposed.  [shared] is the whole k-range -- this
   kernel does not split K. *)
let warp_matmul
  (#m #n : nat)
  (#k : pos)
  (a : chest2 real m k)
  (b : chest2 real n k)
  (rows : pos{rows /?+ m})
  (cols : pos{cols /?+ n})
  (row : natlt (m / rows))
  (col : natlt (n / cols))
  : chest2 real rows cols
= MS.matmul
    (ematrix_subtile a rows k row 0)
    (mtranspose (ematrix_subtile b cols k col 0))

(* [warp_matmul] is exactly the corresponding tile of the whole product, which
   is what lets a per-warp postcondition compose into the launcher's
   whole-matrix one.  This is the only place the physical and transposed views
   of B meet. *)
val warp_matmul_is_subtile
  (#m #n : nat)
  (#k : pos)
  (a : chest2 real m k)
  (b : chest2 real n k)
  (rows : pos{rows /?+ m})
  (cols : pos{cols /?+ n})
  (row : natlt (m / rows))
  (col : natlt (n / cols))
  : Lemma (warp_matmul a b rows cols row col
           == ematrix_subtile (MS.matmul a (mtranspose b)) rows cols row col)
