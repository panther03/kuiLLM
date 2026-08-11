module Kuiops.SuperGEMM.Mm.Spec

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling

module MS = Kuiper.Spec.GEMM

#set-options "--fuel 0 --ifuel 0 --z3rlimit 10"

let mtranspose_subtile #et #rows #cols m trows tcols tr tc =
  Kuiper.Chest.ext
    (mtranspose (ematrix_subtile m trows tcols tr tc))
    (ematrix_subtile (mtranspose m) tcols trows tc tr)

(* Both sides are the (i,j) dot product of row [row*rows+i] of [a] with row
   [col*cols+j] of [b]:

     warp_matmul ... i j
       = sum_p (subtile a) i p * (mtranspose (subtile b)) p j     -- matmul
       = sum_p a (row*rows+i) p * b (col*cols+j) p                -- subtile
     subtile (matmul a (mtranspose b)) ... i j
       = (matmul a (mtranspose b)) (row*rows+i) (col*cols+j)
       = sum_p a (row*rows+i) p * (mtranspose b) p (col*cols+j)
       = sum_p a (row*rows+i) p * b (col*cols+j) p

   [__gmatmul_single_congr] discharges the two sums cell-by-cell; the full-width
   slices make the k-ranges identical, so no k-side reindexing is needed. *)
let warp_matmul_is_subtile #m #n #k a b rows cols row col =
  let lhs = warp_matmul a b rows cols row col in
  let rhs = ematrix_subtile (MS.matmul a (mtranspose b)) rows cols row col in
  introduce forall (i : natlt rows) (j : natlt cols). acc2 lhs i j == acc2 rhs i j
  with begin
    MS.__gmatmul_single_congr #real #real #real 0.0R ( *. ) ( +. )
      (ematrix_subtile a rows k row 0)
      (mtranspose (ematrix_subtile b cols k col 0))
      a (mtranspose b)
      i j (row * rows + i) (col * cols + j) k
  end;
  Kuiper.Chest.ext lhs rhs
