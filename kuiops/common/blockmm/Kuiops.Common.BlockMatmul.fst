module Kuiops.Common.BlockMatmul

(* A chain of tensor-core [emma] calls over the [k]-chunks of a [tm x shared] by
   [shared x tn] product, and its approximation by the exact real matmul.

   Kernels that accumulate a fragment across the shared dimension (Q@K^T and
   P@V in flash attention, the k-loop of a tiled GEMM, ...) compute exactly
   [emma_chain]; [emma_chain_approx] is the only thing needed to relate that to
   [MS.matmul] over reals. *)

open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.TensorCore.Base

module MS = Kuiper.Spec.GEMM

#push-options "--fuel 1 --ifuel 0 --z3rlimit 20"

(* [n] chunks of width [tk] of the shared dimension, accumulated left to right
   into a zero-initialized accumulator fragment. *)
let rec emma_chain
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#tm #tn : pos) (#shared : nat) (tk : pos { tk /? shared })
  (ma : chest2 et_ab tm shared) (mb : chest2 et_ab shared tn)
  (n : nat { n <= shared / tk })
  : GTot (chest2 et_acc tm tn) (decreases n)
= if n = 0 then const (tm @| tn @| INil) zero
  else emma (emma_chain tk ma mb (n - 1))
            (ematrix_subtile ma tm tk 0 (n - 1))
            (ematrix_subtile mb tk tn (n - 1) 0)

(* The zero accumulator approximates the real zero matrix. *)
let const_zero_approx
  (#et : Type0) {| scalar et |} {| real_like et |}
  (tm tn : nat)
  : Lemma ((const (tm @| tn @| INil) (zero #et) <: chest2 et tm tn)
           %~ (const (tm @| tn @| INil) (zero #real) <: chest2 real tm tn))
= let c : chest2 et tm tn = const (tm @| tn @| INil) zero in
  let r : chest2 real tm tn = const (tm @| tn @| INil) zero in
  introduce forall (i : abs (tm @| tn @| INil)). acc c i %~ acc r i
  with (a0 #et)

(* Approximation is preserved by taking the same sub-tile of both sides. *)
let subtile_approx
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#rows #cols : nat)
  (m : chest2 et rows cols) (r : chest2 real rows cols)
  (trows : pos { trows /? rows }) (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
  : Lemma (requires m %~ r)
          (ensures ematrix_subtile m trows tcols tr tc
                   %~ ematrix_subtile r trows tcols tr tc)
= let ms = ematrix_subtile m trows tcols tr tc in
  let rs = ematrix_subtile r trows tcols tr tc in
  introduce forall (ij : abs (trows @| tcols @| INil)). acc ms ij %~ acc rs ij
  with (let (i, (j, ())) = ij in
        assert (acc2 ms i j == acc2 m (tr * trows + i) (tc * tcols + j)))

(* [__matmul_single_tile] over reals is the real counterpart of [emma_chain]:
   both are a left-associated accumulation over the chunks of the shared
   dimension. *)
let rec emma_chain_approx_upto
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| real_like et_ab |}
  {| scalar et_acc |} {| real_like et_acc |}
  (#tm #tn : pos) (#shared : nat) (tk : pos { tk /? shared })
  (ma : chest2 et_ab tm shared) (mb : chest2 et_ab shared tn)
  (ra : chest2 real tm shared) (rb : chest2 real shared tn)
  (n : nat { n <= shared / tk })
  : Lemma (requires ma %~ ra /\ mb %~ rb)
          (ensures emma_chain #et_ab #et_acc tk ma mb n
                   %~ MS.__matmul_single_tile tm tn tk ra rb 0 0 n)
          (decreases n)
= if n = 0
  then begin
    MS.matmul_single_tile_zero_lemma #real tm tn tk ra rb 0 0;
    const_zero_approx #et_acc tm tn
  end
  else begin
    emma_chain_approx_upto #et_ab #et_acc tk ma mb ra rb (n - 1);
    MS.matmul_single_tile_lemma #real tm tn tk ra rb 0 0 n;
    subtile_approx ma ra tm tk 0 (n - 1);
    subtile_approx mb rb tk tn (n - 1) 0;
    emma_approx_lemma
      (emma_chain #et_ab #et_acc tk ma mb (n - 1))
      (ematrix_subtile ma tm tk 0 (n - 1))
      (ematrix_subtile mb tk tn (n - 1) 0)
      (MS.__matmul_single_tile tm tn tk ra rb 0 0 (n - 1))
      (ematrix_subtile ra tm tk 0 (n - 1))
      (ematrix_subtile rb tk tn (n - 1) 0)
  end

#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 20"

(* [__matmul_single_tile] at a single full-size tile is the generic tiled
   accumulation of [Kuiper.Spec.GEMM] with a zero seed. *)
let rec tile_chain_is_gmatmul
  (#et : Type) {| scalar et |}
  (#rows #columns : pos) (#shared : nat)
  (tk : pos { tk /? shared })
  (m1 : chest2 et rows shared) (m2 : chest2 et shared columns)
  (n : nat { n <= shared / tk })
  : Lemma
      (ensures MS.__matmul_single_tile rows columns tk m1 m2 0 0 n
               == MS.__gmatmul_single
                    (const (rows @| columns @| INil) (zero #et))
                    MS.matmul MS.matplus
                    (ematrix_tiled m1 rows tk) (ematrix_tiled m2 tk columns)
                    0 0 n)
      (decreases n)
= assert (rows / rows == 1);
  if n = 0
  then MS.matmul_single_tile_zero_lemma rows columns tk m1 m2 0 0
  else begin
    assert (n <= shared);
    tile_chain_is_gmatmul tk m1 m2 (n - 1);
    MS.matmul_single_tile_lemma rows columns tk m1 m2 0 0 n;
    assert (acc2 (ematrix_tiled m1 rows tk) 0 (n - 1)
            == ematrix_subtile m1 rows tk 0 (n - 1));
    assert (acc2 (ematrix_tiled m2 tk columns) (n - 1) 0
            == ematrix_subtile m2 tk columns (n - 1) 0);
    MS.__gmatmul_single_lemma
      (const (rows @| columns @| INil) (zero #et)) MS.matmul MS.matplus
      (ematrix_tiled m1 rows tk) (ematrix_tiled m2 tk columns) 0 0 n
  end

(* A full-width chunked accumulation over reals is the exact matmul. *)
let matmul_tile_chain_full
  (#rows #columns : pos) (#shared : pos)
  (tk : pos { tk /? shared })
  (ra : chest2 real rows shared) (rb : chest2 real shared columns)
  : Lemma (MS.__matmul_single_tile rows columns tk ra rb 0 0 (shared / tk)
           == MS.matmul ra rb)
= tile_chain_is_gmatmul tk ra rb (shared / tk);
  MS.matmul_tiles_lemma #real (fun _ -> ()) (fun _ _ _ -> ())
    rows columns tk (const (rows @| columns @| INil) (zero #real)) ra rb 0 0;
  assert (equal (ematrix_subtile ra rows shared 0 0) ra);
  assert (equal (ematrix_subtile rb shared columns 0 0) rb);
  assert (equal (MS.matplus (const (rows @| columns @| INil) (zero #real))
                            (MS.matmul ra rb))
                (MS.matmul ra rb))

(* The tensor-core [emma] chain over all chunks of the shared dimension
   approximates the exact real matmul of the two operands. *)
let emma_chain_approx
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| real_like et_ab |}
  {| scalar et_acc |} {| real_like et_acc |}
  (#tm #tn : pos) (#shared : pos) (tk : pos { tk /? shared })
  (ma : chest2 et_ab tm shared) (mb : chest2 et_ab shared tn)
  (ra : chest2 real tm shared) (rb : chest2 real shared tn)
  : Lemma (requires ma %~ ra /\ mb %~ rb)
          (ensures emma_chain #et_ab #et_acc tk ma mb (shared / tk)
                   %~ MS.matmul ra rb)
= emma_chain_approx_upto #et_ab #et_acc tk ma mb ra rb (shared / tk);
  matmul_tile_chain_full tk ra rb

#pop-options

#push-options "--fuel 2 --ifuel 0 --z3rlimit 20"

(* The chain over a single full-width chunk is one fused multiply-add onto a
   zero accumulator -- the shape a kernel produces with [mma_fill] followed by a
   single [mma_sync] over untiled operands. *)
let emma_chain_one
  (#et_ab #et_acc : Type0) {| scalar et_ab |} {| scalar et_acc |}
  (#tm #tn #tk : pos)
  (ma : chest2 et_ab tm tk) (mb : chest2 et_ab tk tn)
  : Lemma (emma_chain #et_ab #et_acc tk ma mb 1
           == emma (const (tm @| tn @| INil) (zero #et_acc)) ma mb)
= assert (equal (ematrix_subtile ma tm tk 0 0) ma);
  assert (equal (ematrix_subtile mb tk tn 0 0) mb)

(* ... and, like any chain, it approximates the exact real matmul. *)
let emma_chain_one_approx
  (#et_ab #et_acc : Type0)
  {| scalar et_ab |} {| real_like et_ab |}
  {| scalar et_acc |} {| real_like et_acc |}
  (#tm #tn #tk : pos)
  (ma : chest2 et_ab tm tk) (mb : chest2 et_ab tk tn)
  (ra : chest2 real tm tk) (rb : chest2 real tk tn)
  : Lemma (requires ma %~ ra /\ mb %~ rb)
          (ensures emma_chain #et_ab #et_acc tk ma mb 1 %~ MS.matmul ra rb)
= emma_chain_approx #et_ab #et_acc #_ #_ #_ #_ #tm #tn #tk tk ma mb ra rb

#pop-options
