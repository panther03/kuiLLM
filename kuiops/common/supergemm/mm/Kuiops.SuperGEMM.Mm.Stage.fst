module Kuiops.SuperGEMM.Mm.Stage
#lang-pulse

(* Module 4 -- Stage.  One thread's contribution to staging ONE k-tile of A
   and of B into one pair of shared buffers, using cp.async.

   Step 1: memory safety only.  All staged contents are existentially
   quantified. *)

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module ML = FStar.Math.Lemmas
module CV = Kuiper.Kernel.GEMM.Copy.Vec2
module FB = Kuiops.GEMM.T.FlipFlopBarrier2

open Kuiper.Array2.Vectorized { row_cells }
open Kuiper.Bijection

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 20"

(* ================================================================== *)
(*  Chunk <-> per-row addressing arithmetic.                          *)
(* ================================================================== *)

(* [geo_ok] and [g_row_step]/[g_a_iters]/[g_t_row]/[g_t_col] come from the
   interface. *)

let in_chunk_at
  (chunk rows cols nthr : pos) (tid : natlt nthr)
  (ij : (natlt rows & natlt cols)) : prop
= CV.in_chunk chunk rows cols nthr tid ij

(* row_step is exact under [geo_ok]. *)
let lemma_rs_exact (rows cols chunk nthr : pos)
  : Lemma (requires geo_ok rows cols chunk nthr)
          (ensures g_row_step cols chunk nthr * cols == chunk * nthr)
= ML.div_exact_r (chunk * nthr) cols

(* a_iters is exact: a_iters * (chunk*nthr) == rows*cols. *)
let lemma_ai_exact (rows cols chunk nthr : pos)
  : Lemma (requires geo_ok rows cols chunk nthr)
          (ensures g_a_iters rows cols chunk nthr * (chunk * nthr) == rows * cols)
= ML.div_exact_r (rows * cols) (chunk * nthr)

(* t_row*cols + t_col == tid*chunk, and the chunk-aligned column bound. *)
let lemma_trow (cols chunk nthr : pos) (tid : nat)
  : Lemma (ensures g_t_row cols chunk nthr tid * cols + g_t_col cols chunk nthr tid
                   == tid * chunk /\
                   g_t_col cols chunk nthr tid < cols)
= ML.lemma_div_mod (tid * chunk) cols

let lemma_tcol_bound (cols chunk nthr : pos) (tid : nat)
  : Lemma (requires cols % chunk == 0)
          (ensures g_t_col cols chunk nthr tid + chunk <= cols)
= let m : pos = cols / chunk in
  ML.div_exact_r cols chunk;              (* cols == chunk * m *)
  let q = (tid * chunk) / cols in
  let tc = (tid * chunk) % cols in
  ML.lemma_div_mod (tid * chunk) cols;    (* tid*chunk == q*cols + tc *)
  ML.modulo_lemma tc cols;
  assert (tc == tid * chunk - q * cols);
  assert (cols == chunk * m);
  assert (tc == chunk * (tid - q * m));   (* nl *)
  ()

(* Forward map:  chunk (tid + s*nthr) <-> row (t_row + s*row_step), col t_col.
   [g_ff_row]/[g_ff_col] name the destination cell of copy-step [s], lane [x]. *)
unfold let g_ff_row (rows cols chunk nthr : pos) (tid s : nat) : nat
  = g_t_row cols chunk nthr tid + s * g_row_step cols chunk nthr
unfold let g_ff_col (cols chunk nthr : pos) (tid x : nat) : nat
  = g_t_col cols chunk nthr tid + x

let lemma_ff (rows cols chunk nthr : pos) (tid : natlt nthr)
             (s : nat) (x : nat)
  : Lemma (requires geo_ok rows cols chunk nthr /\
                    s < g_a_iters rows cols chunk nthr /\ x < chunk)
          (ensures (let row = g_ff_row rows cols chunk nthr tid s in
                    let col = g_ff_col cols chunk nthr tid x in
                    let flat = row * cols + col in
                    row < rows /\ col < cols /\
                    flat == chunk * (tid + s * nthr) + x /\
                    (flat / chunk) % nthr == tid))
= let rs = g_row_step cols chunk nthr in
  let ai = g_a_iters rows cols chunk nthr in
  let tr = g_t_row cols chunk nthr tid in
  let tc = g_t_col cols chunk nthr tid in
  lemma_rs_exact rows cols chunk nthr;      (* rs*cols == chunk*nthr *)
  lemma_ai_exact rows cols chunk nthr;      (* ai*(chunk*nthr) == rows*cols *)
  lemma_trow cols chunk nthr tid;           (* tr*cols + tc == tid*chunk *)
  lemma_tcol_bound cols chunk nthr tid;     (* tc + chunk <= cols *)
  let row = tr + s * rs in
  let col = tc + x in
  let flat = row * cols + col in
  (* flat == chunk*(tid + s*nthr) + x *)
  assert (row * cols == tr * cols + s * rs * cols);
  ML.paren_mul_right s rs cols;             (* s*rs*cols == s*(rs*cols) *)
  assert (row * cols == (tid * chunk - tc) + s * (chunk * nthr));
  assert (flat == tid * chunk + s * (chunk * nthr) + x);
  ML.paren_mul_right s nthr chunk;          (* s*(nthr*chunk)==(s*nthr)*chunk *)
  assert (flat == chunk * (tid + s * nthr) + x);
  (* chunk_idx == tid + s*nthr *)
  ML.lemma_div_plus x (tid + s * nthr) chunk;
  assert (flat / chunk == tid + s * nthr);
  ML.lemma_mod_plus tid s nthr;
  ML.modulo_lemma tid nthr;
  assert ((flat / chunk) % nthr == tid);
  (* bounds *)
  assert (col < cols);
  assert (flat <= chunk * (tid + s * nthr) + (chunk - 1));
  assert (tid + s * nthr <= (nthr - 1) + (ai - 1) * nthr);
  assert ((nthr - 1) + (ai - 1) * nthr == ai * nthr - 1);
  assert (flat <= chunk * (ai * nthr) - 1);
  ML.paren_mul_right ai nthr chunk;
  assert (chunk * (ai * nthr) == ai * (chunk * nthr));
  assert (flat < rows * cols);
  assert (row * cols <= flat);
  assert (row * cols < rows * cols)

(* Backward map: recover (s, x) from an owned cell (i, j). *)
unfold let g_gg_s (cols chunk nthr : pos) (i j : nat) : nat
  = ((i * cols + j) / chunk) / nthr
unfold let g_gg_x (cols chunk nthr : pos) (i j : nat) : nat
  = (i * cols + j) % chunk

let lemma_gg (rows cols chunk nthr : pos) (tid : natlt nthr) (i j : nat)
  : Lemma (requires geo_ok rows cols chunk nthr /\ i < rows /\ j < cols /\
                    (((i * cols + j) / chunk) % nthr == tid))
          (ensures (let s = g_gg_s cols chunk nthr i j in
                    let x = g_gg_x cols chunk nthr i j in
                    s < g_a_iters rows cols chunk nthr /\ x < chunk /\
                    g_ff_row rows cols chunk nthr tid s == i /\
                    g_ff_col cols chunk nthr tid x == j))
= let ai = g_a_iters rows cols chunk nthr in
  let flat = i * cols + j in
  let ci = flat / chunk in
  let s = ci / nthr in
  let x = flat % chunk in
  lemma_ai_exact rows cols chunk nthr;      (* ai*(chunk*nthr) == rows*cols *)
  ML.lemma_div_mod ci nthr;                 (* ci == nthr*(ci/nthr) + ci%nthr *)
  assert (ci == nthr * s + tid);            (* since ci%nthr == tid *)
  ML.lemma_div_mod flat chunk;              (* flat == chunk*ci + x *)
  (* s < ai *)
  assert (flat < rows * cols);
  ML.lemma_mult_lt_left chunk ci (ai * nthr);   (* helper below via bound *)
  assert (chunk * ci <= flat);
  ML.paren_mul_right ai nthr chunk;
  assert (chunk * (ai * nthr) == ai * (chunk * nthr));
  assert (chunk * ci < chunk * (ai * nthr));
  assert (ci < ai * nthr);
  assert (s * nthr <= ci);
  assert (s < ai);
  (* reconstruct via lemma_ff *)
  lemma_ff rows cols chunk nthr tid s x;
  let row = g_ff_row rows cols chunk nthr tid s in
  let col = g_ff_col cols chunk nthr tid x in
  assert (row * cols + col == chunk * (tid + s * nthr) + x);
  assert (tid + s * nthr == ci);            (* ci == nthr*s+tid *)
  assert (row * cols + col == chunk * ci + x);
  assert (row * cols + col == flat);
  assert (col < cols /\ j < cols);
  ML.lemma_div_mod (row * cols + col) cols;
  ML.lemma_div_mod flat cols;
  ML.euclidean_division_definition flat cols

let lemma_ff_inv (rows cols chunk nthr : pos) (tid : natlt nthr) (s x : nat)
  : Lemma (requires geo_ok rows cols chunk nthr /\
                    s < g_a_iters rows cols chunk nthr /\ x < chunk)
          (ensures (let row = g_ff_row rows cols chunk nthr tid s in
                    let col = g_ff_col cols chunk nthr tid x in
                    g_gg_s cols chunk nthr row col == s /\
                    g_gg_x cols chunk nthr row col == x))
= lemma_ff rows cols chunk nthr tid s x;
  let row = g_ff_row rows cols chunk nthr tid s in
  let col = g_ff_col cols chunk nthr tid x in
  let flat = row * cols + col in
  assert (flat == chunk * (tid + s * nthr) + x);
  ML.lemma_div_plus x (tid + s * nthr) chunk;   (* flat/chunk == tid+s*nthr *)
  ML.modulo_lemma x chunk;                       (* flat%chunk == x *)
  ML.lemma_mod_plus x (tid + s * nthr) chunk;
  assert (flat % chunk == x);
  assert (flat / chunk == tid + s * nthr);
  ML.lemma_div_plus tid s nthr;                  (* (tid+s*nthr)/nthr == s *)
  ML.small_div tid nthr

(* ================================================================== *)
(*  The chunk-partition <-> product bijection.                        *)
(* ================================================================== *)

let chunkB (rows cols chunk nthr : pos) (tid : natlt nthr)
  = (ij : (natlt rows & natlt cols){CV.in_chunk chunk rows cols nthr tid ij})
let chunkA (rows cols chunk nthr : pos)
  = (natlt (g_a_iters rows cols chunk nthr) & natlt chunk)

let bij_ff (rows cols chunk nthr : pos) (tid : natlt nthr)
           (_ : squash (geo_ok rows cols chunk nthr))
           (ij : chunkB rows cols chunk nthr tid)
  : GTot (chunkA rows cols chunk nthr)
= lemma_gg rows cols chunk nthr tid ij._1 ij._2;
  (g_gg_s cols chunk nthr ij._1 ij._2, g_gg_x cols chunk nthr ij._1 ij._2)

let bij_gg (rows cols chunk nthr : pos) (tid : natlt nthr)
           (_ : squash (geo_ok rows cols chunk nthr))
           (sx : chunkA rows cols chunk nthr)
  : GTot (chunkB rows cols chunk nthr tid)
= lemma_ff rows cols chunk nthr tid sx._1 sx._2;
  (g_ff_row rows cols chunk nthr tid sx._1, g_ff_col cols chunk nthr tid sx._2)

let chunk_bij (rows cols chunk nthr : pos) (tid : natlt nthr)
              (sq : squash (geo_ok rows cols chunk nthr))
  : (chunkB rows cols chunk nthr tid =~ chunkA rows cols chunk nthr)
= mk_bijection
    (bij_ff rows cols chunk nthr tid sq)
    (bij_gg rows cols chunk nthr tid sq)
    (fun (sx : chunkA rows cols chunk nthr) ->
       lemma_ff rows cols chunk nthr tid sx._1 sx._2;
       lemma_ff_inv rows cols chunk nthr tid sx._1 sx._2)
    (fun (ij : chunkB rows cols chunk nthr tid) ->
       lemma_gg rows cols chunk nthr tid ij._1 ij._2)

open Kuiper.ForEvery
open Pulse.Lib.Forall { elim_forall }
open Pulse.Lib.Pledge

(* One copy-step's destination [row_cells] view (contents [v]). *)
unfold let stage_rc
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr))
  (v : seq et { Seq.length v == SZ.v (chunk et) })
  : slprop
= let ck = SZ.v (chunk et) in
  let _ = lemma_ff rows cols ck nthr tid s 0 in
  let _ = lemma_tcol_bound cols ck nthr tid in
  row_cells m f
    (g_ff_row rows cols ck nthr tid s)
    (g_t_col cols ck nthr tid)
    ck v

let cell_body
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  (m : array2 et l) (em : chest2 et rows cols)
  (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (sx : chunkA rows cols (SZ.v (chunk et)) nthr)
  : slprop
= let ij = (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq).gg sx in
  T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)

ghost
fn stage_rc_intro
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (em : chest2 et rows cols)
  (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr))
  requires
    (forall+ (x : natlt (SZ.v (chunk et))).
      cell_body m em nthr tid sq f (s, x))
  ensures
    (exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
      stage_rc m nthr tid sq f s v)
{
  lemma_ff rows cols (SZ.v (chunk et)) nthr tid s 0;
  lemma_tcol_bound cols (SZ.v (chunk et)) nthr tid;
  let rowix : natlt rows = g_ff_row rows cols (SZ.v (chunk et)) nthr tid s;
  let tcol = g_t_col cols (SZ.v (chunk et)) nthr tid;
  let v : (v:seq et{Seq.length v == SZ.v (chunk et)}) =
    Seq.init_ghost (SZ.v (chunk et)) (fun (x:nat{x < SZ.v (chunk et)}) -> acc2 em rowix (tcol + x));
  Seq.init_ghost_index (SZ.v (chunk et)) (fun (x:nat{x < SZ.v (chunk et)}) -> acc2 em rowix (tcol + x));
  forevery_ext
    (fun (x : natlt (SZ.v (chunk et))) -> cell_body m em nthr tid sq f (s, x))
    (fun (x : natlt (SZ.v (chunk et))) ->
       T.tensor_pts_to_cell m #f
         (idx2 (g_ff_row rows cols (SZ.v (chunk et)) nthr tid s)
               (g_t_col cols (SZ.v (chunk et)) nthr tid + x))
         (Seq.index v x));
  rewrite
    (forall+ (x : natlt (SZ.v (chunk et))).
       T.tensor_pts_to_cell m #f
         (idx2 (g_ff_row rows cols (SZ.v (chunk et)) nthr tid s)
               (g_t_col cols (SZ.v (chunk et)) nthr tid + x))
         (Seq.index v x))
  as
    (row_cells m f
       (g_ff_row rows cols (SZ.v (chunk et)) nthr tid s)
       (g_t_col cols (SZ.v (chunk et)) nthr tid)
       (SZ.v (chunk et)) v);
}

(* forall+ (in-chunk cells at perm [f]) -> per-copy-step row_cells. *)
ghost
fn cells_to_rows
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (em : chest2 et rows cols)
  (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  requires
    (forall+ (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
  ensures
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
      exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
        stage_rc m nthr tid sq f s v)
{
  forevery_iso (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq)
    (fun (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_ext
    (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) ->
       (fun (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid) ->
          T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
         ((chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq).gg sx))
    (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) ->
       cell_body m em nthr tid sq f sx);
  forevery_unflatten'
    (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) ->
       cell_body m em nthr tid sq f sx);
  forevery_map
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       (forall+ (x : natlt (SZ.v (chunk et))). cell_body m em nthr tid sq f (s, x)))
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       (exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
         stage_rc m nthr tid sq f s v))
    (stage_rc_intro m em nthr tid sq f);
}

(* own_strided_chunks (per-thread, 1.0R) -> per-copy-step row_cells. *)
ghost
fn own_chunks_to_rows
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (em : chest2 et rows cols)
  (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  requires FB.own_strided_chunks m em nthr tid
  ensures
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
      exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
        stage_rc m nthr tid sq 1.0R s v)
{
  unfold FB.own_strided_chunks m em nthr tid;
  rewrite
    (forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij}).
       T.tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
  as
    (forall+ (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid).
       T.tensor_pts_to_cell m #1.0R (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  cells_to_rows m em nthr tid sq 1.0R;
}

(* ---- reverse peel : per-copy-step row_cells -> own_strided_chunks ---- *)

let cell_at
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (sx : chunkA rows cols (SZ.v (chunk et)) nthr) (e : et)
  : slprop
= let ij = (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq).gg sx in
  T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) e

ghost
fn intro_ex_cell_at
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (sx : chunkA rows cols (SZ.v (chunk et)) nthr) (e0 : et)
  requires cell_at m nthr tid sq f sx e0
  ensures (exists* (e : et). cell_at m nthr tid sq f sx e)
{
  ()
}

ghost
fn stage_rc_elim
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr))
  (v : seq et { Seq.length v == SZ.v (chunk et) })
  requires stage_rc m nthr tid sq f s v
  ensures
    (forall+ (x : natlt (SZ.v (chunk et))).
       exists* (e : et). cell_at m nthr tid sq f (s, x) e)
{
  lemma_ff rows cols (SZ.v (chunk et)) nthr tid s 0;
  lemma_tcol_bound cols (SZ.v (chunk et)) nthr tid;
  rewrite
    (stage_rc m nthr tid sq f s v)
  as
    (forall+ (x : natlt (SZ.v (chunk et))).
       T.tensor_pts_to_cell m #f
         (idx2 (g_ff_row rows cols (SZ.v (chunk et)) nthr tid s)
               (g_t_col cols (SZ.v (chunk et)) nthr tid + x))
         (Seq.index v x));
  forevery_ext
    (fun (x : natlt (SZ.v (chunk et))) ->
       T.tensor_pts_to_cell m #f
         (idx2 (g_ff_row rows cols (SZ.v (chunk et)) nthr tid s)
               (g_t_col cols (SZ.v (chunk et)) nthr tid + x))
         (Seq.index v x))
    (fun (x : natlt (SZ.v (chunk et))) ->
       cell_at m nthr tid sq f (s, x) (Seq.index v x));
  forevery_map
    (fun (x : natlt (SZ.v (chunk et))) ->
       cell_at m nthr tid sq f (s, x) (Seq.index v x))
    (fun (x : natlt (SZ.v (chunk et))) ->
       (exists* (e : et). cell_at m nthr tid sq f (s, x) e))
    (fun (x : natlt (SZ.v (chunk et))) ->
       intro_ex_cell_at m nthr tid sq f (s, x) (Seq.index v x));
}

(* clamp into range; off-chunk cells get an irrelevant in-range default. *)
let clampn (n : pos) (x : nat) : natlt n = if x < n then x else 0

(* Reconstructed chest: on in-chunk cells it agrees with [efun] via the
   backward map; off-chunk cells are irrelevant. *)
unfold
let em_of
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (efun : chunkA rows cols (SZ.v (chunk et)) nthr -> GTot et)
  : chest2 et rows cols
= let ck = SZ.v (chunk et) in
  let _ = lemma_ai_exact rows cols ck nthr in
  let ai : pos = g_a_iters rows cols ck nthr in
  mk2 (fun (i : natlt rows) (j : natlt cols) ->
         efun (clampn ai (g_gg_s cols ck nthr i j),
               clampn ck (g_gg_x cols ck nthr i j)))

let lemma_em_acc
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (efun : chunkA rows cols (SZ.v (chunk et)) nthr -> GTot et)
  (sx : chunkA rows cols (SZ.v (chunk et)) nthr)
  : Lemma
    (let ij = (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq).gg sx in
     acc2 (em_of nthr tid sq efun) ij._1 ij._2 == efun sx)
= let ck = SZ.v (chunk et) in
  lemma_ai_exact rows cols ck nthr;
  lemma_ff rows cols ck nthr tid sx._1 sx._2;
  lemma_ff_inv rows cols ck nthr tid sx._1 sx._2

ghost
fn cell_at_to_em
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (efun : chunkA rows cols (SZ.v (chunk et)) nthr -> GTot et)
  (sx : chunkA rows cols (SZ.v (chunk et)) nthr)
  requires cell_at m nthr tid sq f sx (efun sx)
  ensures
    (let ij = (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq).gg sx in
     T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2)
       (acc2 (em_of nthr tid sq efun) ij._1 ij._2))
{
  lemma_em_acc nthr tid sq efun sx;
  rewrite
    (cell_at m nthr tid sq f sx (efun sx))
  as
    (let ij = (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq).gg sx in
     T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2)
       (acc2 (em_of nthr tid sq efun) ij._1 ij._2));
}

(* per-copy-step row_cells (perm [f]) -> in-chunk cells (perm [f]) of a
   reconstructed chest. *)
ghost
fn rows_to_cells
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  requires
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
       exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
         stage_rc m nthr tid sq f s v)
  returns em : chest2 et rows cols
  ensures
    (forall+ (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
{
  let vfun =
    forevery_exists
      (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr))
           (v : seq et { Seq.length v == SZ.v (chunk et) }) ->
         stage_rc m nthr tid sq f s v);
  forevery_map
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       stage_rc m nthr tid sq f s (vfun s))
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       (forall+ (x : natlt (SZ.v (chunk et))).
          exists* (e : et). cell_at m nthr tid sq f (s, x) e))
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       stage_rc_elim m nthr tid sq f s (vfun s));
  forevery_flatten'
    (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) ->
       (exists* (e : et). cell_at m nthr tid sq f sx e));
  let efun =
    forevery_exists
      (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) (e : et) ->
         cell_at m nthr tid sq f sx e);
  forevery_map
    (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) ->
       cell_at m nthr tid sq f sx (efun sx))
    (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) ->
       (let ij = (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq).gg sx in
        T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2)
          (acc2 (em_of nthr tid sq efun) ij._1 ij._2)))
    (fun (sx : chunkA rows cols (SZ.v (chunk et)) nthr) ->
       cell_at_to_em m nthr tid sq f efun sx);
  forevery_iso_back
    (chunk_bij rows cols (SZ.v (chunk et)) nthr tid sq)
    (fun (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2)
         (acc2 (em_of nthr tid sq efun) ij._1 ij._2));
  em_of nthr tid sq efun
}

(* per-copy-step row_cells (1.0R) -> live_strided_chunks (pipe_live). *)
ghost
fn rows_to_own_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  requires
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
       exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
         stage_rc m nthr tid sq 1.0R s v)
  ensures FB.live_strided_chunks m nthr tid
{
  let em = rows_to_cells m nthr tid sq 1.0R;
  rewrite
    (forall+ (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid).
       T.tensor_pts_to_cell m #1.0R (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
  as
    (forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij}).
       T.tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  fold (FB.own_strided_chunks m em nthr tid);
  fold (FB.live_strided_chunks m nthr tid);
}

(* ================================================================== *)
(*  Source subtile peel : whole (Frac f) tile <-> mine rows + residual *)
(* ================================================================== *)

(* [m |-> Frac f e]  ->  per-copy rows @f (this thread's chunks)  **
   the residual cells (every other thread's chunks) at [Frac f]. *)
ghost
fn subtile_to_rows
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (#e : chest2 et rows cols)
  requires m |-> Frac f e
  ensures
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
      exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
        stage_rc m nthr tid sq f s v)
    **
    (forall+ (ij : (natlt rows & natlt cols){~(CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij)}).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2))
{
  T.tensor_ilower2 m;
  forevery_flatten'
    (fun (ij : (natlt rows & natlt cols)) ->
       (Cell m (idx2 ij._1 ij._2) |-> Frac f (acc e (idx2 ij._1 ij._2))));
  forevery_ext
    (fun (ij : (natlt rows & natlt cols)) ->
       (Cell m (idx2 ij._1 ij._2) |-> Frac f (acc e (idx2 ij._1 ij._2))))
    (fun (ij : (natlt rows & natlt cols)) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2));
  forevery_refine_split
    (fun (ij : (natlt rows & natlt cols)) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2))
    (fun (ij : (natlt rows & natlt cols)) ->
       CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij);
  rewrite
    (forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij}).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2))
  as
    (forall+ (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2));
  cells_to_rows m e nthr tid sq f;
}

(* Merge the reconstructed in-chunk chest [em] with the residual chest [e]. *)
unfold
let em_merge
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos)
  (nthr : pos) (tid : natlt nthr)
  (em e : chest2 et rows cols)
  : chest2 et rows cols
= mk2 (fun (i:natlt rows) (j:natlt cols) ->
         if CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid (i, j)
         then acc2 em i j else acc2 e i j)

(* Inverse of [subtile_to_rows] : re-assemble the whole tile at [Frac f]. *)
ghost
fn rows_to_subtile
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (#e : chest2 et rows cols)
  requires
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
       exists* (v : seq et { Seq.length v == SZ.v (chunk et) }).
         stage_rc m nthr tid sq f s v)
    **
    (forall+ (ij : (natlt rows & natlt cols){~(CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij)}).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2))
  ensures
    (exists* (e' : chest2 et rows cols). m |-> Frac f e')
{
  let em = rows_to_cells m nthr tid sq f;
  let e' = em_merge #et nthr tid em e;
  forevery_ext
    (fun (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    (fun (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e' ij._1 ij._2));
  rewrite
    (forall+ (ij : chunkB rows cols (SZ.v (chunk et)) nthr tid).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e' ij._1 ij._2))
  as
    (forall+ (ij : (natlt rows & natlt cols){CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij}).
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e' ij._1 ij._2));
  forevery_ext
    (fun (ij : (natlt rows & natlt cols){~(CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij)}) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2))
    (fun (ij : (natlt rows & natlt cols){~(CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij)}) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e' ij._1 ij._2));
  forevery_refine_join'
    (fun (ij : (natlt rows & natlt cols)) ->
       CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij)
    (fun (ij : (natlt rows & natlt cols)) ->
       ~(CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij))
    (fun (ij : (natlt rows & natlt cols){CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij
                                          \/ ~(CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij)}) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e' ij._1 ij._2));
  forevery_unrefine
    #(natlt rows & natlt cols)
    #(fun (ij : (natlt rows & natlt cols)) ->
        CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij
        \/ ~(CV.in_chunk (SZ.v (chunk et)) rows cols nthr tid ij))
    (fun (ij : (natlt rows & natlt cols)) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e' ij._1 ij._2));
  forevery_unflatten'
    (fun (ij : (natlt rows & natlt cols)) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e' ij._1 ij._2));
  rewrite
    (forall+ (r : natlt rows) (c : natlt cols).
       T.tensor_pts_to_cell m #f (idx2 r c) (acc2 e' r c))
  as
    (forall+ (r : natlt rows) (c : natlt cols).
       (Cell m (idx2 r c) |-> Frac f (acc e' (idx2 r c))));
  T.tensor_iraise2 m #f #e';
}

(* ================================================================== *)
(*  Pledge over a forall+.  TODO(upstream): generic, belongs in the   *)
(*  Kuiper pledge library.                                            *)
(* ================================================================== *)
ghost
fn rec collect_pledges (f : slprop) (n : nat) (p : natlt n -> slprop)
  requires forall+ (i : natlt n). pledge0 f (p i)
  ensures  pledge0 f (forall+ (i : natlt n). p i)
  decreases n
{
  if (n = 0) {
    forevery_elim_empty #(natlt n) (fun (i : natlt n) -> pledge0 f (p i));
    return_pledge f emp #_;
    rewrite_pledge emp (forall+ (i : natlt n). p i)
      #emp_inames
      fn _ {
        forevery_intro_empty #(natlt n) p;
      };
  } else {
    forevery_natlt_pop n (fun (i : natlt n) -> pledge0 f (p i));
    collect_pledges f (n - 1) (fun (i : natlt (n - 1)) -> p (natlt_coerce i));
    join_pledge (forall+ (i : natlt (n - 1)). p (natlt_coerce i)) (p (n - 1));
    rewrite_pledge
      ((forall+ (i : natlt (n - 1)). p (natlt_coerce i)) ** p (n - 1))
      (forall+ (i : natlt n). p i)
      #emp_inames
      fn _ {
        forevery_natlt_push n p;
      };
  }
}

(* ================================================================== *)
(*  Per-cell 16-byte alignment of a chunk-aligned strided layout.     *)
(* ================================================================== *)

(* [ck | v]  ==>  [16 | v * size], since [ck * size == 16]. *)
let lemma_div16
  (#et : Type0) {| sized et, has_vec_cpy et |} (v : nat)
  : Lemma (requires SZ.v (chunk et) /? v)
          (ensures 16 /? (v * SZ.v (size #et)))
= let ck = SZ.v (chunk et) in
  let sz_ = SZ.v (size #et) in
  assert (ck * sz_ == 16);
  let q = v / ck in
  lemma_divides_exact ck v;
  assert (v == ck * q);
  ML.paren_mul_right ck q sz_;
  ML.swap_mul q sz_;
  ML.paren_mul_right ck sz_ q;
  assert (v * sz_ == 16 * q);
  lemma_divides_product_l 16 16 q;
  assert (16 /? (16 * q))

(* A chunk-column, chunk-strided, 16-byte-based cell is 16-byte aligned. *)
let lemma_cell_aligned16
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (str : strided_row_major (vtlayout_of_tlayout l))
  (x : array et)
  (i : natlt rows) (j : natlt cols)
  : Lemma
      (requires aligned 16 x /\
                aligned_strided_row_major (SZ.v (chunk et)) str /\
                SZ.v (chunk et) /? j)
      (ensures aligned' 16 x (cell_of_pos l i j))
= let ck = SZ.v (chunk et) in
  let sz_ = SZ.v (size #et) in
  let _ = str.pf i j in
  let off = SZ.v str.offset in
  let stride = SZ.v str.stride in
  let cell = cell_of_pos l i j in
  assert (cell == off + stride * i + j);
  lemma_nat_divides_pos_divides ck stride;
  lemma_nat_divides_pos_divides ck off;
  lemma_divides_product_l ck stride i;
  lemma_div16 #et off;
  lemma_div16 #et (stride * i);
  lemma_div16 #et j;
  ML.distributivity_add_left off (stride * i) sz_;
  ML.distributivity_add_left (off + stride * i) j sz_;
  assert (cell * sz_ == (off * sz_ + (stride * i) * sz_) + j * sz_);
  lemma_divides_sum 16 (off * sz_) ((stride * i) * sz_);
  lemma_divides_sum 16 (off * sz_ + (stride * i) * sz_) (j * sz_);
  lemma_nat_divides_pos_divides 16 (base_address x);
  lemma_divides_sum 16 (base_address x) (cell * sz_);
  lemma_nat_divides_pos_divides 16 (base_address x + cell * sz_)

(* ================================================================== *)
(*  Sendability of the source residual, so it can ride the pledge.    *)
(* ================================================================== *)

(* chunk divides t_col (t_col is a chunk-aligned column). *)
let lemma_tcol_div (cols chunk nthr : pos) (tid : nat)
  : Lemma (requires cols % chunk == 0)
          (ensures chunk /? g_t_col cols chunk nthr tid)
= let m : pos = cols / chunk in
  ML.div_exact_r cols chunk;
  let q = (tid * chunk) / cols in
  let tc = (tid * chunk) % cols in
  ML.lemma_div_mod (tid * chunk) cols;
  ML.modulo_lemma tc cols;
  assert (tc == tid * chunk - q * cols);
  assert (cols == chunk * m);
  assert (tc == chunk * (tid - q * m));
  lemma_divides_product_l chunk chunk (tid - q * m)

module Send = Pulse.Lib.Send
module AC = Pulse.Lib.Array.Core

(* Bridge [is_send_across vis] to plain [is_send] (the visibility retracts
   [process_of]).  Local copy of the unexported [send_of_vis]. *)
let my_send_of_vis (#vis : AC.visibility) (#p : slprop)
  (i : Send.is_send_across vis p) : Send.is_send p
  = fun l l' -> i l l'

unfold let src_residual
  (#et : Type0) {| _s0 : sized et |} {| _v0 : has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (f : perm) (e : chest2 et rows cols)
  : slprop
= let ck = SZ.v (chunk et) in
  forall+ (ij : (natlt rows & natlt cols){~(CV.in_chunk ck rows cols nthr tid ij)}).
    T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2)

let src_residual_send
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (f : perm) (e : chest2 et rows cols)
  (sqg : squash (is_global m))
  : Send.is_send (src_residual m nthr tid f e)
= let ck = SZ.v (chunk et) in
  let per (ij : (natlt rows & natlt cols){~(CV.in_chunk ck rows cols nthr tid ij)})
    : Send.is_send_across gpu_of
        (T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2))
    = T.is_send_across_global_tensor_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2) in
  my_send_of_vis #gpu_of #(src_residual m nthr tid f e)
    (is_send_across_forevery
       (fun (ij : (natlt rows & natlt cols){~(CV.in_chunk ck rows cols nthr tid ij)}) ->
          T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 e ij._1 ij._2))
       gpu_of #per)

(* ================================================================== *)
(*  stage_one : one thread stages ONE operand tile (dst <- src).      *)
(* ================================================================== *)

let natlt_is_between (n : nat) : Lemma (natlt n == between 0 n)
= FStar.RefinementExtensionality.refext
    nat (fun (x:nat) -> x < n) (fun (x:nat) -> 0 <= x /\ x < n);
  assert (x:nat{x < n} == x:nat{0 <= x /\ x < n});
  assert (natlt n == x:nat{x < n});
  assert_norm (between 0 n == x:nat{0 <= x /\ x < n})

(* Weaken a concrete per-step row to an existential one. *)
ghost
fn wrap_ex_rc
  (#et : Type0) {| _sz : sized et |} {| _vc : has_vec_cpy et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  {| T.ctlayout l, strided_row_major (vtlayout_of_tlayout l) |}
  (m : array2 et l) (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr))
  (v : seq et { Seq.length v == SZ.v (chunk et) })
  requires stage_rc m nthr tid sq f s v
  ensures  exists* (w : seq et { Seq.length w == SZ.v (chunk et) }). stage_rc m nthr tid sq f s w
{ () }

(* Assemble the per-step copy pledges (plus the source residual) into the
   whole-tile postcondition. Ghost so it may use [forevery_exists]. *)
ghost
fn finish_stage
  (#et : Type0) {| _sz : sized et |} {| _vc : has_vec_cpy et |}
  (#rows #cols : pos) (#ld : layout2 rows cols)
  {| T.ctlayout ld, strided_row_major (vtlayout_of_tlayout ld) |}
  (m_d : array2 et ld)
  (#ls : layout2 rows cols)
  {| T.ctlayout ls, strided_row_major (vtlayout_of_tlayout ls) |}
  (m_s : array2 et ls)
  (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (f : perm)
  (#e_s : chest2 et rows cols)
  (b : Kuiper.PipelineCopy.pipeline_batch_t)
  (sqg : squash (is_global m_s))
  requires
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
       exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }).
         pledge0 (Kuiper.PipelineCopy.batch_done b)
           (stage_rc m_d nthr tid sq 1.0R s v ** stage_rc m_s nthr tid sq f s v))
    ** src_residual m_s nthr tid f e_s
  ensures pledge0 (Kuiper.PipelineCopy.batch_done b)
            (FB.live_strided_chunks m_d nthr tid ** (exists* e. m_s |-> Frac f e))
{
  let vf = forevery_exists
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) (v : seq et { Seq.length v == (SZ.v (chunk et)) }) ->
       pledge0 (Kuiper.PipelineCopy.batch_done b)
         (stage_rc m_d nthr tid sq 1.0R s v ** stage_rc m_s nthr tid sq f s v));

  collect_pledges (Kuiper.PipelineCopy.batch_done b) (g_a_iters rows cols (SZ.v (chunk et)) nthr)
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       stage_rc m_d nthr tid sq 1.0R s (vf s) ** stage_rc m_s nthr tid sq f s (vf s));

  return_pledge (Kuiper.PipelineCopy.batch_done b) (src_residual m_s nthr tid f e_s)
    #(src_residual_send m_s nthr tid f e_s ());

  join_pledge
    (forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
       stage_rc m_d nthr tid sq 1.0R s (vf s) ** stage_rc m_s nthr tid sq f s (vf s))
    (src_residual m_s nthr tid f e_s);

  rewrite_pledge
    ((forall+ (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)).
        stage_rc m_d nthr tid sq 1.0R s (vf s) ** stage_rc m_s nthr tid sq f s (vf s))
     ** src_residual m_s nthr tid f e_s)
    (FB.live_strided_chunks m_d nthr tid ** (exists* e. m_s |-> Frac f e))
    #emp_inames
    fn _ {
      forevery_unzip
        (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) -> stage_rc m_d nthr tid sq 1.0R s (vf s))
        (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) -> stage_rc m_s nthr tid sq f s (vf s));
      forevery_map
        (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) -> stage_rc m_d nthr tid sq 1.0R s (vf s))
        (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
           exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_d nthr tid sq 1.0R s v)
        (fun s -> wrap_ex_rc m_d nthr tid sq 1.0R s (vf s));
      rows_to_own_chunks m_d nthr tid sq;
      forevery_map
        (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) -> stage_rc m_s nthr tid sq f s (vf s))
        (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
           exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_s nthr tid sq f s v)
        (fun s -> wrap_ex_rc m_s nthr tid sq f s (vf s));
      rows_to_subtile m_s nthr tid sq f;
    };
}

inline_for_extraction noextract
fn stage_one
  (#et : Type0) {| _sz : sized et |} {| _vc : has_vec_cpy et |}
  (#rows #cols : pos) (#ld : layout2 rows cols)
  {| T.ctlayout ld, dstr : strided_row_major (vtlayout_of_tlayout ld) |}
  (m_d : array2 et ld)
  (#ls : layout2 rows cols)
  {| T.ctlayout ls, sstr : strided_row_major (vtlayout_of_tlayout ls) |}
  (m_s : array2 et ls)
  (nthr : pos) (tid : natlt nthr)
  (sq : squash (geo_ok rows cols (SZ.v (chunk et)) nthr))
  (#em_d : chest2 et rows cols)
  (f : perm)
  (#e_s : chest2 et rows cols)
  (t_row t_col row_step a_iters : SZ.t)
  (b : Kuiper.PipelineCopy.pipeline_batch_t)
  (sqh : squash (
     SZ.v t_row == g_t_row cols (SZ.v (chunk et)) nthr tid /\
     SZ.v t_col == g_t_col cols (SZ.v (chunk et)) nthr tid /\
     SZ.v row_step == g_row_step cols (SZ.v (chunk et)) nthr /\
     SZ.v a_iters == g_a_iters rows cols (SZ.v (chunk et)) nthr /\
     SZ.fits rows /\
     aligned 16 (T.core m_d) /\ aligned_strided_row_major (SZ.v (chunk et)) dstr /\
     aligned 16 (T.core m_s) /\ aligned_strided_row_major (SZ.v (chunk et)) sstr /\
     is_global m_s))
  ()
  preserves gpu
  preserves Kuiper.PipelineCopy.batch_live b
  requires FB.own_strided_chunks m_d em_d nthr tid ** (m_s |-> Frac f e_s)
  ensures pledge0 (Kuiper.PipelineCopy.batch_done b)
            (FB.live_strided_chunks m_d nthr tid ** (exists* e. m_s |-> Frac f e))
{
  own_chunks_to_rows m_d em_d nthr tid sq;
  subtile_to_rows m_s nthr tid sq f;

  forevery_zip
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_d nthr tid sq 1.0R s v)
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_s nthr tid sq f s v);

  natlt_is_between (g_a_iters rows cols (SZ.v (chunk et)) nthr);
  forevery_rw_type (natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) (between 0 (SZ.v a_iters))
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       (exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_d nthr tid sq 1.0R s v) **
       (exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_s nthr tid sq f s v));

  Kuiper.For.for_loop' 0sz a_iters
    (fun (s : between 0 (SZ.v a_iters)) ->
       (exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_d nthr tid sq 1.0R s v) **
       (exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }). stage_rc m_s nthr tid sq f s v))
    (fun (s : between 0 (SZ.v a_iters)) ->
       exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }).
         pledge0 (Kuiper.PipelineCopy.batch_done b)
           (stage_rc m_d nthr tid sq 1.0R s v ** stage_rc m_s nthr tid sq f s v))
    (gpu ** Kuiper.PipelineCopy.batch_live b)
    fn x {
       lemma_ff rows cols (SZ.v (chunk et)) nthr tid (SZ.v x) 0;
       lemma_tcol_bound cols (SZ.v (chunk et)) nthr tid;
       lemma_tcol_div cols (SZ.v (chunk et)) nthr tid;
       assert (pure (SZ.v x * SZ.v row_step < rows));
       let di = t_row +^ (x *^ row_step);
       assert (pure (SZ.v di == g_ff_row rows cols (SZ.v (chunk et)) nthr tid (SZ.v x)));
       assert (pure (SZ.v t_col == g_t_col cols (SZ.v (chunk et)) nthr tid));
       lemma_cell_aligned16 dstr (T.core m_d) (SZ.v di) (SZ.v t_col);
       lemma_cell_aligned16 sstr (T.core m_s) (SZ.v di) (SZ.v t_col);
       Kuiops.Array2.Vectorized.Pipelined.array2_vec_cpy_pipelined m_d di t_col m_s di t_col ();
       rewrite each (SZ.v di) as (g_ff_row rows cols (SZ.v (chunk et)) nthr tid (SZ.v x));
       rewrite each (SZ.v t_col) as (g_t_col cols (SZ.v (chunk et)) nthr tid);
     };

  forevery_rw_type (between 0 (SZ.v a_iters)) (natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr))
    (fun (s : natlt (g_a_iters rows cols (SZ.v (chunk et)) nthr)) ->
       exists* (v : seq et { Seq.length v == (SZ.v (chunk et)) }).
         pledge0 (Kuiper.PipelineCopy.batch_done b)
           (stage_rc m_d nthr tid sq 1.0R s v ** stage_rc m_s nthr tid sq f s v));

  finish_stage m_d m_s nthr tid sq f b ();
}

(* ================================================================== *)
(*  stage_tiles : one thread's contribution to staging one k-tile of  *)
(*  A and of B into one pair of shared buffers, then pipeline_commit.  *)
(*                                                                     *)
(*  The div/mod of the reference's hoisted addressing is computed ONCE *)
(*  by the caller (kloop) and threaded in as [t_row]/[t_col] for A and *)
(*  B; the inner per-copy loops only add [s*row_step].  The source     *)
(*  subtiles [mAs]/[mBs] are the kt-th k-tile subviews of global A/B,  *)
(*  extracted and permission-managed by the caller.                    *)
(* ================================================================== *)
inline_for_extraction noextract
fn stage_tiles
  (#et : Type0) {| _sz : sized et |} {| _vc : has_vec_cpy et |}
  (#bm #bk : pos)
  (#ldA : layout2 bm bk)
  {| T.ctlayout ldA, dstrA : strided_row_major (vtlayout_of_tlayout ldA) |}
  (mAd : array2 et ldA)
  (#lsA : layout2 bm bk)
  {| T.ctlayout lsA, sstrA : strided_row_major (vtlayout_of_tlayout lsA) |}
  (mAs : array2 et lsA)
  (#bn : pos)
  (#ldB : layout2 bn bk)
  {| T.ctlayout ldB, dstrB : strided_row_major (vtlayout_of_tlayout ldB) |}
  (mBd : array2 et ldB)
  (#lsB : layout2 bn bk)
  {| T.ctlayout lsB, sstrB : strided_row_major (vtlayout_of_tlayout lsB) |}
  (mBs : array2 et lsB)
  (nthr : pos) (tid : natlt nthr)
  (sqA : squash (geo_ok bm bk (SZ.v (chunk et)) nthr))
  (sqB : squash (geo_ok bn bk (SZ.v (chunk et)) nthr))
  (#emAd : chest2 et bm bk) (fA : perm) (#eAs : chest2 et bm bk)
  (#emBd : chest2 et bn bk) (fB : perm) (#eBs : chest2 et bn bk)
  (a_t_row a_t_col a_row_step a_iters : SZ.t)
  (b_t_row b_t_col b_row_step b_iters : SZ.t)
  (b : Kuiper.PipelineCopy.pipeline_batch_t)
  (sqh : squash (
     SZ.v a_t_row == g_t_row bk (SZ.v (chunk et)) nthr tid /\
     SZ.v a_t_col == g_t_col bk (SZ.v (chunk et)) nthr tid /\
     SZ.v a_row_step == g_row_step bk (SZ.v (chunk et)) nthr /\
     SZ.v a_iters == g_a_iters bm bk (SZ.v (chunk et)) nthr /\
     SZ.v b_t_row == g_t_row bk (SZ.v (chunk et)) nthr tid /\
     SZ.v b_t_col == g_t_col bk (SZ.v (chunk et)) nthr tid /\
     SZ.v b_row_step == g_row_step bk (SZ.v (chunk et)) nthr /\
     SZ.v b_iters == g_a_iters bn bk (SZ.v (chunk et)) nthr /\
     SZ.fits bm /\ SZ.fits bn /\
     aligned 16 (T.core mAd) /\ aligned_strided_row_major (SZ.v (chunk et)) dstrA /\
     aligned 16 (T.core mAs) /\ aligned_strided_row_major (SZ.v (chunk et)) sstrA /\
     is_global mAs /\
     aligned 16 (T.core mBd) /\ aligned_strided_row_major (SZ.v (chunk et)) dstrB /\
     aligned 16 (T.core mBs) /\ aligned_strided_row_major (SZ.v (chunk et)) sstrB /\
     is_global mBs))
  ()
  preserves gpu
  requires FB.own_strided_chunks mAd emAd nthr tid ** (mAs |-> Frac fA eAs) **
           FB.own_strided_chunks mBd emBd nthr tid ** (mBs |-> Frac fB eBs) **
           Kuiper.PipelineCopy.batch_live b
  returns b' : Kuiper.PipelineCopy.pipeline_batch_t
  ensures
    pledge0 (Kuiper.PipelineCopy.batch_done b)
      (FB.live_strided_chunks mAd nthr tid ** FB.live_strided_chunks mBd nthr tid **
       (exists* eA eB. (mAs |-> Frac fA eA) ** (mBs |-> Frac fB eB))) **
    Kuiper.PipelineCopy.batch_committed b **
    Kuiper.PipelineCopy.batch_live b' **
    pure (fst b' == fst b /\ snd b' > snd b)
{
  stage_one mAd mAs nthr tid sqA fA a_t_row a_t_col a_row_step a_iters b () ();
  stage_one mBd mBs nthr tid sqB fB b_t_row b_t_col b_row_step b_iters b () ();
  let b' = Kuiper.PipelineCopy.pipeline_commit #b;
  join_pledge
    (FB.live_strided_chunks mAd nthr tid ** (exists* e. mAs |-> Frac fA e))
    (FB.live_strided_chunks mBd nthr tid ** (exists* e. mBs |-> Frac fB e));
  rewrite_pledge
    ((FB.live_strided_chunks mAd nthr tid ** (exists* e. mAs |-> Frac fA e)) **
     (FB.live_strided_chunks mBd nthr tid ** (exists* e. mBs |-> Frac fB e)))
    (FB.live_strided_chunks mAd nthr tid ** FB.live_strided_chunks mBd nthr tid **
     (exists* eA eB. (mAs |-> Frac fA eA) ** (mBs |-> Frac fB eB)))
    #emp_inames
    fn _ { () };
  b'
}
