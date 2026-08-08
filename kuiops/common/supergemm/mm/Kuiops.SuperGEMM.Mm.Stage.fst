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

(* Hypotheses on the tile geometry.  [cols] is the k-tile width [bk];
   [chunk] the 16-byte vector width; [nthr] the block thread count. *)
let geo_ok (rows cols chunk nthr : pos) : prop =
  cols % chunk == 0 /\                    (* chunk divides a row       *)
  (chunk * nthr) % cols == 0 /\           (* row_step is exact         *)
  (rows * cols) % (chunk * nthr) == 0     (* a_iters is exact          *)

unfold let g_row_step (cols chunk nthr : pos) : nat = (chunk * nthr) / cols
unfold let g_a_iters  (rows cols chunk nthr : pos) : nat = (rows * cols) / (chunk * nthr)
unfold let g_t_row (cols chunk nthr : pos) (tid : nat) : nat = (tid * chunk) / cols
unfold let g_t_col (cols chunk nthr : pos) (tid : nat) : nat = (tid * chunk) % cols

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
