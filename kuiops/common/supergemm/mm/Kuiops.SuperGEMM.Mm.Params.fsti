module Kuiops.SuperGEMM.Mm.Params

(* Derived quantities and the [constraints] predicate of the software-pipelined
   tensor-core GEMM (D = A @ B^T).  See DESIGN.md.

   The five tuning parameters [bm bn bk wm wn skew] stay parameters everywhere.
   [frag = 16] (WMMA 16x16x16), [warp_size = 32] and the cp.async 16-byte
   granule are genuine constants; the pipeline depth is fixed at 2 because
   Kuiper can only express [__pipeline_wait_prior(0)].

   The derived quantities are named here as pure [nat] spec functions so that
   [constraints] and the downstream signatures read cleanly.  The machine-width
   values are computed at the launcher (module 8) from these same expressions;
   the small arithmetic lemmas below turn [constraints] into the individual
   divisibility / fits / bound facts the staging, barrier and epilogue modules
   consume. *)

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

module SZ = Kuiper.SizeT

let frag : nat = 16
inline_for_extraction noextract
let frag_sz : (x:szp{SZ.v x == frag}) = 16sz

(* ---- derived quantities (pure spec) ---- *)

unfold let warps_m (bm wm : szp) : nat = SZ.v bm / SZ.v wm
unfold let warps_n (bn wn : szp) : nat = SZ.v bn / SZ.v wn
unfold let warps   (bm bn wm wn : szp) : nat = warps_m bm wm * warps_n bn wn
unfold let nthr    (bm bn wm wn : szp) : nat = warps bm bn wm wn * SZ.v warp_size

unfold let mfrag (wm : szp) : nat = SZ.v wm / frag
unfold let nfrag (wn : szp) : nat = SZ.v wn / frag
unfold let kstep (bk : szp) : nat = SZ.v bk / frag

unfold let ldt (bk skew : szp) : nat = SZ.v bk + SZ.v skew

unfold let row_step
  (et_ab : Type0) {| sized et_ab, has_vec_cpy et_ab |}
  (bm bn bk wm wn : szp) : nat
  = nthr bm bn wm wn * SZ.v (chunk et_ab) / SZ.v bk

unfold let a_iters
  (et_ab : Type0) {| sized et_ab, has_vec_cpy et_ab |}
  (bm bn bk wm wn : szp) : nat
  = let rs = row_step et_ab bm bn bk wm wn in
    SZ.v bm / (if rs = 0 then 1 else rs)

unfold let b_iters
  (et_ab : Type0) {| sized et_ab, has_vec_cpy et_ab |}
  (bm bn bk wm wn : szp) : nat
  = let rs = row_step et_ab bm bn bk wm wn in
    SZ.v bn / (if rs = 0 then 1 else rs)

unfold let eskew (et_acc : Type0) {| sized et_acc, has_vec_cpy et_acc |} : nat = SZ.v (chunk et_acc)
unfold let lde   (et_acc : Type0) {| sized et_acc, has_vec_cpy et_acc |} (wn : szp) : nat
  = SZ.v wn + eskew et_acc

(* ---- constraints ---- *)

(* Divisibility, fits and thread-count obligations of the pipelined kernel.
   [chunk et_ab /?+ skew] and [chunk et_ab /?+ bk] together give
   [chunk et_ab /?+ ldt], which is what keeps every shared-tile row start
   16-byte aligned for cp.async. *)
let constraints
  (et_ab et_acc : Type0) {| sized et_ab, has_vec_cpy et_ab, sized et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp) : prop
=
  wm /?+ SZ.v bm /\
  wn /?+ SZ.v bn /\
  frag /?+ SZ.v wm /\
  frag /?+ SZ.v wn /\
  frag /?+ SZ.v bk /\
  SZ.v (chunk et_ab) /?+ SZ.v skew /\
  SZ.v (chunk et_ab) /?+ SZ.v bk /\
  SZ.v (chunk et_ab) * nthr bm bn wm wn /?+ (SZ.v bm * SZ.v bk) /\
  SZ.v (chunk et_ab) * nthr bm bn wm wn /?+ (SZ.v bn * SZ.v bk) /\
  nthr bm bn wm wn <= SZ.v max_threads /\
  SZ.fits (SZ.v bm * ldt bk skew) /\
  SZ.fits (SZ.v bn * ldt bk skew) /\
  SZ.fits (warps bm bn wm wn * frag * lde et_acc wn)

(* ---- machine-width sizes for the shared-memory descriptor ---- *)

inline_for_extraction noextract
val ldt_sz (bk skew : szp)
  (#_ : squash (SZ.fits (ldt bk skew)))
  : (x : szp { SZ.v x == ldt bk skew })

inline_for_extraction noextract
val abuf_sz (rows bk skew : szp)
  (#_ : squash (SZ.fits (SZ.v rows * ldt bk skew)))
  : (x : szp { SZ.v x == SZ.v rows * ldt bk skew })

inline_for_extraction noextract
val scratch_sz
  (et_acc : Type0) {| sized et_acc, has_vec_cpy et_acc |}
  (bm bn wm wn : szp)
  (#_ : squash (wm /?+ SZ.v bm /\ wn /?+ SZ.v bn))
  (#_ : squash (SZ.fits (warps bm bn wm wn * frag * lde et_acc wn)))
  : (x : szp { SZ.v x == warps bm bn wm wn * frag * lde et_acc wn })

(* ---- consequences downstream modules need ---- *)

val ldt_pos (bk skew : szp) : Lemma (ldt bk skew > 0)

val chunk_divides_ldt
  (et_ab et_acc : Type0) {| sized et_ab, has_vec_cpy et_ab, sized et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  : Lemma (requires constraints et_ab et_acc bm bn bk wm wn skew)
          (ensures SZ.v (chunk et_ab) /?+ ldt bk skew)

val nthr_pos
  (bm bn wm wn : szp)
  : Lemma (requires wm /?+ SZ.v bm /\ wn /?+ SZ.v bn)
          (ensures nthr bm bn wm wn > 0)

val nthr_le_max_threads
  (et_ab et_acc : Type0) {| sized et_ab, has_vec_cpy et_ab, sized et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  : Lemma (requires constraints et_ab et_acc bm bn bk wm wn skew)
          (ensures nthr bm bn wm wn <= SZ.v max_threads /\
                   SZ.fits (nthr bm bn wm wn))

val warp_divides_nthr
  (bm bn wm wn : szp)
  : Lemma (ensures warp_size /?+ nthr bm bn wm wn)

val bm_ldt_fits
  (et_ab et_acc : Type0) {| sized et_ab, has_vec_cpy et_ab, sized et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  : Lemma (requires constraints et_ab et_acc bm bn bk wm wn skew)
          (ensures SZ.fits (SZ.v bm * ldt bk skew) /\
                   SZ.fits (SZ.v bn * ldt bk skew))

val chunk_nthr_divides_ab
  (et_ab et_acc : Type0) {| sized et_ab, has_vec_cpy et_ab, sized et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  : Lemma (requires constraints et_ab et_acc bm bn bk wm wn skew)
          (ensures SZ.v (chunk et_ab) * nthr bm bn wm wn /?+ (SZ.v bm * SZ.v bk) /\
                   SZ.v (chunk et_ab) * nthr bm bn wm wn /?+ (SZ.v bn * SZ.v bk))
