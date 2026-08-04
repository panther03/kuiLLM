module Kuiops.Matmuls
#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Array2.Strided
open Kuiper.Array2.Strided.Slice
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Float.Casts { float_cast }
open Kuiper.TensorCore
open Kuiper.Tensor.Layout.Alg

module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

inline_for_extraction noextract
fn bt2d_async
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : et -> et -> et)
  (bm bn bk : szp)
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp { tm /?+ bm })
  (tn : szp { tn /?+ bn /\ (bm/tm * (bn/tn) <= max_threads) })
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (batch m n k : szp)
  (#_ : squash (SZ.fits (m * k) /\ SZ.fits (k * n) /\ SZ.fits (m * n)))
  (#_ : squash (SZ.fits (batch * (m * k)) /\ SZ.fits (batch * (k * n))
             /\ SZ.fits (batch * (m * n))))
  (gA : array3 et (l3_batched_row_major (SZ.v batch) (SZ.v m) (SZ.v k)) { is_global gA })
  (gB : array3 et (l3_batched_row_major (SZ.v batch) (SZ.v k) (SZ.v n)) { is_global gB })
  (gC : array3 et (l3_batched_row_major (SZ.v batch) (SZ.v m) (SZ.v n)) { is_global gC })
  (s : stream_t)
  (#eA : chest3 et (SZ.v batch) (SZ.v m) (SZ.v k))
  (#eB : chest3 et (SZ.v batch) (SZ.v k) (SZ.v n))
  (#eC : chest3 et (SZ.v batch) (SZ.v m) (SZ.v n))
  (#fA #fB : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * ((m/bm) * (n/bn)) <= max_blocks) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> eC)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
                   (gC |-> MS.gbmmcomb (fun (x:et) -> x) (fun (x:et) -> x) comb eC eA eB)))

(* [comb2_to] (the plain "D = A@B" combiner, with the accumulator cast to the
   output type) approximates the real-level [comb2]; the casts' [fcast_approx]
   SMT pattern discharges it. Kuiper ships the analogous [lincomb_to_approx2]
   but not this one. *)
let comb2_to_approx2
  (#et_acc #et_cd : Type0)
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  : Lemma (approx2 (MS.comb2_to #et_acc #et_cd) (MS.comb2 #real))
          [SMTPat (approx2 (MS.comb2_to #et_acc #et_cd) (MS.comb2 #real))]
  = ()

(* Kuiper's [MS.lincomb_to_approx2] carries no SMT pattern, so the JIT
   instantiations (which are partial applications, with no room for a lemma
   call) cannot discharge the [approx2] refinement from it. Restate it with a
   pattern. *)
let lincomb_to_approx2
  (#et_acc #et_cd : Type0)
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (alpha beta : et_acc)
  : Lemma (approx2 (MS.lincomb_to #et_acc #et_cd alpha beta)
                   (MS.rlincomb (to_real alpha) (to_real beta)))
          [SMTPat (approx2 (MS.lincomb_to #et_acc #et_cd alpha beta)
                           (MS.rlincomb (to_real alpha) (to_real beta)))]
  = MS.lincomb_to_approx2 #et_acc #et_cd alpha beta


inline_for_extraction noextract
fn tc2d_to_async
  (et_ab et_acc et_cd : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
  (#_ : squash (chunk et_cd /?+ tn))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_size)) <= max_threads))
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (rows shared cols : szp)
  (gA : array2 et_ab (l2_row_major (SZ.v rows) (SZ.v shared)) { is_global gA })
  (gB : array2 et_ab (l2_row_major (SZ.v shared) (SZ.v cols)) { is_global gB })
  (gC : array2 et_cd (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gC })
  (gD : array2 et_cd (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gD })
  (s : stream_t)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned 16 (core gD)))
  (#_ : squash (chunk et_cd /?+ cols))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v shared) (SZ.v cols))
  (#eC : chest2 et_cd (SZ.v rows) (SZ.v cols))
  (#fA #fB #fC : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC) **
    on gpu_loc (live gD)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) ** (gC |-> Frac fC eC) **
        (exists* eD'. (gD |-> eD') **
          pure (eD' %~ MS.mmcomb comb_r
                  (to_real_matrix eC) (to_real_matrix eA) (to_real_matrix eB)))))

#push-options "--split_queries always"
inline_for_extraction noextract
fn tc2d_async
  (et_ab et_acc et_c : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_c, real_like et_c |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_size)) <= max_threads))
  (comb : et_c -> et_acc -> et_c)
  (comb_r : binop real { approx2 comb comb_r })
  (rows shared cols : szp)
  (gA : array2 et_ab (l2_row_major (SZ.v rows) (SZ.v shared)) { is_global gA })
  (gB : array2 et_ab (l2_row_major (SZ.v shared) (SZ.v cols)) { is_global gB })
  (gC : array2 et_c (l2_row_major (SZ.v rows) (SZ.v cols)) { is_global gC })
  (s : stream_t)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#eA : chest2 et_ab (SZ.v rows) (SZ.v shared))
  (#eB : chest2 et_ab (SZ.v shared) (SZ.v cols))
  (#eC : chest2 et_c (SZ.v rows) (SZ.v cols))
  (#fA #fB : perm)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> eC)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc ((gA |-> Frac fA eA) ** (gB |-> Frac fB eB) **
        (exists* eC'. (gC |-> eC') **
          pure (eC' %~ MS.gmmcomb (fun (x:real) -> x) (fun (x:real) -> x) comb_r
            (to_real_matrix eC) (to_real_matrix eA) (to_real_matrix eB)))))
#pop-options