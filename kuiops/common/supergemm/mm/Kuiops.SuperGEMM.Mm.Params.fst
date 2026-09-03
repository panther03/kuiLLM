module Kuiops.SuperGEMM.Mm.Params

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }

module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

#set-options "--z3rlimit 15 --fuel 1 --ifuel 1"

let ldt_sz (bk skew : szp) (#_ : squash (SZ.fits (ldt bk skew)))
  : (x : szp { SZ.v x == ldt bk skew })
  = bk +^ skew

let abuf_sz (rows bk skew : szp) (#_ : squash (SZ.fits (SZ.v rows * ldt bk skew)))
  : (x : szp { SZ.v x == SZ.v rows * ldt bk skew })
  = ML.lemma_mult_le_right (ldt bk skew) 1 (SZ.v rows);
    rows *^ (bk +^ skew)

let warps_m_pos (bm wm : szp)
  : Lemma (requires wm /?+ SZ.v bm) (ensures warps_m bm wm > 0)
= ()

let scratch_sz
  (et_acc : Type0) {| sized et_acc, has_vec_cpy et_acc |}
  (bm bn wm wn : szp)
  (#_ : squash (wm /?+ SZ.v bm /\ wn /?+ SZ.v bn))
  (#_ : squash (SZ.fits (warps bm bn wm wn * frag * lde et_acc wn)))
  : (x : szp { SZ.v x == warps bm bn wm wn * frag * lde et_acc wn })
  = warps_m_pos bm wm;
    warps_m_pos bn wn;
    ML.lemma_mult_le_right (frag * lde et_acc wn) 1 (warps bm bn wm wn);
    ML.lemma_mult_le_right (lde et_acc wn) 1 (warps bm bn wm wn * frag);
    ML.lemma_mult_le_right (lde et_acc wn) 1 (warps bm bn wm wn * SZ.v frag_sz);
    let wa = (bm /^ wm) *^ (bn /^ wn) in
    wa *^ frag_sz *^ (wn +^ chunk et_acc)

let nthr_sz bm bn wm wn
  (#_ : squash (wm /?+ SZ.v bm /\ wn /?+ SZ.v bn))
  (#_ : squash (SZ.fits (nthr bm bn wm wn)))
  : (x : szp { SZ.v x == nthr bm bn wm wn })
  = warps_m_pos bm wm;
    warps_m_pos bn wn;
    ML.lemma_mult_le_right (SZ.v warp_size) 1 (warps bm bn wm wn);
    let wa = (bm /^ wm) *^ (bn /^ wn) in
    wa *^ warp_size

let ldt_pos (bk skew : szp) = ()

let chunk_divides_ldt et_ab et_acc bm bn bk wm wn skew =
  ML.modulo_distributivity (SZ.v bk) (SZ.v skew) (SZ.v (chunk et_ab))

let nthr_pos bm bn wm wn =
  warps_m_pos bm wm;
  warps_m_pos bn wn

let nthr_le_max_threads et_ab et_acc bm bn bk wm wn skew = ()

(* Keep this nonlinear comparison out of the much larger epilogue context. *)
let fits_padded_row
  (w rows cols pad extra : nat)
  : Lemma
      (requires w > 0 /\ extra <= rows * pad /\
                SZ.fits (w * rows * (cols + pad)))
      (ensures SZ.fits (rows * cols + extra))
= ()

let epilogue_band_fits et_ab et_acc bm bn bk wm wn skew =
  warps_m_pos bm wm;
  warps_m_pos bn wn;
  fits_padded_row
    (warps bm bn wm wn) frag (SZ.v wn) (eskew et_acc)
    (SZ.v warp_size)

let warp_divides_nthr bm bn wm wn =
  ML.cancel_mul_mod (warps bm bn wm wn) (SZ.v warp_size)

let bm_ldt_fits et_ab et_acc bm bn bk wm wn skew = ()

let chunk_nthr_divides_ab et_ab et_acc bm bn bk wm wn skew = ()
