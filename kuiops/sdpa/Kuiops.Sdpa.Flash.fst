module Kuiops.Sdpa.Flash

(* Memory-safety-only Kuiper port of the online-softmax update block of the
   bf16 tensor-core flash-attention kernel (etc/tc_flash_attn_fa1.cu, lines
   151-186).  It corresponds to the [if (lane < BM)] body executed by a single
   lane [i]: given the raw QK^T scores for this key tile (row [i] of [Ssh]), it

     1. applies the softmax scale and additive mask, masking-out invalid keys
        to the [-inf] sentinel (score loop);
     2. updates the running max [m] and the correction factor [corr];
     3. turns the scores into probabilities (row [i] of [Psh]) with the
        select-to-zero rule, accumulating the row sum;
     4. updates the running denominator [l], the correction [cw] and [m].

   There is NO functional spec here (exactly like [flashattention_tile] in
   Kuiper.Kernel.FlashAttention): we only verify that every array/ref access is
   in bounds.  The interesting obligation is the mask read, which is only in
   bounds when the key index [kj] is valid; it is guarded by [kj <^ sk]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Floating
open Kuiper.Shape

module SZ = Kuiper.SizeT

(* Clamp a key index into [0, sk) so the mask read is unconditionally in bounds
   (a pure, F*-level [if] refines the result type). *)
inline_for_extraction noextract
let clamp_lt (sk : szp) (x : sz) : szlt sk =
  if x <^ sk then x else 0sz

(* Select-to-zero probability (line 180): masked score is the [-inf] sentinel and
   maps to the literal 0, never [exp(-inf)]. *)
inline_for_extraction noextract
let sel_prob (#et : Type0) {| floating et |} (sv mnew : et) : et =
  if eq sv (neg infinity) then zero else fexp (sv `sub` mnew)

(* One lane's online-softmax update.  Ownership at this point in the program:

   - [gSsh] : row [i] of the [BM x BN] score matrix [Ssh].  Read+write, full
     permission (the lane owns exactly its row -- the 2D subtile pattern).
   - [gPsh] : row [i] of the [BM x BN] probability matrix [Psh].  Write, full
     permission.
   - [gmask]: the whole additive-mask tensor, held read-only with a divided
     fraction (every lane reads it, so we never need to split the resource per
     lane -- a fraction over the whole array is enough).
   - [gm], [gl], [gcw] : the lane's cells of [Msh], [Lsh], [cw].  Read+write.

   [bi], [qh], [qpos] are this lane's fixed mask coordinates (loop-invariant
   across the key loop); only [kj] varies. *)
inline_for_extraction noextract
fn sdpa_flash_softmax_upd
  (#et : Type0) {| scalar et, floating et |}
  (bn : szp)
  (b hq sq sk : szp)
  (#lSsh #lPsh : layout1 bn)
  (#lmask : layout4 b hq sq sk)
  {| ctlayout lSsh, ctlayout lPsh, ctlayout lmask |}
  (gSsh : array1 et lSsh)
  (gPsh : array1 et lPsh)
  (gmask : array4 et lmask)
  (gm gl gcw : ref et)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq)
  (k0 : sz { SZ.fits (SZ.v k0 + SZ.v bn) })
  (cbound : sz)
  (row_active : bool)
  (causal : bool)
  (scale : et)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et)
  (#fmask : perm)
  (#vm #vl #vcw : erased et)
  requires
    gm |-> vm ** gl |-> vl ** gcw |-> vcw
  preserves
    (gmask |-> Frac fmask emask) ** live gSsh ** live gPsh
  ensures
    live gm ** live gl ** live gcw
{
  // Score loop: scale + mask, masking-out invalid keys to the -inf sentinel.
  let mut rowmax : et = neg infinity;
  let mut j : szle bn = 0sz;
  while (!j <^ bn)
    invariant
      (gmask |-> Frac fmask emask) ** live gSsh ** live gPsh **
      live j ** live rowmax
    decreases (bn - !j)
  {
    let vj = !j;
    let jj : szlt bn = vj;
    let kj = k0 +^ vj;
    let sc = tensor_read gSsh (cidx1 jj);
    // The only conditional memory access is the mask read; it is in bounds iff
    // [kj <^ sk].  Since we have no functional spec, we always read an in-bounds
    // mask cell (clamping the key index) and select the score purely: the mask
    // value is discarded whenever the key is invalid.
    let kjb : szlt sk = clamp_lt sk kj;
    let mv = tensor_read gmask (cidx4 bi qh qpos kjb);
    let s : et =
      if (row_active && (not (causal && (kj >^ cbound))) && (kj <^ sk)) {
        (sc `mul` scale) `add` mv
      } else {
        neg #et infinity
      };
    gSsh.(cidx1 jj) <- s;
    rowmax := fmax !rowmax s;
    j := !j +^ 1sz;
  };

  // Online-softmax max/correction update (uses the OLD m still in gm).
  let m_old = !gm;
  let mnew = fmax m_old !rowmax;
  let corr0 = fexp (m_old `sub` mnew);
  // TODO(line 173): clamp [corr] to 0 when it is not finite.  Kuiper only has a
  // GHOST finiteness test ([is_finite]/[kind] returns the erasable [fkind]), so
  // this guard cannot drive concrete control flow.  Needs an extractable
  // [isfinite] on the [floating] typeclass; omitted for now (does not affect
  // memory safety, and there is no functional spec here).
  let corr : et = corr0;

  // Probability loop: select-to-zero probabilities + row sum.
  let mut rowsum : et = zero;
  j := 0sz;
  while (!j <^ bn)
    invariant
      (gmask |-> Frac fmask emask) ** live gSsh ** live gPsh **
      live j ** live rowsum
    decreases (bn - !j)
  {
    let vj = !j;
    let jj : szlt bn = vj;
    let sv = tensor_read gSsh (cidx1 jj);
    let p : et = sel_prob sv mnew;
    gPsh.(cidx1 jj) <- p;
    rowsum := !rowsum `add` p;
    j := !j +^ 1sz;
  };

  // Commit the running denominator, correction and max.
  let l_old = !gl;
  gl := (l_old `mul` corr) `add` !rowsum;
  gcw := corr;
  gm := mnew;
  ()
}
