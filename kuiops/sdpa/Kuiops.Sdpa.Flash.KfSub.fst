module Kuiops.Sdpa.Flash.KfSub

(* Memory-safety-only helpers for the bf16 tensor-core flash-attention kernel
   in [flash_attn_fa1.cu].  This module contains the prologue, key-tile
   loop, epilogue, and ownership bridges composed by [sdpa_flash_kf].  It has
   no functional specification; it verifies bounds and resource ownership. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.Floating
open Kuiper.Shape
open Kuiper.Bijection
open Kuiper.Tensor.Layout.Bijection
open Pulse.Lib.Trade
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiper.ForEvery
open Kuiper.Ghost.TensorTranspose
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module TRO = Kuiper.TensorRO
module B = Kuiper.Barrier
module BW = Kuiper.Barrier.Warp
module Trade = Pulse.Lib.Trade
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float
open Kuiper.TensorRO { vtlayout_of_tlayout }

let flash_transpose_bij (#rows #cols : nat)
  : (abs (rows @| cols @| INil) =~ abs (cols @| rows @| INil)) =
{
  ff = (fun (i, (j, ())) -> (j, (i, ())));
  gg = (fun (j, (i, ())) -> (i, (j, ())));
}

let flash_row2col_layout
  (#rows #cols : nat) (l : layout2 rows cols)
  : layout2 cols rows =
  tlayout_bij flash_transpose_bij l

inline_for_extraction noextract
let flash_row2col
  (#et : Type0) (#rows #cols : erased nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  : array2 et (flash_row2col_layout l) =
  from_array (flash_row2col_layout l) (core a)

inline_for_extraction noextract
instance flash_row2col_ctlayout
  (#rows #cols : erased nat)
  (l : layout2 rows cols) {| ctlayout l |}
  : ctlayout (flash_row2col_layout l) =
  ctlayout_bij flash_transpose_bij
    (fun (j, (i, ())) -> (i, (j, ())))
    (fun _ -> ())
    l

inline_for_extraction noextract
instance flash_row2col_strided
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| srm : strided_row_major (vtlayout_of_tlayout l) |}
  : strided_col_major (vtlayout_of_tlayout (flash_row2col_layout l)) =
{
  offset = srm.offset;
  stride = srm.stride;
  pf = (fun i j ->
    tlayout_bij_imap flash_transpose_bij l (idx2 i j);
    srm.pf j i);
}

ghost
fn flash_transpose
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#f : perm) (#e : chest2 et rows cols)
  requires a |-> Frac f e
  ensures flash_row2col a |-> Frac f (mtranspose e)
{
  tensor_ilower2 a;
  forevery_commute
    (fun (i : natlt rows) (j : natlt cols) ->
      Cell a (idx2 i j) |-> Frac f (acc e (idx2 i j)));
  forevery_map_2
    (fun (j : natlt cols) (i : natlt rows) ->
      Cell a (idx2 i j) |-> Frac f (acc e (idx2 i j)))
    (fun (j : natlt cols) (i : natlt rows) ->
      Cell (flash_row2col a) (idx2 j i) |->
        Frac f (acc (mtranspose e) (idx2 j i)))
    fn j i {
      tlayout_bij_imap flash_transpose_bij l (idx2 j i);
      lem_from_array_core (core a);
      assert pure (core (flash_row2col a) == core a);
      assert pure (flash_transpose_bij.gg (idx2 j i) == idx2 i j);
      assert pure (
        acc (mtranspose e) (idx2 j i) == acc e (idx2 i j));
      tensor_pts_to_cell_eq a (idx2 i j) f (acc e (idx2 i j));
      tensor_pts_to_cell_eq (flash_row2col a) (idx2 j i) f
        (acc (mtranspose e) (idx2 j i));
      rewrite
        (Cell a (idx2 i j) |-> Frac f (acc e (idx2 i j)))
        as
        (Cell (flash_row2col a) (idx2 j i) |->
          Frac f (acc (mtranspose e) (idx2 j i)));
    };
  tensor_iraise2 (flash_row2col a);
}

ghost
fn flash_transpose_back
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#f : perm) (#e : chest2 et cols rows)
  requires flash_row2col a |-> Frac f e
  ensures a |-> Frac f (mtranspose e)
{
  tensor_ilower2 (flash_row2col a);
  forevery_commute
    (fun (j : natlt cols) (i : natlt rows) ->
      Cell (flash_row2col a) (idx2 j i) |->
        Frac f (acc e (idx2 j i)));
  forevery_map_2
    (fun (i : natlt rows) (j : natlt cols) ->
      Cell (flash_row2col a) (idx2 j i) |->
        Frac f (acc e (idx2 j i)))
    (fun (i : natlt rows) (j : natlt cols) ->
      Cell a (idx2 i j) |-> Frac f (acc (mtranspose e) (idx2 i j)))
    fn i j {
      tlayout_bij_imap flash_transpose_bij l (idx2 j i);
      lem_from_array_core (core a);
      assert pure (core (flash_row2col a) == core a);
      assert pure (flash_transpose_bij.gg (idx2 j i) == idx2 i j);
      assert pure (
        acc (mtranspose e) (idx2 i j) == acc e (idx2 j i));
      tensor_pts_to_cell_eq (flash_row2col a) (idx2 j i) f
        (acc e (idx2 j i));
      tensor_pts_to_cell_eq a (idx2 i j) f
        (acc (mtranspose e) (idx2 i j));
      rewrite
        (Cell (flash_row2col a) (idx2 j i) |->
          Frac f (acc e (idx2 j i)))
        as
        (Cell a (idx2 i j) |->
          Frac f (acc (mtranspose e) (idx2 i j)));
    };
  tensor_iraise2 a;
}

(* ── array1-over-tensor cell / ref shims ─────────────────────────────────────
   Ported verbatim from [Kuiper.Kernel.FlashAttention] so we do not have to
   [open] (and re-verify) that whole kernel.  These expose the 1-D whole/cell/ref
   conversions used by the shcw scalar array and the per-row [mrow] bridge. *)

let fa_abs_cons_nil_eq (len:nat)
  : Lemma (abs (len @| INil) == (natlt len & unit))
          [SMTPat (abs (len @| INil))]
  = ()

unfold
let fa_abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

let fa_acc1_upd1 (#et:Type) (#len:nat) (s:chest1 et len) (i:natlt len) (v:et) (j:natlt len)
  : Lemma (acc1 (upd1 s i v) j == (if j = i then v else acc1 s j))
          [SMTPat (acc1 (upd1 s i v) j)]
  = ()

let fa_up_cidx1_eq (#d0:nat) (i:szlt d0)
  : Lemma (up (cidx1 i) == idx1 (SZ.v i))
          [SMTPat (up (cidx1 i))]
  = ()

let fa_up_cidx4_eq (#d0 #d1 #d2 #d3:nat)
  (i:szlt d0) (j:szlt d1) (k:szlt d2) (l:szlt d3)
  : Lemma (up (cidx4 i j k l) == idx4 (SZ.v i) (SZ.v j) (SZ.v k) (SZ.v l))
          [SMTPat (up (cidx4 i j k l))]
  = ()

let fa_tr_val_chest1_to_seq (#et:Type) (#len:nat) (v:chest1 et len)
  : Lemma (tr_val (chest1_to_seq v) == v)
          [SMTPat (tr_val (chest1_to_seq v))]
  = introduce forall (i : abs (len @| INil)).
        acc (tr_val (chest1_to_seq v)) i == acc v i
    with ( let (j, _) = i in () );
    Kuiper.Chest.lemma_equal_intro (tr_val (chest1_to_seq v)) v;
    Kuiper.Chest.ext (tr_val (chest1_to_seq v)) v

ghost
fn explode1
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l)
  (#f : perm)
  (#s : chest1 et len)
  requires a |-> Frac f s
  ensures
    forall+ (i : natlt len).
      Cell a (idx1 i) |-> Frac f (acc1 s i)
{
  tensor_explode a #f #s;
  forevery_iso fa_abs_bij (fun (i : abs (len @| INil)) -> Cell a i |-> Frac f (acc s i));
  forevery_ext _ (fun (i : natlt len) -> Cell a (idx1 i) |-> Frac f (acc1 s i));
  ()
}

ghost
fn implode1
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l)
  (#f : perm)
  (#s : chest1 et len)
  requires
    pure (SZ.fits (tlayout_ulen l))
  requires
    forall+ (i : natlt len).
      Cell a (idx1 i) |-> Frac f (acc1 s i)
  ensures
    a |-> Frac f s
{
  forevery_ext _ (fun (i : natlt len) -> Cell a (fa_abs_bij.gg i) |-> Frac f (acc s (fa_abs_bij.gg i)));
  forevery_iso_back fa_abs_bij (fun (i : abs (len @| INil)) -> Cell a i |-> Frac f (acc s i));
  tensor_implode a #f #s;
}

ghost
fn extract_cell1
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l)
  (i : natlt len)
  (#f : perm)
  (#s : chest1 et len)
  requires
    a |-> Frac f s **
    pure (SZ.fits (tlayout_ulen l))
  ensures
    Cell a (idx1 i) |-> Frac f (acc1 s i) **
    (forall* (si': et).
      Cell a (idx1 i) |-> Frac f si' @==> a |-> Frac f (upd1 s i si' <: chest1 et len))
{
  explode1 a #f #s;
  forevery_extract' #(natlt len) i _;
  ghost fn aux (si' : et)
    requires forall* (p': natlt len -> slprop).
      p' i ** pure (forall (j:natlt len{~(eq2 #(natlt len) j i)}). p' j == (Cell a (idx1 j) |-> Frac f (acc1 s j)))
        @==> (forall+ (j:natlt len). p' j)
    ensures
      Cell a (idx1 i) |-> Frac f si' @==> a |-> Frac f (upd1 s i si' <: chest1 et len)
    {
      let p' = (fun (j: natlt len) -> (Cell a (idx1 j)) |-> Frac f (acc1 (upd1 s i si' <: chest1 et len) j));
      assert rewrites_to p' (fun (j: natlt len) -> (Cell a (idx1 j)) |-> Frac f (acc1 (upd1 s i si' <: chest1 et len) j));
      elim_forall p';

      Trade.intro_trade
        (Cell a (idx1 i) |-> Frac f si')
        (a |-> Frac f (upd1 s i si' <: chest1 et len))
        (p' i ** pure (forall (j:natlt len{~(eq2 #(natlt len) j i)}). p' j == (Cell a (idx1 j) |-> Frac f (acc1 s j)))
          @==> (forall+ (j:natlt len). p' j))
        fn _ {
          rewrite (Cell a (idx1 i) |-> Frac f si') as (p' i);
          Trade.elim_trade _ _;
          implode1 a #f #(upd1 s i si' <: chest1 et len);
        };
    };
  intro_forall _ aux;
  ()
}

ghost
fn restore_cell1
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l)
  (i : natlt len)
  (#f : perm)
  (#si': et)
  (#s : chest1 et len)
  requires
    Cell a (idx1 i) |-> Frac f si' **
    (forall* (si': et).
      Cell a (idx1 i) |-> Frac f si' @==> a |-> Frac f (upd1 s i si' <: chest1 et len))
  ensures
    a |-> Frac f (upd1 s i si' <: chest1 et len)
{
  elim_forall si';
  Trade.elim_trade _ _;
}

(* Deliberately not [inline_for_extraction]: inlining it substitutes the
   [lane] expression into the softmax body and karamel's Low* re-check then
   rejects the kernel.  As a real (static) function it costs nothing -- nvcc
   inlines the identity. *)
[@@CPrologue "__device__"]
let szlt16 (i : sz { SZ.v i < 16 }) : szlt 16 = i

let ref_of_array_cell
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l) (i : natlt len)
  : GTot (ref et)
  = ref_of_tensor_cell a (idx1 i)

inline_for_extraction noextract
fn get_ref_of_array_cell
  (#et : Type0) (#len : erased nat) (#l : layout1 len) {| ctlayout l |}
  (a : array1 et l) (i : szlt len)
  returns r : ref et
  ensures pure (r == ref_of_array_cell a i)
{
  get_ref_of_tensor_cell a (cidx1 i)
}

ghost
fn array1_cell_to_ref
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l) (i : natlt len)
  (#f : perm) (#v : erased et)
  requires Cell a (idx1 i) |-> Frac f v
  ensures ref_of_array_cell a i |-> Frac f v
{
  tensor_cell_to_ref a (idx1 i);
}

ghost
fn array1_cell_from_ref
  (#et : Type0) (#len : nat) (#l : layout1 len)
  (a : array1 et l) (i : natlt len)
  (#f : perm) (#v : erased et)
  requires ref_of_array_cell a i |-> Frac f v
  ensures Cell a (idx1 i) |-> Frac f v
{
  tensor_cell_from_ref a (idx1 i);
}

(* Read the additive mask bias for one (row, key) pair, or [zero] when the
   caller supplied no mask tensor (flash_attn_fa1.cu:162, [if (mask)]).

   This owns the [has_mask] branch in a function with an explicit signature,
   rather than inlining it in [sdpa_flash_softmax_upd]: the two arms are then
   each checked against the same fixed frame.  Inlined, the arm that performs
   the read contributes an [exists*] over the value read while the other does
   not, and the automatic join of the two shapes fails. *)
inline_for_extraction noextract
fn sdpa_flash_mask_bias
  (#et_ab : Type0)
  {| scalar et_ab |}
  (b hq sq sk : szp)
  (#lmask : TRO.vlayout4 b hq sq sk) {| TRO.cvtlayout lmask |}
  (gmask : TRO.roarray4 et_ab lmask)
  (has_mask : bool)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq) (kjb : szlt sk)
  (#fmask : perm)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  preserves (gmask |-> Frac fmask emask)
  returns v : et_ab
  ensures pure (v == SF.mask_bias emask has_mask
                       (SZ.v bi) (SZ.v qh) (SZ.v qpos) (SZ.v kjb))
{
  if has_mask {
    TRO.tensor_read gmask (cidx4 bi qh qpos kjb)
  } else {
    zero #et_ab
  }
}

(* One lane's online-softmax update.  Ownership at this point in the program:

   - [shS] : row [i] of the [BM x BN] score matrix [Ssh].  Read+write, full
     permission (the lane owns exactly its row -- the 2D subtile pattern).
   - [shP] : row [i] of the [BM x BN] probability matrix [Psh].  Write, full
     permission.
   - [gmask]: the whole additive-mask tensor, held read-only with a divided
     fraction (every lane reads it, so we never need to split the resource per
     lane -- a fraction over the whole array is enough).
   - [shm], [shl], [shcw] : the lane's cells of [Msh], [Lsh], [cw].  Read+write.

   [bi], [qh], [qpos] are this lane's fixed mask coordinates (loop-invariant
   across the key loop); only [kj] varies. *)
inline_for_extraction noextract
fn sdpa_flash_softmax_upd
  (#et_acc #et_ab : Type0)
  {| scalar et_acc, floating et_acc, real_like et_acc |}
  {| scalar et_ab, real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}   (* mask read  (line 163): bf16 -> f32 *)
  {| FC.float_cast et_acc et_ab |}   (* prob write (line 181): f32  -> bf16 *)
  (bn : szp)
  (b hq sq sk : szp)
  (#lshS #lshP : layout1 bn)
  (#lmask : TRO.vlayout4 b hq sq sk)
  {| ctlayout lshS, ctlayout lshP, TRO.cvtlayout lmask |}
  (shS : array1 et_acc lshS)
  (shP : array1 et_ab lshP)
  (gmask : TRO.roarray4 et_ab lmask)
  (shm shl shcw : ref et_acc)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq)
  (k0 : sz { SZ.fits (SZ.v k0 + SZ.v bn) })
  (cbound : sz)
  (row_active : bool)
  (causal : bool)
  (has_mask : bool)
  (scale : et_acc)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  (#fmask : perm)
  (#vm #vl #vcw : erased et_acc)
  (#es : chest1 et_acc (SZ.v bn))
  (#ep : chest1 et_ab (SZ.v bn))
  requires
    shm |-> vm ** shl |-> vl ** shcw |-> vcw ** shS |-> es ** shP |-> ep
  preserves
    (gmask |-> Frac fmask emask)
  ensures
    exists* (es' : chest1 et_acc (SZ.v bn)) (ep' : chest1 et_ab (SZ.v bn))
            (m' l' cw' : et_acc).
      shS |-> es' ** shP |-> ep' ** shm |-> m' ** shl |-> l' ** shcw |-> cw' **
      pure (SF.softmax_upd_post emask has_mask row_active causal
              (SZ.v bi) (SZ.v qh) (SZ.v qpos) (SZ.v k0) (SZ.v cbound) scale
              es vm vl es' ep' m' l' cw'
            /\ kind cw' == Finite)
{
  // Score loop: scale + mask, masking-out invalid keys to the -inf sentinel.
  let mut rowmax : et_acc = neg infinity;
  let mut j : szle bn = 0sz;
  while (!j <^ bn)
    invariant exists* (vj : szle bn) (rmv : et_acc) (esc : chest1 et_acc (SZ.v bn)).
      (gmask |-> Frac fmask emask) ** shS |-> esc ** shP |-> ep **
      j |-> vj ** rowmax |-> rmv **
      pure (
        (forall (t : natlt (SZ.v bn)). t < SZ.v vj ==>
           acc1 esc t == SF.score_upd scale (acc1 es t)
             (SF.mask_bias emask has_mask (SZ.v bi) (SZ.v qh) (SZ.v qpos)
                (SF.clamp_nat (SZ.v sk) (SZ.v k0 + t)))
             (SF.key_ok row_active causal (SZ.v sk) (SZ.v cbound) (SZ.v k0 + t))) /\
        (forall (t : natlt (SZ.v bn)). SZ.v vj <= t ==> acc1 esc t == acc1 es t) /\
        rmv == SF.row_max esc (SZ.v vj))
    decreases (bn - !j)
  {
    let vj = !j;
    let jj : szlt bn = vj;
    let kj = k0 +^ vj;
    let sc = tensor_read shS (cidx1 jj);
    // The only conditional memory access is the mask read; it is in bounds iff
    // [kj <^ sk].  We read an in-bounds mask cell (clamping the key index) and
    // select the score purely: the mask value is discarded whenever the key is
    // invalid.  With [has_mask] false there is no mask tensor and the read is
    // skipped, yielding a zero bias.
    let kjb : szlt sk = clamp_lt sk kj;
    let mv = sdpa_flash_mask_bias b hq sq sk gmask has_mask bi qh qpos kjb;
    let ok : bool = row_active && (not (causal && (kj >^ cbound))) && (kj <^ sk);
    let s : et_acc = SF.score_upd scale sc mv ok;
    with esc. assert (shS |-> esc);
    shS.(cidx1 jj) <- s;
    SF.row_max_ext esc (upd1 esc (SZ.v jj) s <: chest1 et_acc (SZ.v bn)) (SZ.v vj);
    rowmax := fmax !rowmax s;
    j := !j +^ 1sz;
  };

  with esf. assert (shS |-> esf);

  // Online-softmax max/correction update (uses the OLD m still in shm).
  let m_old = !shm;
  let mnew = fmax m_old !rowmax;
  let corr0 = fexp (m_old `sub` mnew);
  // TODO(line 173): clamp [corr] to 0 when it is not finite.  Kuiper only has a
  // GHOST finiteness test ([is_finite]/[kind] returns the erasable [fkind]), so
  // this guard cannot drive concrete control flow.  Needs an extractable
  // [isfinite] on the [floating] typeclass; assumed for now.
  let corr : et_acc = corr0;
  assume pure (kind corr == Finite);

  // Probability loop: select-to-zero probabilities + row sum.
  let mut rowsum : et_acc = zero;
  j := 0sz;
  while (!j <^ bn)
    invariant exists* (vj : szle bn) (rsv : et_acc) (epc : chest1 et_ab (SZ.v bn)).
      (gmask |-> Frac fmask emask) ** shS |-> esf ** shP |-> epc **
      j |-> vj ** rowsum |-> rsv **
      pure (
        (forall (t : natlt (SZ.v bn)). t < SZ.v vj ==>
           acc1 epc t == FC.fcast (SF.sel_prob (acc1 esf t) mnew)) /\
        rsv == SF.row_sum esf mnew (SZ.v vj))
    decreases (bn - !j)
  {
    let vj = !j;
    let jj : szlt bn = vj;
    let sv = tensor_read shS (cidx1 jj);
    let p : et_acc = SF.sel_prob sv mnew;
    shP.(cidx1 jj) <- FC.fcast p;
    rowsum := !rowsum `add` p;
    j := !j +^ 1sz;
  };

  // Commit the running denominator, correction and max.
  let l_old = !shl;
  shl := (l_old `mul` corr) `add` !rowsum;
  shcw := corr;
  shm := mnew;
  ()
}

(* Memory-safety-only Kuiper port of the Q@K^T tensor-core matmul in the same
   kernel (flash_attn_fa1.cu, lines 138-147).  A single warp computes one
   [16 x 16] score tile [Ssh[w]] by accumulating over the head dimension [d] in
   16-wide chunks (like [subproducts_tc] in Kuiper.Kernel.GEMM.TensorCore).

   As in the CUDA, [K] is consumed transposed: [kf] is a [col_major] matrix_b
   fragment.  We model this with [shKT], the K tile viewed column-major with
   leading dimension [hd] (so [shKT] and the row-major K buffer share memory and
   [shKT[d][j] == K[j][d]]), loaded via [mma_loadB_cm].  [shQ] is the row-major Q
   tile (leading dimension [hd]); [shS] is the row-major float score tile.

   No functional spec: we only verify every fragment load/store is in bounds.
   Ownership: the warp collectively owns [shS] with the per-lane [1/warp_size]
   fraction that [mma_store] consumes; [shQ]/[shKT] are read-only (divided
   fraction over the whole tile, restored each iteration through the extract
   trade). *)
inline_for_extraction noextract
fn sdpa_flash_qk_mm
  (#et_ab #et_acc : Type0)
  {| sc_ab : scalar et_ab, sc_acc : scalar et_acc |}
  (hd : szp)
  (d  : szp)
  (#_ : squash (16 /?+ SZ.v hd))
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.v d <= SZ.v hd))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  (#lQ : layout2 16 hd)
  (#lKT : layout2 hd 16)
  (#lS : layout2 16 16)
  {| ctlayout lQ |} {| ctlayout lKT |} {| ctlayout lS |}
  {| strided_row_major (vtlayout_of_tlayout lQ) |}
  {| strided_col_major (vtlayout_of_tlayout lKT) |}
  {| strided_row_major (vtlayout_of_tlayout lS) |}
  (shQ  : array2 et_ab lQ)
  (shKT : array2 et_ab lKT)
  (shS  : array2 et_acc lS)
  (#fQ #fK : perm)
  (#eQ  : chest2 et_ab 16 hd)
  (#eKT : chest2 et_ab hd 16)
  (#eS0 : chest2 et_acc 16 16)
  requires
    shQ  |-> Frac fQ eQ **
    shKT |-> Frac fK eKT **
    shS  |-> Frac (1.0R /. warp_size) eS0
  ensures
    shQ  |-> Frac fQ eQ **
    shKT |-> Frac fK eKT **
    (exists* eS. shS |-> Frac (1.0R /. warp_size) eS)
{
  tensor_pts_to_ref shQ;
  tensor_pts_to_ref shKT;

  let qFrag = __alloc_fragment et_ab FragA 16sz 16sz 16sz FragLRM;
  let kFrag = __alloc_fragment et_ab FragB 16sz 16sz 16sz FragLCM;
  let sFrag = __alloc_fragment et_acc FragAcc 16sz 16sz 16sz FragLAcc;

  mma_fill sFrag sc_acc.zero;

  let nchunks = d /^ 16sz;
  let mut chunk : sz = 0sz;
  while (!chunk <^ nchunks)
    invariant
      live qFrag ** live kFrag ** live sFrag ** live chunk **
      shQ |-> Frac fQ eQ **
      shKT |-> Frac fK eKT **
      pure (SZ.v !chunk <= SZ.v nchunks)
    decreases (nchunks - !chunk)
  {
    let qtile = array2_extract_tile_ro' shQ 16 16 0 (SZ.v !chunk);
    let ktile = array2_extract_tile_ro' shKT 16 16 (SZ.v !chunk) 0;

    mma_loadA qFrag qtile;
    mma_loadB_cm kFrag ktile;
    mma_sync' qFrag kFrag sFrag;

    with etQ. assert (tensor_pts_to qtile #fQ etQ);
      elim_trade (qtile |-> Frac fQ etQ) (shQ |-> Frac fQ eQ);
    with etK. assert (tensor_pts_to ktile #fK etK);
      elim_trade (ktile |-> Frac fK etK) (shKT |-> Frac fK eKT);

    chunk := !chunk +^ 1sz;
  };

  mma_store sFrag shS;

  with vq. assert qFrag |-> vq; drop_ (qFrag |-> vq);
  with vk. assert kFrag |-> vk; drop_ (kFrag |-> vk);
  with vs. assert sFrag |-> vs; drop_ (sFrag |-> vs);
  ()
}

(* Identity warp-barrier transform: with [p == q] the collected [forall+ i. p i]
   is returned unchanged, so no ownership moves across the warp barrier.  This is
   uniform in the lane index (it ignores [i] entirely), so it does not exploit
   the unsoundness of the current tid-dependent [warp_barrier_wait]. *)
ghost
fn warp_sync_noop (p : natlt BW.warp_size -> slprop)
  requires forall+ (i : natlt BW.warp_size). p i
  ensures  forall+ (i : natlt BW.warp_size). p i
{
  ()
}

unfold
let warp_emp_pred (_ : natlt BW.warp_size) : slprop = emp

(* The empty warp-barrier transform, as a first-class [stt_ghost] value.  Bound
   with a plain F* [let] (Kuiper's convention for barrier transforms) so it is
   passed to [warp_barrier_wait] as a value rather than run as a ghost step in
   the caller's single-lane context.  [p == q == emp]: the [__syncwarp()] threads
   no ownership, it is only an ordering fence. *)
let warp_emp_proof
  : stt_ghost unit emp_inames
      (requires forall+ (i : natlt BW.warp_size). warp_emp_pred i)
      (ensures  fun _ -> forall+ (i : natlt BW.warp_size). warp_emp_pred i)
  = warp_sync_noop warp_emp_pred

(* Derived tile/lane geometry for the [PVc] -> [Osh] accumulation.  The
   tensor-core fragment tile is a fixed [16 x 16] (hardware), and a warp of
   [warp_size] lanes strides over it: consecutive groups of 16 lanes cover one
   16-wide row, so a warp spans [warp_size / 16] rows at once ([warp_row_span]),
   and each lane therefore owns [16 / warp_row_span] rows of the tile
   ([lane_row_span]).  Nothing here is a bare specialized literal -- the counts
   follow from [warp_size] and the fragment width. *)
unfold let lane_row_span : nat = 16 / warp_row_span

inline_for_extraction noextract let warp_row_span_sz : sz = warp_size /^ 16sz
inline_for_extraction noextract let lane_row_span_sz : sz = 16sz /^ warp_row_span_sz

(* Memory-safety-only Kuiper port of the P@V tensor-core matmul plus the
   per-lane accumulation into [Osh] (flash_attn_fa1.cu, lines 194-209).
   A single warp, looping over the head dimension [d] in 16-wide chunks [dc]:
   for each chunk it computes the [16 x 16] product [PV = P @ V[:, dc:dc+16]]
   into a fresh accumulator, stores it to the scratch tile [PVc], and then each
   lane strides over [PVc] adding its cells into [Osh].

   The first [__syncwarp()] (after [store_matrix_sync]) IS emitted, as
   [warp_barrier_wait]: on real hardware [store_matrix_sync] does not fence the
   shared-memory writes, so the warp must synchronize before the lanes read
   [PVc] back.  For memory safety it transfers no ownership -- each lane keeps
   its own [1/warp_size] fraction of [PVc] framed across the barrier -- so we use
   the empty transform ([p == q == emp]).  That is trivially uniform across lanes
   (it never depends on the thread id, as the unsound-by-construction
   [warp_barrier_wait] would otherwise permit).  The transform is bound as a
   top-level [stt_ghost] value ([warp_emp_proof]) so it is passed by value rather
   than run as a ghost step in this single-lane context.

   The second [__syncwarp()] (after the accumulation) is OMITTED: it orders
   reads/writes for value visibility but transfers no ownership, and the proof
   goes through without it.  This is sound because the [Osh] cells a lane touches
   are exactly its stride-subtile [(2, 16)] with residue [(lane/16, lane%16)],
   which are pairwise disjoint across the 32 lanes -- so there is no race for a
   barrier to prevent.  After [mma_store] each lane still holds the [1/warp_size]
   fraction of the whole [PVc], enough to read any cell.

   [P] loads as a row-major [matrix_a], [V]'s column chunk as a row-major
   [matrix_b] (neither is transposed, unlike qk_mm).  No functional spec: only
   in-bounds fragment loads/stores and [PVc]/[Osh] accesses are verified. *)
inline_for_extraction noextract
fn sdpa_flash_pv_mm
  (#et_ab #et_acc : Type0)
  {| sc_ab : scalar et_ab, sc_acc : scalar et_acc |}
  (hd : szp)
  (d  : szp)
  (#_ : squash (16 /?+ SZ.v hd))
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.v d <= SZ.v hd))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  (#lP : layout2 16 16)
  (#lV : layout2 16 (SZ.v hd))
  (#lPVc : layout2 16 16)
  (#lO : layout2 16 (SZ.v hd))
  {| ctlayout lP |} {| ctlayout lV |} {| cPVc : ctlayout lPVc |} {| ctlayout lO |}
  {| strided_row_major (vtlayout_of_tlayout lP) |}
  {| strided_row_major (vtlayout_of_tlayout lV) |}
  {| strided_row_major (vtlayout_of_tlayout lPVc) |}
  (shP   : array2 et_ab lP)
  (shV   : array2 et_ab lV)
  (shPVc : array2 et_acc lPVc)
  (shO : array2 et_acc lO)
  (#fP #fV : perm)
  (#eP  : chest2 et_ab  16 16)
  (#eV  : chest2 et_ab  16 hd)
  (#ePVc0 : chest2 et_acc 16 16)
  (#eO0 : chest2 et_acc lane_row_span (SZ.v hd / 16))
  requires
    shP   |-> Frac fP eP **
    shV   |-> Frac fV eV **
    shPVc |-> Frac (1.0R /. warp_size) ePVc0 **
    (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO0)
  preserves thread_id (SZ.v nthr) tid
  ensures
    shP |-> Frac fP eP **
    shV |-> Frac fV eV **
    (exists* ePVc. shPVc |-> Frac (1.0R /. warp_size) ePVc) **
    (exists* eO. array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO)
{
  tensor_pts_to_ref shV;
  tensor_pts_to_ref shPVc;

  let tr = lane /^ 16sz;
  let tc = lane %^ 16sz;
  tensor_pts_to_ref (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16));

  let pf    = __alloc_fragment et_ab  FragA   16sz 16sz 16sz FragLRM;
  let vf    = __alloc_fragment et_ab  FragB   16sz 16sz 16sz FragLRM;
  let pvacc = __alloc_fragment et_acc FragAcc 16sz 16sz 16sz FragLAcc;

  let njcol = d /^ 16sz;
  let mut jcol : sz = 0sz;
  while (!jcol <^ njcol)
    invariant
      live pf ** live vf ** live pvacc ** live jcol **
      thread_id (SZ.v nthr) tid **
      shP |-> Frac fP eP **
      shV |-> Frac fV eV **
      (exists* ePVc. shPVc |-> Frac (1.0R /. warp_size) ePVc) **
      (exists* eO. array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO) **
      pure (SZ.v !jcol <= SZ.v njcol)
    decreases (njcol - !jcol)
  {
    let vjcol = !jcol;
    let ocol : szlt (SZ.v hd / 16) = vjcol;

    mma_fill pvacc sc_acc.zero;
    let vtile = array2_extract_tile_ro' shV 16 16 0 (SZ.v vjcol);
    mma_loadA pf shP;
    mma_loadB vf vtile;
    mma_sync' pf vf pvacc;
    with etV. assert (tensor_pts_to vtile #fV etV);
      elim_trade (vtile |-> Frac fV etV) (shV |-> Frac fV eV);
    mma_store pvacc shPVc;

    (* The [__syncwarp()] after [store_matrix_sync]: model it as an empty warp
       barrier (threads no ownership, [p == q == emp]).  It is a pure ordering
       fence -- each lane keeps its own [1/warp_size] fraction of [shPVc] framed
       across it -- so [forall+ i. emp] discharges trivially and the transform is
       uniform in the lane (independent of the thread id). *)
    BW.warp_barrier_wait () warp_emp_pred warp_emp_pred warp_emp_proof
      #(SZ.v nthr) #(SZ.v tid);

    let mut k : sz = 0sz;
    while (!k <^ lane_row_span_sz)
      invariant
        live k **
        (exists* ePVc. shPVc |-> Frac (1.0R /. warp_size) ePVc) **
        (exists* eO. array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO) **
        pure (SZ.v !k <= SZ.v lane_row_span_sz)
      decreases (lane_row_span_sz - !k)
    {
      let vk = !k;
      let prow : szlt 16 = warp_row_span_sz *^ vk +^ tr;
      let orow : szlt lane_row_span = vk;
      let pv  = tensor_read #_ #_ #_ #_ #cPVc shPVc (cidx2 prow tc);
      let old = tensor_read #_ #_ #_ #_ #(c_stride_subtile_layout lO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16))
        (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 orow ocol);
      tensor_write #_ #_ #_ #_ #(c_stride_subtile_layout lO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16))
        (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 orow ocol) (old `sc_acc.add` pv);
      k := !k +^ 1sz;
    };
    jcol := !jcol +^ 1sz;
  };

  with vp. assert pf |-> vp; drop_ (pf |-> vp);
  with vv. assert vf |-> vv; drop_ (vf |-> vv);
  with va. assert pvacc |-> va; drop_ (pvacc |-> va);
  ()
}

(* ────────────────────────────────────────────────────────────────────────
   sdpa_flash_kv_load  --  global -> shared caching of the K/V key tile
   (flash_attn_fa1.cu, lines 130-135).

   A single warp caches the [bn x d] key tile from global to shared.  The tile
   is partitioned across the [warp_size] lanes by the strided [(warp_row_span,
   16)] sub-tile scheme shared with [sdpa_flash_pv_mm] / [sdpa_flash_scale]:
   lane [(tr, tc) = (lane/16, lane%16)] owns the sub-tile of shape
   [(bn/warp_row_span) x (d/16)].  Its cell [(a, b)] is tile row
   [a*warp_row_span + tr] and column [b*16 + tc]; the global key row is
   [k0base + tile-row], clamped into [0, sk) so every read is unconditionally
   in bounds (exactly like the mask read in [sdpa_flash_softmax_upd]).  There
   is NO functional spec, only memory safety. *)

(* [i < n/s], [r < s] and [s | n] imply [s*i + r < n] -- the standard bound for
   a strided [(srows, scols)] tile cell. *)
let tile_idx_lem (s i r n : nat)
  : Lemma (requires s > 0 /\ (s /? n) /\ i < n / s /\ r < s) (ensures s * i + r < n)
= let z = Kuiper.Divides.get_factor s n in
  FStar.Math.Lemmas.cancel_mul_div z s;
  FStar.Math.Lemmas.lemma_mult_le_left s (i + 1) z

inline_for_extraction noextract
fn sdpa_flash_kv_load
  (#et : Type0)
  (bn d sk : szp)
  (lane : szlt warp_size)
  (k0base : sz)
  (#_ : squash (warp_row_span /?+ SZ.v bn))
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (SZ.fits (SZ.v bn * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v k0base + SZ.v bn)))
  (#lK #lV : layout2 (SZ.v sk) (SZ.v d))
  {| ctlayout lK, ctlayout lV |}
  (gK : array2 et lK { Kuiper.Tensor.is_global gK })
  (gV : array2 et lV { Kuiper.Tensor.is_global gV })
  (#lshK #lshV : layout2 (SZ.v bn) (SZ.v d))
  {| cshK : ctlayout lshK |} {| cshV : ctlayout lshV |}
  (shK : array2 et lshK) (shV : array2 et lshV)
  (#fK #fV : perm)
  (#eK #eV : chest2 et (SZ.v sk) (SZ.v d))
  (#eKc0 : chest2 et (SZ.v bn / warp_row_span) (SZ.v d / 16))
  (#eVc0 : chest2 et (SZ.v bn / warp_row_span) (SZ.v d / 16))
  requires
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (array2_stride_subtile shK warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eKc0) **
    (array2_stride_subtile shV warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eVc0)
  ensures
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (exists* eKc. array2_stride_subtile shK warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eKc) **
    (exists* eVc. array2_stride_subtile shV warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eVc)
{
  let tr = lane /^ 16sz;
  let tc = lane %^ 16sz;
  let nrow : sz = bn /^ warp_row_span_sz;
  let ncol : sz = d  /^ 16sz;

  let mut a : sz = 0sz;
  while (!a <^ nrow)
    invariant
      live a **
      (gK |-> Frac fK eK) **
      (gV |-> Frac fV eV) **
      (exists* eKc. array2_stride_subtile shK warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eKc) **
      (exists* eVc. array2_stride_subtile shV warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eVc) **
      pure (SZ.v !a <= SZ.v bn / warp_row_span)
    decreases (nrow - !a)
  {
    let va0 = !a;
    let arow : szlt (SZ.v bn / warp_row_span) = va0;
    tile_idx_lem warp_row_span (SZ.v arow) (SZ.v tr) (SZ.v bn);
    let trow : szlt bn = warp_row_span_sz *^ arow +^ tr;
    let kr : szlt sk = clamp_lt sk (k0base +^ trow);

    let mut b : sz = 0sz;
    while (!b <^ ncol)
      invariant
        live b **
        (gK |-> Frac fK eK) **
        (gV |-> Frac fV eV) **
        (exists* eKc. array2_stride_subtile shK warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eKc) **
        (exists* eVc. array2_stride_subtile shV warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eVc) **
        pure (SZ.v !b <= SZ.v d / 16)
      decreases (ncol - !b)
    {
      let vb0 = !b;
      let bcol : szlt (SZ.v d / 16) = vb0;
      tile_idx_lem 16 (SZ.v bcol) (SZ.v tc) (SZ.v d);
      let dd : szlt d = 16sz *^ bcol +^ tc;

      let vk = tensor_read gK (cidx2 kr dd);
      tensor_write #_ #_ #_ #_
        #(c_stride_subtile_layout lshK #cshK
            warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16))
        (array2_stride_subtile shK warp_row_span 16
          (SZ.v lane / 16) (SZ.v lane % 16))
        (cidx2 arow bcol) vk;
      let vv = tensor_read gV (cidx2 kr dd);
      tensor_write #_ #_ #_ #_
        #(c_stride_subtile_layout lshV #cshV
            warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16))
        (array2_stride_subtile shV warp_row_span 16
          (SZ.v lane / 16) (SZ.v lane % 16))
        (cidx2 arow bcol) vv;
      b := !b +^ 1sz;
    };
    a := !a +^ 1sz;
  };
  ()
}

(* ────────────────────────────────────────────────────────────────────────
   sdpa_flash_scale  --  online-softmax rescale of the output tile
   (flash_attn_fa1.cu, lines 190-191).

   The CUDA loop [for idx = lane; idx < BM*HD; idx += WARP] does the in-place
   row-broadcast multiply [Osh[idx] *= cw[idx / HD]] over the [16 x hd] output
   tile.  The per-lane ownership of the [O] tile is exactly the strided
   [(warp_row_span, 16)] subtile of [sdpa_flash_pv_mm] -- so [O] never changes
   representation between the rescale and the [P@V] accumulation, and no barrier
   has to move it.  Lane [(tr, tc) = (lane/16, lane%16)] owns the subtile of
   shape [lane_row_span x (hd/16)]; its cell [(orow, ocol)] is global row
   [orow*warp_row_span + tr], so the correction weight it multiplies by is
   [cw[orow*warp_row_span + tr]].

   The correction weights [cw] form a length-16 row vector; every lane reads the
   whole vector, so it is passed with a divided read-only fraction over the
   entire array -- no need to split it into per-lane cells since it is read
   exclusively.  Memory safety only. *)
inline_for_extraction noextract
fn sdpa_flash_scale
  (#et : Type0) {| scalar et |}
  (hd : szp)
  (#_ : squash (16 /?+ SZ.v hd))
  (#_ : squash (SZ.fits (16 * SZ.v hd)))
  (lane : szlt warp_size)
  (#lcw : layout1 16)
  (#lO : layout2 16 (SZ.v hd))
  {| ctlayout lcw |} {| ctlayout lO |}
  (shO : array2 et lO)
  (shcw : array1 et lcw)
  (#fcw : perm)
  (#ecw : chest1 et 16)
  (#eO0 : chest2 et lane_row_span (SZ.v hd / 16))
  requires
    (shcw |-> Frac fcw ecw) **
    (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO0)
  ensures
    (shcw |-> Frac fcw ecw) **
    (exists* eO. array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO)
{
  let tr = lane /^ 16sz;
  let ncol : sz = hd /^ 16sz;

  let mut orow : sz = 0sz;
  while (!orow <^ lane_row_span_sz)
    invariant
      live orow **
      (shcw |-> Frac fcw ecw) **
      (exists* eO. array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO) **
      pure (SZ.v !orow <= lane_row_span)
    decreases (lane_row_span_sz - !orow)
  {
    let vorow = !orow;
    let vor : szlt lane_row_span = vorow;
    let irow : szlt 16 = warp_row_span_sz *^ vor +^ tr;
    let cwv = tensor_read shcw (cidx1 irow);

    let mut ocol : sz = 0sz;
    while (!ocol <^ ncol)
      invariant
        live ocol **
        (shcw |-> Frac fcw ecw) **
        (exists* eO. array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16) |-> Frac 1.0R eO) **
        pure (SZ.v !ocol <= SZ.v hd / 16)
      decreases (ncol - !ocol)
    {
      let vocol = !ocol;
      let oc : szlt (SZ.v hd / 16) = vocol;
      let ov = tensor_read #_ #_ #_ #_ #(c_stride_subtile_layout lO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16))
        (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 vor oc);
      tensor_write #_ #_ #_ #_ #(c_stride_subtile_layout lO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16))
        (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 vor oc) (ov `mul` cwv);
      ocol := !ocol +^ 1sz;
    };
    orow := !orow +^ 1sz;
  };
  ()
}

(* ────────────────────────────────────────────────────────────────────────
   Warp-barrier ownership helpers.

   The five leaf functions are per-lane, and each [__syncwarp()] between them is
   the only place inter-lane ownership may move (a single lane cannot collect
   another lane's fraction on its own).  These ghosts implement, at the warp
   level (over [forall+ (i:natlt warp_size)]), the representation changes the
   adjacent stages disagree on.  They are the transforms fed to the (unsound-by-
   construction) [warp_barrier_wait]; every one is uniform in the lane index, so
   none exploits the tid-dependence the barrier would otherwise permit. *)

(* Collect the [warp_size] exclusive [(warp_row_span, 16)] stride sub-tiles that
   [kv_load] / [scale] / [pv_mm] own (lane [i] owns residue [(i/16, i%16)]) into
   the whole tile.  This is the [array2_stride_untile'] of the FA [rows_gather],
   with the lane index factored [warp_size = warp_row_span * 16]. *)
ghost
fn warp_gather_stride
  (#et:Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#_ : squash (warp_row_span /? rows))
  (#_ : squash (16 /? cols))
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i:natlt BW.warp_size).
       exists* (r:chest2 et (rows / warp_row_span) (cols / 16)).
         array2_stride_subtile a warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ensures
    exists* (e:chest2 et rows cols). a |-> Frac 1.0R e
{
  let rf = forevery_exists
    (fun (i:natlt BW.warp_size) (r:chest2 et (rows / warp_row_span) (cols / 16)) ->
       array2_stride_subtile a warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r);
  forevery_ext #(natlt BW.warp_size)
    (fun (i:natlt BW.warp_size) ->
       array2_stride_subtile a warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R (rf i))
    (fun (i:natlt BW.warp_size) ->
       array2_stride_subtile a warp_row_span 16 (i / 16) (i % 16)
         |-> Frac 1.0R (rf ((i / 16) * 16 + (i % 16))));
  forevery_factor' BW.warp_size warp_row_span 16
    (fun (tr:natlt warp_row_span) (tc:natlt 16) ->
       array2_stride_subtile a warp_row_span 16 tr tc |-> Frac 1.0R (rf (tr * 16 + tc)));
  array2_stride_untile' a warp_row_span 16
    (fun (tr:natlt warp_row_span) (tc:natlt 16) -> rf (tr * 16 + tc));
}

(* Split the whole tile back into the [warp_size] exclusive [(warp_row_span, 16)]
   stride sub-tiles.  Inverse of [warp_gather_stride]; the [array2_stride_tile]
   of the FA [rows_split]. *)
ghost
fn warp_split_stride
  (#et:Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (#_ : squash (warp_row_span /? rows))
  (#_ : squash (16 /? cols))
  requires
    exists* (e:chest2 et rows cols). a |-> Frac 1.0R e
  ensures
    forall+ (i:natlt BW.warp_size).
      exists* (r:chest2 et (rows / warp_row_span) (cols / 16)).
        array2_stride_subtile a warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r
{
  with e. assert (a |-> Frac 1.0R e);
  array2_stride_tile a warp_row_span 16;
  forevery_unfactor' BW.warp_size warp_row_span 16
    (fun (tr:natlt warp_row_span) (tc:natlt 16) ->
       array2_stride_subtile a warp_row_span 16 tr tc
         |-> Frac 1.0R (ematrix_stride_subtile e warp_row_span 16 tr tc));
  forevery_map #(natlt BW.warp_size)
    (fun (i:natlt BW.warp_size) ->
       array2_stride_subtile a warp_row_span 16 (i / 16) (i % 16)
         |-> Frac 1.0R (ematrix_stride_subtile e warp_row_span 16 (i / 16) (i % 16)))
    (fun (i:natlt BW.warp_size) ->
       exists* (r:chest2 et (rows / warp_row_span) (cols / 16)).
         array2_stride_subtile a warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
    fn i { () };
}

(* ── barrier 1 : kv_load -> qk_mm (CUDA line 136) ─────────────────────────────
   K and V arrive as the [warp_size] exclusive [(warp_row_span,16)] stride
   sub-tiles [kv_load] just filled.  qk_mm reads Q@K^T, so it wants K viewed
   COLUMN-major ([row2col shK], the same storage) and shared read-only across the
   warp; V is not touched by qk_mm but is likewise re-shared now (it stays a
   read-only fraction until pv_mm, so no lane needs it exclusive in between).
   Both are matmul *inputs* whose fraction qk_mm/pv_mm accept as arbitrary, so no
   perm token has to be matched here. *)
unfold let b1_pre (#et:Type0) (d:szp) (#_ : squash (16 /?+ SZ.v d))
  (#lK #lV : layout2 16 (SZ.v d))
  (shK : array2 et lK) (shV : array2 et lV)
  (i : natlt BW.warp_size) : slprop
= (exists* (r:chest2 et (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shK warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ** (exists* (r:chest2 et (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shV warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)

unfold let b1_post (#et:Type0) (d:szp)
  (#lK #lV : layout2 16 (SZ.v d))
  (shK : array2 et lK) (shV : array2 et lV)
  (i : natlt BW.warp_size) : slprop
= (exists* (s:chest2 et (SZ.v d) 16). flash_row2col shK |-> Frac (1.0R /. BW.warp_size) s)
  ** (exists* (s:chest2 et 16 (SZ.v d)). shV |-> Frac (1.0R /. BW.warp_size) s)

ghost
fn barrier1_transform
  (#et:Type0) (d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#lK #lV : layout2 16 (SZ.v d))
  {| ctlayout lK |} {| ctlayout lV |}
  (shK : array2 et lK) (shV : array2 et lV)
  requires forall+ (i:natlt BW.warp_size). b1_pre d shK shV i
  ensures  forall+ (i:natlt BW.warp_size). b1_post d shK shV i
{
  forevery_unzip
    (fun (i:natlt BW.warp_size) ->
       exists* (r:chest2 et (16 / warp_row_span) (SZ.v d / 16)).
         array2_stride_subtile shK warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
    (fun (i:natlt BW.warp_size) ->
       exists* (r:chest2 et (16 / warp_row_span) (SZ.v d / 16)).
         array2_stride_subtile shV warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r);

  warp_gather_stride shK;
  with eK. assert (shK |-> Frac 1.0R eK);
  flash_transpose shK;
  tensor_share_n (flash_row2col shK) BW.warp_size;
  forevery_map #(natlt BW.warp_size)
    (fun (_:natlt BW.warp_size) -> flash_row2col shK |-> Frac (1.0R /. BW.warp_size) (mtranspose eK))
    (fun (_:natlt BW.warp_size) ->
       exists* (s:chest2 et (SZ.v d) 16). flash_row2col shK |-> Frac (1.0R /. BW.warp_size) s)
    fn i { () };

  warp_gather_stride shV;
  with eV. assert (shV |-> Frac 1.0R eV);
  tensor_share_n shV BW.warp_size;
  forevery_map #(natlt BW.warp_size)
    (fun (_:natlt BW.warp_size) -> shV |-> Frac (1.0R /. BW.warp_size) eV)
    (fun (_:natlt BW.warp_size) ->
       exists* (s:chest2 et 16 (SZ.v d)). shV |-> Frac (1.0R /. BW.warp_size) s)
    fn i { () };

  forevery_zip
    (fun (i:natlt BW.warp_size) ->
       exists* (s:chest2 et (SZ.v d) 16). flash_row2col shK |-> Frac (1.0R /. BW.warp_size) s)
    (fun (i:natlt BW.warp_size) ->
       exists* (s:chest2 et 16 (SZ.v d)). shV |-> Frac (1.0R /. BW.warp_size) s);
}

(* The K/V ownership transform as a first-class [stt_ghost] value, to hand to
   [warp_barrier_wait].  [d]/[shK]/[shV] are IMPLICIT so this is passed as the
   bare name [barrier1_proof]: Pulse then recovers the implicits by unifying
   against the barrier's expected [p]/[q] and passes it as a value.  Written as
   an explicit ghost application ([barrier1_proof d shK shV]) it would instead be
   sequenced as a ghost step in the single-lane caller, demanding the whole-warp
   [forall+ i. b1_pre i] there (which no single lane holds). *)
let barrier1_proof
  (#et:Type0) (#d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#lK #lV : layout2 16 (SZ.v d))
  {| ctlayout lK |} {| ctlayout lV |}
  (#shK : array2 et lK)
  (#shV : array2 et lV)
  : stt_ghost unit emp_inames
      (requires forall+ (i:natlt BW.warp_size). b1_pre d shK shV i)
      (ensures  fun _ -> forall+ (i:natlt BW.warp_size). b1_post d shK shV i)
  = barrier1_transform d shK shV

inline_for_extraction noextract
fn sdpa_flash_barrier1
  (#et:Type0) (d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#lK #lV : layout2 16 (SZ.v d))
  {| ctlayout lK |} {| ctlayout lV |}
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  (shK : array2 et lK) (shV : array2 et lV)
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires b1_pre d shK shV (SZ.v lane % BW.warp_size)
  ensures  b1_post d shK shV (SZ.v lane % BW.warp_size)
{
  rewrite each (SZ.v lane % BW.warp_size) as (SZ.v tid % BW.warp_size);
  BW.warp_barrier_wait () (b1_pre d shK shV) (b1_post d shK shV)
    barrier1_proof #(SZ.v nthr) #(SZ.v tid);
}

(* ── barrier 2 : qk_mm -> softmax (CUDA line 148) ─────────────────────────────
   K is done being read (as [row2col shK], shared col-major fraction), so it goes
   back to the [warp_size] exclusive stride sub-tiles [kv_load] will overwrite next
   iteration.  The score tile [shS], produced by qk_mm as a whole-tile [1/warp]
   fraction over all 32 lanes, is handed to the 16 active [softmax] lanes: each
   lane [i < 16] receives its row [i] at full permission (plus the [mextract_row]
   restore wand, framed across softmax and consumed by barrier 3); lanes 16..31
   receive [emp].  Only shK/shS move; shV, shP, shO, ... are framed in [jt_body]. *)

(* One score/probability row [i] (i < 16) as its [1 x 16] write-subtile at full
   permission.  [jt_body] locally bridges this to the [array1] [mrow] that
   [softmax_upd] wants (via [mextract_row]/wand), so the barriers only ever move
   clean subtile [pts_to]s -- no [mrow]/wand plumbing crosses a warp barrier. *)
(* Lift 16 active [forall+] entries to a full 32-lane [forall+] (inactive = emp)
   and the reverse.  Generic over the per-lane payload [p]. *)
ghost
fn lift_16to32 (p : natlt 16 -> slprop)
  requires forall+ (i:natlt 16). p i
  ensures  forall+ (i:natlt BW.warp_size). when__ (i < 16) (fun _ -> p i)
{
  forevery_natlt_extend BW.warp_size p;
  forevery_unrefine_pred' #(natlt BW.warp_size)
    (fun (i:natlt BW.warp_size) -> i < 16)
    (fun (i:natlt BW.warp_size) (_:squash (i < 16)) -> p i);
}

ghost
fn lower_32to16 (p : natlt 16 -> slprop)
  requires forall+ (i:natlt BW.warp_size). when__ (i < 16) (fun _ -> p i)
  ensures  forall+ (i:natlt 16). p i
{
  forevery_refine_split (fun (i:natlt BW.warp_size) -> when__ (i < 16) (fun _ -> p i))
    (fun (i:natlt BW.warp_size) -> i < 16);
  drop_ (forall+ (i:natlt BW.warp_size { ~(i < 16) }). when__ (i < 16) (fun _ -> p i));
  forevery_ext #(i:natlt BW.warp_size { i < 16 })
    (fun (i:natlt BW.warp_size { i < 16 }) -> when__ (i < 16) (fun _ -> p i))
    (fun (i:natlt BW.warp_size { i < 16 }) -> p i);
  forevery_natlt_restrict BW.warp_size p;
}


(* Whole 1/warp fraction (all 32 lanes) <-> 16 exclusive row-subtiles.  Used by
   b2 (whole->rows, forward), b3/b5 (rows->whole, reverse). *)
ghost
fn whole32_to_rows16 (#et:Type0) (#l : layout2 16 16)
(shA : array2 et l)
  requires forall+ (i:natlt BW.warp_size). (exists* (e:chest2 et 16 16). shA |-> Frac (1.0R /. BW.warp_size) e)
  ensures  forall+ (i:natlt 16). row_subtile shA i
{
  tensor_gather_n_underspec shA BW.warp_size;
  with eS. assert (shA |-> Frac 1.0R eS);
  array2_tile shA 1 16;
  forevery_unfactor' 16 16 1
    (fun (tr:natlt 16) (tc:natlt 1) ->
       array2_subtile shA 1 16 tr tc |-> Frac 1.0R (ematrix_subtile eS 1 16 tr tc));
  forevery_ext #(natlt 16)
    (fun (i:natlt 16) ->
       array2_subtile shA 1 16 (i / 1) (i % 1) |-> Frac 1.0R (ematrix_subtile eS 1 16 (i / 1) (i % 1)))
    (fun (i:natlt 16) ->
       array2_subtile shA 1 16 i 0 |-> Frac 1.0R (ematrix_subtile eS 1 16 i 0));
  forevery_map #(natlt 16)
    (fun (tid:natlt 16) ->
       array2_subtile shA 1 16 tid 0 |-> Frac 1.0R (ematrix_subtile eS 1 16 tid 0))
    (fun (tid:natlt 16) -> row_subtile shA tid)
    fn tid { () };
}

ghost
fn rows16_to_whole32 (#et:Type0) (#l : layout2 16 16)
  {| c : ctlayout l |}
  (shA : array2 et l)
  requires forall+ (i:natlt 16). row_subtile shA i
  ensures  forall+ (i:natlt BW.warp_size). (exists* (e:chest2 et 16 16). shA |-> Frac (1.0R /. BW.warp_size) e)
{
  assert pure (SZ.fits (tlayout_ulen l));
  let rf = forevery_exists #(natlt 16)
    (fun (i:natlt 16) (r:chest2 et 1 16) -> array2_subtile shA 1 16 i 0 |-> Frac 1.0R r);
  forevery_ext #(natlt 16)
    (fun (i:natlt 16) -> array2_subtile shA 1 16 i 0 |-> Frac 1.0R (rf i))
    (fun (i:natlt 16) -> array2_subtile shA 1 16 (i / 1) (i % 1) |-> Frac 1.0R (rf (i / 1)));
  forevery_factor' 16 16 1
    (fun (tr:natlt 16) (tc:natlt 1) -> array2_subtile shA 1 16 tr tc |-> Frac 1.0R (rf tr));
  array2_untile' shA 1 16 (fun (tr:natlt 16) (tc:natlt 1) -> rf tr);
  tensor_share_n shA BW.warp_size;
  forevery_map #(natlt BW.warp_size)
    (fun (i:natlt BW.warp_size) ->
       shA |-> Frac (1.0R /. BW.warp_size) (ematrix_from_tiles 1 16 (fun (tr:natlt 16) (tc:natlt 1) -> rf tr)))
    (fun (i:natlt BW.warp_size) -> exists* (e:chest2 et 16 16). shA |-> Frac (1.0R /. BW.warp_size) e)
    fn i { () };
}

(* shcw scalar array (length 16): whole 1/warp fraction (all 32) <-> 16 exclusive
   cells (via [Cell]).  Used by b3 (cells->whole) and b5 (whole->cells). *)
ghost
fn cells16_to_whole32 (#et:Type0) (#lcw:layout1 16)
  (shcw : array1 et lcw)
  requires (forall+ (i:natlt 16). cell_full shcw i) ** pure (SZ.fits (tlayout_ulen lcw))
  ensures  forall+ (i:natlt BW.warp_size). (exists* (e:chest1 et 16). shcw |-> Frac (1.0R /. BW.warp_size) e)
{
  let vf = forevery_exists #(natlt 16)
    (fun (i:natlt 16) (v:et) -> Cell shcw (idx1 i) |-> Frac 1.0R v);
  let s : chest1 et 16 = mk1 vf;
  forevery_ext #(natlt 16)
    (fun (i:natlt 16) -> Cell shcw (idx1 i) |-> Frac 1.0R (vf i))
    (fun (i:natlt 16) -> Cell shcw (idx1 i) |-> Frac 1.0R (acc1 s i));
  implode1 shcw #1.0R #s;
  tensor_share_n shcw BW.warp_size;
  forevery_map #(natlt BW.warp_size)
    (fun (i:natlt BW.warp_size) -> shcw |-> Frac (1.0R /. BW.warp_size) s)
    (fun (i:natlt BW.warp_size) -> exists* (e:chest1 et 16). shcw |-> Frac (1.0R /. BW.warp_size) e)
    fn i { () };
}

ghost
fn whole32_to_cells16 (#et:Type0) (#lcw:layout1 16)
  (shcw : array1 et lcw)
  requires forall+ (i:natlt BW.warp_size). (exists* (e:chest1 et 16). shcw |-> Frac (1.0R /. BW.warp_size) e)
  ensures  forall+ (i:natlt 16). cell_full shcw i
{
  tensor_gather_n_underspec shcw BW.warp_size;
  with s. assert (shcw |-> Frac 1.0R s);
  explode1 shcw #1.0R #s;
  forevery_map #(natlt 16)
    (fun (i:natlt 16) -> Cell shcw (idx1 i) |-> Frac 1.0R (acc1 s i))
    (fun (i:natlt 16) -> cell_full shcw i)
    fn i { () };
}

unfold let b2_pre (#et_ab #et_acc:Type0) (d:szp)
  (#lK : layout2 16 (SZ.v d)) (#lS : layout2 16 16)
  (shK : array2 et_ab lK)
  (shS : array2 et_acc lS)
  (i : natlt BW.warp_size) : slprop
= (exists* (s:chest2 et_ab (SZ.v d) 16). flash_row2col shK |-> Frac (1.0R /. BW.warp_size) s)
  ** (exists* (e:chest2 et_acc 16 16). shS |-> Frac (1.0R /. BW.warp_size) e)

unfold let b2_post (#et_ab #et_acc:Type0) (d:szp) (#_ : squash (16 /?+ SZ.v d))
  (#lK : layout2 16 (SZ.v d)) (#lS : layout2 16 16)
  (shK : array2 et_ab lK)
  (shS : array2 et_acc lS)
  (i : natlt BW.warp_size) : slprop
= (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shK warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ** when__ (i < 16) (fun _ -> row_subtile shS i)

ghost
fn barrier2_transform
  (#et_ab #et_acc:Type0) (d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#lK : layout2 16 (SZ.v d)) (#lS : layout2 16 16)
  {| ctlayout lK |} {| ctlayout lS |}
  (shK : array2 et_ab lK)
  (shS : array2 et_acc lS)
  requires forall+ (i:natlt BW.warp_size). b2_pre d shK shS i
  ensures  forall+ (i:natlt BW.warp_size). b2_post d shK shS i
{
  forevery_unzip
    (fun (i:natlt BW.warp_size) ->
       exists* (s:chest2 et_ab (SZ.v d) 16). flash_row2col shK |-> Frac (1.0R /. BW.warp_size) s)
    (fun (i:natlt BW.warp_size) ->
       exists* (e:chest2 et_acc 16 16). shS |-> Frac (1.0R /. BW.warp_size) e);

  (* shK: shared col-major fraction -> exclusive stride sub-tiles *)
  tensor_gather_n_underspec (flash_row2col shK) BW.warp_size;
  with sKc. assert (flash_row2col shK |-> Frac 1.0R sKc);
  flash_transpose_back shK;
  warp_split_stride shK;

  (* shS: whole-tile 1/warp fraction -> per-row full permission (16 active) + emp *)
  whole32_to_rows16 shS;
  lift_16to32 (row_subtile shS);

  forevery_zip
    (fun (i:natlt BW.warp_size) ->
       exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
         array2_stride_subtile shK warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
    (fun (i:natlt BW.warp_size) -> when__ (i < 16) (fun _ -> row_subtile shS i));
}

let barrier2_proof
  (#et_ab #et_acc:Type0) (#d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#lK : layout2 16 (SZ.v d)) (#lS : layout2 16 16)
  {| ctlayout lK |} {| ctlayout lS |}
  (#shK : array2 et_ab lK)
  (#shS : array2 et_acc lS)
  : stt_ghost unit emp_inames
      (requires forall+ (i:natlt BW.warp_size). b2_pre d shK shS i)
      (ensures  fun _ -> forall+ (i:natlt BW.warp_size). b2_post d shK shS i)
  = barrier2_transform d shK shS

inline_for_extraction noextract
fn sdpa_flash_barrier2
  (#et_ab #et_acc:Type0) (d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#lK : layout2 16 (SZ.v d)) (#lS : layout2 16 16)
  {| ctlayout lK |} {| ctlayout lS |}
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  (shK : array2 et_ab lK)
  (shS : array2 et_acc lS)
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires b2_pre d shK shS (SZ.v lane % BW.warp_size)
  ensures  b2_post d shK shS (SZ.v lane % BW.warp_size)
{
  BW.warp_barrier_wait () (b2_pre d shK shS) (b2_post d shK shS)
    barrier2_proof #(SZ.v nthr) #(SZ.v tid);
  rewrite each (SZ.v tid % BW.warp_size) as (SZ.v lane % BW.warp_size);
}

(* ── Barrier 3 (CUDA line 188): softmax_upd -> scale ───────────────────────────
   The three arrays the softmax lanes exclusively owned per-row (shS, shP) or
   per-cell (shcw) are returned to the collective whole-tile [1/warp] fraction so
   the following [scale] (and next-iteration [qk_mm]) can read them. *)
unfold let b3_pre (#et_ab #et_acc:Type0) (#lcw:layout1 16)
  (#lS #lP : layout2 16 16)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  (i : natlt BW.warp_size) : slprop
= when__ (i < 16) (fun _ -> row_subtile shS i)
  ** when__ (i < 16) (fun _ -> row_subtile shP i)
  ** when__ (i < 16) (fun _ -> cell_full shcw i)

unfold let b3_post (#et_ab #et_acc:Type0) (#lcw:layout1 16)
  (#lS #lP : layout2 16 16)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  (i : natlt BW.warp_size) : slprop
= (exists* (e:chest2 et_acc 16 16). shS |-> Frac (1.0R /. BW.warp_size) e)
  ** (exists* (e:chest2 et_ab 16 16). shP |-> Frac (1.0R /. BW.warp_size) e)
  ** (exists* (e:chest1 et_acc 16). shcw |-> Frac (1.0R /. BW.warp_size) e)

ghost
fn barrier3_transform
  (#et_ab #et_acc:Type0) (#lcw:layout1 16) (#_ : squash (SZ.fits (tlayout_ulen lcw)))
  (#lS #lP : layout2 16 16)
  {| ctlayout lS |} {| ctlayout lP |}
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  requires forall+ (i:natlt BW.warp_size). b3_pre shS shP shcw i
  ensures  forall+ (i:natlt BW.warp_size). b3_post shS shP shcw i
{
  forevery_unzip
    (fun (i:natlt BW.warp_size) -> when__ (i < 16) (fun _ -> row_subtile shS i))
    (fun (i:natlt BW.warp_size) ->
       when__ (i < 16) (fun _ -> row_subtile shP i)
       ** when__ (i < 16) (fun _ -> cell_full shcw i));
  forevery_unzip
    (fun (i:natlt BW.warp_size) -> when__ (i < 16) (fun _ -> row_subtile shP i))
    (fun (i:natlt BW.warp_size) -> when__ (i < 16) (fun _ -> cell_full shcw i));

  lower_32to16 (row_subtile shS);
  rows16_to_whole32 shS;
  lower_32to16 (row_subtile shP);
  rows16_to_whole32 shP;
  lower_32to16 (cell_full shcw);
  cells16_to_whole32 shcw;

  forevery_zip
    (fun (i:natlt BW.warp_size) -> exists* (e:chest2 et_ab 16 16). shP |-> Frac (1.0R /. BW.warp_size) e)
    (fun (i:natlt BW.warp_size) -> exists* (e:chest1 et_acc 16). shcw |-> Frac (1.0R /. BW.warp_size) e);
  forevery_zip
    (fun (i:natlt BW.warp_size) -> exists* (e:chest2 et_acc 16 16). shS |-> Frac (1.0R /. BW.warp_size) e)
    (fun (i:natlt BW.warp_size) ->
       (exists* (e:chest2 et_ab 16 16). shP |-> Frac (1.0R /. BW.warp_size) e)
       ** (exists* (e:chest1 et_acc 16). shcw |-> Frac (1.0R /. BW.warp_size) e));
}

let barrier3_proof
  (#et_ab #et_acc:Type0) (#lcw:layout1 16) (#_ : squash (SZ.fits (tlayout_ulen lcw)))
  (#lS #lP : layout2 16 16)
  {| ctlayout lS |} {| ctlayout lP |}
  (#shS : array2 et_acc lS)
  (#shP : array2 et_ab lP)
  (#shcw : array1 et_acc lcw)
  : stt_ghost unit emp_inames
      (requires forall+ (i:natlt BW.warp_size). b3_pre shS shP shcw i)
      (ensures  fun _ -> forall+ (i:natlt BW.warp_size). b3_post shS shP shcw i)
  = barrier3_transform shS shP shcw

inline_for_extraction noextract
fn sdpa_flash_barrier3
  (#et_ab #et_acc:Type0) (#lcw:layout1 16) (#_ : squash (SZ.fits (tlayout_ulen lcw)))
  (#lS #lP : layout2 16 16)
  {| ctlayout lS |} {| ctlayout lP |}
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires b3_pre shS shP shcw (SZ.v lane % BW.warp_size)
  ensures  b3_post shS shP shcw (SZ.v lane % BW.warp_size)
{
  rewrite each (SZ.v lane % BW.warp_size) as (SZ.v tid % BW.warp_size);
  BW.warp_barrier_wait () (b3_pre shS shP shcw) (b3_post shS shP shcw)
    barrier3_proof #(SZ.v nthr) #(SZ.v tid);
}

(* ── Barrier 4 (CUDA line 192): scale -> pv_mm ─────────────────────────────────
   A pure ordering fence: [scale] and [pv_mm] both use [shO] as the same exclusive
   stride sub-tile, and shP/shV/shcw keep their (framed) forms across it, so no
   ownership moves -- [p == q == emp]. *)
inline_for_extraction noextract
fn sdpa_flash_barrier4
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  preserves thread_id (SZ.v nthr) tid
  requires emp
  ensures  emp
{
  BW.warp_barrier_wait () warp_emp_pred warp_emp_pred warp_emp_proof
    #(SZ.v nthr) #(SZ.v tid);
}

(* ── Barrier 5 (CUDA line 208): loop edge -> next iteration ────────────────────
   Restores the three arrays whose form pv_mm/scale left "collective" back to the
   loop-invariant resting forms the next iteration expects: shV to the exclusive
   stride sub-tiles kv_load overwrites, shP to per-row (softmax writes), shcw to
   per-cell (softmax writes).  shK/shS/shO already rest in their invariant form. *)
unfold let b5_pre (#et_ab #et_acc:Type0) (d:szp) (#lcw:layout1 16)
  (#lV : layout2 16 (SZ.v d)) (#lP : layout2 16 16)
  (shV : array2 et_ab lV)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  (i : natlt BW.warp_size) : slprop
= (exists* (s:chest2 et_ab 16 (SZ.v d)). shV |-> Frac (1.0R /. BW.warp_size) s)
  ** (exists* (e:chest2 et_ab 16 16). shP |-> Frac (1.0R /. BW.warp_size) e)
  ** (exists* (e:chest1 et_acc 16). shcw |-> Frac (1.0R /. BW.warp_size) e)

unfold let b5_post (#et_ab #et_acc:Type0) (d:szp) (#_ : squash (16 /?+ SZ.v d)) (#lcw:layout1 16)
  (#lV : layout2 16 (SZ.v d)) (#lP : layout2 16 16)
  (shV : array2 et_ab lV)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  (i : natlt BW.warp_size) : slprop
= (exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
     array2_stride_subtile shV warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ** when__ (i < 16) (fun _ -> row_subtile shP i)
  ** when__ (i < 16) (fun _ -> cell_full shcw i)

ghost
fn barrier5_transform
  (#et_ab #et_acc:Type0) (d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#lcw:layout1 16)
  (#lV : layout2 16 (SZ.v d)) (#lP : layout2 16 16)
  {| ctlayout lV |} {| ctlayout lP |}
  (shV : array2 et_ab lV)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  requires forall+ (i:natlt BW.warp_size). b5_pre d shV shP shcw i
  ensures  forall+ (i:natlt BW.warp_size). b5_post d shV shP shcw i
{
  forevery_unzip
    (fun (i:natlt BW.warp_size) -> exists* (s:chest2 et_ab 16 (SZ.v d)). shV |-> Frac (1.0R /. BW.warp_size) s)
    (fun (i:natlt BW.warp_size) ->
       (exists* (e:chest2 et_ab 16 16). shP |-> Frac (1.0R /. BW.warp_size) e)
       ** (exists* (e:chest1 et_acc 16). shcw |-> Frac (1.0R /. BW.warp_size) e));
  forevery_unzip
    (fun (i:natlt BW.warp_size) -> exists* (e:chest2 et_ab 16 16). shP |-> Frac (1.0R /. BW.warp_size) e)
    (fun (i:natlt BW.warp_size) -> exists* (e:chest1 et_acc 16). shcw |-> Frac (1.0R /. BW.warp_size) e);

  (* shV: whole 1/warp fraction -> exclusive stride sub-tiles *)
  tensor_gather_n_underspec shV BW.warp_size;
  warp_split_stride shV;

  (* shP: whole -> per-row (16 active) + emp *)
  whole32_to_rows16 shP;
  lift_16to32 (row_subtile shP);

  (* shcw: whole -> per-cell (16 active) + emp *)
  whole32_to_cells16 shcw;
  lift_16to32 (cell_full shcw);

  forevery_zip
    (fun (i:natlt BW.warp_size) -> when__ (i < 16) (fun _ -> row_subtile shP i))
    (fun (i:natlt BW.warp_size) -> when__ (i < 16) (fun _ -> cell_full shcw i));
  forevery_zip
    (fun (i:natlt BW.warp_size) ->
       exists* (r:chest2 et_ab (16 / warp_row_span) (SZ.v d / 16)).
         array2_stride_subtile shV warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
    (fun (i:natlt BW.warp_size) ->
       when__ (i < 16) (fun _ -> row_subtile shP i)
       ** when__ (i < 16) (fun _ -> cell_full shcw i));
}

let barrier5_proof
  (#et_ab #et_acc:Type0) (#d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#lcw:layout1 16)
  (#lV : layout2 16 (SZ.v d)) (#lP : layout2 16 16)
  {| ctlayout lV |} {| ctlayout lP |}
  (#shV : array2 et_ab lV)
  (#shP : array2 et_ab lP)
  (#shcw : array1 et_acc lcw)
  : stt_ghost unit emp_inames
      (requires forall+ (i:natlt BW.warp_size). b5_pre d shV shP shcw i)
      (ensures  fun _ -> forall+ (i:natlt BW.warp_size). b5_post d shV shP shcw i)
  = barrier5_transform d shV shP shcw

inline_for_extraction noextract
fn sdpa_flash_barrier5
  (#et_ab #et_acc:Type0) (d:szp)
  (#_ : squash (16 /?+ SZ.v d)) (#lcw:layout1 16)
  (#lV : layout2 16 (SZ.v d)) (#lP : layout2 16 16)
  {| ctlayout lV |} {| ctlayout lP |}
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  (shV : array2 et_ab lV)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires b5_pre d shV shP shcw (SZ.v lane % BW.warp_size)
  ensures  b5_post d shV shP shcw (SZ.v lane % BW.warp_size)
{
  BW.warp_barrier_wait () (b5_pre d shV shP shcw) (b5_post d shV shP shcw)
    barrier5_proof #(SZ.v nthr) #(SZ.v tid);
  rewrite each (SZ.v tid % BW.warp_size) as (SZ.v lane % BW.warp_size);
}

(* ── single-lane when__ intro/elim (from Kuiper.Sparse.SPMM) ──────────────────
   [jt_body] holds one lane's slice of each barrier's [forall+]; inside the
   [if lane < 16] guard it must unwrap / rewrap the [when__ (i<16)] payloads. *)
ghost
fn when__elim_true (b:bool{b == true}) (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures q ()
{ rewrite when__ b q as q (); }

ghost
fn when__elim_false (b:bool{b == false}) (q : squash (b2t b) -> slprop)
  requires when__ b q
  ensures emp
{ rewrite when__ b q as emp; }

ghost
fn when__intro_true (b:bool{b == true}) (q : squash (b2t b) -> slprop)
  requires q ()
  ensures when__ b q
{ rewrite q () as when__ b q; }

ghost
fn when__intro_false (b:bool{b == false}) (q : squash (b2t b) -> slprop)
  requires emp
  ensures when__ b q
{ rewrite emp as when__ b q; }

ghost
fn when_elim_true (b:bool{b == true}) (q : slprop)
  requires when_ b q
  ensures q
{
  rewrite when_ b q as q;
}

ghost
fn when_elim_false (b:bool{b == false}) (q : slprop)
  requires when_ b q
  ensures emp
{
  rewrite when_ b q as emp;
}

ghost
fn when_intro_true (b:bool{b == true}) (q : slprop)
  requires q
  ensures when_ b q
{
  rewrite q as when_ b q;
}

ghost
fn when_intro_false (b:bool{b == false}) (q : slprop)
  ensures when_ b q
{
  rewrite emp as when_ b q;
}

(* ── sdpa_flash_jt_body : one key-tile iteration (CUDA lines 128-210) ──────────
   The body of the [for jt] loop, executed by a single lane.  It chains the five
   leaf functions with the five warp-barrier separators:

     kv_load ; barrier1 ; qk_mm ; barrier2 ; (if lane<16 softmax_upd) ;
     barrier3 ; scale ; barrier4 ; pv_mm ; barrier5

   Because this is a loop body its pre- and post-condition are the same resting
   invariant [jt_rest] (the state each barrier leaves for the next iteration).
   HD is packed to D here (hd = d): the K/V/Q/O tiles are all [16 x d].

   Element types match the CUDA exactly: scores/max/sum/corr/O/PVc are [et_acc]
   (f32-like); K/V/Q/P/mask are [et_ab] (bf16-like). *)
(* Re-index a whole strided [(warp_row_span, 16)] sub-tile from lane residue [i]
   to [j] when [i == j] (they always are: a lane's [SZ.v lane % warp_size] equals
   [SZ.v lane]).  Bridges the barrier convention (residue [SZ.v lane % warp_size])
   to the leaf convention (residue [SZ.v lane]); it moves no ownership. *)
ghost
fn stride_reindex
  (#et:Type0) (#cols:nat) (#l:layout2 16 cols)
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (16 /?+ cols))
  (shX : array2 et l)
  (i j : natlt BW.warp_size)
  requires
    pure (i == j) **
    (exists* (r:chest2 et (16 / warp_row_span) (cols / 16)).
       array2_stride_subtile shX warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
  ensures
    (exists* (r:chest2 et (16 / warp_row_span) (cols / 16)).
       array2_stride_subtile shX warp_row_span 16 (j / 16) (j % 16) |-> Frac 1.0R r)
{
  with r. assert (array2_stride_subtile shX warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r);
  rewrite (array2_stride_subtile shX warp_row_span 16 (i / 16) (i % 16) |-> Frac 1.0R r)
       as (array2_stride_subtile shX warp_row_span 16 (j / 16) (j % 16) |-> Frac 1.0R r);
}

(* Re-index a [when__ (i<16)] row / cell payload from lane residue [i] to [j]
   when [i == j].  Same purpose as [stride_reindex] for the [when__]-guarded
   softmax rows/cells that the barriers state at [SZ.v lane % warp_size]. *)
ghost
fn row_reindex (#et:Type0) (#l : layout2 16 16)
  (shA : array2 et l) (i j : natlt BW.warp_size)
  requires pure (i == j) ** when__ (i < 16) (fun _ -> row_subtile shA i)
  ensures  when__ (j < 16) (fun _ -> row_subtile shA j)
{
  rewrite (when__ (i < 16) (fun _ -> row_subtile shA i))
       as (when__ (j < 16) (fun _ -> row_subtile shA j));
}

ghost
fn cell_reindex (#et:Type0) (#l:layout1 16) (shA : array1 et l) (i j : natlt BW.warp_size)
  requires pure (i == j) ** when__ (i < 16) (fun _ -> cell_full shA i)
  ensures  when__ (j < 16) (fun _ -> cell_full shA j)
{
  rewrite (when__ (i < 16) (fun _ -> cell_full shA i))
       as (when__ (j < 16) (fun _ -> cell_full shA j));
}

(* Online-softmax update for a single active lane [lane16 < 16].  Factored into
   its own function so the [mrow]/ref plumbing (and its local [let]-bound refs)
   stay behind a clean spec: the [jt] body can call it inside [if lane < 16]
   without the branch leaking existentials that would block the branch join. *)
inline_for_extraction noextract
fn sdpa_flash_softmax_active
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (b hq sq sk : szp)
  (#lgmask : TRO.vlayout4 b hq sq sk) {| TRO.cvtlayout lgmask |}
  (#lcw #lm #ll : layout1 16) {| ctlayout lcw |} {| ctlayout lm |} {| ctlayout ll |}
  (#lS #lP : layout2 16 16)
  {| cS : ctlayout lS |} {| cP : ctlayout lP |}
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  (shm : array1 et_acc lm)
  (shl : array1 et_acc ll)
  (gmask : TRO.roarray4 et_ab lgmask)
  (lane16 : szlt 16)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq)
  (k0 : sz) (#_ : squash (SZ.fits (SZ.v k0 + 16)))
  (cbound : sz) (row_active : bool) (causal : bool) (has_mask : bool) (scale : et_acc)
  (#fmask : perm)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  preserves (gmask |-> Frac fmask emask)
  requires
    row_subtile shS (SZ.v lane16) ** row_subtile shP (SZ.v lane16)
    ** cell_full shcw (SZ.v lane16) ** cell_full shm (SZ.v lane16) ** cell_full shl (SZ.v lane16)
  ensures
    row_subtile shS (SZ.v lane16) ** row_subtile shP (SZ.v lane16)
    ** cell_full shcw (SZ.v lane16) ** cell_full shm (SZ.v lane16) ** cell_full shl (SZ.v lane16)
{
  unfold (row_subtile shS (SZ.v lane16));
  unfold (row_subtile shP (SZ.v lane16));
  unfold (cell_full shcw (SZ.v lane16));
  unfold (cell_full shm (SZ.v lane16));
  unfold (cell_full shl (SZ.v lane16));

  with rS. assert (array2_subtile shS 1 16 (SZ.v lane16) 0 |-> Frac 1.0R rS);
  mextract_row (array2_subtile shS 1 16 (SZ.v lane16) 0) 0;
  with rP. assert (array2_subtile shP 1 16 (SZ.v lane16) 0 |-> Frac 1.0R rP);
  mextract_row (array2_subtile shP 1 16 (SZ.v lane16) 0) 0;

  array1_cell_to_ref shm (SZ.v lane16);
  let rm = get_ref_of_array_cell shm lane16;
  array1_cell_to_ref shl (SZ.v lane16);
  let rl = get_ref_of_array_cell shl lane16;
  array1_cell_to_ref shcw (SZ.v lane16);
  let rcw = get_ref_of_array_cell shcw lane16;

  with vmr0. assert (ref_of_array_cell shm (SZ.v lane16) |-> Frac 1.0R vmr0);
  rewrite (ref_of_array_cell shm (SZ.v lane16) |-> Frac 1.0R vmr0) as (rm |-> Frac 1.0R vmr0);
  with vlr0. assert (ref_of_array_cell shl (SZ.v lane16) |-> Frac 1.0R vlr0);
  rewrite (ref_of_array_cell shl (SZ.v lane16) |-> Frac 1.0R vlr0) as (rl |-> Frac 1.0R vlr0);
  with vcr0. assert (ref_of_array_cell shcw (SZ.v lane16) |-> Frac 1.0R vcr0);
  rewrite (ref_of_array_cell shcw (SZ.v lane16) |-> Frac 1.0R vcr0) as (rcw |-> Frac 1.0R vcr0);

  sdpa_flash_softmax_upd 16sz b hq sq sk
    #_ #_ #_
    #(ctlayout_slice
      (subtile_layout lS 1 16 (SZ.v lane16) 0)
      #(c_subtile_layout lS #cS 1 16 (SZ.v lane16) 0) 0 0)
    #(ctlayout_slice
      (subtile_layout lP 1 16 (SZ.v lane16) 0)
      #(c_subtile_layout lP #cP 1 16 (SZ.v lane16) 0) 0 0)
    #solve
    (mrow (array2_subtile shS 1 16 (SZ.v lane16) 0) 0)
    (mrow (array2_subtile shP 1 16 (SZ.v lane16) 0) 0)
    gmask rm rl rcw bi qh qpos k0 cbound row_active causal has_mask scale;

  with vSr. assert ((mrow (array2_subtile shS 1 16 (SZ.v lane16) 0) 0
                     <: array1 et_acc (mrow_layout (array2_subtile shS 1 16 (SZ.v lane16) 0) 0))
                    |-> Frac 1.0R vSr);
  rewrite ((mrow (array2_subtile shS 1 16 (SZ.v lane16) 0) 0
            <: array1 et_acc (mrow_layout (array2_subtile shS 1 16 (SZ.v lane16) 0) 0)) |-> Frac 1.0R vSr)
       as ((mrow (array2_subtile shS 1 16 (SZ.v lane16) 0) 0
            <: array1 et_acc (mrow_layout (array2_subtile shS 1 16 (SZ.v lane16) 0) 0)) |-> Frac 1.0R (tr_val (chest1_to_seq vSr)));
  elim_forall (chest1_to_seq vSr);
  Trade.elim_trade ((mrow (array2_subtile shS 1 16 (SZ.v lane16) 0) 0
            <: array1 et_acc (mrow_layout (array2_subtile shS 1 16 (SZ.v lane16) 0) 0)) |-> Frac 1.0R (tr_val (chest1_to_seq vSr))) _;

  with vPr. assert ((mrow (array2_subtile shP 1 16 (SZ.v lane16) 0) 0
                     <: array1 et_ab (mrow_layout (array2_subtile shP 1 16 (SZ.v lane16) 0) 0))
                    |-> Frac 1.0R vPr);
  rewrite ((mrow (array2_subtile shP 1 16 (SZ.v lane16) 0) 0
            <: array1 et_ab (mrow_layout (array2_subtile shP 1 16 (SZ.v lane16) 0) 0)) |-> Frac 1.0R vPr)
       as ((mrow (array2_subtile shP 1 16 (SZ.v lane16) 0) 0
            <: array1 et_ab (mrow_layout (array2_subtile shP 1 16 (SZ.v lane16) 0) 0)) |-> Frac 1.0R (tr_val (chest1_to_seq vPr)));
  elim_forall (chest1_to_seq vPr);
  Trade.elim_trade ((mrow (array2_subtile shP 1 16 (SZ.v lane16) 0) 0
            <: array1 et_ab (mrow_layout (array2_subtile shP 1 16 (SZ.v lane16) 0) 0)) |-> Frac 1.0R (tr_val (chest1_to_seq vPr))) _;

  with vmr. assert (rm |-> Frac 1.0R vmr);
  rewrite (rm |-> Frac 1.0R vmr) as (ref_of_array_cell shm (SZ.v lane16) |-> Frac 1.0R vmr);
  array1_cell_from_ref shm (SZ.v lane16);
  with vlr. assert (rl |-> Frac 1.0R vlr);
  rewrite (rl |-> Frac 1.0R vlr) as (ref_of_array_cell shl (SZ.v lane16) |-> Frac 1.0R vlr);
  array1_cell_from_ref shl (SZ.v lane16);
  with vcr. assert (rcw |-> Frac 1.0R vcr);
  rewrite (rcw |-> Frac 1.0R vcr) as (ref_of_array_cell shcw (SZ.v lane16) |-> Frac 1.0R vcr);
  array1_cell_from_ref shcw (SZ.v lane16);

  fold (row_subtile shS (SZ.v lane16));
  fold (row_subtile shP (SZ.v lane16));
  fold (cell_full shcw (SZ.v lane16));
  fold (cell_full shm (SZ.v lane16));
  fold (cell_full shl (SZ.v lane16));
}

(* Guarded wrapper: run the online-softmax update on the [< 16] active lanes and
   no-op on the rest.  It owns the branch: with an *explicit* pre==post the two
   arms are each checked against a fixed (non-dependent) [when__] frame, avoiding
   the fragile automatic join that mixes the near-identical [when__] payloads. *)
inline_for_extraction noextract
fn sdpa_flash_softmax_maybe
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (b hq sq sk : szp)
  (#lgmask : TRO.vlayout4 b hq sq sk) {| TRO.cvtlayout lgmask |}
  (#lcw #lm #ll : layout1 16) {| ctlayout lcw |} {| ctlayout lm |} {| ctlayout ll |}
  (#lS #lP : layout2 16 16)
  {| ctlayout lS |} {| ctlayout lP |}
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shcw : array1 et_acc lcw)
  (shm : array1 et_acc lm)
  (shl : array1 et_acc ll)
  (gmask : TRO.roarray4 et_ab lgmask)
  (lane : szlt warp_size)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq)
  (k0 : sz) (#_ : squash (SZ.fits (SZ.v k0 + 16)))
  (cbound : sz) (row_active : bool) (causal : bool) (has_mask : bool) (scale : et_acc)
  (#fmask : perm)
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  preserves (gmask |-> Frac fmask emask)
  requires
    when__ (SZ.v lane < 16) (fun _ -> row_subtile shS (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> row_subtile shP (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> cell_full shcw (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> cell_full shm (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> cell_full shl (SZ.v lane))
  ensures
    when__ (SZ.v lane < 16) (fun _ -> row_subtile shS (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> row_subtile shP (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> cell_full shcw (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> cell_full shm (SZ.v lane))
    ** when__ (SZ.v lane < 16) (fun _ -> cell_full shl (SZ.v lane))
{
  let active = lane <^ 16sz;
  if active {
    when__elim_true (SZ.v lane < 16) (fun _ -> row_subtile shS (SZ.v lane));
    when__elim_true (SZ.v lane < 16) (fun _ -> row_subtile shP (SZ.v lane));
    when__elim_true (SZ.v lane < 16) (fun _ -> cell_full shcw (SZ.v lane));
    when__elim_true (SZ.v lane < 16) (fun _ -> cell_full shm (SZ.v lane));
    when__elim_true (SZ.v lane < 16) (fun _ -> cell_full shl (SZ.v lane));

    sdpa_flash_softmax_active b hq sq sk shS shP shcw shm shl gmask
      (szlt16 lane) bi qh qpos k0 cbound row_active causal has_mask scale;

    when__intro_true (SZ.v lane < 16) (fun _ -> row_subtile shS (SZ.v lane));
    when__intro_true (SZ.v lane < 16) (fun _ -> row_subtile shP (SZ.v lane));
    when__intro_true (SZ.v lane < 16) (fun _ -> cell_full shcw (SZ.v lane));
    when__intro_true (SZ.v lane < 16) (fun _ -> cell_full shm (SZ.v lane));
    when__intro_true (SZ.v lane < 16) (fun _ -> cell_full shl (SZ.v lane));
  } else {
    assert pure ((SZ.v lane < 16) == false);
    when__elim_false (SZ.v lane < 16) (fun _ -> row_subtile shS (SZ.v lane));
    when__elim_false (SZ.v lane < 16) (fun _ -> row_subtile shP (SZ.v lane));
    when__elim_false (SZ.v lane < 16) (fun _ -> cell_full shcw (SZ.v lane));
    when__elim_false (SZ.v lane < 16) (fun _ -> cell_full shm (SZ.v lane));
    when__elim_false (SZ.v lane < 16) (fun _ -> cell_full shl (SZ.v lane));

    when__intro_false (SZ.v lane < 16) (fun _ -> row_subtile shS (SZ.v lane));
    when__intro_false (SZ.v lane < 16) (fun _ -> row_subtile shP (SZ.v lane));
    when__intro_false (SZ.v lane < 16) (fun _ -> cell_full shcw (SZ.v lane));
    when__intro_false (SZ.v lane < 16) (fun _ -> cell_full shm (SZ.v lane));
    when__intro_false (SZ.v lane < 16) (fun _ -> cell_full shl (SZ.v lane));
  }
}

inline_for_extraction noextract
fn sdpa_flash_jt_body
  (#et_ab #et_acc : Type0)
  {| scalar et_acc |} {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  {| FC.float_cast et_acc et_ab |}
  (d sk : szp) (b hq sq : szp)
  (#lgK #lgV : layout2 (SZ.v sk) (SZ.v d))
  (#lgmask : TRO.vlayout4 b hq sq sk)
  (#lcw #lm #ll : layout1 16)
  (#lK #lV : layout2 16 (SZ.v d))
  (#lS #lP #lPVc : layout2 16 16)
  (#lO : layout2 16 (SZ.v d))
  (#_ : squash (16 /?+ SZ.v d))
  (#_ : squash (warp_row_span /?+ 16))
  (#_ : squash (SZ.fits (16 * SZ.v d)))
  (#_ : squash (SZ.fits (tlayout_ulen lcw)))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (valid_frag_et_dims et_ab FragA 16 16 16))
  (#_ : squash (valid_frag_et_dims et_ab FragB 16 16 16))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc 16 16 16))
  {| ctlayout lgK |} {| ctlayout lgV |} {| TRO.cvtlayout lgmask |}
  {| ctlayout lcw |} {| ctlayout lm |} {| ctlayout ll |}
  {| ctlayout lK |} {| ctlayout lV |} {| ctlayout lS |}
  {| ctlayout lP |} {| ctlayout lPVc |} {| ctlayout lO |}
  {| strided_row_major (vtlayout_of_tlayout lK) |} {| strided_row_major (vtlayout_of_tlayout lV) |}
  {| strided_row_major (vtlayout_of_tlayout lS) |} {| strided_row_major (vtlayout_of_tlayout lP) |}
  {| strided_row_major (vtlayout_of_tlayout lPVc) |}
  (lane : szlt warp_size) (nthr : szp) (tid : szlt nthr)
  (shK : array2 et_ab lK)
  (shV : array2 et_ab lV)
  (shS : array2 et_acc lS)
  (shP : array2 et_ab lP)
  (shO : array2 et_acc lO)
  (shQ : array2 et_ab (l2_row_major 16 (SZ.v d)))
  (shPVc : array2 et_acc lPVc)
  (shcw : array1 et_acc lcw)
  (shm : array1 et_acc lm)
  (shl : array1 et_acc ll)
  (gK : array2 et_ab lgK { Kuiper.Tensor.is_global gK })
  (gV : array2 et_ab lgV { Kuiper.Tensor.is_global gV })
  (gmask : TRO.roarray4 et_ab lgmask)
  (bi : szlt b) (qh : szlt hq) (qpos : szlt sq)
  (k0 : sz) (#_ : squash (SZ.fits (SZ.v k0 + 16)))
  (cbound : sz) (row_active : bool) (causal : bool) (has_mask : bool) (scale : et_acc)
  (#fQ #fKg #fVg #fmask : perm)
  (#eQ : chest2 et_ab 16 (SZ.v d))
  (#eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (#emask : chest (b @| hq @| sq @| sk @| INil) et_ab)
  preserves thread_id (SZ.v nthr) tid
  requires pure (SZ.v lane == SZ.v tid % BW.warp_size)
  requires
    jt_rest #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask (SZ.v lane)
  ensures
    jt_rest #et_ab #et_acc d sk b hq sq shK shV shS shP shO shQ shPVc shcw shm shl
      gK gV gmask #fQ #fKg #fVg #fmask #eQ #eKg #eVg #emask (SZ.v lane)
{
  assert pure (SZ.v lane % BW.warp_size == SZ.v lane);

  (* K/V shared load (leaf uses residue [SZ.v lane]); then bridge to the barrier
     residue and run barrier 1 (K/V -> readable). *)
  sdpa_flash_kv_load 16sz d sk lane k0 gK gV shK shV;
  stride_reindex shK (SZ.v lane) (SZ.v lane % BW.warp_size);
  stride_reindex shV (SZ.v lane) (SZ.v lane % BW.warp_size);
  sdpa_flash_barrier1 d lane nthr tid shK shV;

  (* Q@K^T score matmul, then barrier 2 (K -> stride sub-tiles, S -> rows). *)
  sdpa_flash_qk_mm d d shQ (flash_row2col shK) shS;
  sdpa_flash_barrier2 d lane nthr tid shK shS;

  (* Bridge barrier-2 outputs (K stride, S row) back to the leaf residue. *)
  stride_reindex shK (SZ.v lane % BW.warp_size) (SZ.v lane);
  row_reindex shS (SZ.v lane % BW.warp_size) (SZ.v lane);

  (* Online-softmax update on the 16 active lanes (guarded wrapper owns the if). *)
  sdpa_flash_softmax_maybe b hq sq sk shS shP shcw shm shl gmask
    lane bi qh qpos k0 cbound row_active causal has_mask scale;

  (* Bridge S/P rows + cw cell to the barrier residue, run barrier 3
     (rows/cells -> whole tiles). *)
  row_reindex shS (SZ.v lane) (SZ.v lane % BW.warp_size);
  row_reindex shP (SZ.v lane) (SZ.v lane % BW.warp_size);
  cell_reindex shcw (SZ.v lane) (SZ.v lane % BW.warp_size);
  sdpa_flash_barrier3 lane nthr tid shS shP shcw;

  (* Rescale, barrier 4, P@V (all use residue [SZ.v lane]). *)
  sdpa_flash_scale d lane shO shcw;
  sdpa_flash_barrier4 lane nthr tid;
  sdpa_flash_pv_mm d d lane nthr tid shP shV shPVc shO;

  (* Barrier 5, then bridge its outputs (V stride, P row, cw cell) back to the
     resting residue [SZ.v lane]. *)
  sdpa_flash_barrier5 d lane nthr tid shV shP shcw;
  stride_reindex shV (SZ.v lane % BW.warp_size) (SZ.v lane);
  row_reindex shP (SZ.v lane % BW.warp_size) (SZ.v lane);
  cell_reindex shcw (SZ.v lane % BW.warp_size) (SZ.v lane);
}
