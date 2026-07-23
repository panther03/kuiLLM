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

module SZ = Kuiper.SizeT
module BW = Kuiper.Barrier.Warp

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
  (#et : Type0) {| scalar et, floating et |}
  (bn : szp)
  (b hq sq sk : szp)
  (#lshS #lshP : layout1 bn)
  (#lmask : layout4 b hq sq sk)
  {| ctlayout lshS, ctlayout lshP, ctlayout lmask |}
  (shS : array1 et lshS)
  (shP : array1 et lshP)
  (gmask : array4 et lmask)
  (shm shl shcw : ref et)
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
    shm |-> vm ** shl |-> vl ** shcw |-> vcw
  preserves
    (gmask |-> Frac fmask emask) ** live shS ** live shP
  ensures
    live shm ** live shl ** live shcw
{
  // Score loop: scale + mask, masking-out invalid keys to the -inf sentinel.
  let mut rowmax : et = neg infinity;
  let mut j : szle bn = 0sz;
  while (!j <^ bn)
    invariant
      (gmask |-> Frac fmask emask) ** live shS ** live shP **
      live j ** live rowmax
    decreases (bn - !j)
  {
    let vj = !j;
    let jj : szlt bn = vj;
    let kj = k0 +^ vj;
    let sc = tensor_read shS (cidx1 jj);
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
    shS.(cidx1 jj) <- s;
    rowmax := fmax !rowmax s;
    j := !j +^ 1sz;
  };

  // Online-softmax max/correction update (uses the OLD m still in shm).
  let m_old = !shm;
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
      (gmask |-> Frac fmask emask) ** live shS ** live shP **
      live j ** live rowsum
    decreases (bn - !j)
  {
    let vj = !j;
    let jj : szlt bn = vj;
    let sv = tensor_read shS (cidx1 jj);
    let p : et = sel_prob sv mnew;
    shP.(cidx1 jj) <- p;
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
   kernel (etc/tc_flash_attn_fa1.cu, lines 138-147).  A single warp computes one
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
  (shQ  : array2 et_ab (l2_row_major 16 hd))
  (shKT : array2 et_ab (l2_col_major hd 16))
  (shS  : array2 et_acc (l2_row_major 16 16))
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
unfold let warp_row_span : nat = SZ.v warp_size / 16
unfold let lane_row_span : nat = 16 / warp_row_span

inline_for_extraction noextract let warp_row_span_sz : sz = warp_size /^ 16sz
inline_for_extraction noextract let lane_row_span_sz : sz = 16sz /^ warp_row_span_sz

(* Memory-safety-only Kuiper port of the P@V tensor-core matmul plus the
   per-lane accumulation into [Osh] (etc/tc_flash_attn_fa1.cu, lines 194-209).
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
  (lane : szlt warp_size)
  (shP   : array2 et_ab  (l2_row_major 16 16))
  (shV   : array2 et_ab  (l2_row_major 16 hd))
  (shPVc : array2 et_acc (l2_row_major 16 16))
  (shO : array2 et_acc (l2_row_major 16 hd))
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
  preserves thread_id BW.warp_size (SZ.v lane)
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
  let cstr = c_stride_subtile_layout (l2_row_major 16 hd) warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16);
  tensor_pts_to_ref (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16));

  let pf    = __alloc_fragment et_ab  FragA   16sz 16sz 16sz FragLRM;
  let vf    = __alloc_fragment et_ab  FragB   16sz 16sz 16sz FragLRM;
  let pvacc = __alloc_fragment et_acc FragAcc 16sz 16sz 16sz FragLAcc;

  let njcol = d /^ 16sz;
  let mut jcol : sz = 0sz;
  while (!jcol <^ njcol)
    invariant
      live pf ** live vf ** live pvacc ** live jcol **
      thread_id BW.warp_size (SZ.v lane) **
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
      #(BW.warp_size) #(SZ.v lane);

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
      let pv  = tensor_read #_ #_ #_ #_ #(c_l2_row_major 16 16sz) shPVc (cidx2 prow tc);
      let old = tensor_read #_ #_ #_ #_ #cstr (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 orow ocol);
      tensor_write #_ #_ #_ #_ #cstr (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 orow ocol) (old `sc_acc.add` pv);
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
   (etc/tc_flash_attn_fa1.cu, lines 130-135).

   The CUDA loop [for idx = lane; idx < BN*D; idx += WARP] partitions the
   [bn * d] logical cells of the shared key tile across the [warp_size] lanes
   by flattened row-major residue: lane owns exactly the cells [idx] with
   [idx % warp_size == lane].  Over the [bn x d] tile that owned set is skewed
   (it is not a rectangular sub-tile whenever [gcd(warp_size, d) < d]), so we
   first *reshape* the tile -- a pure ghost view of the same storage -- into a
   [R x warp_size] matrix ([R = bn*d/warp_size]).  Under that view "residue
   == lane" becomes "column [lane]", a clean [array2_stride_subtile], which is
   the per-lane ownership this function consumes.

   The reshape is [tensor_apply_bij] instantiated with the standard flatten /
   unflatten index bijection, reusing Kuiper's proven [flatten]/[unflatten];
   no data moves.  There is NO functional spec, only memory safety: the global
   row index is clamped into [0, sk) so every read is unconditionally in
   bounds, exactly like the mask read in [sdpa_flash_softmax_upd]. *)

(* Refinement coercions across a proven-equal size (pure identity on values). *)
inline_for_extraction noextract
let szc (#m #n : nat) (#_ : squash (m == n)) (x : natlt m) : natlt n = x

inline_for_extraction noextract
let szc_sz (#m #n : nat) (#_ : squash (m == n)) (x : szlt m) : szlt n = x

(* [a < n*d] and [d>0] imply [a/d < n]. *)
let div_lt_lem (a n d : nat)
  : Lemma (requires d > 0 /\ a < n * d) (ensures a / d < n)
= if a / d >= n then begin
    FStar.Math.Lemmas.lemma_mult_le_right d n (a / d);
    FStar.Math.Lemmas.multiply_fractions a d
  end

(* The [bn*d]  <->  [R x warp_size] index bijection (both flatten to the same
   flat range of size [bn*d == R*warp_size]). *)
inline_for_extraction noextract
let kv_bij (bn d r : szp)
  (pf : squash (SZ.v bn * SZ.v d == SZ.v r * SZ.v warp_size))
  : (abs (bn @| d @| INil) =~ abs (r @| warp_size @| INil))
= mk_bijection
    (fun (i : abs (bn @| d @| INil)) ->
       unflatten (r @| warp_size @| INil) (szc (flatten (bn @| d @| INil) i)))
    (fun (i : abs (r @| warp_size @| INil)) ->
       unflatten (bn @| d @| INil) (szc (flatten (r @| warp_size @| INil) i)))
    (fun _ -> ())
    (fun _ -> ())

(* Concrete inverse of [kv_bij] (for the [ctlayout] of the reshaped view),
   written as direct [sz] arithmetic (unflatten of a row-major [bn x d]):
   flat index [p = a*warp_size + c] maps to [(p/d, p%d)]. *)
inline_for_extraction noextract
let kv_fconc (bn d r : szp)
  (pf   : squash (SZ.v bn * SZ.v d == SZ.v r * SZ.v warp_size))
  (fits : squash (SZ.fits (SZ.v bn * SZ.v d)))
  (x : conc (r @| warp_size @| INil))
  : conc (bn @| d @| INil)
= let (a, (c, _)) = x in
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v warp_size) (SZ.v a) (SZ.v r - 1);
  let p : szlt (bn *^ d) = a *^ warp_size +^ c in
  div_lt_lem (SZ.v p) (SZ.v bn) (SZ.v d);
  ((p /^ d, (p %^ d, ())))

inline_for_extraction noextract
let kv_fconc_correct (bn d r : szp)
  (pf   : squash (SZ.v bn * SZ.v d == SZ.v r * SZ.v warp_size))
  (fits : squash (SZ.fits (SZ.v bn * SZ.v d)))
  (x : conc (r @| warp_size @| INil))
  : squash (up (kv_fconc bn d r pf fits x) == (kv_bij bn d r pf).gg (up x))
= ()

inline_for_extraction noextract
instance kv_ctl (bn d r : szp)
  (pf   : squash (SZ.v bn * SZ.v d == SZ.v r * SZ.v warp_size))
  (fits : squash (SZ.fits (SZ.v bn * SZ.v d)))
  : ctlayout (tlayout_bij (kv_bij bn d r pf) (l2_row_major bn d))
= ctlayout_bij (kv_bij bn d r pf) (kv_fconc bn d r pf fits) (kv_fconc_correct bn d r pf fits)
    (l2_row_major bn d)

(* The [R x warp_size] reshaped view: same storage as [sh], different layout. *)
inline_for_extraction noextract
let kv_view (#et : Type0) (bn d r : szp)
  (pf : squash (SZ.v bn * SZ.v d == SZ.v r * SZ.v warp_size))
  (sh : array2 et (l2_row_major bn d))
  : array2 et (tlayout_bij (kv_bij bn d r pf) (l2_row_major bn d))
= from_array (tlayout_bij (kv_bij bn d r pf) (l2_row_major bn d)) (core sh)

(* Lane [lane]'s owned column of the reshaped view: a [R x 1] stride sub-tile. *)
inline_for_extraction noextract
let kv_col (#et : Type0) (bn d r : szp)
  (pf : squash (SZ.v bn * SZ.v d == SZ.v r * SZ.v warp_size))
  (sh : array2 et (l2_row_major bn d))
  (lane : szlt warp_size)
  : array2 et (stride_subtile_layout
                 (tlayout_bij (kv_bij bn d r pf) (l2_row_major bn d))
                 1 (SZ.v warp_size) 0 (SZ.v lane))
= array2_stride_subtile (kv_view bn d r pf sh) 1 (SZ.v warp_size) 0 (SZ.v lane)

fn sdpa_flash_kv_load
  (#et : Type0)
  (bn d r sk : szp)
  (lane : szlt warp_size)
  (k0base : sz)
  (#_ : squash (SZ.v bn * SZ.v d == SZ.v r * SZ.v warp_size))
  (#_ : squash (SZ.fits (SZ.v bn * SZ.v d)))
  (#_ : squash (SZ.fits (SZ.v k0base + SZ.v bn)))
  (#lK #lV : layout2 (SZ.v sk) (SZ.v d))
  {| ctlayout lK, ctlayout lV |}
  (gK : array2 et lK { Kuiper.Tensor.is_global gK })
  (gV : array2 et lV { Kuiper.Tensor.is_global gV })
  (shK shV : array2 et (l2_row_major bn d))
  (#fK #fV : perm)
  (#eK #eV : chest2 et (SZ.v sk) (SZ.v d))
  (#eKc0 : chest2 et (SZ.v r / 1) (SZ.v warp_size / SZ.v warp_size))
  (#eVc0 : chest2 et (SZ.v r / 1) (SZ.v warp_size / SZ.v warp_size))
  requires
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (kv_col bn d r () shK lane |-> Frac 1.0R eKc0) **
    (kv_col bn d r () shV lane |-> Frac 1.0R eVc0)
  ensures
    (gK |-> Frac fK eK) **
    (gV |-> Frac fV eV) **
    (exists* eKc. kv_col bn d r () shK lane |-> Frac 1.0R eKc) **
    (exists* eVc. kv_col bn d r () shV lane |-> Frac 1.0R eVc)
{
  let cstrK = c_stride_subtile_layout
    (tlayout_bij (kv_bij bn d r ()) (l2_row_major bn d)) #(kv_ctl bn d r () ())
    1 (SZ.v warp_size) 0 (SZ.v lane);

  let mut a : sz = 0sz;
  while (!a <^ r)
    invariant
      live a **
      (gK |-> Frac fK eK) **
      (gV |-> Frac fV eV) **
      (exists* eKc. kv_col bn d r () shK lane |-> Frac 1.0R eKc) **
      (exists* eVc. kv_col bn d r () shV lane |-> Frac 1.0R eVc) **
      pure (SZ.v !a <= SZ.v r)
    decreases (r - !a)
  {
    let va0 = !a;
    let va : szlt r = va0;
    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v warp_size) (SZ.v va) (SZ.v r - 1);
    let p : szlt (bn *^ d) = warp_size *^ va +^ lane;
    div_lt_lem (SZ.v p) (SZ.v bn) (SZ.v d);
    let j  : szlt bn = p /^ d;
    let dd : szlt d  = p %^ d;
    let kr : szlt sk = clamp_lt sk (k0base +^ j);

    let vk = tensor_read gK (cidx2 kr dd);
    tensor_write #_ #_ #_ #_ #cstrK (kv_col bn d r () shK lane) (cidx2 va 0sz) vk;
    let vv = tensor_read gV (cidx2 kr dd);
    tensor_write #_ #_ #_ #_ #cstrK (kv_col bn d r () shV lane) (cidx2 va 0sz) vv;
    a := !a +^ 1sz;
  };
  ()
}

(* ────────────────────────────────────────────────────────────────────────
   sdpa_flash_scale  --  online-softmax rescale of the output tile
   (etc/tc_flash_attn_fa1.cu, lines 190-191).

   The CUDA loop [for idx = lane; idx < BM*HD; idx += WARP] does the in-place
   row-broadcast multiply [Osh[idx] *= cw[idx / HD]] over the [bm x hd] output
   tile.  The per-lane ownership of the [O] tile is exactly the strided one of
   [sdpa_flash_kv_load], so we reuse the same [bm*hd <-> R x warp_size] reshape:
   lane owns column [lane] ([kv_col]) with read+write permission.

   The correction weights [cw] form a length-[bm] row vector; every lane reads
   the whole vector (each cell [i = idx/hd] it touches), so it is passed with a
   divided read-only fraction over the entire array -- no need to split it into
   per-lane cells since it is read exclusively.  Memory safety only. *)
fn sdpa_flash_scale
  (#et : Type0) {| scalar et |}
  (bm hd r : szp)
  (lane : szlt warp_size)
  (#_ : squash (SZ.v bm * SZ.v hd == SZ.v r * SZ.v warp_size))
  (#_ : squash (SZ.fits (SZ.v bm * SZ.v hd)))
  (#lcw : layout1 (SZ.v bm))
  {| ctlayout lcw |}
  (shO : array2 et (l2_row_major bm hd))
  (shcw : array1 et lcw)
  (#fcw : perm)
  (#ecw : chest1 et (SZ.v bm))
  (#eOc0 : chest2 et (SZ.v r / 1) (SZ.v warp_size / SZ.v warp_size))
  requires
    (shcw |-> Frac fcw ecw) **
    (kv_col bm hd r () shO lane |-> Frac 1.0R eOc0)
  ensures
    (shcw |-> Frac fcw ecw) **
    (exists* eOc. kv_col bm hd r () shO lane |-> Frac 1.0R eOc)
{
  let cstrO = c_stride_subtile_layout
    (tlayout_bij (kv_bij bm hd r ()) (l2_row_major bm hd)) #(kv_ctl bm hd r () ())
    1 (SZ.v warp_size) 0 (SZ.v lane);

  let mut a : sz = 0sz;
  while (!a <^ r)
    invariant
      live a **
      (shcw |-> Frac fcw ecw) **
      (exists* eOc. kv_col bm hd r () shO lane |-> Frac 1.0R eOc) **
      pure (SZ.v !a <= SZ.v r)
    decreases (r - !a)
  {
    let va0 = !a;
    let va : szlt r = va0;
    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v warp_size) (SZ.v va) (SZ.v r - 1);
    let p : szlt (bm *^ hd) = warp_size *^ va +^ lane;
    div_lt_lem (SZ.v p) (SZ.v bm) (SZ.v hd);
    let i : szlt bm = p /^ hd;

    let cwv = tensor_read shcw (cidx1 i);
    let ov  = tensor_read #_ #_ #_ #_ #cstrO (kv_col bm hd r () shO lane) (cidx2 va 0sz);
    tensor_write #_ #_ #_ #_ #cstrO (kv_col bm hd r () shO lane) (cidx2 va 0sz) (ov `mul` cwv);
    a := !a +^ 1sz;
  };
  ()
}
