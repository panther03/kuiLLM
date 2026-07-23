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
  (shK shV : array2 et (l2_row_major bn d))
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
  let cstr = c_stride_subtile_layout (l2_row_major bn d) warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16);
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
      tensor_write #_ #_ #_ #_ #cstr (array2_stride_subtile shK warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 arow bcol) vk;
      let vv = tensor_read gV (cidx2 kr dd);
      tensor_write #_ #_ #_ #_ #cstr (array2_stride_subtile shV warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 arow bcol) vv;
      b := !b +^ 1sz;
    };
    a := !a +^ 1sz;
  };
  ()
}

(* ────────────────────────────────────────────────────────────────────────
   sdpa_flash_scale  --  online-softmax rescale of the output tile
   (etc/tc_flash_attn_fa1.cu, lines 190-191).

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
fn sdpa_flash_scale
  (#et : Type0) {| scalar et |}
  (hd : szp)
  (#_ : squash (16 /?+ SZ.v hd))
  (#_ : squash (SZ.fits (16 * SZ.v hd)))
  (lane : szlt warp_size)
  (#lcw : layout1 16)
  {| ctlayout lcw |}
  (shO : array2 et (l2_row_major 16 hd))
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
  let cstr = c_stride_subtile_layout (l2_row_major 16 hd) warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16);
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
      let ov = tensor_read #_ #_ #_ #_ #cstr (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 vor oc);
      tensor_write #_ #_ #_ #_ #cstr (array2_stride_subtile shO warp_row_span 16 (SZ.v lane / 16) (SZ.v lane % 16)) (cidx2 vor oc) (ov `mul` cwv);
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
