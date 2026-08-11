module Kuiops.SuperGEMM.Mm.SplitK.Kernel

(* Split-K GEMM pass 1: partial products into an fp32 [(splits*m, n)] workspace.

   The grid is the non-split grid replicated [splits] times.  Kuiper's
   [kernel_desc] has a 1-D block index, so the [blockIdx.z] of the reference is
   flattened into [bid]; viewing the workspace as a [(splits*m, n)] matrix makes
   the ordinary row-major block decode [bid / (n/bn)] land on exactly the
   [(z, block_row)] pair, so the workspace ownership partition is the plain
   block/warp tiling of [gW] with no split-specific reindexing.

   Split [z] reduces the k-range [[z*ks, (z+1)*ks)] with [ks == k/splits]: A and
   B are sliced to that range with [array2_extract_tile_ro'] and [kloop] is
   reused verbatim at [k := ks].  Equal splits (rather than the reference's
   proportional floor division) are a deliberate strengthening: the whole
   development already requires [bm | m], [bn | n], [bk | k], and requiring
   [(splits*bk) | k] keeps every k-tile count exact.

   There is no epilogue: the accumulator and the workspace are both fp32, so
   [Store.store_warp_tile] drains the fragments straight to global memory. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorCore
open Kuiper.ForEvery
open Pulse.Lib.Array { length }

open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Kernel.Desc { kernel_desc }
open Kuiper.Tensor.Tiling { array2_extract_tile_ro', subtile_layout, array2_subtile }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile, warp_tile }
open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { warp_tile_pts_to, warp_tile_approximates }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.SplitK.Output { ws_warp_live, ws_warp_approximates,
  split_ws_to_warps, gather_ws_approximates }
open Kuiops.SuperGEMM.Mm.SplitK.WsLemmas { ws_target, ws_warp_target }
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState { epilogue_warp_input }
open Kuiops.SuperGEMM.Mm.SplitK.Store { store_warp_tile }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.Barrier
  { skewed_view, pipe_live, pipe_q, pipe_contract_c, pipe_p_to_q_transform_c }
open Kuiops.SuperGEMM.Mm.Stage { geo_ok }
open Kuiops.SuperGEMM.Mm.KLoop { kloop, acc_len_reveal, acc_len_alloc }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KLoop { populate_acc_with_zero }
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState { fragarrayAcc_approximates }
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module B = Kuiper.Barrier
module P = Kuiops.SuperGEMM.Mm.Params
module SH = Kuiops.SuperGEMM.Mm.SplitK.Shared
module ML = FStar.Math.Lemmas

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

let divides_helper
  (d : pos)
  (a b r c : int)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
  = Kuiper.Divides.lemma_divides_product_l d b r;
    Kuiper.Divides.lemma_divides_sum d a (b * r);
    Kuiper.Divides.lemma_divides_sum d (a + b * r) c

(* ---------------------------------------------------------------------- *)
(* pure arithmetic bridges                                                *)
(* ---------------------------------------------------------------------- *)

(* a < b*c /\ c > 0 ==> a/c < b. *)
let div_ub (a b c : nat)
  : Lemma (requires c > 0 /\ a < b * c) (ensures a / c < b)
= ML.lemma_div_mod a c;
  ML.lemma_mult_lt_left c (a / c) b

(* Row blocks of the replicated workspace split evenly across the splits. *)
let mws_row_blocks (m bm splits : pos)
  : Lemma (requires bm /?+ m)
          (ensures (splits * m) / bm == splits * (m / bm))
= ML.lemma_div_exact m bm;
  ML.paren_mul_right splits (m / bm) bm;
  ML.cancel_mul_div (splits * (m / bm)) bm

(* [out_ok] (the workspace warp-tile partition side conditions) from the
   ordinary tiling facts. *)
let out_tiling_facts (bm bn : pos) (wm wn : szp) (m n splits : pos)
  : Lemma
      (requires bm /?+ m /\ bn /?+ n /\ SZ.v wm /?+ bm /\ SZ.v wn /?+ bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn)
      (ensures SH.out_ok bm bn wm wn (splits * m) n)
=
  ML.lemma_div_exact (SZ.v wm) frag;
  ML.lemma_div_exact (SZ.v wn) frag;
  ML.lemma_div_exact m bm;
  ML.paren_mul_right splits (m / bm) bm;
  ML.cancel_mul_div (splits * (m / bm)) bm;
  ML.lemma_mod_mul_distr_l m splits bm;
  Kuiper.Divides.lemma_divides_trans frag (SZ.v wm) bm;
  Kuiper.Divides.lemma_divides_trans frag (SZ.v wn) bn

let bok_bounds (mws n bm bn nblk bid : nat)
  : Lemma
    (requires mws > 0 /\ n > 0 /\ bm > 0 /\ bn > 0 /\
              mws % bm == 0 /\ n % bn == 0 /\
              nblk == mws / bm * (n / bn) /\ bid < nblk)
    (ensures  n / bn > 0 /\ mws / bm > 0 /\
              bid / (n / bn) < mws / bm /\ bid % (n / bn) < n / bn)
= ML.lemma_div_mod mws bm;
  ML.lemma_div_mod n bn;
  div_ub bid (mws / bm) (n / bn);
  ML.lemma_mod_lt bid (n / bn)

let div_compose (m bm wm : pos)
  : Lemma (requires bm % wm == 0 /\ m % bm == 0)
          (ensures m / wm == (m / bm) * (bm / wm))
= let a = m / bm in let b = bm / wm in
  ML.lemma_div_exact bm wm;
  ML.lemma_div_exact m bm;
  assert (m == a * (b * wm));
  ML.paren_mul_right a b wm;
  ML.cancel_mul_div (a * b) wm

let grow_bound (m bm wm block_row warp_m : nat)
  : Lemma (requires wm > 0 /\ bm > 0 /\ m > 0 /\ bm % wm == 0 /\ m % bm == 0 /\
                    block_row < m / bm /\ warp_m < bm / wm)
          (ensures block_row * (bm / wm) + warp_m < m / wm)
= div_compose m bm wm; let b = bm / wm in
  ML.distributivity_add_left block_row 1 b;
  ML.lemma_mult_le_right b (block_row + 1) (m / bm)

let subtile_approximates
  (#et : Type0) {| scalar et, real_like et |} (#rows #cols : nat)
  (em : chest2 et rows cols) (rm : chest2 real rows cols)
  (trows : pos{trows /? rows}) (tcols : pos{tcols /? cols})
  (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
  : Lemma (requires em %~ rm)
          (ensures ematrix_subtile em trows tcols tr tc
                   %~ ematrix_subtile rm trows tcols tr tc)
= ()

(* ---------------------------------------------------------------------- *)
(* setup / teardown : GPU-level pre/post transforms                       *)
(* ---------------------------------------------------------------------- *)

#push-options "--split_queries no"
ghost
fn setup
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn wm wn : szp)
  (#_ : squash (SH.out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (#_ : squash (aligned 16 (core gA) /\ aligned 16 (core gB) /\ aligned 16 (core gW)))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  ()
  norewrite
  requires
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    live gW
  ensures
    (forall+ (bid : natlt nblk).
      SH.block_pre gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid) ** pure True
{
  let mfw = wm /^ frag_sz;
  let nfw = wn /^ frag_sz;
  assert pure (SZ.v mfw == mfrag wm /\ SZ.v nfw == nfrag wn);
  assert pure (SZ.v mfw * SZ.v frag_sz == SZ.v wm);
  assert pure (SZ.v nfw * SZ.v frag_sz == SZ.v wn);

  tensor_share_n gA (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_share_n gB (nblk * nthr);
  forevery_factor (nblk * nthr) nblk nthr
    (fun _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);

  split_ws_to_warps gW bm bn frag_sz frag_sz mfw nfw nblk nthr ();

  rewrite each (SZ.v frag_sz) as frag;
  rewrite each (SZ.v mfw) as (mfrag wm);
  rewrite each (SZ.v nfw) as (nfrag wn);

  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      ws_warp_live gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn) bid tid);
  forevery_zip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      ws_warp_live gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn) bid tid);

  forevery_map_2
    #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      (gB |-> Frac (fB /. (nblk * nthr)) eB **
       ws_warp_live gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn) bid tid))
    (fun bid tid ->
      SH.kpre1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr bid tid)
    fn bid tid {
      ();
    };
}
#pop-options

#push-options "--split_queries no"
ghost
fn teardown
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  {| real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gW : array2 et_acc lW)
  (bm bn wm wn : szp)
  (#_ : squash (SH.out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (#_ : squash (SZ.fits lW.ulen))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  ()
  requires
    (forall+ (bid : natlt nblk).
      SH.block_post gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid) ** pure True
  ensures
    (gA |-> Frac fA eA) **
    (gB |-> Frac fB eB) **
    (exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v n)).
       gW |-> eW ** pure (eW %~ rW))
{
  let mfw = wm /^ frag_sz;
  let nfw = wn /^ frag_sz;
  assert pure (SZ.v mfw == mfrag wm /\ SZ.v nfw == nfrag wn);
  assert pure (SZ.v mfw * SZ.v frag_sz == SZ.v wm);
  assert pure (SZ.v nfw * SZ.v frag_sz == SZ.v wn);

  forevery_map_2
    #(natlt nblk) #(natlt nthr)
    (fun bid tid ->
      SH.kpost1 gA eA gB eB gW bm bn wm wn fA fB nblk nthr rW bid tid)
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      (gB |-> Frac (fB /. (nblk * nthr)) eB **
       ws_warp_approximates gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn)
         bid tid rW))
    fn bid tid {
      ();
    };

  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      ws_warp_approximates gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn)
        bid tid rW);
  forevery_unzip_2
    #(natlt nblk) #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      ws_warp_approximates gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn)
        bid tid rW);

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_gather_n gA (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_gather_n gB (nblk * nthr);

  rewrite each (mfrag wm) as (SZ.v mfw);
  rewrite each (nfrag wn) as (SZ.v nfw);
  rewrite each frag as (SZ.v frag_sz);
  gather_ws_approximates gW bm bn frag_sz frag_sz mfw nfw nblk nthr rW ();
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* bcontract : the block barrier contract, at the split's k-range          *)
(* ---------------------------------------------------------------------- *)

let bcontract
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#m #n #k : szp)
  (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (bm bn bk wm wn skew : szp)
  (mws splits ks : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v mws == SZ.v splits * SZ.v m /\ SZ.v k == SZ.v splits * SZ.v ks /\
                SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\ SZ.v bk /?+ SZ.v ks))
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (bid : natlt nblk)
  (ptrs : c_shmems (SH.shmems_desc et_ab et_acc bm bn bk wm wn skew))
  : B.contract (SZ.v nthr)
=
  let num_n = SZ.v n / SZ.v bn in
  let num_m = SZ.v m / SZ.v bm in
  mws_row_blocks (SZ.v m) (SZ.v bm) (SZ.v splits);
  ML.lemma_div_mod (SZ.v n) (SZ.v bn);
  ML.lemma_div_mod (SZ.v m) (SZ.v bm);
  div_ub bid (SZ.v splits * num_m) num_n;
  let brg = bid / num_n in
  div_ub brg (SZ.v splits) num_m;
  ML.cancel_mul_div (SZ.v splits) (SZ.v ks);
  pipe_contract_c m n ks bm bn bk skew
    (ematrix_subtile eA (SZ.v m) (SZ.v ks) 0 (brg / num_m))
    (ematrix_subtile eB (SZ.v n) (SZ.v ks) 0 (brg / num_m))
    (brg % num_m) (bid % num_n)
    (SH.sar_a0 bm bn bk wm wn skew ptrs) (SH.sar_a1 bm bn bk wm wn skew ptrs)
    (SH.sar_b0 bm bn bk wm wn skew ptrs) (SH.sar_b1 bm bn bk wm wn skew ptrs)
    (SZ.v nthr) ()

(* ---------------------------------------------------------------------- *)
(* kf : one thread's kernel body                                          *)
(* ---------------------------------------------------------------------- *)

#push-options "--split_queries no --z3rlimit 20"
inline_for_extraction noextract
fn kf
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA) (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB) (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (gW : array2 et_acc lW)
  (rA : chest2 real (SZ.v m) (SZ.v k) { eA %~ rA })
  (rB : chest2 real (SZ.v n) (SZ.v k) { eB %~ rB })
  (bm bn bk wm wn skew : szp)
  (splits ks : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v mws == SZ.v splits * SZ.v m /\ SZ.v k == SZ.v splits * SZ.v ks))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn /\
                SZ.v bk /?+ SZ.v ks /\ SZ.v bk <= SZ.v ks))
  (#_ : squash (SH.out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)))
  (#_ : squash (SZ.fits (SZ.v m * SZ.v ks) /\ SZ.fits (SZ.v n * SZ.v ks) /\
                SZ.fits lA.ulen /\ SZ.fits lB.ulen /\ SZ.fits lW.ulen))
  (#_ : squash (is_global gA /\ is_global gB))
  (#_ : squash (aligned_strided_row_major (SZ.v (chunk et_ab)) str_A /\
                aligned_strided_row_major (SZ.v (chunk et_ab)) str_B /\
                SZ.v (chunk et_ab) /?+ SZ.v ks))
  (#_ : squash (geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn) /\
                geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA frag frag frag /\
                valid_frag_et_dims et_ab FragB frag frag frag /\
                valid_frag_et_dims et_acc FragAcc frag frag frag /\
                valid_frag_et_comb et_ab et_acc /\
                SZ.fits (SZ.v wm / frag * (SZ.v wn / frag))))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (SH.shmems_desc et_ab et_acc bm bn bk wm wn skew) { c_shmems_inv sh })
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
  requires
    gpu **
    SH.kpre gA eA gB eB gW bm bn bk wm wn skew fA fB nblk nthr sh bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok (bcontract eA eB bm bn bk wm wn skew mws splits ks nthr nblk (SZ.v bid) sh) **
    B.barrier_state 0
  ensures
    gpu **
    SH.kpost gA eA gB eB gW bm bn bk wm wn skew fA fB nblk nthr
      (SZ.v ks / SZ.v bk) sh
      (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ()) bid tid **
    thread_id (SZ.v nthr) (SZ.v tid) **
    block_id (SZ.v nblk) (SZ.v bid) **
    B.barrier_tok (bcontract eA eB bm bn bk wm wn skew mws splits ks nthr nblk (SZ.v bid) sh) **
    B.barrier_state (SZ.v ks / SZ.v bk)
{
  unfold SH.shared_thread_live bm bn bk wm wn skew sh nthr (SZ.v tid);
  SH.shared_buffers_aligned16 bm bn bk wm wn skew sh ();

  assert pure (mfrag wm * frag == SZ.v wm);
  assert pure (nfrag wn * frag == SZ.v wn);

  (* ---- index decode ----
     [gW] is the [(splits*m, n)] workspace, so the ordinary row-major block
     decode already carries the split index in the high part of the row-block
     coordinate: [bid / (n/bn) == z * (m/bm) + block_row]. *)
  let num_n = n /^ bn;
  let num_m = m /^ bm;
  mws_row_blocks (SZ.v m) (SZ.v bm) (SZ.v splits);
  assert pure (SZ.v mws / SZ.v bm == SZ.v splits * SZ.v num_m);
  div_ub (SZ.v bid) (SZ.v splits * SZ.v num_m) (SZ.v num_n);
  let brg : szlt (SZ.v mws / SZ.v bm) = bid /^ num_n;
  let block_col : szlt (SZ.v n / SZ.v bn) = bid %^ num_n;
  div_ub (SZ.v brg) (SZ.v splits) (SZ.v num_m);
  let z : szlt splits = brg /^ num_m;
  let block_row : szlt (SZ.v m / SZ.v bm) = brg %^ num_m;

  let wid = tid /^ warp_size;
  assert pure (SZ.v nthr == warps bm bn wm wn * SZ.v warp_size);
  div_ub (SZ.v tid) (warps bm bn wm wn) (SZ.v warp_size);
  let wnn = bn /^ wn;
  div_ub (SZ.v wid) (SZ.v bm / SZ.v wm) (SZ.v wnn);
  let warp_m : szlt (SZ.v bm / SZ.v wm) = wid /^ wnn;
  let warp_n : szlt (SZ.v bn / SZ.v wn) = wid %^ wnn;

  (* ---- slice A and B to this split's k-range ---- *)
  ML.cancel_mul_div (SZ.v splits) (SZ.v ks);
  let gA' = array2_extract_tile_ro' gA (SZ.v m) (SZ.v ks) 0 (SZ.v z);
  let gB' = array2_extract_tile_ro' gB (SZ.v n) (SZ.v ks) 0 (SZ.v z);
  subtile_approximates eA rA (SZ.v m) (SZ.v ks) 0 (SZ.v z);
  subtile_approximates eB rB (SZ.v n) (SZ.v ks) 0 (SZ.v z);

  (* ---- hoisted staging addressing ---- *)
  let nthrc = P.nthr_sz bm bn wm wn;
  let ch : szp = chunk et_ab;
  let a_t_row   = (tid *^ ch) /^ bk;
  let a_t_col   = (tid *^ ch) %^ bk;
  let a_row_step = (ch *^ nthrc) /^ bk;
  let a_iters   = (bm *^ bk) /^ (ch *^ nthrc);
  let b_iters   = (bn *^ bk) /^ (ch *^ nthrc);

  let accFrags = __alloc_array_fragment et_acc FragAcc frag_sz frag_sz frag_sz FragLAcc ((wm /^ frag_sz) *^ (wn /^ frag_sz));
  acc_len_alloc wm wn;
  acc_len_reveal wm wn;
  populate_acc_with_zero #et_acc frag_sz frag_sz frag_sz (wm /^ frag_sz) (wn /^ frag_sz) accFrags;
  rewrite each SZ.v (wm /^ frag_sz) as (SZ.v wm / frag);
  rewrite each SZ.v (wn /^ frag_sz) as (SZ.v wn / frag);

  grow_bound (SZ.v m) (SZ.v bm) (SZ.v wm) (SZ.v block_row) (SZ.v warp_m);
  grow_bound (SZ.v n) (SZ.v bn) (SZ.v wn) (SZ.v block_col) (SZ.v warp_n);
  let grow : (g:erased nat { reveal g < SZ.v m / SZ.v wm }) =
    hide (SZ.v block_row * (SZ.v bm / SZ.v wm) + SZ.v warp_m);
  let gcol : (g:erased nat { reveal g < SZ.v n / SZ.v wn }) =
    hide (SZ.v block_col * (SZ.v bn / SZ.v wn) + SZ.v warp_n);

  assert pure (SZ.v brg / SZ.v num_m == SZ.v z);
  assert pure (SZ.v brg % SZ.v num_m == SZ.v block_row);
  assert pure (SZ.v brg == SZ.v bid / (SZ.v n / SZ.v bn));
  assert pure (SZ.v block_col == SZ.v bid % (SZ.v n / SZ.v bn));
  rewrite (B.barrier_tok (bcontract eA eB bm bn bk wm wn skew mws splits ks nthr nblk (SZ.v bid) sh))
       as (B.barrier_tok
             (pipe_contract_c m n ks bm bn bk skew
                (ematrix_subtile eA (SZ.v m) (SZ.v ks) 0 (SZ.v z))
                (ematrix_subtile eB (SZ.v n) (SZ.v ks) 0 (SZ.v z))
                (SZ.v block_row) (SZ.v block_col)
                (SH.sar_a0 bm bn bk wm wn skew sh) (SH.sar_a1 bm bn bk wm wn skew sh)
                (SH.sar_b0 bm bn bk wm wn skew sh) (SH.sar_b1 bm bn bk wm wn skew sh)
                (SZ.v nthr) ()));

  let (sA0, (sA1, (sB0, (sB1, srest)))) = sh;
  assert rewrites_to sA0 (SH.sar_a0 bm bn bk wm wn skew sh);
  assert rewrites_to sA1 (SH.sar_a1 bm bn bk wm wn skew sh);
  assert rewrites_to sB0 (SH.sar_b0 bm bn bk wm wn skew sh);
  assert rewrites_to sB1 (SH.sar_b1 bm bn bk wm wn skew sh);

  Kuiper.Divides.lemma_divides_trans (SZ.v wm) (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_divides_trans (SZ.v wn) (SZ.v bn) (SZ.v n);
  (* the k-slice shifts each operand's base offset by [z*ks], which stays
     [chunk]-aligned because [chunk | bk | ks]. *)
  Kuiper.Divides.lemma_divides_product_r (SZ.v (chunk et_ab)) (SZ.v z) (SZ.v ks);
  divides_helper (SZ.v (chunk et_ab)) str_A.offset str_A.stride (0 * SZ.v m)
    (SZ.v z * SZ.v ks);
  divides_helper (SZ.v (chunk et_ab)) str_B.offset str_B.stride (0 * SZ.v n)
    (SZ.v z * SZ.v ks);
  kloop #et_ab #et_acc #_ #_ #_ #_ #_ #_ #m #n #ks bm bn bk wm wn skew
    gA' #(ematrix_subtile eA (SZ.v m) (SZ.v ks) 0 (SZ.v z))
    gB' #(ematrix_subtile eB (SZ.v n) (SZ.v ks) 0 (SZ.v z))
    sA0 sA1 sB0 sB1
    accFrags
    (fun x -> x)
    (fA /. (nblk * nthr)) (fB /. (nblk * nthr))
    nthr tid block_row block_col warp_m warp_n
    (ematrix_subtile rA (SZ.v m) (SZ.v ks) 0 (SZ.v z))
    (ematrix_subtile rB (SZ.v n) (SZ.v ks) 0 (SZ.v z))
    grow gcol
    a_t_row a_t_col a_row_step a_iters
    a_t_row a_t_col a_row_step b_iters
    () () () () () () () () () () ();

  rewrite (B.barrier_tok
             (pipe_contract_c m n ks bm bn bk skew
                (ematrix_subtile eA (SZ.v m) (SZ.v ks) 0 (SZ.v z))
                (ematrix_subtile eB (SZ.v n) (SZ.v ks) 0 (SZ.v z))
                (SZ.v block_row) (SZ.v block_col)
                sA0 sA1 sB0 sB1 (SZ.v nthr) ()))
       as (B.barrier_tok (bcontract eA eB bm bn bk wm wn skew mws splits ks nthr nblk (SZ.v bid) sh));

  (* release the A/B slices back to the whole operands *)
  Kuiper.TradeHelpers.ambig_trade_elim ();
  Kuiper.TradeHelpers.ambig_trade_elim ();

  (* ---- drain the accumulator straight into the workspace warp tile ---- *)
  unfold (fragarrayAcc_approximates (SZ.v wm / frag) (SZ.v wn / frag) accFrags
            (warp_matmul (ematrix_subtile rA (SZ.v m) (SZ.v ks) 0 (SZ.v z))
                         (ematrix_subtile rB (SZ.v n) (SZ.v ks) 0 (SZ.v z))
                         (SZ.v wm) (SZ.v wn) (reveal grow) (reveal gcol)));
  with ems. assert (accFrags |-> ems);

  unfold (ws_warp_live gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn)
            (SZ.v bid) (SZ.v tid));
  Kuiper.Divides.lemma_divides_product_r (SZ.v bm) (SZ.v splits) (SZ.v m);
  Kuiper.Divides.lemma_divides_product_r (SZ.v wm) (SZ.v splits) (SZ.v m);
  Kuiper.Divides.lemma_divides_product_r (SZ.v ks) (SZ.v splits) (SZ.v ks);
  ML.lemma_div_mod (SZ.v brg) (SZ.v num_m);
  assert pure (SZ.v brg == SZ.v z * (SZ.v m / SZ.v bm) + SZ.v block_row);
  assert pure (SZ.v brg < SZ.v mws / SZ.v bm);
  ws_warp_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB
    (SZ.v bm) (SZ.v bn) (SZ.v wm) (SZ.v wn)
    (SZ.v z) (SZ.v block_row) (SZ.v block_col) (SZ.v warp_m) (SZ.v warp_n) () ();
  with ews. assert (warp_tile_pts_to gW (SZ.v bm) (SZ.v bn) frag frag
                      (mfrag wm) (nfrag wn) (SZ.v bid) (SZ.v tid / warp_size) ews);
  unfold (warp_tile_pts_to gW (SZ.v bm) (SZ.v bn) frag frag
            (mfrag wm) (nfrag wn) (SZ.v bid) (SZ.v tid / warp_size) ews);

  let wtile = warp_tile (block_tile gW (SZ.v bm) (SZ.v bn) (SZ.v bid))
                (mfrag wm * frag) (nfrag wn * frag) (SZ.v tid / warp_size);
  rewrite each _ as wtile;

  store_warp_tile (wm /^ frag_sz) (wn /^ frag_sz) wtile accFrags
    (warp_matmul (ematrix_subtile rA (SZ.v m) (SZ.v ks) 0 (SZ.v z))
                 (ematrix_subtile rB (SZ.v n) (SZ.v ks) 0 (SZ.v z))
                 (SZ.v wm) (SZ.v wn) (reveal grow) (reveal gcol)) ();

  rewrite each wtile as _;
  with ews'. fold (warp_tile_pts_to gW (SZ.v bm) (SZ.v bn) frag frag
                     (mfrag wm) (nfrag wn) (SZ.v bid) (SZ.v tid / warp_size) ews');
  fold (warp_tile_approximates gW (SZ.v bm) (SZ.v bn) frag frag
          (mfrag wm) (nfrag wn) (SZ.v bid) (SZ.v tid / warp_size)
          (epilogue_warp_input (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ())
            (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn) (SZ.v bid) (SZ.v tid)));
  fold (ws_warp_approximates gW (SZ.v bm) (SZ.v bn) frag frag (mfrag wm) (nfrag wn)
          (SZ.v bid) (SZ.v tid)
          (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ()));

  with ems'. assert (accFrags |-> ems');
  drop_ (accFrags |-> ems');

  fold SH.shared_thread_final bm bn bk wm wn skew sh nthr (SZ.v ks / SZ.v bk) (SZ.v tid);
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* mk_kernel : assemble the [kernel_desc]                                  *)
(* ---------------------------------------------------------------------- *)

let geo_facts
  (et_ab et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (bm bn bk wm wn skew : szp)
  : Lemma
      (requires
        P.constraints et_ab et_acc bm bn bk wm wn skew /\
        (SZ.v (chunk et_ab) * P.nthr bm bn wm wn) % SZ.v bk == 0)
      (ensures
        geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn) /\
        geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn))
=
  P.nthr_pos bm bn wm wn;
  P.chunk_nthr_divides_ab et_ab et_acc bm bn bk wm wn skew

#push-options "--fuel 1 --ifuel 1 --z3rlimit 15 --split_queries always"
inline_for_extraction noextract
let mk_kernel
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab,
     scalar et_acc, has_vec_cpy et_acc, real_like et_acc |}
  (#m #n #k #mws : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k)) {| T.ctlayout lA |}
       {| str_A : strided_row_major (vtlayout_of_tlayout lA) |}
  (gA : array2 et_ab lA { is_global gA }) (#eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k)) {| T.ctlayout lB |}
       {| str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (gB : array2 et_ab lB { is_global gB }) (#eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (gW : array2 et_acc lW { is_global gW })
  (rA : chest2 real (SZ.v m) (SZ.v k) { eA %~ rA })
  (rB : chest2 real (SZ.v n) (SZ.v k) { eB %~ rB })
  (bm bn bk wm wn skew : szp)
  (splits ks : szp)
  (#sqc : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#sq_ws : squash (SZ.v mws == SZ.v splits * SZ.v m /\ SZ.v k == SZ.v splits * SZ.v ks))
  (#sq_div : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn /\
                SZ.v bk /?+ SZ.v ks /\ SZ.v bk <= SZ.v ks))
  (#sq_fits : squash (SZ.fits (SZ.v m * SZ.v ks) /\ SZ.fits (SZ.v n * SZ.v ks) /\
                SZ.fits lA.ulen /\ SZ.fits lB.ulen /\ SZ.fits lW.ulen))
  (#sq_al16 : squash (aligned 16 (core gA) /\ aligned 16 (core gB) /\ aligned 16 (core gW)))
  (#sq_asAB : squash (aligned_strided_row_major (SZ.v (chunk et_ab)) str_A /\
                aligned_strided_row_major (SZ.v (chunk et_ab)) str_B /\
                SZ.v (chunk et_ab) /?+ SZ.v ks))
  (#sq_cnbk : squash ((SZ.v (chunk et_ab) * P.nthr bm bn wm wn) % SZ.v bk == 0))
  (#sq_vf : squash (valid_frag_et_dims et_ab FragA frag frag frag /\
                valid_frag_et_dims et_ab FragB frag frag frag /\
                valid_frag_et_dims et_acc FragAcc frag frag frag /\
                valid_frag_et_comb et_ab et_acc /\
                SZ.fits (SZ.v wm / frag * (SZ.v wn / frag))))
  (fA fB : perm)
  (nblk : szp{SZ.v nblk == SZ.v mws / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (#_ : squash (SZ.v nblk <= SZ.v max_blocks))
  ()
  : kernel_desc
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       live gW)
      (gA |-> Frac fA eA **
       gB |-> Frac fB eB **
       (exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v n)).
          gW |-> eW **
          pure (eW %~ ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ())))
=
  geo_facts et_ab et_acc bm bn bk wm wn skew;
  P.nthr_pos bm bn wm wn;
  P.nthr_le_max_threads et_ab et_acc bm bn bk wm wn skew;
  P.bm_ldt_fits et_ab et_acc bm bn bk wm wn skew;
  out_tiling_facts (SZ.v bm) (SZ.v bn) wm wn (SZ.v m) (SZ.v n) (SZ.v splits);
  mws_row_blocks (SZ.v m) (SZ.v bm) (SZ.v splits);
  ML.cancel_mul_div (SZ.v splits) (SZ.v ks);
  let sq_out : squash (SH.out_ok (SZ.v bm) (SZ.v bn) wm wn (SZ.v mws) (SZ.v n)) = () in
  let sq_bcon : squash (SZ.v mws == SZ.v splits * SZ.v m /\ SZ.v k == SZ.v splits * SZ.v ks /\
                        SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                        SZ.v bk /?+ SZ.v ks) = () in
  let sq_lw : squash (SZ.fits lW.ulen) = () in
  let sq_al : squash (aligned 16 (core gA) /\ aligned 16 (core gB) /\ aligned 16 (core gW)) = () in
  let sq_glob : squash (is_global gA /\ is_global gB) = () in
  let sq_geo : squash
      (geo_ok (SZ.v bm) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn) /\
       geo_ok (SZ.v bn) (SZ.v bk) (SZ.v (chunk et_ab)) (P.nthr bm bn wm wn)) = () in
  {
    nblk;
    nthr;

    shmems_desc = SH.shmems_desc et_ab et_acc bm bn bk wm wn skew;

    kpre  = (fun sh bid tid ->
      SH.kpre gA eA gB eB gW bm bn bk wm wn skew #sqc #sq_out fA fB nblk nthr sh bid tid);
    kpost = (fun sh bid tid ->
      SH.kpost gA eA gB eB gW bm bn bk wm wn skew #sqc #sq_out fA fB nblk nthr
        (SZ.v ks / SZ.v bk) sh
        (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ()) bid tid);

    barrier_contract = (fun _bid ptrs ->
      bcontract eA eB bm bn bk wm wn skew mws splits ks #sqc #sq_bcon nthr nblk _bid ptrs);
    barrier_count = (fun _bid -> SZ.v ks / SZ.v bk);
    barrier_ok = (fun _bid ptrs ->
      let num_n = SZ.v n / SZ.v bn in
      let num_m = SZ.v m / SZ.v bm in
      bok_bounds (SZ.v mws) (SZ.v n) (SZ.v bm) (SZ.v bn) (SZ.v nblk) _bid;
      div_ub (_bid / num_n) (SZ.v splits) num_m;
      pipe_p_to_q_transform_c m n ks bm bn bk skew
        (ematrix_subtile eA (SZ.v m) (SZ.v ks) 0 (_bid / num_n / num_m))
        (ematrix_subtile eB (SZ.v n) (SZ.v ks) 0 (_bid / num_n / num_m))
        (_bid / num_n % num_m) (_bid % num_n)
        (SH.sar_a0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_a1 bm bn bk wm wn skew #sqc ptrs)
        (SH.sar_b0 bm bn bk wm wn skew #sqc ptrs) (SH.sar_b1 bm bn bk wm wn skew #sqc ptrs)
        (SZ.v nthr) ());

    frame = pure True;

    block_pre  = (fun bid ->
      SH.block_pre gA eA gB eB gW bm bn wm wn #sq_out fA fB nblk nthr bid);
    block_post = (fun bid ->
      SH.block_post gA eA gB eB gW bm bn wm wn #sq_out fA fB nblk nthr
        (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ()) bid);

    setup = setup #_ #et_acc #_ gA eA gB eB gW bm bn wm wn #sq_out #sq_al fA fB nblk nthr;
    teardown = teardown #_ #et_acc #_ gA eA gB eB gW bm bn wm wn #sq_out #sq_lw fA fB nblk nthr
      (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ());

    block_frame = (fun ptrs _bid -> SH.block_frame bm bn bk wm wn skew #sqc ptrs);
    block_setup = (fun sh bid ->
      SH.block_setup gA eA gB eB gW bm bn bk wm wn skew #sqc #sq_out fA fB nblk nthr sh bid);
    block_teardown = (fun sh bid ->
      SH.block_teardown gA eA gB eB gW bm bn bk wm wn skew #sqc #sq_out fA fB nblk nthr
        (SZ.v ks / SZ.v bk) sh
        (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ()) bid);

    f = kf gA #eA gB #eB gW rA rB bm bn bk wm wn skew splits ks
          #sqc #sq_ws #sq_div #sq_out #sq_fits #sq_glob #sq_asAB #sq_geo #sq_vf
          fA fB nblk nthr;

    block_pre_sendable = solve;
    block_post_sendable = solve;
    kpre_sendable = (fun sh sh_inv bid tid ->
      SH.kpre_sendable gA eA gB eB gW bm bn bk wm wn skew #sqc #sq_out
        fA fB nblk nthr sh #sh_inv bid tid);
    kpost_sendable = (fun sh sh_inv bid tid ->
      SH.kpost_sendable gA eA gB eB gW bm bn bk wm wn skew #sqc #sq_out
        fA fB nblk nthr (SZ.v ks / SZ.v bk) sh #sh_inv
        (ws_target (SZ.v mws) (SZ.v splits) (SZ.v ks) rA rB ()) bid tid);
  }
#pop-options
