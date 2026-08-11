module Kuiops.SuperGEMM.Mm.Epi.Shared

(* C-augmented per-thread / per-block pre- and postconditions for the epilogue
   variant of the software-pipelined tensor-core GEMM.

   Everything about the shared memory, the pipeline and the staging is
   [Kuiops.SuperGEMM.Mm.Shared] verbatim; this module only layers a read-only
   C operand on top:

     - the pre gains a [Frac (fC /. (nblk * nthr))] read share of C per thread
       (C may be read non-injectively -- a broadcast view is exactly a
       non-injective [imap] -- so a whole-tensor read share is what is needed,
       not a per-lane partition);
     - the post's real target becomes [chest_comb comb_r] of the warp's C
       window and the warp's slice of [A @ B^T] instead of a [chest_map]. *)

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.Chest { chest2, chest_comb }
open Kuiops.SuperGEMM.Mm.Output { output_lane_live', output_lane_approximates' }
open Kuiops.SuperGEMM.Mm.Params
open Kuiops.SuperGEMM.Mm.Shared
open Kuiops.SuperGEMM.Mm.Epi.SharedGather { shared_gather, output_lane_approximates_sendable', shared_thread_final_sendable }
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul }
open Kuiops.SuperGEMM.Mm.Epi.EpilogueLemmas { lane_c_target }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module RO = Kuiper.TensorRO
module P = Kuiops.SuperGEMM.Mm.Params

(* ---- the per-lane real output target ----
   Same [bid]/[tid] decoding as [Shared.lane_target]; the [chest_map] over the
   warp's product is replaced by a [chest_comb] against the warp's C window
   ([lane_c_target] decodes [bid]/[tid] identically).  [coerce_eq] only
   reshapes the column count [nfrag wn * frag == 1 * wn]. *)
let lane_target_c
  (#m #n #k : szp)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (comb_r : real -> real -> real)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : chest2 real (mfrag wm * frag) (1 * SZ.v wn)
=
  let num_n = SZ.v n / SZ.v bn in
  let block_row = bid / num_n in
  let block_col = bid % num_n in
  let wnn = SZ.v bn / SZ.v wn in
  let wid = tid / warp_size in
  let warp_m = wid / wnn in
  let warp_n = wid % wnn in
  div_ub bid (SZ.v m / SZ.v bm) num_n;
  div_ub wid (SZ.v bm / SZ.v wm) wnn;
  grow_bound (SZ.v m) (SZ.v bm) (SZ.v wm) block_row warp_m;
  grow_bound (SZ.v n) (SZ.v bn) (SZ.v wn) block_col warp_n;
  Kuiper.Divides.lemma_divides_trans (SZ.v wm) (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_divides_trans (SZ.v wn) (SZ.v bn) (SZ.v n);
  let grow : natlt (SZ.v m / SZ.v wm) = block_row * (SZ.v bm / SZ.v wm) + warp_m in
  let gcol : natlt (SZ.v n / SZ.v wn) = block_col * (SZ.v bn / SZ.v wn) + warp_n in
  coerce_eq ()
    (chest_comb comb_r
      (lane_c_target rC bm bn wm wn nblk nthr bid tid)
      (warp_matmul rA rB (mfrag wm * frag) (nfrag wn * frag) grow gcol))

(* ---- global-side per-thread pre/post (sh-free) ---- *)

unfold
let kpre1_c
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid **
  (gC |-> Frac (fC /. (nblk * nthr)) eC)

unfold
let kpost1_c
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (comb_r : real -> real -> real)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB fC : perm)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= (gA |-> Frac (fA /. (nblk * nthr)) eA) **
  (gB |-> Frac (fB /. (nblk * nthr)) eB) **
  (gC |-> Frac (fC /. (nblk * nthr)) eC) **
  output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid
    (lane_target_c rC rA rB comb_r bm bn wm wn nblk nthr bid tid) **
  pure (aligned 16 (core gA)) **
  pure (aligned 16 (core gB)) **
  pure (aligned 16 (core gD))

(* ---- full per-thread pre/post: global side + shared ownership ---- *)

unfold
let kpre_c
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid **
  (gC |-> Frac (fC /. (nblk * nthr)) eC)

unfold
let kpost_c
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (comb_r : real -> real -> real)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (fA fB fC : perm)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk) (tid : natlt nthr)
  : slprop
= kpost1_c gA eA gB eB gC eC gD comb_r bm bn wm wn fA fB fC rC rA rB nblk nthr bid tid **
  shared_thread_final bm bn bk wm wn skew sh nthr (last_ktiles k bk ()) tid

(* ---- block-level pre/post (sh-free) ---- *)

unfold
let block_pre_c
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpre1_c gA eA gB eB gC eC gD bm bn wm wn fA fB fC nblk nthr bid tid

unfold
let block_post_c
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (comb_r : real -> real -> real)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB fC : perm)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk)
  : slprop
= forall+ (tid : natlt nthr).
    kpost1_c gA eA gB eB gC eC gD comb_r bm bn wm wn fA fB fC rC rA rB nblk nthr bid tid

(* ---- block setup / teardown ----
   [Mm.Shared.block_setup]/[block_teardown] do all the shared-memory work; the
   C read share is simply carried alongside, unzipped on the way in and zipped
   back on the way out. *)

#push-options "--split_queries no"
ghost
fn block_setup_c
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (bm bn bk wm wn skew : szp)
  (#sqc : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#sqd : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk)
  ()
  requires
    live_c_shmems sh **
    block_pre_c gA eA gB eB gC eC gD bm bn wm wn fA fB fC nblk nthr bid
  ensures
    (forall+ (tid : natlt nthr).
      kpre_c gA eA gB eB gC eC gD bm bn bk wm wn skew
        fA fB fC nblk nthr sh bid tid) **
    block_frame bm bn bk wm wn skew sh
{
  forevery_unzip #(natlt (SZ.v nthr))
    (fun tid -> kpre1 gA eA gB eB gD bm bn wm wn fA fB nblk nthr bid tid)
    (fun _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  block_setup gA eA gB eB gD bm bn bk wm wn skew #sqc #sqd fA fB nblk nthr sh bid ();
  forevery_zip #(natlt (SZ.v nthr))
    (fun tid ->
      kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid)
    (fun _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
}
#pop-options

#push-options "--split_queries no"
ghost
fn block_teardown_c
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (comb_r : real -> real -> real)
  (bm bn bk wm wn skew : szp)
  (#sqc : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#sqd : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                  SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (fA fB fC : perm)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (bid : natlt nblk)
  ()
  requires
    (forall+ (tid : natlt nthr).
      kpost_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew
        fA fB fC rC rA rB nblk nthr sh bid tid) **
    block_frame bm bn bk wm wn skew sh
  ensures
    live_c_shmems sh **
    block_post_c gA eA gB eB gC eC gD comb_r bm bn wm wn fA fB fC rC rA rB nblk nthr bid
{
  forevery_map #(natlt (SZ.v nthr))
    (fun tid ->
      kpost_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew
        fA fB fC rC rA rB nblk nthr sh bid tid)
    (fun tid ->
      kpost1_c gA eA gB eB gC eC gD comb_r bm bn wm wn fA fB fC rC rA rB nblk nthr bid tid **
      shared_thread_final bm bn bk wm wn skew sh nthr (last_ktiles k bk ()) tid)
    fn tid {
      unfold (kpost_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew
        fA fB fC rC rA rB nblk nthr sh bid tid);
    };
  forevery_unzip #(natlt (SZ.v nthr))
    (fun tid ->
      kpost1_c gA eA gB eB gC eC gD comb_r bm bn wm wn fA fB fC rC rA rB nblk nthr bid tid)
    (fun tid -> shared_thread_final bm bn bk wm wn skew sh nthr (last_ktiles k bk ()) tid);
  shared_gather bm bn bk wm wn skew #sqc nthr sh (last_ktiles k bk ()) ();
}
#pop-options

(* ------------------------------------------------------------------ *)
(* Block-sendability                                                   *)
(* ------------------------------------------------------------------ *)

let kpre_sendable_c
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC { RO.is_global gC }) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD { is_global gD })
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n))
  (fA fB fC : perm)
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpre_c gA eA gB eB gC eC gD bm bn bk wm wn skew
        fA fB fC nblk nthr sh bid tid)
=
  let base_send =
    kpre_sendable gA eA gB eB gD bm bn bk wm wn skew #_ #_
      fA fB nblk nthr sh #_ bid tid in
  let pC = gC |-> Frac (fC /. (nblk * nthr)) eC in
  let sendC : is_send_across block_of pC =
    send_across_if_send_across_gpu pC
      (RO.is_send_across_global_tensor gC #(fC /. (nblk * nthr)) eC) in
  is_send_across_star
    (kpre gA eA gB eB gD bm bn bk wm wn skew fA fB nblk nthr sh bid tid)
    pC
    #base_send #sendC

#push-options "--split_queries always"
let kpost1_sendable_c
  (#et_ab : Type0) {| scalar et_ab, has_vec_cpy et_ab |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC { RO.is_global gC }) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD { is_global gD })
  (comb_r : real -> real -> real)
  (bm bn wm wn : szp)
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (fA fB fC : perm)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of (
      kpost1_c gA eA gB eB gC eC gD comb_r bm bn wm wn
        fA fB fC rC rA rB nblk nthr bid tid)
=
  (* As in [Mm.Shared.kpost1_sendable]: bind the global read shares and their
     sendability before any real-approximation fact enters scope. *)
  let pA = gA |-> Frac (fA /. (nblk * nthr)) eA in
  let pB = gB |-> Frac (fB /. (nblk * nthr)) eB in
  let pC = gC |-> Frac (fC /. (nblk * nthr)) eC in
  let pAlignedA = pure (aligned 16 (core gA)) in
  let pAlignedB = pure (aligned 16 (core gB)) in
  let pAlignedD = pure (aligned 16 (core gD)) in
  let pPure = pAlignedA ** pAlignedB ** pAlignedD in
  let sendA : is_send_across block_of pA =
    send_across_if_send_across_gpu pA
      (is_send_across_global_tensor gA #(fA /. (nblk * nthr)) eA) in
  let sendB : is_send_across block_of pB =
    send_across_if_send_across_gpu pB
      (is_send_across_global_tensor gB #(fB /. (nblk * nthr)) eB) in
  let sendC : is_send_across block_of pC =
    send_across_if_send_across_gpu pC
      (RO.is_send_across_global_tensor gC #(fC /. (nblk * nthr)) eC) in
  let sendAlignedA : is_send_across block_of pAlignedA =
    is_send_across_placeless pAlignedA
      #(placeless_pure (aligned 16 (core gA))) in
  let sendAlignedB : is_send_across block_of pAlignedB =
    is_send_across_placeless pAlignedB
      #(placeless_pure (aligned 16 (core gB))) in
  let sendAlignedD : is_send_across block_of pAlignedD =
    is_send_across_placeless pAlignedD
      #(placeless_pure (aligned 16 (core gD))) in
  let sendPure =
    is_send_across_star pAlignedA (pAlignedB ** pAlignedD)
      #sendAlignedA
      #(is_send_across_star pAlignedB pAlignedD #sendAlignedB #sendAlignedD) in
  let rD = lane_target_c rC rA rB comb_r bm bn wm wn nblk nthr bid tid in
  FStar.Math.Lemmas.lemma_div_exact (SZ.v wm) frag;
  let output_send =
    output_lane_approximates_sendable' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 #_
      nblk nthr bid tid rD in
  let pOutput =
    output_lane_approximates' gD (SZ.v bm) (SZ.v bn) frag (SZ.v wn) (mfrag wm) 1 bid tid rD in
  let sendOutputPure =
    is_send_across_star pOutput pPure #output_send #sendPure in
  let sendCOutput =
    is_send_across_star pC (pOutput ** pPure) #sendC #sendOutputPure in
  let sendBC =
    is_send_across_star pB (pC ** pOutput ** pPure) #sendB #sendCOutput in
  is_send_across_star pA (pB ** pC ** pOutput ** pPure) #sendA #sendBC
#pop-options

let kpost_sendable_c
  (#et_ab #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, scalar et_acc, has_vec_cpy et_acc |}
  (#et_c : Type0) {| scalar et_c |}
  (#et_d : Type0) {| scalar et_d, has_vec_cpy et_d, real_like et_d |}
  (#m #n #k : szp)
  (#lA : layout2 (SZ.v m) (SZ.v k))
  (#lB : layout2 (SZ.v n) (SZ.v k))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strided_row_major (vtlayout_of_tlayout lD) |}
  (gA : array2 et_ab lA { is_global gA }) (eA : chest2 et_ab (SZ.v m) (SZ.v k))
  (gB : array2 et_ab lB { is_global gB }) (eB : chest2 et_ab (SZ.v n) (SZ.v k))
  (gC : RO.roarray2 et_c lC { RO.is_global gC }) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD { is_global gD })
  (comb_r : real -> real -> real)
  (bm bn bk wm wn skew : szp)
  (#_ : squash (constraints et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v bk /?+ SZ.v k /\ SZ.v bk <= SZ.v k))
  (fA fB fC : perm)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (sh : c_shmems (shmems_desc et_ab et_acc bm bn bk wm wn skew))
  (#_ : squash (c_shmems_inv sh))
  (bid : natlt nblk) (tid : natlt nthr)
  : is_send_across block_of
      (kpost_c gA eA gB eB gC eC gD comb_r bm bn bk wm wn skew
        fA fB fC rC rA rB nblk nthr sh bid tid)
=
  let base_send =
    kpost1_sendable_c gA eA gB eB gC eC gD comb_r bm bn wm wn #_
      fA fB fC rC rA rB nblk nthr bid tid in
  let shared_send =
    shared_thread_final_sendable bm bn bk wm wn skew nthr sh #_ (last_ktiles k bk ()) tid in
  is_send_across_star
    (kpost1_c gA eA gB eB gC eC gD comb_r bm bn wm wn
      fA fB fC rC rA rB nblk nthr bid tid)
    (shared_thread_final bm bn bk wm wn skew sh nthr (last_ktiles k bk ()) tid)
    #base_send #shared_send
