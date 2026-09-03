module Kuiops.HReducePoly.Exact

(* Exact reduction over the innermost dimension of a tensor. Each outer
   (batch) index is reduced by one CUDA block. *)

#lang-pulse

open Pulse.Lib.Pledge
open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Functions
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg
open Kuiper.Bijection { ( =~ ) }
open Kuiops.HReducePoly.Spec

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier
module ML = FStar.Math.Lemmas

(* The tree-reduction invariant: a shared-memory cell holds exactly the
   reduction of the range of per-thread partials it covers. *)
let exact_red (#et:Type0) (#n:nat) (f : et -> et -> et) (parts : lseq et n)
  (i j : nat) (x : et) : prop
  = i < j /\ j <= n ==> x == rfold1 f (Seq.slice parts i j)

let exact_red_comb (#et:Type0) (#n:nat)
  (f : (et -> et -> et) { is_associative f }) (parts : lseq et n)
  (i j k : nat) (x y : et)
  : Lemma (requires i < j /\ j < k /\ exact_red f parts i j x /\ exact_red f parts j k y)
          (ensures exact_red f parts i k (f x y))
  = if k <= n then begin
      lem_append_slice parts i j k;
      rfold1_append f (Seq.slice parts i j) (Seq.slice parts j k)
    end

(* ------------------------------------------------------------------ *)
(* Kernel spec plumbing.                                               *)
(* ------------------------------------------------------------------ *)


unfold
let kpre
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (output : tensor et_o lout)
  (va : chest (snoc_shape d cols) et_i)
  (vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = a |-> Frac ((1 /. rows) /. nth) va **
    if_ (bool_eq #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)) **
    exists* (v : et).
      tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

unfold
let kpost
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : et -> et -> et)
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (output : tensor et_o lout)
  (va : chest (snoc_shape d cols) et_i)
  (vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = a |-> Frac ((1 /. rows) /. nth) va **
    if_ (bool_eq #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid)
    )

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn kf
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (index : conc d -> szlt cols -> conc (snoc_shape d cols))
  (index_up : (i:conc d -> j:szlt cols ->
    Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin)
  (output : tensor et_o lout)
  (va : chest (snoc_shape d cols) et_i)
  (vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    pure (c_shmems_inv shmem) **
    kpre f pre_map post_map rows cols nth a output va vout shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix
      (exact_red f (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid))))
      nth (from_array (l1_forward nth) shmem._1)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost f pre_map post_map rows cols nth a output va vout shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix
      (exact_red f (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid))))
      nth (from_array (l1_forward nth) shmem._1)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre f pre_map post_map rows cols nth a output va vout shmem bid tid;
  let (gsa, _) = shmem;

  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  let batch = cunflatten cd bid;
  let parts : erased (lseq et nth) =
    partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid));

  let psum : et = fold_block cd f pre_map cols nth index index_up a batch tid;
  assert pure (up batch == unflatten d (SZ.v bid));
  tensor_write_cell sa (cidx1 tid) psum;

  let mut n : szlt 32 = 0sz;

  forevery_singleton_intro'
    #(x:nat{tid <= x /\ x < tid + 1})
    (fun x -> tensor_pts_to_cell sa ((x <: natlt nth), ()) (seq![psum] @! (x - tid)))
    tid;
  fold array1_pts_to_slice sa tid (tid+1) seq![psum];

  (**)Seq.init_ghost_index (SZ.v nth)
  (**)  (fun (i:nat{i < SZ.v nth}) -> rfold1 f
  (**)    (block (SZ.v cols) (SZ.v nth)
  (**)      (lseq_map pre_map (inner_seq va (unflatten d (SZ.v bid)))) i));
  (**)rfold1_singleton f (Seq.index parts (SZ.v tid));
  (**)Seq.lemma_index_slice parts (SZ.v tid) (SZ.v tid + 1) 0;
  (**)assert pure (Seq.equal (Seq.slice parts (SZ.v tid) (SZ.v tid + 1))
  (**)                       (Seq.create 1 (Seq.index parts (SZ.v tid))));

  (**)fold (array1_pts_to_slice_red (exact_red f parts) sa tid (tid + 1));
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_red (exact_red f parts) sa tid (min (tid + pow2 !n) nth));

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_red (exact_red f parts) sa tid (min (tid + pow2 !n) nth)) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration f (exact_red f parts) (exact_red_comb f parts) nth sa tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_red (exact_red f parts) sa tid (min (tid + pow2 it) nth)))
  as
    (if_ (bool_eq #nat tid 0) (array1_pts_to_slice_red (exact_red f parts) sa 0 nth));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  if (tid = 0sz) {
    if_elim_true' (bool_eq #nat tid 0) (array1_pts_to_slice_red (exact_red f parts) sa 0 nth);
    if_elim_true' (bool_eq #nat tid 0)
      (Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid)));
    unfold array1_pts_to_slice_red (exact_red f parts) sa 0 nth;
    unfold array1_pts_to_slice_red_inner (exact_red f parts) sa 0 nth;
    (**)partials_reduces f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d (SZ.v bid));
    (**)assert pure (Seq.equal (Seq.slice parts 0 (SZ.v nth)) parts);
    let red = array1_read_from_slice sa 0sz;
    rewrite
      Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid))
    as
      Cell output (up batch) |-> acc vout (unflatten d (SZ.v bid));
    tensor_write_cell output batch (post_map red);
    assert pure (post_map red == reduced f pre_map post_map va (unflatten d (SZ.v bid)));
    rewrite
      Cell output (up batch) |-> post_map red
    as
      Cell output (unflatten d (SZ.v bid)) |->
        reduced f pre_map post_map va (unflatten d (SZ.v bid));
    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let css : erased (chest1 et nth) = hide (seq_to_chest1 (reveal ss));
    forevery_refine_ext'
      #nat
      #(fun (k:nat) -> 0 <= k /\ k < nth)
      (fun (k:nat) -> k < nth)
      _;
    forevery_ext
      (fun (k:natlt nth) ->
        tensor_pts_to_cell sa ((k <: natlt nth), ()) (ss @! (k - 0)))
      (fun (k:natlt nth) ->
        tensor_pts_to_cell sa (abs_bij.gg k) (acc (reveal css) (abs_bij.gg k)));
    forevery_iso_back (abs_bij #nth)
      (fun (i : abs (nth @| INil)) -> tensor_pts_to_cell sa i (acc (reveal css) i));
    tensor_implode sa;
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    if_intro_true' (bool_eq #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d (SZ.v bid)) |->
        reduced f pre_map post_map va (unflatten d (SZ.v bid))
    )
  } else {
    if_elim_false' (bool_eq #nat tid 0) (array1_pts_to_slice_red (exact_red f parts) sa 0 nth);
    if_elim_false' (bool_eq #nat tid 0)
      (Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid)));
    if_intro_false' (bool_eq #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d (SZ.v bid)) |->
        reduced f pre_map post_map va (unflatten d (SZ.v bid))
    );
    ();
  };
}
#pop-options

ghost
fn block_setup
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (a |-> Frac (1 /. rows) va **
     Cell output (unflatten d bid) |-> acc vout (unflatten d bid))
  ensures
    (forall+ (i : natlt nth).
      kpre f pre_map post_map rows cols nth a output va vout shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  tensor_share_n a (SZ.v nth);

  forevery_if_intro #(natlt nth) 0 (fun _ ->
    Cell output (unflatten d bid) |-> acc vout (unflatten d bid));
  forevery_ext
    (fun tid -> if_ (bool_eq #(natlt nth) tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)))
    (fun tid -> if_ (bool_eq #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)));

  forevery_zip (fun _ -> a |-> Frac ((1 /. rows) /. nth) va) _;

  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
  (fun tid ->
    a |-> Frac ((1 /. rows) /. nth) va **
    if_ (bool_eq #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)))
  _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (a |-> Frac ((1 /. rows) /. nth) va **
       if_ (bool_eq #nat tid 0)
         (Cell output (unflatten d bid) |-> acc vout (unflatten d bid))) **
      tensor_pts_to_cell (from_array (l1_forward nth) gsa)
        (abs_bij.gg (tid <: natlt nth))
        (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) ->
      kpre f pre_map post_map rows cols nth a output va vout shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}


ghost
fn block_teardown
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : nat) (#d : shape rank)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth).
      kpost f pre_map post_map rows cols nth a output va vout shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (a |-> Frac (1 /. rows) va **
     Cell output (unflatten d bid) |->
       reduced f pre_map post_map va (unflatten d bid))
{
  forevery_unzip _ _;

  tensor_gather_n a (SZ.v nth);

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (bool_eq #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        Cell output (unflatten d bid) |->
          reduced f pre_map post_map va (unflatten d bid)))
    (fun tid ->
      if_ (bool_eq #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        Cell output (unflatten d bid) |->
          reduced f pre_map post_map va (unflatten d bid)));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      Cell output (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid)
  );

  tensor_concr (from_array (l1_forward nth) shmem._1);
  rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

ghost
fn setup
  (#et_i #et_o : Type0)
  (#rank : nat) (#d : shape rank)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vout : chest d et_o)
  ()
  norewrite
  requires
    a |-> va ** output |-> vout
  ensures
    (forall+ (bid : natlt rows).
      a |-> Frac (1 /. rows) va **
      Cell output (unflatten d bid) |-> acc vout (unflatten d bid)) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n a (SZ.v rows);
  tensor_explode output;
  forevery_iso (flatten_bij d)
    (fun i -> Cell output i |-> acc vout i);
  forevery_rw_size (sizeof d) (SZ.v rows);
  forevery_zip
    (fun (_ : natlt rows) -> a |-> Frac (1 /. rows) va)
    (fun (bid : natlt rows) ->
      Cell output (unflatten d bid) |-> acc vout (unflatten d bid));
}

ghost
fn teardown
  (#et_i #et #et_o : Type0)
  (#rank : nat) (#d : shape rank)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  ()
  norewrite
  requires
    (forall+ (bid : natlt rows).
      a |-> Frac (1 /. rows) va **
      Cell output (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid)) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    a |-> va **
    output |-> mk d (fun i -> reduced f pre_map post_map va i)
{
  forevery_unzip _ _;
  tensor_gather_n a (SZ.v rows);
  forevery_rw_size (SZ.v rows) (sizeof d);
  forevery_iso_back (flatten_bij d)
    (fun i -> Cell output i |-> reduced f pre_map post_map va i);
  let result = mk d (fun i -> reduced f pre_map post_map va i);
  forevery_map
    (fun i -> Cell output i |-> reduced f pre_map post_map va i)
    (fun i -> Cell output i |-> acc result i)
    fn _ { () };
  tensor_implode output;
}

inline_for_extraction noextract
let kernel
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d)
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (index : conc d -> szlt cols -> conc (snoc_shape d cols))
  (index_up : (i:conc d -> j:szlt cols ->
    Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin { is_global a })
  (output : tensor et_o lout { is_global output })
  (#va : chest (snoc_shape d cols) et_i)
  (#vout : chest d et_o)
  : kernel_desc
      (a |-> va ** output |-> vout)
      (a |-> va ** output |-> mk d (fun i -> reduced f pre_map post_map va i))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et
        (exact_red f (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d bid)))
        nth (from_array _ shmem._1)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix
        (exact_red f (partials f pre_map (SZ.v cols) (SZ.v nth) va (unflatten d bid)))
        nth #(l1_forward nth) (from_array _ shmem._1)));

    f = kf cd f pre_map post_map rows cols nth index index_up
      a output va vout;

    block_pre  = (fun bid ->
      a |-> Frac (1 /. rows) va **
      Cell (output <: tensor et_o lout) (unflatten d bid) |->
        acc vout (unflatten d bid));
    block_post = (fun bid ->
      a |-> Frac (1 /. rows) va **
      Cell (output <: tensor et_o lout) (unflatten d bid) |->
        reduced f pre_map post_map va (unflatten d bid));
    setup      = setup rows cols a #va output #vout;
    teardown   = teardown f pre_map post_map rows cols a #va output;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup
      f pre_map post_map rows cols nth a #va output #vout;
    block_teardown = block_teardown
      f pre_map post_map rows cols nth a #va output #vout;

    kpre = kpre f pre_map post_map rows cols nth a output va vout;
    kpost = kpost f pre_map post_map rows cols nth a output va vout;
    frame = pure (SZ.fits (tlayout_ulen lout));

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

inline_for_extraction noextract
fn reduce
  (#et_i #et #et_o : Type0) {| sized et |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d { batches_ok d })
  (f : (et -> et -> et) { is_associative f })
  (pre_map : et_i -> et)
  (post_map : et -> et_o)
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (index : conc d -> szlt cols -> conc (snoc_shape d cols))
  (index_up : (i:conc d -> j:szlt cols ->
    Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin { is_global a })
  (output : tensor et_o lout { is_global output })
  (s : stream_t)
  (#va : chest (snoc_shape d cols) et_i)
  (#vout : chest d et_o)
  (#e : Kuiops.Epoch.epoch_t)
  preserves
    cpu ** stream_live s
  requires Kuiops.Epoch.epoch_live s e
  requires
    on gpu_loc (a |-> va) ** on gpu_loc (output |-> vout)
  ensures
    Kuiops.Epoch.epoch_live s (Kuiops.Epoch.epoch_next e) **
    pledge0 (Kuiops.Epoch.epoch_flushed s (Kuiops.Epoch.epoch_next e))
      (on gpu_loc (
        (a |-> va) **
        (output |-> mk d (fun i -> reduced f pre_map post_map va i))))
{
  let rows = Kuiper.Shape.csizeof cd;
  Kuiops.Kernel.launch (kernel cd f pre_map post_map rows cols nth
    index index_up a output) s;
}
