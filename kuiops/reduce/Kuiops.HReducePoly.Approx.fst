module Kuiops.HReducePoly.Approx

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
open Kuiops.Maps { approx1 }

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier

(* ------------------------------------------------------------------ *)
(* Pure approximation lemmas.                                          *)
(* ------------------------------------------------------------------ *)

let seq_slice_approx
  (#et : Type0) {| scalar et, real_like et |}
  (s : seq et) (t : seq real) (i j : nat { i <= j /\ j <= Seq.length s })
  : Lemma (requires s %~ t) (ensures Seq.slice s i j %~ Seq.slice t i j)
  = introduce forall (k:nat). k < j - i ==>
      (Seq.slice s i j @! k) %~ (Seq.slice t i j @! k)
    with introduce _ ==> _
    with _. (
      Seq.lemma_index_slice s i j k;
      Seq.lemma_index_slice t i j k
    )

let seq_map_approx
  (#a #b : Type0) {| scalar a, real_like a, scalar b, real_like b |}
  (g : a -> b) (g_r : real -> real)
  (s : seq a) (t : seq real)
  : Lemma (requires s %~ t /\ (forall (x:a) (r:real). x %~ r ==> g x %~ g_r r))
          (ensures seq_map g s %~ seq_map g_r t)
  = Seq.init_ghost_index (Seq.length s) (fun (i:nat{i < Seq.length s}) -> g (s @! i));
    Seq.init_ghost_index (Seq.length t) (fun (i:nat{i < Seq.length t}) -> g_r (t @! i))

let rec fold_left_approx
  (#et : Type0) {| scalar et, real_like et |}
  (f : et -> et -> et) (f_r : (real -> real -> real) { approx2 f f_r })
  (s : seq et) (t : seq real) (acc : et) (acc_r : real)
  : Lemma (requires s %~ t /\ acc %~ acc_r)
          (ensures seq_fold_left f acc s %~ seq_fold_left f_r acc_r t)
          (decreases Seq.length s)
  = match view_seq s, view_seq t with
    | SNil, SNil -> ()
    | SCons hd tl, SCons hd' tl' ->
      fold_left_approx f f_r tl tl' (f acc hd) (f_r acc_r hd')

let rfold1_approx
  (#et : Type0) {| scalar et, real_like et |}
  (f : et -> et -> et) (f_r : (real -> real -> real) { approx2 f f_r })
  (s : seq et { Seq.length s > 0 }) (t : seq real)
  : Lemma (requires (s <: seq et) %~ t) (ensures rfold1 f s %~ rfold1 f_r t)
  = seq_slice_approx s t 1 (Seq.length s);
    fold_left_approx f f_r
      (Seq.slice s 1 (Seq.length s)) (Seq.slice t 1 (Seq.length t))
      (s @! 0) (t @! 0)

(* The concrete per-thread partial approximates the real one. *)
let partials_approx
  (#et_i : Type0) {| scalar et_i, real_like et_i |}
  (#et : Type0) {| scalar et, real_like et |}
  (#rank : nat) (#d : shape rank)
  (f : et -> et -> et) (f_r : (real -> real -> real) { approx2 f f_r })
  (pre_map : et_i -> et) (pre_map_r : real -> real)
  (cols nth : pos { nth <= cols })
  (va : chest (snoc_shape d cols) et_i)
  (vr : chest (snoc_shape d cols) real)
  (batch : abs d)
  (tid : natlt nth)
  : Lemma (requires va %~ vr /\
                    (forall (x:et_i) (r:real). x %~ r ==> pre_map x %~ pre_map_r r))
          (ensures (partials f pre_map cols nth va batch @! tid)
                   %~ (partials f_r pre_map_r cols nth vr batch @! tid))
  = let si = inner_seq va batch in
    let ti = inner_seq vr batch in
    assert (si %~ ti);
    seq_map_approx pre_map pre_map_r si ti;
    bnd_mono cols nth tid;
    bnd_le cols nth (tid + 1);
    seq_slice_approx (lseq_map pre_map si) (lseq_map pre_map_r ti)
      (bnd cols nth tid) (bnd cols nth (tid + 1));
    rfold1_approx f f_r
      (block cols nth (lseq_map pre_map si) tid)
      (block cols nth (lseq_map pre_map_r ti) tid);
    Seq.init_ghost_index nth (fun (i:nat{i<nth}) ->
      rfold1 f (block cols nth (lseq_map pre_map si) i));
    Seq.init_ghost_index nth (fun (i:nat{i<nth}) ->
      rfold1 f_r (block cols nth (lseq_map pre_map_r ti) i))

(* ------------------------------------------------------------------ *)
(* The tree-reduction invariant, over the reals.                       *)
(* ------------------------------------------------------------------ *)

let approx_red
  (#et : Type0) {| scalar et, real_like et |} (#n : nat)
  (f_r : real -> real -> real) (rparts : lseq real n)
  (i j : nat) (x : et) : prop
  = i < j /\ j <= n ==> x %~ rfold1 f_r (Seq.slice rparts i j)

let approx_red_comb
  (#et : Type0) {| scalar et, real_like et |} (#n : nat)
  (f : et -> et -> et)
  (f_r : (real -> real -> real) { is_associative f_r /\ approx2 f f_r })
  (rparts : lseq real n)
  (i j k : nat) (x y : et)
  : Lemma (requires i < j /\ j < k /\
                    approx_red f_r rparts i j x /\ approx_red f_r rparts j k y)
          (ensures approx_red f_r rparts i k (f x y))
  = if k <= n then begin
      lem_append_slice rparts i j k;
      rfold1_append f_r (Seq.slice rparts i j) (Seq.slice rparts j k)
    end

(* ------------------------------------------------------------------ *)
(* Kernel spec plumbing.                                               *)
(* ------------------------------------------------------------------ *)

let approx_out
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#rank : nat) (#d : shape rank) (#cols : pos)
  (f_r : real -> real -> real)
  (pre_map_r post_map_r : real -> real)
  (vr : chest (snoc_shape d cols) real)
  (i : abs d) (v : et_o) : prop
  = v %~ reduced f_r pre_map_r post_map_r vr i

unfold
let kpre
  (#et_i #et #et_o : Type0) {| scalar et |}
  (#rank : nat) (#d : shape rank)
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
    if_ (op_Equality #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)) **
    exists* (v : et).
      tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

unfold
let kpost
  (#et_i #et : Type0) {| scalar et |}
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#rank : nat) (#d : shape rank)
  (f_r : real -> real -> real)
  (pre_map_r post_map_r : real -> real)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (output : tensor et_o lout)
  (va : chest (snoc_shape d cols) et_i)
  (vr : chest (snoc_shape d cols) real)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = a |-> Frac ((1 /. rows) /. nth) va **
    if_ (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      (exists* (v : et_o).
        Cell output (unflatten d bid) |-> v **
        pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v))
    )

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn kf
  (#et_i : Type0) {| scalar et_i, real_like et_i |}
  (#et : Type0) {| scalar et, real_like et |}
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d)
  (f : et -> et -> et)
  (f_r : (real -> real -> real) { is_associative f_r /\ approx2 f f_r })
  (pre_map : et_i -> et)
  (pre_map_r : (real -> real)
    { forall (x:et_i) (r:real). x %~ r ==> pre_map x %~ pre_map_r r })
  (post_map : et -> et_o)
  (post_map_r : (real -> real)
    { forall (x:et) (r:real). x %~ r ==> post_map x %~ post_map_r r })
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
  (vr : chest (snoc_shape d cols) real { va %~ vr })
  (vout : chest d et_o)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    kpre rows cols nth a output va vout shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix
      (approx_red #et f_r
        (partials f_r pre_map_r (SZ.v cols) (SZ.v nth) vr (unflatten d (SZ.v bid))))
      nth (from_array (l1_forward nth) shmem._1)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost f_r pre_map_r post_map_r rows cols nth a output va vr shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix
      (approx_red #et f_r
        (partials f_r pre_map_r (SZ.v cols) (SZ.v nth) vr (unflatten d (SZ.v bid))))
      nth (from_array (l1_forward nth) shmem._1)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre rows cols nth a output va vout shmem bid tid;
  let (gsa, _) = shmem;

  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  let batch = cunflatten cd bid;
  let rparts : erased (lseq real nth) =
    partials f_r pre_map_r (SZ.v cols) (SZ.v nth) vr (unflatten d (SZ.v bid));

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
  (**)partials_approx f f_r pre_map pre_map_r (SZ.v cols) (SZ.v nth)
  (**)  va vr (unflatten d (SZ.v bid)) (SZ.v tid);
  (**)rfold1_singleton f_r (Seq.index rparts (SZ.v tid));
  (**)Seq.lemma_index_slice rparts (SZ.v tid) (SZ.v tid + 1) 0;
  (**)assert pure (Seq.equal (Seq.slice rparts (SZ.v tid) (SZ.v tid + 1))
  (**)                       (Seq.create 1 (Seq.index rparts (SZ.v tid))));

  (**)fold (array1_pts_to_slice_red (approx_red #et f_r rparts) sa tid (tid + 1));
  (**)if_intro_true' (div_pow2 !n tid)
  (**)  (array1_pts_to_slice_red (approx_red #et f_r rparts) sa tid (min (tid + pow2 !n) nth));

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid)
        (array1_pts_to_slice_red (approx_red #et f_r rparts) sa tid (min (tid + pow2 !n) nth)) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration f (approx_red #et f_r rparts) (approx_red_comb f f_r rparts) nth sa tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid)
      (array1_pts_to_slice_red (approx_red #et f_r rparts) sa tid (min (tid + pow2 it) nth)))
  as
    (if_ (op_Equality #nat tid 0)
      (array1_pts_to_slice_red (approx_red #et f_r rparts) sa 0 nth));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  if (tid = 0sz) {
    if_elim_true' (op_Equality #nat tid 0)
      (array1_pts_to_slice_red (approx_red #et f_r rparts) sa 0 nth);
    if_elim_true' (op_Equality #nat tid 0)
      (Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid)));
    unfold array1_pts_to_slice_red (approx_red #et f_r rparts) sa 0 nth;
    unfold array1_pts_to_slice_red_inner (approx_red #et f_r rparts) sa 0 nth;
    (**)partials_reduces f_r pre_map_r (SZ.v cols) (SZ.v nth) vr (unflatten d (SZ.v bid));
    (**)assert pure (Seq.equal (Seq.slice rparts 0 (SZ.v nth)) rparts);
    let red = array1_read_from_slice sa 0sz;
    rewrite
      Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid))
    as
      Cell output (up batch) |-> acc vout (unflatten d (SZ.v bid));
    tensor_write_cell output batch (post_map red);
    assert pure (approx_out f_r pre_map_r post_map_r vr
      (unflatten d (SZ.v bid)) (post_map red));
    rewrite
      Cell output (up batch) |-> post_map red
    as
      Cell output (unflatten d (SZ.v bid)) |-> post_map red;
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
    if_intro_true' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      (exists* (v : et_o).
        Cell output (unflatten d (SZ.v bid)) |-> v **
        pure (approx_out f_r pre_map_r post_map_r vr (unflatten d (SZ.v bid)) v))
    )
  } else {
    if_elim_false' (op_Equality #nat tid 0)
      (array1_pts_to_slice_red (approx_red #et f_r rparts) sa 0 nth);
    if_elim_false' (op_Equality #nat tid 0)
      (Cell output (unflatten d (SZ.v bid)) |-> acc vout (unflatten d (SZ.v bid)));
    if_intro_false' (op_Equality #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      (exists* (v : et_o).
        Cell output (unflatten d (SZ.v bid)) |-> v **
        pure (approx_out f_r pre_map_r post_map_r vr (unflatten d (SZ.v bid)) v))
    );
    ();
  };
}
#pop-options

ghost
fn block_setup
  (#et_i #et #et_o : Type0) {| scalar et |}
  (#rank : nat) (#d : shape rank)
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
      kpre rows cols nth a output va vout shmem bid i) **
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
    (fun tid -> if_ (op_Equality #(natlt nth) tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)))
    (fun tid -> if_ (op_Equality #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)));

  forevery_zip (fun _ -> a |-> Frac ((1 /. rows) /. nth) va) _;

  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
  (fun tid ->
    a |-> Frac ((1 /. rows) /. nth) va **
    if_ (op_Equality #nat tid 0)
      (Cell output (unflatten d bid) |-> acc vout (unflatten d bid)))
  _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (a |-> Frac ((1 /. rows) /. nth) va **
       if_ (op_Equality #nat tid 0)
         (Cell output (unflatten d bid) |-> acc vout (unflatten d bid))) **
      tensor_pts_to_cell (from_array (l1_forward nth) gsa)
        (abs_bij.gg (tid <: natlt nth))
        (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) ->
      kpre rows cols nth a output va vout shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}

ghost
fn block_teardown
  (#et_i #et : Type0) {| scalar et |}
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#rank : nat) (#d : shape rank)
  (f_r : real -> real -> real)
  (pre_map_r post_map_r : real -> real)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin : tlayout (snoc_shape d cols))
  (#lout : tlayout d)
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vr : chest (snoc_shape d cols) real)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth).
      kpost f_r pre_map_r post_map_r rows cols nth a output va vr shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (a |-> Frac (1 /. rows) va **
     (exists* (v : et_o).
       Cell output (unflatten d bid) |-> v **
       pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v)))
{
  forevery_unzip _ _;

  tensor_gather_n a (SZ.v nth);

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equality #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        (exists* (v : et_o).
          Cell output (unflatten d bid) |-> v **
          pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v))))
    (fun tid ->
      if_ (op_Equality #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        (exists* (v : et_o).
          Cell output (unflatten d bid) |-> v **
          pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      (exists* (v : et_o).
        Cell output (unflatten d bid) |-> v **
        pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v))
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
fn forevery_drop_pure
  (#a : Type0)
  (p : a -> slprop)
  (q : a -> prop)
  requires forall+ (x:a). p x ** pure (q x)
  ensures forall+ (x:a). p x
{
  forevery_map
    (fun (x:a) -> p x ** pure (q x))
    p
    fn x { drop_ (pure (q x)) }
}

ghost
fn teardown
  (#et_i : Type0)
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#rank : nat) (#d : shape rank)
  (f_r : real -> real -> real)
  (pre_map_r post_map_r : real -> real)
  (rows : szp { SZ.v rows == sizeof d /\ rows <= max_blocks })
  (cols : szp)
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (a : tensor et_i lin)
  (#va : chest (snoc_shape d cols) et_i)
  (output : tensor et_o lout)
  (#vr : chest (snoc_shape d cols) real)
  ()
  norewrite
  requires
    (forall+ (bid : natlt rows).
      a |-> Frac (1 /. rows) va **
      (exists* (v : et_o).
        Cell output (unflatten d bid) |-> v **
        pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v))) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    a |-> va **
    (exists* (vout' : chest d et_o).
      output |-> vout' **
      pure (forall (i : abs d).
        acc vout' i %~ reduced f_r pre_map_r post_map_r vr i))
{
  forevery_unzip _ _;
  tensor_gather_n a (SZ.v rows);

  let g = forevery_exists
    (fun (bid : natlt rows) (v : et_o) ->
      Cell output (unflatten d bid) |-> v **
      pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v));

  let result : erased (chest d et_o) =
    hide (mk d (fun (i : abs d) -> g (flatten d i)));

  forevery_extract_pure
    (fun (bid : natlt rows) ->
      Cell output (unflatten d bid) |-> g bid **
      pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) (g bid)))
    (fun (bid : natlt rows) ->
      acc (reveal result) (unflatten d bid)
        %~ reduced f_r pre_map_r post_map_r vr (unflatten d bid))
    fn bid { () };

  forevery_drop_pure
    (fun (bid : natlt rows) -> Cell output (unflatten d bid) |-> g bid)
    (fun (bid : natlt rows) ->
      approx_out f_r pre_map_r post_map_r vr (unflatten d bid) (g bid));

  forevery_ext
    (fun (bid : natlt rows) -> Cell output (unflatten d bid) |-> g bid)
    (fun (bid : natlt rows) ->
      Cell output (unflatten d bid) |-> acc (reveal result) (unflatten d bid));

  forevery_rw_size (SZ.v rows) (sizeof d);
  forevery_iso_back (flatten_bij d)
    (fun (i : abs d) -> Cell output i |-> acc (reveal result) i);
  tensor_implode output;

  assert pure (forall (i : abs d).
    acc (reveal result) i %~ reduced f_r pre_map_r post_map_r vr i);
  ()
}

inline_for_extraction noextract
let kernel
  (#et_i : Type0) {| scalar et_i, real_like et_i |}
  (#et : Type0) {| scalar et, real_like et |}
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#rank : erased nat) (#d : shape rank)
  (cd : cshape d)
  (f : et -> et -> et)
  (f_r : (real -> real -> real) { is_associative f_r /\ approx2 f f_r })
  (pre_map : et_i -> et)
  (pre_map_r : (real -> real)
    { forall (x:et_i) (r:real). x %~ r ==> pre_map x %~ pre_map_r r })
  (post_map : et -> et_o)
  (post_map_r : (real -> real)
    { forall (x:et) (r:real). x %~ r ==> post_map x %~ post_map_r r })
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
  (#vr : chest (snoc_shape d cols) real { va %~ vr })
  (#vout : chest d et_o)
  : kernel_desc
      (a |-> va ** output |-> vout)
      (a |-> va **
        (exists* (vout' : chest d et_o).
          output |-> vout' **
          pure (forall (i : abs d).
            acc vout' i %~ reduced f_r pre_map_r post_map_r vr i)))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et
        (approx_red #et f_r
          (partials f_r pre_map_r (SZ.v cols) (SZ.v nth) vr (unflatten d bid)))
        nth (from_array _ shmem._1)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix
        (approx_red #et f_r
          (partials f_r pre_map_r (SZ.v cols) (SZ.v nth) vr (unflatten d bid)))
        nth #(l1_forward nth) (from_array _ shmem._1)));

    f = kf cd f f_r pre_map pre_map_r post_map post_map_r rows cols nth
      index index_up a output va vr vout;

    block_pre  = (fun bid ->
      a |-> Frac (1 /. rows) va **
      Cell (output <: tensor et_o lout) (unflatten d bid) |->
        acc vout (unflatten d bid));
    block_post = (fun bid ->
      a |-> Frac (1 /. rows) va **
      (exists* (v : et_o).
        Cell (output <: tensor et_o lout) (unflatten d bid) |-> v **
        pure (approx_out f_r pre_map_r post_map_r vr (unflatten d bid) v)));
    setup      = setup rows cols a #va output #vout;
    teardown   = teardown f_r pre_map_r post_map_r rows cols a #va output #vr;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup rows cols nth a #va output #vout;
    block_teardown = block_teardown
      f_r pre_map_r post_map_r rows cols nth a #va output #vr;

    kpre = kpre rows cols nth a output va vout;
    kpost = kpost f_r pre_map_r post_map_r rows cols nth a output va vr;
    frame = pure (SZ.fits (tlayout_ulen lout));

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

(* [index]/[index_up] build a flat index of the input shape from a batch index
   and a position along the reduced axis. [Kuiops.HReducePoly.Spec.conc_snoc]
   does exactly this, but recurses over an erased shape and so is not
   extractable; callers pass a monomorphic builder from
   [Kuiops.HReducePoly.Index] instead. *)
inline_for_extraction noextract
fn reduce
  (#et_i : Type0) {| scalar et_i, real_like et_i |}
  (#et : Type0) {| scalar et, real_like et |}
  (#et_o : Type0) {| scalar et_o, real_like et_o |}
  (#r : erased nat)
  (#d : shape r)
  (cd : cshape d { batches_ok d })
  (f : et -> et -> et)
  (f_r : (real -> real -> real) { is_associative f_r /\ approx2 f f_r })
  (pre_map : et_i -> et)
  (pre_map_r : (real -> real) { approx1 pre_map pre_map_r })
  (post_map : et -> et_o)
  (post_map_r : (real -> real) { approx1 post_map post_map_r })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (index : conc d -> szlt cols -> conc (snoc_shape d cols))
  (index_up : (i:conc d -> j:szlt cols ->
    Lemma (up (index i j) == abs_snoc (up i) (SZ.v j))))
  (#lin : tlayout (snoc_shape d cols)) {| ctlayout lin |}
  (#lout : tlayout d) {| ctlayout lout |}
  (input : tensor et_i lin { is_global input })
  (output : tensor et_o lout { is_global output })
  (s : stream_t)
  (#vin : chest (snoc_shape d cols) et_i)
  (#vr : chest (snoc_shape d cols) real)
  (#vout : chest d et_o)
  (#e : epoch_t)
  preserves cpu ** stream_live s ** epoch_live s e
  requires on gpu_loc (input |-> vin) ** on gpu_loc (output |-> vout)
  requires pure (vin %~ vr)
  ensures
    pledge0 (epoch_done s e)
      (on gpu_loc (
        (input |-> vin) **
        (exists* (vout' : chest d et_o).
          (output |-> vout') **
          pure (out_approx f_r pre_map_r post_map_r vr vout'))))
{
  let rows = Kuiper.Shape.csizeof cd;
  launch (kernel cd f f_r pre_map pre_map_r post_map post_map_r rows cols nth
    index index_up input output #vin #vr) s;
}
