module Kuiops.SuperGEMM.Mm.SplitK.Reduce

(* Split-K pass 2: sum the [splits] fp32 partials in the workspace and narrow
   to the output element type.

   One thread per 128-bit granule of D.  The workspace is the [(splits*m, n)]
   fp32 matrix written by pass 1, so granule [(di, dj)] of D reads workspace
   rows [s*m + di] for [s < splits].  The workspace is read-only and shared by
   the whole grid; only D is partitioned, into [1 x chunk et_d] subtiles.

   This is a [kernel_desc_n]: the jobs are independent, there is no shared
   memory and no barrier, and Kuiper's cast does the blocking (1024 threads per
   block with a bounds guard), which subsumes the reference's grid-stride
   loop. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiops.Array2.Vectorized { array2_vec_read, array2_vec_write }
open Kuiper.Tensor
open Kuiops.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.ForEvery
open Kuiper.Tensor.Tiling { array2_subtile, array2_tile, array2_untile_underspec, subtile_layout }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiper.Kernel.Desc { kernel_desc }
open Kuiper.Kernel.Casts { kernel_desc_n }
open Kuiops.Array.LocalAligned { local_aligned16 }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module A = Pulse.Lib.Array
module ML = FStar.Math.Lemmas
open Kuiops.Approx.Share { approx_pts_to, approx_pts_to_sendable, approx_share, approx_gather }

module RL = Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas
module SL = Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

let divides_helper (d : pos) (a b r c : int)
  : Lemma (requires d /? a /\ d /? b /\ d /? c)
          (ensures d /? (a + b * r + c))
= Kuiper.Divides.lemma_divides_product_l d b r;
  Kuiper.Divides.lemma_divides_sum d a (b * r);
  Kuiper.Divides.lemma_divides_sum d (a + b * r) c

let div_step (d : pos) (a b : nat)
  : Lemma (requires d /? a /\ d /? b /\ a < b) (ensures a + d <= b)
= let x = Kuiper.Divides.get_factor d a in
  let y = Kuiper.Divides.get_factor d b in
  if y <= x then ML.lemma_mult_le_left d y x
  else begin
    ML.lemma_mult_le_left d (x + 1) y;
    ML.distributivity_add_right d x 1
  end

let chunk_div16 (et : Type0) {| sized et, hvc : has_vec_cpy et |}
  : Lemma (chunk et /? 16)
= assert (chunk et * size #et == 16);
  introduce exists (z : int). chunk et * z == 16
  with (size #et) and ()

let scale_align (d : pos) (c : nat) (s : pos)
  : Lemma (requires d /? c /\ d * s == 16)
          (ensures 16 /?+ (c * s))
= let z = Kuiper.Divides.get_factor d c in
  calc (==) {
    c * s;
    == { }
    (d * z) * s;
    == { ML.paren_mul_right d z s }
    d * (z * s);
    == { ML.swap_mul z s }
    d * (s * z);
    == { ML.paren_mul_right d s z }
    (d * s) * z;
  }

(* [aligned' 16] of a cell whose column index is [chunk]-aligned, in a
   [chunk]-aligned row-major layout over a 16-byte-aligned base. *)
let cell_aligned16
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (l : layout2 rows cols)
  {| str : strided_row_major (vtlayout_of_tlayout l) |}
  (gm : array2 et l)
  (i : natlt rows) (j : natlt cols)
  : Lemma (requires aligned 16 (T.core gm) /\
                    aligned_strided_row_major (chunk et) str /\
                    chunk et /? j)
          (ensures aligned' 16 (T.core gm) (cell_of_pos l i j))
= str.pf i j;
  divides_helper (chunk et) str.offset str.stride i j;
  scale_align (chunk et) (cell_of_pos l i j) (size #et);
  Kuiper.Divides.lemma_divides_sum 16
    (base_address (T.core gm)) (cell_of_pos l i j * size #et)

(* ---------------------------------------------------------------------- *)
(* per-thread output capability: one 128-bit granule of D                   *)
(* ---------------------------------------------------------------------- *)

let div_ub (a b c : nat)
  : Lemma (requires c > 0 /\ a < b * c) (ensures a / c < b)
= ML.lemma_div_mod a c;
  ML.lemma_mult_lt_left c (a / c) b

unfold
let gran_ub (m : nat) (gc : pos) : prop =
  forall (x : nat). {:pattern (x / gc)} x < m * gc ==> x / gc < m

let gran_ub_lemma (m : nat) (gc : pos) : Lemma (gran_ub m gc)
= introduce forall (x : nat). x < m * gc ==> x / gc < m
  with introduce _ ==> _
  with div_ub x m gc

unfold
let gcols (et_d : Type0) {| sized et_d, has_vec_cpy et_d |} (n : szp) : nat
= SZ.v n / SZ.v (chunk et_d)

let d_gran
  (#et_d : Type0) {| sized et_d, has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (di : natlt (SZ.v m)) (dj : natlt (gcols et_d n))
  : slprop
= exists* (v : chest2 et_d 1 (SZ.v (chunk et_d))).
    array2_subtile gD 1 (SZ.v (chunk et_d)) di dj |-> v

(* The same granule, pinned to the real matrix pass 2 must produce. *)
let d_gran_at
  (#et_d : Type0) {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD)
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (rD : chest2 real (SZ.v m) (SZ.v n))
  (di : natlt (SZ.v m)) (dj : natlt (gcols et_d n))
  : slprop
= exists* (v : chest2 et_d 1 (SZ.v (chunk et_d))).
    (array2_subtile gD 1 (SZ.v (chunk et_d)) di dj |-> v) **
    pure (v %~ ematrix_subtile rD 1 (SZ.v (chunk et_d)) di dj)

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1"
ghost
fn rsetup
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| sized et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW)
  (gD : array2 et_d lD)
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (#fW : perm) (rW : chest2 real (SZ.v mws) (SZ.v n))
  ()
  norewrite
  requires approx_pts_to gW fW rW ** live gD
  ensures
    (forall+ (i : natlt njobs).
      approx_pts_to gW ((fW /. 2) /. njobs) rW **
      d_gran gD () (i / gcols et_d n) (i % gcols et_d n)) **
    approx_pts_to gW (fW /. 2) rW
{
  with (eD : chest2 _ _ _). assert gD |-> eD;
  array2_tile gD 1 (SZ.v (chunk et_d)) #eD #1.0R;
  forevery_unfactor' (SZ.v njobs) (SZ.v m / 1) (gcols et_d n) _;
  forevery_map
    (fun (i : natlt njobs) ->
      array2_subtile gD 1 (SZ.v (chunk et_d))
        (i / gcols et_d n) (i % gcols et_d n)
        |-> ematrix_subtile eD 1 (SZ.v (chunk et_d))
              (i / gcols et_d n) (i % gcols et_d n))
    (fun (i : natlt njobs) ->
      d_gran gD () (i / gcols et_d n) (i % gcols et_d n))
    fn i {
      fold (d_gran gD () (i / gcols et_d n) (i % gcols et_d n));
    };
  approx_share gW fW rW (SZ.v njobs);
  forevery_zip
    (fun (i : natlt njobs) -> approx_pts_to gW ((fW /. 2) /. njobs) rW)
    (fun (i : natlt njobs) ->
      d_gran gD () (i / gcols et_d n) (i % gcols et_d n));
}
#pop-options

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1"
ghost
fn rteardown
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW)
  (gD : array2 et_d lD)
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (rD : chest2 real (SZ.v m) (SZ.v n))
  (#_ : squash (SZ.fits lD.ulen))
  (#fW : perm) (rW : chest2 real (SZ.v mws) (SZ.v n))
  ()
  norewrite
  requires
    (forall+ (i : natlt njobs).
      approx_pts_to gW ((fW /. 2) /. njobs) rW **
      d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n)) **
    approx_pts_to gW (fW /. 2) rW
  ensures approx_pts_to gW fW rW **
          (exists* (eD : chest2 et_d (SZ.v m) (SZ.v n)).
             gD |-> eD ** pure (eD %~ rD))
{
  forevery_unzip
    (fun (i : natlt njobs) -> approx_pts_to gW ((fW /. 2) /. njobs) rW)
    (fun (i : natlt njobs) ->
      d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n));
  approx_gather gW fW rW (SZ.v njobs);
  forevery_map
    (fun (i : natlt njobs) ->
      d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n))
    (fun (i : natlt njobs) ->
      exists* (v : chest2 et_d 1 (SZ.v (chunk et_d))).
        (array2_subtile gD 1 (SZ.v (chunk et_d))
          (i / gcols et_d n) (i % gcols et_d n) |-> v) **
        pure (v %~ ematrix_subtile rD 1 (SZ.v (chunk et_d))
                     (i / gcols et_d n) (i % gcols et_d n)))
    fn i {
      unfold (d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n));
    };
  forevery_factor' (SZ.v njobs) (SZ.v m / 1) (gcols et_d n)
    (fun (tr : natlt (SZ.v m / 1)) (tc : natlt (gcols et_d n)) ->
      exists* (v : chest2 et_d 1 (SZ.v (chunk et_d))).
        (array2_subtile gD 1 (SZ.v (chunk et_d)) tr tc |-> v) **
        pure (v %~ ematrix_subtile rD 1 (SZ.v (chunk et_d)) tr tc));
  Kuiops.SuperGEMM.Mm.SplitK.Gather.array2_untile_approximates
    gD 1 (SZ.v (chunk et_d)) rD;
}
#pop-options

let gran_col_bound (c : pos) (n : nat) (dj y : nat)
  : Lemma (requires c /?+ n /\ dj < n / c /\ y < c)
          (ensures dj * c + y < n)
= let z = Kuiper.Divides.get_factor c n in
  ML.cancel_mul_div c z;
  ML.lemma_mult_le_right c (dj + 1) z;
  ML.swap_mul z c;
  ML.distributivity_add_left dj 1 c

let gran_col_bound_all (c : pos) (n : nat) (dj : nat)
  : Lemma (requires c /?+ n /\ dj < n / c)
          (ensures forall (y : natlt c). dj * c + y < n)
= introduce forall (y : natlt c). dj * c + y < n
  with gran_col_bound c n dj y

(* Add one vector read to a contiguous run of the wider accumulator buffer.
   Keeping this generic inner loop separate avoids asking the solver to carry
   the tensor and kernel contracts through the pointwise sequence update. *)
#push-options "--z3rlimit 10 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn acc_chunk
  (#et_acc : Type0) {| scalar et_acc |}
  (cd ca : szp)
  (abuf vbuf : A.array et_acc)
  (g : nat -> nat -> GTot et_acc)
  (sv : erased nat)
  (ev : SZ.t { SZ.v ev + SZ.v ca <= SZ.v cd })
  (#bv : erased (seq et_acc))
  (#av0 : erased (seq et_acc))
  ()
  preserves vbuf |-> bv
  requires abuf |-> av0
  requires pure (
    A.length abuf == SZ.v cd /\ A.length vbuf == SZ.v ca /\
    Seq.length av0 == SZ.v cd /\ Seq.length bv == SZ.v ca /\
    (forall (x : natlt (SZ.v ca)). Seq.index bv x == g (SZ.v ev + x) sv) /\
    (forall (e : natlt (SZ.v cd)).
       Seq.index av0 e
       == SL.sum_upto (g e) (if e < SZ.v ev then sv + 1 else sv)))
  ensures exists* (av : seq et_acc).
    abuf |-> av **
    pure (Seq.length av == SZ.v cd /\
          (forall (e : natlt (SZ.v cd)).
             Seq.index av e
             == SL.sum_upto (g e)
                  (if e < SZ.v ev + SZ.v ca then sv + 1 else sv)))
{
  let mut xc = 0sz;
  while (!xc <^ ca)
    invariant exists* (xv : SZ.t) (av : seq et_acc).
      xc |-> xv ** abuf |-> av **
      pure (SZ.v xv <= SZ.v ca /\ Seq.length av == SZ.v cd /\
            (forall (e : natlt (SZ.v cd)).
               Seq.index av e
               == SL.sum_upto (g e)
                    (if e < SZ.v ev + SZ.v xv then sv + 1 else sv)))
    decreases (SZ.v ca - SZ.v !xc)
  {
    let xv = !xc;
    let b = A.op_Dot_Lparen_Rparen vbuf xv;
    with av. assert abuf |-> av;
    let ax : szlt (SZ.v cd) = ev +^ xv;
    let a = A.op_Dot_Lparen_Rparen abuf ax;
    A.op_Dot_Lparen_Rparen_Less_Minus abuf ax (add a b);
    xc := !xc +^ 1sz;
  };
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* per-thread body                                                          *)
(* ---------------------------------------------------------------------- *)

unfold
let rk_sq
  (#et_acc #et_d : Type0)
  {| scalar et_acc, has_vec_cpy et_acc |}
  {| scalar et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (#lD : layout2 (SZ.v m) (SZ.v n))
  {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gW : array2 et_acc lW)
  (gD : array2 et_d lD)
  (splits : szp)
  : prop
= SZ.v (chunk et_d) /?+ SZ.v n /\
  SZ.v (chunk et_acc) /?+ SZ.v (chunk et_d) /\
  SZ.v mws == SZ.v splits * SZ.v m /\
  aligned 16 (T.core gW) /\ aligned 16 (T.core gD) /\
  aligned_strided_row_major (SZ.v (chunk et_acc)) strW /\
  aligned_strided_row_major (SZ.v (chunk et_d)) strD /\
  SZ.fits lD.ulen /\ SZ.fits lW.ulen

#push-options "--z3rlimit 20 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn rkf_at
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gW : array2 et_acc lW)
  (gD : array2 et_d lD)
  (splits : szp)
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (sq : squash (rk_sq gW gD splits))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (#fW : perm) (#eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (rW : chest2 real (SZ.v mws) (SZ.v n) { eW %~ rW })
  (i : szlt njobs)
  ()
  preserves gpu
  preserves gW |-> Frac fW eW
  requires d_gran gD () (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n)
  ensures d_gran_at gD ()
            (RL.gran_target (SZ.v m) (SZ.v splits) rW post_map_r)
            (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n)
{
  let gc : szp = n /^ chunk et_d;
  let di : szlt m = i /^ gc;
  let dj : szlt (SZ.v n / SZ.v (chunk et_d)) = i %^ gc;

  unfold (d_gran gD () (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n));
  rewrite each (SZ.v i / gcols et_d n) as (SZ.v di);
  rewrite each (SZ.v i % gcols et_d n) as (SZ.v dj);

  let zacc : et_acc = zero #et_acc;
  let zd : et_d = zero #et_d;
  let mut vbuf = [| zacc; chunk et_acc |];
  let mut abuf = [| zacc; chunk et_d |];
  let mut obuf = [| zd; chunk et_d |];
  A.pts_to_len vbuf;
  A.pts_to_len abuf;
  A.pts_to_len obuf;
  local_aligned16 #et_acc vbuf;
  local_aligned16 #et_d obuf;

  let mut sc = 0sz;
  while (!sc <^ splits)
    invariant exists* (sv : SZ.t) (av bv : seq et_acc).
      sc |-> sv ** abuf |-> av ** vbuf |-> bv **
      pure (SZ.v sv <= SZ.v splits /\
            Seq.length av == SZ.v (chunk et_d) /\
            Seq.length bv == SZ.v (chunk et_acc) /\
            (forall (e : natlt (SZ.v (chunk et_d))).
               Seq.index av e
               == SL.sum_upto
                    (RL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
                       (SZ.v dj * SZ.v (chunk et_d)) e)
                    (SZ.v sv)))
    decreases (SZ.v splits - SZ.v !sc)
  {
    let sv = !sc;
    let wrow : szlt mws = sv *^ m +^ di;
    let mut ec = 0sz;
    while (!ec <^ chunk et_d)
      invariant exists* (ev : SZ.t) (av bv : seq et_acc).
        ec |-> ev ** abuf |-> av ** vbuf |-> bv **
        pure (SZ.v ev <= SZ.v (chunk et_d) /\ SZ.v (chunk et_acc) /? SZ.v ev /\
              Seq.length av == SZ.v (chunk et_d) /\
              Seq.length bv == SZ.v (chunk et_acc) /\
              (forall (e : natlt (SZ.v (chunk et_d))).
                 Seq.index av e
                 == SL.sum_upto
                      (RL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
                         (SZ.v dj * SZ.v (chunk et_d)) e)
                      (if e < SZ.v ev then SZ.v sv + 1 else SZ.v sv)))
      decreases (SZ.v (chunk et_d) - SZ.v !ec)
    {
      let ev = !ec;
      let wcol : szlt (SZ.v n - SZ.v (chunk et_acc) + 1) = dj *^ chunk et_d +^ ev;
      assert pure (SZ.v wcol == SZ.v dj * SZ.v (chunk et_d) + SZ.v ev);
      Kuiper.Divides.lemma_divides_product_r (SZ.v (chunk et_acc)) (SZ.v dj) (SZ.v (chunk et_d));
      Kuiper.Divides.lemma_divides_sum (SZ.v (chunk et_acc))
        (SZ.v dj * SZ.v (chunk et_d)) (SZ.v ev);
      assert pure (SZ.v (chunk et_acc) /? SZ.v wcol);
      divides_helper (SZ.v (chunk et_acc)) strW.offset strW.stride
        (SZ.v wrow) (SZ.v wcol);
      cell_aligned16 lW gW (SZ.v wrow) (SZ.v wcol);
      array2_vec_read gW wrow wcol vbuf;
      RL.ws_row_bound (SZ.v m) (SZ.v splits) (SZ.v mws) (SZ.v sv) (SZ.v di);
      div_step (SZ.v (chunk et_acc)) (SZ.v ev) (SZ.v (chunk et_d));
      acc_chunk (chunk et_d) (chunk et_acc) abuf vbuf
        (RL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
          (SZ.v dj * SZ.v (chunk et_d)))
        (SZ.v sv) ev ();
      Kuiper.Divides.lemma_divides_sum (SZ.v (chunk et_acc))
        (SZ.v ev) (SZ.v (chunk et_acc));
      div_step (SZ.v (chunk et_acc)) (SZ.v ev) (SZ.v (chunk et_d));
      ec := !ec +^ chunk et_acc;
    };
    sc := !sc +^ 1sz;
  };

  let mut yc = 0sz;
  while (!yc <^ chunk et_d)
    invariant exists* (yv : SZ.t) (av : seq et_acc) (ov : seq et_d).
      yc |-> yv ** abuf |-> av ** obuf |-> ov **
      pure (SZ.v yv <= SZ.v (chunk et_d) /\
            Seq.length av == SZ.v (chunk et_d) /\
            Seq.length ov == SZ.v (chunk et_d) /\
            (forall (e : natlt (SZ.v (chunk et_d))).
               Seq.index av e
               == SL.sum_upto
                    (RL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
                       (SZ.v dj * SZ.v (chunk et_d)) e)
                    (SZ.v splits)) /\
            (forall (y : natlt (SZ.v (chunk et_d))).
               y < SZ.v yv ==> Seq.index ov y == post_map (Seq.index av y)))
    decreases (SZ.v (chunk et_d) - SZ.v !yc)
  {
    let yv = !yc;
    with av. assert abuf |-> av;
    let a = abuf.(yv);
    with ov. assert obuf |-> ov;
    A.op_Dot_Lparen_Rparen_Less_Minus obuf yv (post_map a);
    yc := !yc +^ 1sz;
  };

  Kuiper.Divides.lemma_divides_product_r (SZ.v (chunk et_d)) (SZ.v dj) (SZ.v (chunk et_d));
  divides_helper (SZ.v (chunk et_d)) strD.offset strD.stride (SZ.v di * 1)
    (SZ.v dj * SZ.v (chunk et_d));
  cell_aligned16 (subtile_layout lD 1 (SZ.v (chunk et_d)) (SZ.v di) (SZ.v dj))
    (array2_subtile gD 1 (SZ.v (chunk et_d)) (SZ.v di) (SZ.v dj)) 0 0;
  with ov. assert obuf |-> ov;
  array2_vec_write (array2_subtile gD 1 (SZ.v (chunk et_d)) (SZ.v di) (SZ.v dj))
    0sz 0sz obuf ov ();

  with v'. assert
    (array2_subtile gD 1 (SZ.v (chunk et_d)) (SZ.v di) (SZ.v dj) |-> v');
  gran_col_bound_all (SZ.v (chunk et_d)) (SZ.v n) (SZ.v dj);
  RL.ws_run_sum_all (SZ.v m) (SZ.v splits) eW (SZ.v di)
    (SZ.v dj * SZ.v (chunk et_d)) (SZ.v (chunk et_d)) (SZ.v splits);
  RL.gran_approx (SZ.v m) (SZ.v splits) eW rW post_map post_map_r
    (SZ.v (chunk et_d)) v' (SZ.v di) (SZ.v dj) ();
  fold (d_gran_at gD ()
          (RL.gran_target (SZ.v m) (SZ.v splits) rW post_map_r)
          (SZ.v di) (SZ.v dj));
  rewrite each (SZ.v di) as (SZ.v i / gcols et_d n);
  rewrite each (SZ.v dj) as (SZ.v i % gcols et_d n);
}
#pop-options

(* The device body as the descriptor needs it: with the workspace contents
   existential, so that [kpre] does not have to name them.  The body itself is
   [rkf_at], unchanged -- naming the witness here rather than inside keeps the
   inner loops' proof context exactly as it was. *)
inline_for_extraction noextract
fn rkf
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gW : array2 et_acc lW)
  (gD : array2 et_d lD)
  (splits : szp)
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (sq : squash (rk_sq gW gD splits))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (#fW : perm)
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  (i : szlt njobs)
  ()
  preserves gpu
  preserves approx_pts_to gW ((fW /. 2) /. njobs) rW
  requires d_gran gD () (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n)
  ensures d_gran_at gD ()
            (RL.gran_target (SZ.v m) (SZ.v splits) rW post_map_r)
            (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n)
{
  unfold (approx_pts_to gW ((fW /. 2) /. njobs) rW);
  with eW. assert (gW |-> Frac ((fW /. 2) /. njobs) eW);
  rkf_at gW gD splits post_map post_map_r () njobs
    #((fW /. 2) /. njobs) #eW rW i ();
  fold (approx_pts_to gW ((fW /. 2) /. njobs) rW);
}

(* ---------------------------------------------------------------------- *)
(* kernel descriptor                                                        *)
(* ---------------------------------------------------------------------- *)

let d_gran_sendable
  (#et_d : Type0) {| sized et_d, has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD { is_global gD })
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (di : natlt (SZ.v m)) (dj : natlt (gcols et_d n))
  : is_send_across gpu_of (d_gran gD () di dj)
= let tl = array2_subtile gD 1 (SZ.v (chunk et_d)) di dj in
  let ff (v : chest2 et_d 1 (SZ.v (chunk et_d)))
    : is_send_across gpu_of (tl |-> v)
  = is_send_across_global_tensor tl #1.0R v in
  is_send_across_exists (fun (v : chest2 et_d 1 (SZ.v (chunk et_d))) -> tl |-> v) #ff

let d_gran_at_sendable
  (#et_d : Type0) {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gD : array2 et_d lD { is_global gD })
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (rD : chest2 real (SZ.v m) (SZ.v n))
  (di : natlt (SZ.v m)) (dj : natlt (gcols et_d n))
  : is_send_across gpu_of (d_gran_at gD () rD di dj)
= let tl = array2_subtile gD 1 (SZ.v (chunk et_d)) di dj in
  let rt = ematrix_subtile rD 1 (SZ.v (chunk et_d)) di dj in
  let ff (v : chest2 et_d 1 (SZ.v (chunk et_d)))
    : is_send_across gpu_of ((tl |-> v) ** pure (v %~ rt))
  = is_send_across_star (tl |-> v) (pure (v %~ rt))
      #(is_send_across_global_tensor tl #1.0R v)
      #(is_send_across_placeless (pure (v %~ rt)) #(placeless_pure (v %~ rt))) in
  is_send_across_exists
    (fun (v : chest2 et_d 1 (SZ.v (chunk et_d))) -> (tl |-> v) ** pure (v %~ rt)) #ff

let rk_sendable
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| sized et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW { is_global gW })
  (gD : array2 et_d lD { is_global gD })
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (fW : perm) (rW : chest2 real (SZ.v mws) (SZ.v n))
  (i : natlt njobs)
  : is_send_across gpu_of
      (approx_pts_to gW ((fW /. 2) /. njobs) rW **
       d_gran gD () (i / gcols et_d n) (i % gcols et_d n))
= is_send_across_star
    (approx_pts_to gW ((fW /. 2) /. njobs) rW)
    (d_gran gD () (i / gcols et_d n) (i % gcols et_d n))
    #(approx_pts_to_sendable gW ((fW /. 2) /. njobs) rW)
    #(d_gran_sendable gD () (i / gcols et_d n) (i % gcols et_d n))

let rk_post_sendable
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW { is_global gW })
  (gD : array2 et_d lD { is_global gD })
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (fW : perm) (rW : chest2 real (SZ.v mws) (SZ.v n))
  (rD : chest2 real (SZ.v m) (SZ.v n))
  (i : natlt njobs)
  : is_send_across gpu_of
      (approx_pts_to gW ((fW /. 2) /. njobs) rW **
       d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n))
= is_send_across_star
    (approx_pts_to gW ((fW /. 2) /. njobs) rW)
    (d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n))
    #(approx_pts_to_sendable gW ((fW /. 2) /. njobs) rW)
    #(d_gran_at_sendable gD () rD (i / gcols et_d n) (i % gcols et_d n))

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1"
inline_for_extraction noextract
let mk_reduce_kernel
  (#et_acc #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gW : array2 et_acc lW { is_global gW })
  (gD : array2 et_d lD { is_global gD })
  (splits : szp)
  (post_map : et_acc -> et_d)
  (post_map_r : real -> real { post_map %~ post_map_r })
  (sq : squash (rk_sq gW gD splits))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) /\
                 njobs <= max_blocks * max_threads })
  (fW : perm)
  (rW : chest2 real (SZ.v mws) (SZ.v n))
  ()
  : kernel_desc
      ((exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v n)).
          (gW |-> Frac fW eW) ** pure (eW %~ rW)) ** live gD)
      ((exists* (eW : chest2 et_acc (SZ.v mws) (SZ.v n)).
          (gW |-> Frac fW eW) ** pure (eW %~ rW)) **
       (exists* (eD : chest2 et_d (SZ.v m) (SZ.v n)).
          gD |-> eD **
          pure (eD %~ RL.gran_target (SZ.v m) (SZ.v splits) rW post_map_r)))
= {
    nthr = njobs;
    frame = approx_pts_to gW (fW /. 2) rW;
    kpre = (fun (i : natlt njobs) ->
      approx_pts_to gW ((fW /. 2) /. njobs) rW **
      d_gran gD () (i / gcols et_d n) (i % gcols et_d n));
    kpost = (fun (i : natlt njobs) ->
      approx_pts_to gW ((fW /. 2) /. njobs) rW **
      d_gran_at gD () (RL.gran_target (SZ.v m) (SZ.v splits) rW post_map_r)
        (i / gcols et_d n) (i % gcols et_d n));
    setup = rsetup gW gD () njobs #fW rW;
    teardown = rteardown gW gD () njobs
      (RL.gran_target (SZ.v m) (SZ.v splits) rW post_map_r) #() #fW rW;
    f = rkf gW gD splits post_map post_map_r () njobs #fW rW;
    kpre_sendable = rk_sendable gW gD () njobs fW rW;
    kpost_sendable = rk_post_sendable gW gD () njobs fW rW
      (RL.gran_target (SZ.v m) (SZ.v splits) rW post_map_r);
  } <: kernel_desc_n _ _
#pop-options
