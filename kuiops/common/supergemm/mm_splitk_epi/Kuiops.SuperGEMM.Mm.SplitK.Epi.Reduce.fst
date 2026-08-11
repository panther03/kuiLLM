module Kuiops.SuperGEMM.Mm.SplitK.Epi.Reduce

(* Split-K pass 2 WITH the epilogue: sum the [splits] accumulator-typed
   partials in the workspace, combine the complete sum with the C view, and
   narrow to the output element type.

   This is [Kuiops.SuperGEMM.Mm.SplitK.Reduce] with the unary [post_map]
   replaced by a binary [comb : et_c -> et_acc -> et_d] whose first argument is
   read from an arbitrary read-only C VIEW.  Everything that does not touch the
   epilogue is IMPORTED from that module rather than restated: the per-thread
   granule capability [d_gran] / [d_gran_at], its sendability proofs, the
   granule-index arithmetic ([gcols], [gran_ub]), the alignment lemmas, and the
   setup/teardown of D and the workspace.  Only the per-thread body is written
   out again, because the [comb] enters in the middle of it.

   WHY THE EPILOGUE IS IN PASS 2.  [comb] is affine in the accumulated value,
   so applying it in pass 1 -- which produces [splits] PARTIAL sums, none of
   which has seen the whole k range -- would add the [beta * C] term once per
   split instead of once in total.  Pass 2 is the first point at which a
   complete k reduction exists for an output element, so it is the only place
   the epilogue can go.  See [.Epi.ReduceLemmas].

   THE C VIEW.  [gC] is a [rotensor] over an arbitrary [vtlayout]: its index
   map is not assumed injective, contiguous or aligned.  Non-injectivity is
   exactly what broadcasting is, and a broadcast bias is what every [addmm] in
   the pipeline supplies, so this one function covers both the dense and the
   broadcast C without a second entry point.  C is therefore read SCALARLY, one
   [RO.tensor_read] per element -- that is the definition of the C read, and is
   the path the reference itself blesses as "correct and simpler".  The D store
   stays 128-bit vectorized. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Array2.Vectorized { array2_vec_read, array2_vec_write }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.ForEvery
open Kuiper.Tensor.Tiling { array2_subtile, subtile_layout }
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiper.Kernel.Desc { kernel_desc }
open Kuiper.Kernel.Casts { kernel_desc_n }
open Kuiops.Array.LocalAligned { local_aligned16 }
open Kuiops.SuperGEMM.Mm.SplitK.Reduce {
  gcols, gran_ub, gran_ub_lemma, d_gran, d_gran_at, rk_sq,
  divides_helper, div_step, cell_aligned16,
  rsetup, rteardown, d_gran_sendable, d_gran_at_sendable }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module RO = Kuiper.TensorRO
module A = Pulse.Lib.Array
module ML = FStar.Math.Lemmas
module RL = Kuiops.SuperGEMM.Mm.SplitK.ReduceLemmas
module EL = Kuiops.SuperGEMM.Mm.SplitK.Epi.ReduceLemmas
module SL = Kuiops.SuperGEMM.Mm.SplitK.SpecLemmas

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

(* ---------------------------------------------------------------------- *)
(* setup / teardown: the no-epi ones, plus the read-only share of C         *)
(* ---------------------------------------------------------------------- *)

ghost
fn resetup
  (#et_acc #et_c #et_d : Type0)
  {| sized et_acc, has_vec_cpy et_acc, sized et_c, sized et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW)
  (gC : RO.roarray2 et_c lC)
  (gD : array2 et_d lD)
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (#fW : perm) (#eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (#fC : perm) (#eC : chest2 et_c (SZ.v m) (SZ.v n))
  ()
  norewrite
  requires (gW |-> Frac fW eW) ** (gC |-> Frac fC eC) ** live gD
  ensures
    (forall+ (i : natlt njobs).
      (gC |-> Frac (fC /. njobs) eC) **
      ((gW |-> Frac (fW /. njobs) eW) **
       d_gran gD () (i / gcols et_d n) (i % gcols et_d n))) ** pure True
{
  rsetup gW gD () njobs #fW #eW ();
  RO.tensor_share_n gC (SZ.v njobs);
  forevery_zip
    (fun (i : natlt njobs) -> gC |-> Frac (fC /. njobs) eC)
    (fun (i : natlt njobs) ->
      (gW |-> Frac (fW /. njobs) eW) **
      d_gran gD () (i / gcols et_d n) (i % gcols et_d n));
}

ghost
fn reteardown
  (#et_acc #et_c #et_d : Type0)
  {| sized et_acc, has_vec_cpy et_acc |}
  {| sized et_c |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW)
  (gC : RO.roarray2 et_c lC)
  (gD : array2 et_d lD)
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (rD : chest2 real (SZ.v m) (SZ.v n))
  (#_ : squash (SZ.fits lD.ulen))
  (#fW : perm) (#eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (#fC : perm) (#eC : chest2 et_c (SZ.v m) (SZ.v n))
  ()
  norewrite
  requires
    (forall+ (i : natlt njobs).
      (gC |-> Frac (fC /. njobs) eC) **
      ((gW |-> Frac (fW /. njobs) eW) **
       d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n))) ** pure True
  ensures (gW |-> Frac fW eW) ** (gC |-> Frac fC eC) **
          (exists* (eD : chest2 et_d (SZ.v m) (SZ.v n)).
             gD |-> eD ** pure (eD %~ rD))
{
  forevery_unzip
    (fun (i : natlt njobs) -> gC |-> Frac (fC /. njobs) eC)
    (fun (i : natlt njobs) ->
      (gW |-> Frac (fW /. njobs) eW) **
      d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n));
  RO.tensor_gather_n gC (SZ.v njobs);
  rteardown gW gD () njobs rD #() #fW #eW ();
}

(* [dj] indexes granules, so its column base plus any within-granule offset is
   in range.  Needed as an explicit step because it is nonlinear. *)
let gran_col_bound (cd : pos) (n : nat) (dj y : nat)
  : Lemma (requires cd /? n /\ dj < n / cd /\ y < cd)
          (ensures dj * cd + y < n)
= let z = Kuiper.Divides.get_factor cd n in
  ML.cancel_mul_div cd z;
  ML.lemma_mult_le_right cd (dj + 1) z;
  ML.swap_mul z cd;
  ML.distributivity_add_left dj 1 cd

let gran_col_bound_all (cd : pos) (n : nat) (dj : nat)
  : Lemma (requires cd /? n /\ dj < n / cd)
          (ensures forall (y : natlt cd). dj * cd + y < n)
= introduce forall (y : natlt cd). dj * cd + y < n
  with gran_col_bound cd n dj y

(* ---------------------------------------------------------------------- *)
(* the innermost accumulation step, with nothing else in scope              *)
(* ---------------------------------------------------------------------- *)

(* Add one [chunk et_acc]-wide vector read into the [chunk et_d]-wide running
   fold.  Deliberately stated over two plain local buffers and a ghost family
   [g] of per-column split terms, with no tensor, permission or view anywhere
   in the context: this is the one VC in pass 2 that quantifies over both the
   granule width and the vector width, and keeping its context minimal is what
   keeps it cheap.  [inline_for_extraction], so it disappears into the caller.

   TODO(upstream): this is a generic "fold a vector read into an accumulator
   run" and belongs next to [Kuiper.Array2.Vectorized]. *)
#push-options "--z3rlimit 10 --fuel 1 --ifuel 1"
inline_for_extraction noextract
fn acc_chunk
  (#et_acc : Type0) {| scalar et_acc |}
  (cd ca : szp)
  (abuf : A.array et_acc)
  (vbuf : A.array et_acc)
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
    (abuf |-> av) **
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
    let b = A.op_Array_Access vbuf xv #1.0R #bv;
    with av. assert abuf |-> av;
    let a = A.op_Array_Access abuf (ev +^ xv) #1.0R #av;
    A.op_Array_Assignment abuf (ev +^ xv) (add a b) #av;
    xc := !xc +^ 1sz;
  };
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* phase 1 of the body: the split reduction                                 *)
(* ---------------------------------------------------------------------- *)

(* Sum the [splits] workspace rows that contribute to granule [(di, dj)] into
   [abuf].  Split out of [rkf] so that the reduction and the epilogue are two
   separate, small verification conditions instead of one large one: the
   combined query was both slow and brittle.  Nothing here mentions C, D or
   the combiner -- this phase is IDENTICAL to the no-epi kernel's, which is
   the point. *)
#push-options "--z3rlimit 20 --fuel 1 --ifuel 1 --split_queries no"
inline_for_extraction noextract
fn accumulate
  (#et_acc #et_d : Type0)
  {| scalar et_acc, has_vec_cpy et_acc |}
  {| sized et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (gW : array2 et_acc lW)
  (abuf : A.array et_acc)
  (vbuf : A.array et_acc)
  (splits : szp)
  (di : szlt m)
  (dj : szlt (SZ.v n / SZ.v (chunk et_d)))
  (#fW : perm) (#eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (sq : squash (
     SZ.v (chunk et_d) /?+ SZ.v n /\
     SZ.v (chunk et_acc) /?+ SZ.v (chunk et_d) /\
     SZ.v mws == SZ.v splits * SZ.v m /\
     aligned 16 (T.core gW) /\
     aligned_strided_row_major (SZ.v (chunk et_acc)) strW /\
     SZ.fits lW.ulen))
  (#av0 : erased (seq et_acc))
  (#bv0 : erased (seq et_acc))
  ()
  preserves gpu
  preserves gW |-> Frac fW eW
  requires abuf |-> av0 ** (vbuf |-> bv0)
  requires pure (
    A.length abuf == SZ.v (chunk et_d) /\ A.length vbuf == SZ.v (chunk et_acc) /\
    Seq.length av0 == SZ.v (chunk et_d) /\ Seq.length bv0 == SZ.v (chunk et_acc) /\
    aligned 16 vbuf /\
    (forall (e : natlt (SZ.v (chunk et_d))).
       Seq.index av0 e == (zero <: et_acc)))
  ensures exists* (av bv : seq et_acc).
    (abuf |-> av) ** (vbuf |-> bv) **
    pure (Seq.length av == SZ.v (chunk et_d) /\
          Seq.length bv == SZ.v (chunk et_acc) /\
          (forall (e : natlt (SZ.v (chunk et_d))).
             Seq.index av e
             == SL.sum_upto
                  (EL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
                     (SZ.v dj * SZ.v (chunk et_d)) e)
                  (SZ.v splits)))
{
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
                    (EL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
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
                      (EL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
                         (SZ.v dj * SZ.v (chunk et_d)) e)
                      (if e < SZ.v ev then SZ.v sv + 1 else SZ.v sv)))
      decreases (SZ.v (chunk et_d) - SZ.v !ec)
    {
      let ev = !ec;
      let wcol : szlt (SZ.v n - SZ.v (chunk et_acc) + 1) = dj *^ chunk et_d +^ ev;
      Kuiper.Divides.lemma_divides_product_r (SZ.v (chunk et_acc)) (SZ.v dj) (SZ.v (chunk et_d));
      Kuiper.Divides.lemma_divides_sum (SZ.v (chunk et_acc))
        (SZ.v dj * SZ.v (chunk et_d)) (SZ.v ev);
      assert pure (SZ.v wcol == SZ.v dj * SZ.v (chunk et_d) + SZ.v ev);
      assert pure (SZ.v (chunk et_acc) /? SZ.v wcol);
      divides_helper (SZ.v (chunk et_acc)) strW.offset strW.stride
        (SZ.v wrow) (SZ.v wcol);
      cell_aligned16 lW gW (SZ.v wrow) (SZ.v wcol);
      array2_vec_read gW wrow wcol vbuf;
      RL.ws_row_bound (SZ.v m) (SZ.v splits) (SZ.v mws) (SZ.v sv) (SZ.v di);
      (* The vector just read is split [sv]'s contribution to columns
         [ev .. ev + chunk et_acc) of this granule. *)
      acc_chunk (chunk et_d) (chunk et_acc) abuf vbuf
        (EL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
           (SZ.v dj * SZ.v (chunk et_d)))
        (SZ.v sv) ev ();
      Kuiper.Divides.lemma_divides_sum (SZ.v (chunk et_acc))
        (SZ.v ev) (SZ.v (chunk et_acc));
      div_step (SZ.v (chunk et_acc)) (SZ.v ev) (SZ.v (chunk et_d));
      ec := !ec +^ chunk et_acc;
    };
    sc := !sc +^ 1sz;
  };

}
#pop-options

(* ---------------------------------------------------------------------- *)
(* per-thread body                                                          *)
(* ---------------------------------------------------------------------- *)

#push-options "--z3rlimit 20 --fuel 1 --ifuel 1 --split_queries no"
inline_for_extraction noextract
fn rkf
  (#et_acc #et_c #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| scalar et_c, real_like et_c |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n)) {| RO.cvtlayout lC |}
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gW : array2 et_acc lW)
  (gC : RO.roarray2 et_c lC)
  (gD : array2 et_d lD)
  (splits : szp)
  (comb : et_c -> et_acc -> et_d)
  (comb_r : binop real { approx2 comb comb_r })
  (sq : squash (rk_sq gW gD splits))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (#fW : perm) (#eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (#fC : perm) (#eC : chest2 et_c (SZ.v m) (SZ.v n))
  (rW : chest2 real (SZ.v mws) (SZ.v n) { eW %~ rW })
  (rC : chest2 real (SZ.v m) (SZ.v n) { eC %~ rC })
  (i : szlt njobs)
  ()
  preserves gpu
  preserves gC |-> Frac (fC /. njobs) eC
  preserves gW |-> Frac (fW /. njobs) eW
  requires d_gran gD () (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n)
  ensures d_gran_at gD ()
            (EL.gran_target (SZ.v m) (SZ.v splits) rC rW comb_r)
            (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n)
{
  let gc : szp = n /^ chunk et_d;
  let di : szlt m = i /^ gc;
  let dj : szlt (SZ.v n / SZ.v (chunk et_d)) = i %^ gc;

  unfold (d_gran gD () (SZ.v i / gcols et_d n) (SZ.v i % gcols et_d n));
  assert pure (SZ.v (chunk et_acc) /?+ SZ.v (chunk et_d));
  assert pure (SZ.v (chunk et_d) /?+ SZ.v n);
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

  accumulate gW abuf vbuf splits di dj () ();

  (* The epilogue.  [abuf] now holds the COMPLETE k reduction for this
     granule, so this is the first and only point at which [comb] may be
     applied.  C is read scalarly through the view's index map, once per
     element; the store below is still a single 128-bit write. *)
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
                    (EL.ws_run (SZ.v m) (SZ.v splits) eW (SZ.v di)
                       (SZ.v dj * SZ.v (chunk et_d)) e)
                    (SZ.v splits)) /\
            (forall (y : natlt (SZ.v (chunk et_d))).
               y < SZ.v yv ==>
                 Seq.index ov y
                 == comb (acc2 eC (SZ.v di) (SZ.v dj * SZ.v (chunk et_d) + y))
                         (Seq.index av y)))
    decreases (SZ.v (chunk et_d) - SZ.v !yc)
  {
    let yv = !yc;
    gran_col_bound (SZ.v (chunk et_d)) (SZ.v n) (SZ.v dj) (SZ.v yv);
    let ccol : szlt n = dj *^ chunk et_d +^ yv;
    let cv = RO.tensor_read gC (di, (ccol, ()));
    with av. assert abuf |-> av;
    let a = A.op_Array_Access abuf yv #1.0R #av;
    with ov. assert obuf |-> ov;
    A.op_Array_Assignment obuf yv (comb cv a) #ov;
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
  EL.ws_run_sum_all (SZ.v m) (SZ.v splits) eW (SZ.v di)
    (SZ.v dj * SZ.v (chunk et_d)) (SZ.v (chunk et_d)) (SZ.v splits);
  EL.gran_approx (SZ.v m) (SZ.v splits) eC rC eW rW comb comb_r
    (SZ.v (chunk et_d)) v' (SZ.v di) (SZ.v dj) ();
  fold (d_gran_at gD ()
          (EL.gran_target (SZ.v m) (SZ.v splits) rC rW comb_r)
          (SZ.v di) (SZ.v dj));
  assert pure (SZ.v gc == SZ.v n / SZ.v (chunk et_d));
  assert pure (SZ.v di == SZ.v i / gcols et_d n);
  assert pure (SZ.v dj == SZ.v i % gcols et_d n);
  rewrite each (SZ.v di) as (SZ.v i / gcols et_d n);
  rewrite each (SZ.v dj) as (SZ.v i % gcols et_d n);
}
#pop-options

(* ---------------------------------------------------------------------- *)
(* kernel descriptor                                                        *)
(* ---------------------------------------------------------------------- *)

let rk_sendable
  (#et_acc #et_c #et_d : Type0)
  {| sized et_acc, has_vec_cpy et_acc, sized et_c, sized et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW { is_global gW })
  (gC : RO.roarray2 et_c lC { RO.is_global gC })
  (gD : array2 et_d lD { is_global gD })
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (fW : perm) (eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (fC : perm) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (i : natlt njobs)
  : is_send_across gpu_of
      ((gC |-> Frac (fC /. njobs) eC) **
       ((gW |-> Frac (fW /. njobs) eW) **
        d_gran gD () (i / gcols et_d n) (i % gcols et_d n)))
= is_send_across_star
    (gC |-> Frac (fC /. njobs) eC)
    ((gW |-> Frac (fW /. njobs) eW) **
     d_gran gD () (i / gcols et_d n) (i % gcols et_d n))
    #(RO.is_send_across_global_tensor gC #(fC /. njobs) eC)
    #(is_send_across_star
        (gW |-> Frac (fW /. njobs) eW)
        (d_gran gD () (i / gcols et_d n) (i % gcols et_d n))
        #(is_send_across_global_tensor gW #(fW /. njobs) eW)
        #(d_gran_sendable gD () (i / gcols et_d n) (i % gcols et_d n)))

let rk_post_sendable
  (#et_acc #et_c #et_d : Type0)
  {| sized et_acc, has_vec_cpy et_acc |}
  {| sized et_c |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n))
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n))
  (#lD : layout2 (SZ.v m) (SZ.v n))
  (gW : array2 et_acc lW { is_global gW })
  (gC : RO.roarray2 et_c lC { RO.is_global gC })
  (gD : array2 et_d lD { is_global gD })
  (sq : squash (SZ.v (chunk et_d) /?+ SZ.v n))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) })
  (fW : perm) (eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (fC : perm) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (rD : chest2 real (SZ.v m) (SZ.v n))
  (i : natlt njobs)
  : is_send_across gpu_of
      ((gC |-> Frac (fC /. njobs) eC) **
       ((gW |-> Frac (fW /. njobs) eW) **
        d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n)))
= is_send_across_star
    (gC |-> Frac (fC /. njobs) eC)
    ((gW |-> Frac (fW /. njobs) eW) **
     d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n))
    #(RO.is_send_across_global_tensor gC #(fC /. njobs) eC)
    #(is_send_across_star
        (gW |-> Frac (fW /. njobs) eW)
        (d_gran_at gD () rD (i / gcols et_d n) (i % gcols et_d n))
        #(is_send_across_global_tensor gW #(fW /. njobs) eW)
        #(d_gran_at_sendable gD () rD (i / gcols et_d n) (i % gcols et_d n)))

inline_for_extraction noextract
let mk_reduce_kernel
  (#et_acc #et_c #et_d : Type0)
  {| scalar et_acc, real_like et_acc, has_vec_cpy et_acc |}
  {| scalar et_c, real_like et_c |}
  {| scalar et_d, real_like et_d, has_vec_cpy et_d |}
  (#m #n #mws : szp)
  (#lW : layout2 (SZ.v mws) (SZ.v n)) {| T.ctlayout lW |}
       {| strW : strided_row_major (vtlayout_of_tlayout lW) |}
  (#lC : RO.vlayout2 (SZ.v m) (SZ.v n)) {| RO.cvtlayout lC |}
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
       {| strD : strided_row_major (vtlayout_of_tlayout lD) |}
  (gW : array2 et_acc lW { is_global gW })
  (gC : RO.roarray2 et_c lC { RO.is_global gC })
  (gD : array2 et_d lD { is_global gD })
  (splits : szp)
  (comb : et_c -> et_acc -> et_d)
  (comb_r : binop real { approx2 comb comb_r })
  (sq : squash (rk_sq gW gD splits))
  (njobs : szp { SZ.v njobs == SZ.v m * gcols et_d n /\
                 gran_ub (SZ.v m) (gcols et_d n) /\
                 njobs <= max_blocks * max_threads })
  (fW : perm) (eW : chest2 et_acc (SZ.v mws) (SZ.v n))
  (fC : perm) (eC : chest2 et_c (SZ.v m) (SZ.v n))
  (rW : chest2 real (SZ.v mws) (SZ.v n) { eW %~ rW })
  (rC : chest2 real (SZ.v m) (SZ.v n) { eC %~ rC })
  ()
  : kernel_desc
      ((gW |-> Frac fW eW) ** (gC |-> Frac fC eC) ** live gD)
      ((gW |-> Frac fW eW) ** (gC |-> Frac fC eC) **
       (exists* (eD : chest2 et_d (SZ.v m) (SZ.v n)).
          gD |-> eD **
          pure (eD %~ EL.gran_target (SZ.v m) (SZ.v splits) rC rW comb_r)))
= {
    nthr = njobs;
    frame = pure True;
    kpre = (fun (i : natlt njobs) ->
      (gC |-> Frac (fC /. njobs) eC) **
      ((gW |-> Frac (fW /. njobs) eW) **
       d_gran gD () (i / gcols et_d n) (i % gcols et_d n)));
    kpost = (fun (i : natlt njobs) ->
      (gC |-> Frac (fC /. njobs) eC) **
      ((gW |-> Frac (fW /. njobs) eW) **
       d_gran_at gD () (EL.gran_target (SZ.v m) (SZ.v splits) rC rW comb_r)
         (i / gcols et_d n) (i % gcols et_d n)));
    setup = resetup gW gC gD () njobs #fW #eW #fC #eC;
    teardown = reteardown gW gC gD () njobs
      (EL.gran_target (SZ.v m) (SZ.v splits) rC rW comb_r) #() #fW #eW #fC #eC;
    f = rkf gW gC gD splits comb comb_r () njobs #fW #eW #fC #eC rW rC;
    kpre_sendable = rk_sendable gW gC gD () njobs fW eW fC eC;
    kpost_sendable = rk_post_sendable gW gC gD () njobs fW eW fC eC
      (EL.gran_target (SZ.v m) (SZ.v splits) rC rW comb_r);
  } <: kernel_desc_n _ _
