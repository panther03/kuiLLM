module Kuiops.SuperGEMM.Mm.SplitK.Gather

(* Gather [k] equal fractional shares of a tensor back into one, keeping the
   functional content.

   [Kuiper.Tensor.tensor_gather_n] needs every share to carry the *same* chest,
   and [tensor_gather_n_underspec] throws the contents away.  A warp that drains
   its accumulator with [mma_store] leaves 32 independently existentially
   quantified shares that all approximate the same real matrix, which fits
   neither.  [unify_shares] closes the gap: given one named share, it rewrites
   every share of a [forevery] to that same chest with [tensor_pts_to_eq], after
   which the ordinary [tensor_gather_n] applies.

   TODO(upstream): this belongs next to [tensor_gather_n] in Kuiper.Tensor. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiper.ForEvery

module SZ = Kuiper.SizeT

#push-options "--fuel 1 --ifuel 1 --z3rlimit 10"

ghost
fn rec unify_shares
  (#et : Type0) (#r : nat) (#d : shape r) (#l : tlayout d)
  (a : tensor et l) (k : nat) (#f #f0 : perm) (e : chest d et)
  requires
    (forall+ (_ : natlt k). exists* (s : chest d et). a |-> Frac f s) **
    (a |-> Frac f0 e)
  ensures
    (forall+ (_ : natlt k). a |-> Frac f e) **
    (a |-> Frac f0 e)
  decreases k
{
  if (k = 0) {
    forevery_elim_empty #(natlt k) (fun _ -> exists* (s : chest d et). a |-> Frac f s);
    forevery_intro_empty #(natlt k) (fun _ -> a |-> Frac f e);
  } else {
    forevery_natlt_pop k (fun _ -> exists* (s : chest d et). a |-> Frac f s);
    unify_shares a (k - 1) #f #f0 e;
    with s. assert (a |-> Frac f s);
    tensor_pts_to_eq a #f f0 #s #e;
    forevery_natlt_push k (fun _ -> a |-> Frac f e);
  }
}

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 10"

(* Gather [k] lane shares that each approximate [rm] into a single share that
   approximates [rm]: the shape a warp leaves behind after [mma_store]. *)
ghost
fn array2_gather_n_approximates
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (k : pos) (#f : perm)
  (rm : chest2 real rows cols)
  requires
    forall+ (_ : natlt k).
      exists* (s : chest2 et rows cols). a |-> Frac (f /. k) s ** pure (s %~ rm)
  ensures
    exists* (s : chest2 et rows cols). a |-> Frac f s ** pure (s %~ rm)
{
  forevery_natlt_pop k
    (fun (_ : natlt k) ->
      exists* (s : chest2 et rows cols). a |-> Frac (f /. k) s ** pure (s %~ rm));
  with e0. assert (a |-> Frac (f /. k) e0);
  forevery_map
    (fun (_ : natlt (k - 1)) ->
      exists* (s : chest2 et rows cols). a |-> Frac (f /. k) s ** pure (s %~ rm))
    (fun (_ : natlt (k - 1)) ->
      exists* (s : chest2 et rows cols). a |-> Frac (f /. k) s)
    fn _ { () };
  unify_shares a (k - 1) #(f /. k) #(f /. k) e0;
  forevery_natlt_push k (fun (_ : natlt k) -> a |-> Frac (f /. k) e0);
  tensor_gather_n a k #f #e0;
}

#pop-options

#push-options "--fuel 1 --ifuel 1 --z3rlimit 10"

(* TODO(upstream): a copy of the [array2_untile_approximates] of
   [Kuiops.Kernel.GEMM.TensorCore2D.To.KernelDesc.Teardown], which has no
   [.fsti] entry and so cannot be reused. *)
ghost
fn array2_untile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? cols})
  {| enumerable (natlt (rows / trows)),
     enumerable (natlt (cols / tcols)) |}
  (r : chest2 real rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (#_ : squash (SZ.fits (rows / trows)))
  (#_ : squash (SZ.fits (cols / tcols)))
  requires
    forall+ (tr : natlt (rows / trows))
             (tc : natlt (cols / tcols)).
      exists* (em : chest2 et trows tcols).
        array2_subtile m trows tcols tr tc |-> em **
        pure (em %~ ematrix_subtile r trows tcols tr tc)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
{
  let ff = forevery_exists_2
    #(natlt (rows / trows)) #_ #(natlt (cols / tcols)) #_
    (fun tr tc (em : chest2 et trows tcols) ->
      array2_subtile m trows tcols tr tc |-> em **
      pure (em %~ ematrix_subtile r trows tcols tr tc));
  forevery_extract_pure_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      ff tr tc %~ ematrix_subtile r trows tcols tr tc)
    fn tr tc { () };
  assert pure (forall (tr : natlt (rows / trows))
                      (tc : natlt (cols / tcols)).
    ff tr tc %~ ematrix_subtile r trows tcols tr tc);
  forevery_map_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc)
    fn tr tc { () };
  array2_untile' m trows tcols ff;
  assert pure (ematrix_from_tiles trows tcols ff %~ r);
}

#pop-options
