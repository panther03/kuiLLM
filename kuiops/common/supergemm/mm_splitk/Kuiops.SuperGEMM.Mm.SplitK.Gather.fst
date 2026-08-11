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
open Kuiper.ForEvery

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
