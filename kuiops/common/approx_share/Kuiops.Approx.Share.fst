module Kuiops.Approx.Share

(* Sharing a read capability whose contents are known only up to [%~].

   A float kernel's postcondition cannot name the exact bits it wrote, only the
   reals they approximate, so ownership of its output comes back as
   [exists* e. (a |-> Frac f e) ** pure (e %~ r)].  That existential is fatal
   for a dependent launch: [kernel_desc] fixes [kpre] before [setup] runs, so a
   [kpre] mentioning a concrete [e] can only be satisfied after the host has
   redeemed the producer's pledge -- which means a [sync_stream], which means
   the pair cannot be captured in a CUDA graph.  Pushing the existential inside
   [kpre] removes the need to name [e] at all.

   Doing that costs a way to split and re-join the capability, which is what
   this module provides.  Splitting is easy.  Re-joining is not: the [k]
   per-thread existentials are independent, so nothing says the witnesses agree,
   and gathering them underspecified would lose the [%~ r] fact.  [approx_gather]
   fixes this by having the caller retain one share -- in the descriptor's
   [frame], which never leaves the host -- and reconciling the anonymous
   gathered witness against it with [tensor_pts_to_eq].

   TODO: upstream to Kuiper. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery
open Kuiper.Approximates

module T = Kuiper.Tensor

(* Read access to [a] at fraction [f], with the contents pinned only up to
   [%~ rm].  This is Kuiper's [( |~> )] under a name that inference can see. *)
let approx_pts_to
  (#et : Type0) (#r : nat) (#d : shape r) (#l : tlayout d)
  {| scalar et, real_like et |}
  (a : tensor et l)
  (f : perm)
  (rm : chest d real)
  : slprop
= exists* (e : chest d et). (a |-> Frac f e) ** pure (e %~ rm)

let approx_pts_to_sendable
  (#et : Type0) (#r : nat) (#d : shape r) (#l : tlayout d)
  {| scalar et, real_like et |}
  (a : tensor et l { is_global a })
  (f : perm)
  (rm : chest d real)
  : is_send_across gpu_of (approx_pts_to a f rm)
= let ff (e : chest d et)
    : is_send_across gpu_of ((a |-> Frac f e) ** pure (e %~ rm))
  = is_send_across_star (a |-> Frac f e) (pure (e %~ rm))
      #(is_send_across_global_tensor a #f e)
      #(is_send_across_placeless (pure (e %~ rm)) #(placeless_pure (e %~ rm))) in
  is_send_across_exists
    (fun (e : chest d et) -> (a |-> Frac f e) ** pure (e %~ rm)) #ff

(* Split [f] into a retained half and [k] equal shares of the other half. *)
ghost
fn approx_share
  (#et : Type0) (#r : nat) (#d : shape r) (#l : tlayout d)
  {| scalar et, real_like et |}
  (a : tensor et l)
  (f : perm)
  (rm : chest d real)
  (k : pos)
  requires approx_pts_to a f rm
  ensures (forall+ (_ : natlt k). approx_pts_to a ((f /. 2) /. k) rm) **
          approx_pts_to a (f /. 2) rm
{
  unfold (approx_pts_to a f rm);
  with e. assert (a |-> Frac f e);
  tensor_share_n a 2;
  forevery_natlt_pop 2 (fun (_ : natlt 2) -> a |-> Frac (f /. 2) e);
  forevery_singleton_elim (fun (_ : natlt 1) -> a |-> Frac (f /. 2) e);
  fold (approx_pts_to a (f /. 2) rm);
  tensor_share_n a k;
  forevery_map
    (fun (_ : natlt k) -> a |-> Frac ((f /. 2) /. k) e)
    (fun (_ : natlt k) -> approx_pts_to a ((f /. 2) /. k) rm)
    fn _ {
      fold (approx_pts_to a ((f /. 2) /. k) rm);
    };
}

(* The inverse.  The [k] shares carry independent witnesses, so they are
   gathered underspecified and the result is then rewritten to the retained
   share's witness, which is the one still known to approximate [rm]. *)
ghost
fn approx_gather
  (#et : Type0) (#r : nat) (#d : shape r) (#l : tlayout d)
  {| scalar et, real_like et |}
  (a : tensor et l)
  (f : perm)
  (rm : chest d real)
  (k : pos)
  requires (forall+ (_ : natlt k). approx_pts_to a ((f /. 2) /. k) rm) **
           approx_pts_to a (f /. 2) rm
  ensures approx_pts_to a f rm
{
  forevery_map
    (fun (_ : natlt k) -> approx_pts_to a ((f /. 2) /. k) rm)
    (fun (_ : natlt k) -> exists* (e : chest d et). a |-> Frac ((f /. 2) /. k) e)
    fn _ {
      unfold (approx_pts_to a ((f /. 2) /. k) rm);
    };
  tensor_gather_n_underspec a k #(f /. 2);
  with e1. assert (tensor_pts_to a #(f /. 2) e1);
  unfold (approx_pts_to a (f /. 2) rm);
  with e0. _;
  tensor_pts_to_eq a #(f /. 2) (f /. 2) #e1 #e0;
  forevery_singleton_intro (fun (_ : natlt 1) -> a |-> Frac (f /. 2) e0);
  forevery_natlt_push 2 (fun (_ : natlt 2) -> a |-> Frac (f /. 2) e0);
  tensor_gather_n a 2;
  fold (approx_pts_to a f rm);
}
