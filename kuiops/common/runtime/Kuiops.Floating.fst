module Kuiops.Floating

open Kuiper.Floating

(* Kuiper currently exposes finiteness only through the ghost [kind] model.
   KuiLLM also needs the corresponding executable IEEE test in generated CUDA.
   Equality with oneself rejects NaNs, and the other two comparisons reject
   the signed infinities.  Keeping the correspondence theorem here makes this
   small project-specific trusted boundary explicit. *)
inline_for_extraction noextract
let fisfinite (#t : Type0) {| floating t |} (x : t) : bool =
  eq x x && not (eq x infinity) && not (eq x (neg infinity))

assume val fisfinite_spec
  (#t : Type0) {| floating t |} (x : t)
  : Lemma (ensures fisfinite x <==> Finite? (kind x))
          [SMTPat (fisfinite x)]
