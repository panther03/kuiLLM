module Kuiops.Elementwise

#lang-pulse
open Kuiper

inline_for_extraction noextract
let silu (#t:Type0) {| floating t |} (x : t) : t =
  mul x (div one (add one (fexp (sub zero x))))

inline_for_extraction noextract
let neg (#t:Type0) {| floating t |} (x : t) : t = sub zero x

inline_for_extraction noextract
let square (#t:Type0) {| scalar t |} (x : t) : t = x `mul` x

inline_for_extraction noextract
let add_alpha (#t:Type0) {| scalar t |} (x : t) (alpha: t) (y: t): t = x `add` (alpha `mul` y)