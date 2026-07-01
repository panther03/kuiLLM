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

// torch.bool is stored as a single byte; we model it as u8 with 0/1 values.
inline_for_extraction noextract
let of_bool (b : bool) : u8 = if b then 1uy else 0uy

inline_for_extraction noextract
let lt_u8 (#t:Type0) {| scalar t |} (x : t) (y : t) : u8 = of_bool (lt x y)

inline_for_extraction noextract
let le_u8 (#t:Type0) {| scalar t |} (x : t) (y : t) : u8 = of_bool (lte x y)

inline_for_extraction noextract
let eq_u8 (#t:Type0) {| scalar t |} (x : t) (y : t) : u8 = of_bool (eq x y)

inline_for_extraction noextract
let band (x : u8) (y : u8) : u8 = FStar.UInt8.logand x y

inline_for_extraction noextract
let bor (x : u8) (y : u8) : u8 = FStar.UInt8.logor x y

inline_for_extraction noextract
let bnot (x : u8) : u8 = FStar.UInt8.logxor x 1uy

// where c x y := x if the boolean-as-byte c is nonzero, else y.
inline_for_extraction noextract
let bwhere (#t:Type0) (c : u8) (x : t) (y : t) : t =
  if FStar.UInt8.gt c 0uy then x else y