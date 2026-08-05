module Kuiops.HReducePoly.Ops

open Kuiper
open Kuiper.Functions

inline_for_extraction noextract
val add_u8 (x y : u8) : Tot u8
val add_u8_assoc (x y z : u8) :
  Lemma (add_u8 (add_u8 x y) z == add_u8 x (add_u8 y z))

inline_for_extraction noextract
val add_u16 (x y : u16) : Tot u16
val add_u16_assoc (x y z : u16) :
  Lemma (add_u16 (add_u16 x y) z == add_u16 x (add_u16 y z))

inline_for_extraction noextract
val add_u32 (x y : u32) : Tot u32
val add_u32_assoc (x y z : u32) :
  Lemma (add_u32 (add_u32 x y) z == add_u32 x (add_u32 y z))

inline_for_extraction noextract
val add_u64 (x y : u64) : Tot u64
val add_u64_assoc (x y z : u64) :
  Lemma (add_u64 (add_u64 x y) z == add_u64 x (add_u64 y z))

inline_for_extraction noextract
val mul_u8 (x y : u8) : Tot u8
val mul_u8_assoc (x y z : u8) :
  Lemma (mul_u8 (mul_u8 x y) z == mul_u8 x (mul_u8 y z))

inline_for_extraction noextract
val mul_u16 (x y : u16) : Tot u16
val mul_u16_assoc (x y z : u16) :
  Lemma (mul_u16 (mul_u16 x y) z == mul_u16 x (mul_u16 y z))

inline_for_extraction noextract
val mul_u32 (x y : u32) : Tot u32
val mul_u32_assoc (x y z : u32) :
  Lemma (mul_u32 (mul_u32 x y) z == mul_u32 x (mul_u32 y z))

inline_for_extraction noextract
val mul_u64 (x y : u64) : Tot u64
val mul_u64_assoc (x y z : u64) :
  Lemma (mul_u64 (mul_u64 x y) z == mul_u64 x (mul_u64 y z))

inline_for_extraction noextract
val and_u8 (x y : u8) : Tot u8
val and_u8_assoc (x y z : u8) :
  Lemma (and_u8 (and_u8 x y) z == and_u8 x (and_u8 y z))

inline_for_extraction noextract
val or_u8 (x y : u8) : Tot u8
val or_u8_assoc (x y z : u8) :
  Lemma (or_u8 (or_u8 x y) z == or_u8 x (or_u8 y z))

(* Approximate operators: generic over any [scalar] element type, with the
   associativity obligation discharged over the reals only. *)

inline_for_extraction noextract
val add_a (#et : Type0) {| scalar et |} (x y : et) : Tot et
let add_a_r (x y : real) : Tot real = x +. y
val add_a_r_assoc (x y z : real) :
  Lemma (add_a_r (add_a_r x y) z == add_a_r x (add_a_r y z))
val add_a_ok (et : Type0) {| scalar et, real_like et |} :
  Lemma (approx2 (add_a #et) add_a_r)

inline_for_extraction noextract
val mul_a (#et : Type0) {| scalar et |} (x y : et) : Tot et
let mul_a_r (x y : real) : Tot real = x *. y
val mul_a_r_assoc (x y z : real) :
  Lemma (mul_a_r (mul_a_r x y) z == mul_a_r x (mul_a_r y z))
val mul_a_ok (et : Type0) {| scalar et, real_like et |} :
  Lemma (approx2 (mul_a #et) mul_a_r)
