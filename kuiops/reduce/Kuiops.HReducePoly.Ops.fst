module Kuiops.HReducePoly.Ops

open Kuiper
module M = FStar.Math.Lemmas
module U = FStar.UInt

let mul_mod_assoc (#n : pos) (a b c : U.uint_t n) :
  Lemma (U.mul_mod (U.mul_mod a b) c == U.mul_mod a (U.mul_mod b c))
=
  M.lemma_mod_mul_distr_l (a * b) c (pow2 n);
  M.lemma_mod_mul_distr_r a (b * c) (pow2 n)

inline_for_extraction noextract
let add_u8 = FStar.UInt8.add_mod
let add_u8_assoc x y z =
  Kuiper.Math.add_mod_assoc (FStar.UInt8.v x) (FStar.UInt8.v y) (FStar.UInt8.v z);
  FStar.UInt8.v_inj (add_u8 (add_u8 x y) z) (add_u8 x (add_u8 y z))

inline_for_extraction noextract
let add_u16 = FStar.UInt16.add_mod
let add_u16_assoc x y z =
  Kuiper.Math.add_mod_assoc (FStar.UInt16.v x) (FStar.UInt16.v y) (FStar.UInt16.v z);
  FStar.UInt16.v_inj (add_u16 (add_u16 x y) z) (add_u16 x (add_u16 y z))

inline_for_extraction noextract
let add_u32 = FStar.UInt32.add_mod
let add_u32_assoc x y z =
  Kuiper.Math.add_mod_assoc (FStar.UInt32.v x) (FStar.UInt32.v y) (FStar.UInt32.v z);
  FStar.UInt32.v_inj (add_u32 (add_u32 x y) z) (add_u32 x (add_u32 y z))

inline_for_extraction noextract
let add_u64 = FStar.UInt64.add_mod
let add_u64_assoc x y z =
  Kuiper.Math.add_mod_assoc (FStar.UInt64.v x) (FStar.UInt64.v y) (FStar.UInt64.v z);
  FStar.UInt64.v_inj (add_u64 (add_u64 x y) z) (add_u64 x (add_u64 y z))

inline_for_extraction noextract
let mul_u8 = FStar.UInt8.mul_mod
let mul_u8_assoc x y z =
  mul_mod_assoc (FStar.UInt8.v x) (FStar.UInt8.v y) (FStar.UInt8.v z);
  FStar.UInt8.v_inj (mul_u8 (mul_u8 x y) z) (mul_u8 x (mul_u8 y z))

inline_for_extraction noextract
let mul_u16 = FStar.UInt16.mul_mod
let mul_u16_assoc x y z =
  mul_mod_assoc (FStar.UInt16.v x) (FStar.UInt16.v y) (FStar.UInt16.v z);
  FStar.UInt16.v_inj (mul_u16 (mul_u16 x y) z) (mul_u16 x (mul_u16 y z))

inline_for_extraction noextract
let mul_u32 = FStar.UInt32.mul_mod
let mul_u32_assoc x y z =
  mul_mod_assoc (FStar.UInt32.v x) (FStar.UInt32.v y) (FStar.UInt32.v z);
  FStar.UInt32.v_inj (mul_u32 (mul_u32 x y) z) (mul_u32 x (mul_u32 y z))

inline_for_extraction noextract
let mul_u64 = FStar.UInt64.mul_mod
let mul_u64_assoc x y z =
  mul_mod_assoc (FStar.UInt64.v x) (FStar.UInt64.v y) (FStar.UInt64.v z);
  FStar.UInt64.v_inj (mul_u64 (mul_u64 x y) z) (mul_u64 x (mul_u64 y z))

inline_for_extraction noextract
let and_u8 = FStar.UInt8.logand
let and_u8_assoc x y z =
  FStar.UInt.logand_associative
    (FStar.UInt8.v x) (FStar.UInt8.v y) (FStar.UInt8.v z);
  FStar.UInt8.v_inj (and_u8 (and_u8 x y) z) (and_u8 x (and_u8 y z))

inline_for_extraction noextract
let or_u8 = FStar.UInt8.logor
let or_u8_assoc x y z =
  FStar.UInt.logor_associative
    (FStar.UInt8.v x) (FStar.UInt8.v y) (FStar.UInt8.v z);
  FStar.UInt8.v_inj (or_u8 (or_u8 x y) z) (or_u8 x (or_u8 y z))
