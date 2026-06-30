module Kuiops.Common

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor

module SZ = Kuiper.SizeT

let rec abs_set_at (#r : nat) (#d : shape r) (dim : natlt r) (idx : natlt (d @! dim)) (x : abs d)
  : GTot (abs d)
         (decreases r)
  = assert r > 0;
    let (i1, i2) = x <: natlt (d @! 0) & abs (tail d) in
    if dim == 0 then
      (idx, i2)
    else
      (i1, abs_set_at #_ #(tail d) (dim - 1) idx i2)

inline_for_extraction noextract
let rec conc_set_at (#r : erased nat) (#d : shape r) (dim : szlt r) (idx : szlt (d @! dim)) (x : conc d)
  : Tot (c : conc d {up c == abs_set_at dim idx (up x)})
         (decreases (SZ.v dim))
  = assert r > 0;
    let (i1,i2) = x <: szlt (d @! 0) & conc (tail d) in
    if dim = 0sz then
      (idx, i2)
    else (
      let x1 = conc_set_at #_ #(tail d) (dim -^ 1sz) idx i2 in
      (i1, (x1 <: (conc (tail d))))
    )

let rec shape_le (#r: erased nat) (d1 d2: shape r): prop =
  match d1 with 
  | INil -> true 
  | ICons d1h d1t -> (match d2 with 
    | ICons d2h d2t -> d1h <= d2h /\ shape_le d1t d2t)

// let rec abs_le (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (x: abs d1): GTot (abs d2) =
//   match d1 with
//   | INil -> ()
//   | ICons _ _ -> (
//     let xh,xt = x <: natlt (d1 @! 0) & abs (tail d1) in
//     (xh <: natlt (d2 @! 0)), abs_le #(r-1) (tail d1) (tail d2) xt
//   )

let rec abs_set_at2 (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (dim : natlt r) (idx : natlt (d2 @! dim)) (x : abs d1)
  : GTot (abs d2)
         (decreases dim)
  = assert r > 0;
    assert (d1 @! 0) <= (d2 @! 0);
    let (i1, i2) = x <: natlt (d1 @! 0) & abs (tail d1) in    
    let i1: natlt (d2 @! 0) = i1 in
    if dim == 0 then (
      // TODO: how can I justify to fstar that abs (tail d1) is 
      // a subtype of abs (tail d2)? im not even sure how to write that 
      // as a provable proposition
      assume (tail d1) == (tail d2); 
      (idx, (i2 <: abs (tail d2)))
    ) else
      (i1, abs_set_at2 #_ (tail d1) (tail d2) (dim - 1) idx i2)

inline_for_extraction noextract
let rec conc_set_at2 (#r : erased nat) (d1 d2 : shape r { shape_le d1 d2 }) (dim : szlt r) (idx : szlt (d2 @! dim)) (x : conc d1)
  : Tot (c : conc d2 {up c == abs_set_at2 d1 d2 dim idx (up x)})
         (decreases (SZ.v dim))
  = assert r > 0;
    let (i1,i2) = x <: szlt (d1 @! 0) & conc (tail d1) in
    let i1 = i1 <: szlt (d2 @! 0) in
    if dim = 0sz then (
      // TODO see above with abs_set_at2
      assume (tail d1) == (tail d2);
      (idx, i2)
    ) else (
      let x1 = conc_set_at2 #_ (tail d1) (tail d2) (dim -^ 1sz) idx i2 in
      (i1, (x1 <: (conc (tail d2))))
    )