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

// Embed an abstract index of d1 into the (pointwise larger) shape d2. Every
// coordinate i < d1 @! k <= d2 @! k, so the same value is in range for d2.
let rec abs_le (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (x: abs d1)
  : GTot (abs d2)
         (decreases r)
  = match d1 with
    | INil -> ()
    | ICons _ _ ->
      let i1, i2 = x <: natlt (d1 @! 0) & abs (tail d1) in
      ((i1 <: natlt (d2 @! 0)), abs_le (tail d1) (tail d2) i2)

let abs_le_cons (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (x: abs d1)
  : Lemma (requires ICons? d1)
          (ensures (let i1, i2 = x <: natlt (d1 @! 0) & abs (tail d1) in
                    abs_le d1 d2 x == ((i1 <: natlt (d2 @! 0)), abs_le (tail d1) (tail d2) i2)))
  = ()

let up_cons (#r : nat) (d : shape r { ICons? d }) (v : conc d)
  : Lemma (let i1, i2 = v <: szlt (d @! 0) & conc (tail d) in
           up v == ((SZ.v i1 <: natlt (d @! 0)), up i2))
  = ()

let conc_cons (#r : erased nat) (#d : shape r { ICons? d })
  (p : szlt (d @! 0) & conc (tail d)) : conc d
  = p

// Concrete counterpart of abs_le, commuting with up. The recursion is driven by
// the (non-erasable) cshapes: matching the erasable `shape` directly to build an
// informative `conc` would force a ghost computation.
inline_for_extraction noextract
let rec conc_le (#r : erased nat) (#d1 #d2 : shape r { shape_le d1 d2 })
  (cd1 : cshape d1) (cd2 : cshape d2) (x : conc d1)
  : Tot (c : conc d2 {up c == abs_le d1 d2 (up x)})
  = match cd1 with
    | CNil -> ()
    | CCons _ ct1 ->
      (match cd2 with
       | CCons _ ct2 ->
         assert (r > 0);
         let i1, i2 = x <: szlt (d1 @! 0) & conc (tail d1) in
         let res : conc d2 = conc_cons #_ #d2 ((i1 <: szlt (d2 @! 0)), conc_le ct1 ct2 i2) in
         abs_le_cons d1 d2 (up x);
         up_cons d1 x;
         up_cons d2 res;
         res)

let rec abs_set_at2 (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (dim : natlt r) (idx : natlt (d2 @! dim)) (x : abs d1)
  : GTot (abs d2)
         (decreases dim)
  = assert r > 0;
    assert (d1 @! 0) <= (d2 @! 0);
    let (i1, i2) = x <: natlt (d1 @! 0) & abs (tail d1) in    
    let i1: natlt (d2 @! 0) = i1 in
    if dim == 0 then
      (idx, abs_le (tail d1) (tail d2) i2)
    else
      (i1, abs_set_at2 #_ (tail d1) (tail d2) (dim - 1) idx i2)

let abs_set_at2_cons (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (dim : natlt r)
  (idx : natlt (d2 @! dim)) (x : abs d1)
  : Lemma (ensures (let i1, i2 = x <: natlt (d1 @! 0) & abs (tail d1) in
                    abs_set_at2 d1 d2 dim idx x ==
                      (if dim = 0
                       then ((idx <: natlt (d2 @! 0)), abs_le (tail d1) (tail d2) i2)
                       else ((i1 <: natlt (d2 @! 0)),
                             abs_set_at2 (tail d1) (tail d2) (dim - 1) idx i2))))
  = ()

inline_for_extraction noextract
let rec conc_set_at2 (#r : erased nat) (#d1 #d2 : shape r { shape_le d1 d2 })
  (cd1 : cshape d1) (cd2 : cshape d2) (dim : szlt r) (idx : szlt (d2 @! dim)) (x : conc d1)
  : Tot (c : conc d2 {up c == abs_set_at2 d1 d2 dim idx (up x)})
         (decreases (SZ.v dim))
  = assert r > 0;
    match cd1 with
    | CCons _ ct1 ->
      (match cd2 with
       | CCons _ ct2 ->
         let (i1,i2) = x <: szlt (d1 @! 0) & conc (tail d1) in
         abs_set_at2_cons d1 d2 (SZ.v dim) idx (up x);
         up_cons d1 x;
         let res : conc d2 =
           if dim = 0sz then
             conc_cons #_ #d2 ((idx <: szlt (d2 @! 0)), conc_le ct1 ct2 i2)
           else
             conc_cons #_ #d2 ((i1 <: szlt (d2 @! 0)),
                               conc_set_at2 ct1 ct2 (dim -^ 1sz) idx i2) in
         up_cons d2 res;
         res)