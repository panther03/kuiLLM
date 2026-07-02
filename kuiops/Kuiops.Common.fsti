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

inline_for_extraction noextract
let conc_cons (#r : erased nat) (#d : shape r { ICons? d })
  (p : szlt (d @! 0) & conc (tail d)) : conc d
  = p

// Concrete counterpart of abs_le, commuting with up. The recursion is driven by
// the (non-erasable) cshapes: matching the erasable `shape` directly to build an
// informative `conc` would force a ghost computation.
inline_for_extraction noextract
[@@strict_on_arguments [3]]
let rec conc_le (#r : erased nat) (#d1 #d2 : shape r { shape_le d1 d2 })
  (cd1 : cshape d1) (cd2 : cshape d2) (x : conc d1)
  : Tot (c : conc d2 {up c == abs_le d1 d2 (up x)})
  = match cd1 with
    | CNil -> ()
    | CCons #_ #_ _ #t1 ct1 ->
      (match cd2 with
       | CCons #_ #_ _ #t2 ct2 ->
         assert (r > 0);
         // Destructure *and* rebuild through the cshape-bound tail shapes `t1`/
         // `t2` (which reduce to concrete shapes at a JIT instantiation) rather
         // than `conc (tail d)`: `tail` does not reduce during extraction, so
         // `conc (tail d)` leaves the tuple tail as an unreduced recursive `conc`
         // type that F* casts to Top and karamel then rejects (Warning 26).
         let i1, i2 = x <: szlt (d1 @! 0) & conc t1 in
         let res : szlt (d2 @! 0) & conc t2 = ((i1 <: szlt (d2 @! 0)), conc_le ct1 ct2 i2) in
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
[@@strict_on_arguments [3]]
let rec conc_set_at2 (#r : erased nat) (#d1 #d2 : shape r { shape_le d1 d2 })
  (cd1 : cshape d1) (cd2 : cshape d2) (dim : szlt r) (idx : szlt (d2 @! dim)) (x : conc d1)
  : Tot (c : conc d2 {up c == abs_set_at2 d1 d2 dim idx (up x)})
         (decreases (SZ.v dim))
  = assert r > 0;
    match cd1 with
    | CCons #_ #_ _ #t1 ct1 ->
      (match cd2 with
       | CCons #_ #_ _ #t2 ct2 ->
         let (i1,i2) = x <: szlt (d1 @! 0) & conc t1 in
         abs_set_at2_cons d1 d2 (SZ.v dim) idx (up x);
         up_cons d1 x;
         // Build through the cshape-bound tail shape `t2` (concrete at a JIT
         // instantiation), not `conc (tail d2)`, which extraction leaves stuck.
         let res : szlt (d2 @! 0) & conc t2 =
           if dim = 0sz then
             ((idx <: szlt (d2 @! 0)), conc_le ct1 ct2 i2)
           else
             ((i1 <: szlt (d2 @! 0)),
              conc_set_at2 ct1 ct2 (dim -^ 1sz) idx i2) in
         up_cons d2 res;
         res)

(* ----------------------------------------------------------------------- *)
(* Coordinate get / narrow: the pieces the cat kernel needs, expressed so no
   `conc (modulo_i dim d)` type ever appears in extracted code (that type stays
   a stuck application at extraction time and karamel casts it to `any`). The
   recursions are cshape-driven, exactly like `conc_le`/`conc_set_at2`. *)

(* modulo_i unfolding facts, used to line up the shapes narrowed across `dim`. *)
let modulo_zero (#n : pos) (d : shape n)
  : Lemma (modulo_i 0 d == tail d)
  = match d with | ICons _ _ -> ()

let modulo_succ (#n : nat) (dim : natlt n { dim > 0 }) (d : shape n)
  : Lemma (modulo_i dim d == ICons (d @! 0) (modulo_i (dim - 1) (tail d)))
  = match d with | ICons _ _ -> ()

(* Read the coordinate along `dim`. *)
let rec abs_get_at (#r : nat) (#d : shape r) (dim : natlt r) (x : abs d)
  : GTot (natlt (d @! dim))
         (decreases dim)
  = let (h, t) = x <: natlt (d @! 0) & abs (tail d) in
    if dim = 0 then h
    else abs_get_at #(r-1) #(tail d) (dim - 1) t

inline_for_extraction noextract
[@@strict_on_arguments [2]]
let rec conc_get_at (#r : erased nat) (#d : shape r) (cd : cshape d) (dim : szlt r) (x : conc d)
  : Tot (v : szlt (d @! dim) { SZ.v v == abs_get_at (SZ.v dim) (up x) })
         (decreases (SZ.v dim))
  = match cd with
    | CCons #_ #_ _ #t ct ->
      let (h, tl) = x <: szlt (d @! 0) & conc t in
      up_cons d x;
      if dim = 0sz then (h <: szlt (d @! dim))
      else conc_get_at ct (dim -^ 1sz) tl

(* Rebuild an index of `d1` from an index of `d2`, where `d1`/`d2` agree on every
   axis except `dim` (captured by `modulo_i dim d1 == modulo_i dim d2`): copy all
   off-`dim` coordinates, and set the `dim` coordinate to `v`. *)
let rec abs_narrow (#r : nat) (dim : natlt r) (d1 d2 : shape r)
  (v : natlt (d1 @! dim)) (x : abs d2 { modulo_i dim d1 == modulo_i dim d2 })
  : GTot (abs d1)
         (decreases dim)
  = let (h, t) = x <: natlt (d2 @! 0) & abs (tail d2) in
    if dim = 0 then (
      modulo_zero d1; modulo_zero d2;
      (v, (t <: abs (tail d1)))
    ) else (
      modulo_succ dim d1; modulo_succ dim d2;
      ((h <: natlt (d1 @! 0)), abs_narrow #(r-1) (dim - 1) (tail d1) (tail d2) v t)
    )

inline_for_extraction noextract
[@@strict_on_arguments [4]]
let rec conc_narrow (#r : erased nat) (dim : szlt r) (#d1 #d2 : shape r)
  (cd1 : cshape d1) (cd2 : cshape d2)
  (pf : squash (modulo_i (SZ.v dim) d1 == modulo_i (SZ.v dim) d2))
  (v : sz) (pfv : squash (SZ.v v < d1 @! dim)) (x : conc d2)
  : Tot (c : conc d1 { up c == abs_narrow (SZ.v dim) d1 d2 (SZ.v v) (up x) })
         (decreases (SZ.v dim))
  = let vv : szlt (d1 @! dim) = v in
    match cd1 with
    | CCons #_ #_ _ #t1 ct1 ->
      (match cd2 with
       | CCons #_ #_ _ #t2 ct2 ->
         let (h, tl) = x <: szlt (d2 @! 0) & conc t2 in
         up_cons d2 x;
         let res : szlt (d1 @! 0) & conc t1 =
           if dim = 0sz then (
             modulo_zero d1; modulo_zero d2;
             ((vv <: szlt (d1 @! 0)), (tl <: conc t1))
           ) else (
             modulo_succ (SZ.v dim) d1; modulo_succ (SZ.v dim) d2;
             ((h <: szlt (d1 @! 0)),
              conc_narrow (dim -^ 1sz) ct1 ct2 () v () tl)
           ) in
         up_cons d1 res;
         res)