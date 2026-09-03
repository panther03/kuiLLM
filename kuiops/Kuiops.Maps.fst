module Kuiops.Maps

(* Unary maps for the [pre_map]/[post_map] slots of the Kuiops kernels.

   An [amap] bundles a concrete map over a scalar element type, its model over
   the reals, and the proof that the former approximates the latter. Composition
   carries the proof along, so a JIT instantiation only writes an [amap_comp]
   chain and discharges no proof obligation at extraction time.

   Only ops whose real model is expressible with Kuiper's approximation rules
   live here: [sin] and [cos] have no real counterpart, so they cannot be fused
   into a reduction specified over the reals. [rsqrt] is the exception -- see
   its explicitly documented approximation rule below. *)

open Kuiper

module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module Sqrt = FStar.Math.Sqrt

(* [f] approximates [g]: it maps any value approximating [r] to a value
   approximating [g r]. The unary analogue of [Kuiper.Approximates.approx2]. *)
let approx1
  (#a #b : Type0) {| scalar a, real_like a, scalar b, real_like b |}
  (f : a -> b) (g : real -> real) : prop
  = forall (x:a) (r:real). x %~ r ==> f x %~ g r

inline_for_extraction noextract
noeq type amap (t : Type0) {| scalar t, real_like t |} = {
  mf  : t -> t;
  mr  : real -> real;
  mok : squash (approx1 mf mr);
}

inline_for_extraction noextract
let mk_amap (#t:Type0) {| scalar t, real_like t |}
  (f : t -> t) (g : real -> real)
  (pf : (x:t -> r:real -> Lemma (requires x %~ r) (ensures f x %~ g r)))
  : amap t =
  FStar.Classical.forall_intro_2 (FStar.Classical.move_requires_2 pf);
  { mf = f; mr = g; mok = () }

(* The concrete half, carrying its approximation proof in the result type: this
   is what a kernel's refined [pre_map]/[post_map] argument expects. *)
inline_for_extraction noextract
let amap_f (#t:Type0) {| scalar t, real_like t |} (m : amap t)
  : (f : (t -> t) { approx1 f m.mr })
  = let _ = m.mok in m.mf

inline_for_extraction noextract
let amap_id (t:Type0) {| scalar t, real_like t |} : amap t =
  mk_amap (fun x -> x) (fun r -> r) (fun _ _ -> ())

(* [amap_comp g f] applies [f] first, then [g]. *)
inline_for_extraction noextract
let amap_comp (#t:Type0) {| scalar t, real_like t |} (g f : amap t) : amap t =
  let _ = f.mok in
  let _ = g.mok in
  mk_amap (fun x -> g.mf (f.mf x)) (fun r -> g.mr (f.mr r)) (fun _ _ -> ())

(* --- Maps with a constant baked in at instantiation time --- *)

inline_for_extraction noextract
let amap_add (#t:Type0) {| scalar t, real_like t |} (c : t) : amap t =
  mk_amap (fun x -> add x c) (fun r -> r +. to_real c)
    (fun x r -> to_real_ok c; a_add x c r (to_real c))

inline_for_extraction noextract
let amap_mul (#t:Type0) {| scalar t, real_like t |} (c : t) : amap t =
  mk_amap (fun x -> mul x c) (fun r -> r *. to_real c)
    (fun x r -> to_real_ok c; a_mul x c r (to_real c))

inline_for_extraction noextract
let amap_sub (#t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |} (c : t) : amap t =
  mk_amap (fun x -> sub x c) (fun r -> r -. to_real c)
    (fun x r -> to_real_ok c; sub_approx x c r (to_real c))

(* Division by a constant. The divisor is an integer so that its real value --
   and hence the non-zeroness [( /. )] demands -- is known: [of_literal] is
   uninterpreted, so a decimal divisor could not be discharged. *)
inline_for_extraction noextract
let amap_div (#t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |}
  (n : I64.t { I64.v n <> 0 }) : amap t =
  mk_amap (fun x -> div x (of_int n)) (fun r -> r /. Real.of_int (I64.v n))
    (fun x r ->
      let _ = of_int_approx #t n in
      div_approx x (of_int n) r (Real.of_int (I64.v n)))

(* --- Constant-free maps --- *)

inline_for_extraction noextract
let amap_neg (t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |} : amap t =
  mk_amap (fun x -> sub zero x) (fun r -> 0.0R -. r)
    (fun x r ->
      let _ : squash (v_approximates (zero #t) 0.0R) = a0 in
      sub_approx zero x 0.0R r)

inline_for_extraction noextract
let amap_square (t:Type0) {| scalar t, real_like t |} : amap t =
  mk_amap (fun x -> mul x x) (fun r -> r *. r)
    (fun x r -> a_mul x x r r)

(* [fmax] rather than the comparison-based [Kuiops.ElementwiseOps.relu]: the
   [real_like] rules relate [fmax] to [rmax] but say nothing about [lte]. *)
inline_for_extraction noextract
let amap_relu (t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |} : amap t =
  mk_amap (fun x -> fmax x zero) (fun r -> rmax r 0.0R)
    (fun x r ->
      let _ : squash (v_approximates (zero #t) 0.0R) = a0 in
      fmax_approx x zero r 0.0R)

inline_for_extraction noextract
let amap_silu (t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |} : amap t =
  mk_amap
    (fun x -> mul x (div one (add one (fexp (sub zero x)))))
    (fun r -> r *. (1.0R /. (1.0R +. exp (0.0R -. r))))
    (fun x r ->
      let _ : squash (v_approximates (zero #t) 0.0R) = a0 in
      let _ : squash (v_approximates (one #t) 1.0R) = a1 in
      sub_approx zero x 0.0R r;
      exp_approx (sub zero x) (0.0R -. r);
      a_add one (fexp (sub zero x)) 1.0R (exp (0.0R -. r));
      div_approx one (add one (fexp (sub zero x))) 1.0R
        (1.0R +. exp (0.0R -. r));
      a_mul x (div one (add one (fexp (sub zero x)))) r
        (1.0R /. (1.0R +. exp (0.0R -. r))))

(* --- Maps parameterised by the reduced length --- *)

(* Every Kuiper [sz] fits in a [uint32], so the round trip through [uint64] and
   the reinterpretation as a signed [int64] are both value-preserving. *)
inline_for_extraction noextract
let sz_to_i64 (n : SZ.t) : (m : I64.t { I64.v m == SZ.v n }) =
  FStar.Int.Cast.uint64_to_int64 (FStar.SizeT.sizet_to_uint64 n)

(* [mean]: divide by the number of reduced elements. *)
inline_for_extraction noextract
let amap_div_sz (#t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |}
  (n : SZ.t { SZ.v n > 0 }) : amap t =
  mk_amap (fun x -> div x (of_int (sz_to_i64 n)))
    (fun r -> r /. Real.of_int (SZ.v n))
    (fun x r ->
      let _ = of_int_approx #t (sz_to_i64 n) in
      div_approx x (of_int (sz_to_i64 n)) r (Real.of_int (SZ.v n)))

(* --- Reciprocal square root --- *)

(* [amap] requires a total real model, whereas real inverse square root is only
   defined on positives.  Extend it with zero outside its domain; this choice
   is proof-only and the positive branch is the one RMSNorm intends to use. *)
inline_for_extraction noextract
let real_rsqrt_total (r : real) : Tot real =
  if r >. 0.0R then Sqrt.rsqrt r else 0.0R

let real_rsqrt_total_positive (r : Sqrt.rpos)
  : Lemma (ensures real_rsqrt_total r == Sqrt.rsqrt r)
  = ()

(* ASSUMED. Kuiper's [floating_real_like] hierarchy does not yet relate its
   concrete [rsqrt] operation to [FStar.Math.Sqrt.rsqrt].  Only this
   floating-point approximation rule remains an axiom here. *)
assume val rsqrt_approx (#t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |}
  (x : t) (r : real)
  : Lemma (requires x %~ r /\ r >. 0.0R)
          (ensures rsqrt x %~ (Sqrt.rsqrt r <: real))

(* GAP. The [pre_map]/[post_map] slots demand an unconditional [approx1], but
   [rsqrt_approx] only holds on positives, and the reduction's specification
   says nothing about the sign of the value reaching the post-map. Closing this
   properly needs a domain-restricted map in the kernel specification; until
   then the non-positive case is admitted here, and this is the only unsound
   step in [Kuiops.Maps]. *)
inline_for_extraction noextract
let amap_rsqrt (t:Type0)
  {| scalar t, floating t, real_like t, floating_real_like t |} : amap t =
  mk_amap (fun x -> rsqrt x) real_rsqrt_total
    (fun x r ->
      if r >. 0.0R then begin
        rsqrt_approx x r;
        real_rsqrt_total_positive r
      end else
        admit ())
