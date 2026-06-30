module Kuiops.Cat

(* Implementation of the verified torch.cat kernel; see Kuiops.Cat.fsti for the
   public spec (abs_cat / cat_chest) and the cat_gpu signature. *)

#lang-pulse
open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Kuiper.Shareable

open Kuiper.Kernel.TMap

module SZ = Kuiper.SizeT

(* ----------------------------------------------------------------------- *)
(* Index machinery: split/rebuild a concrete index across dimension `dim`,
   carrying the abstract round-trip relationship in the return type (so the
   per-element function below verifies by unfolding the spec). *)

let rec gg_eq (#r : nat) (dim : natlt r) (d : shape r)
  (j : szlt (d @! dim)) (m : conc (modulo_i dim d))
  : Lemma (ensures c_bring_forward_gg dim d j m == (conc_bring_forward_bij dim d).gg (j, m))
          (decreases dim)
  = if dim = 0 then ()
    else (let (h2, t2) = m <: szlt (d @! 0) & conc (tail (modulo_i dim d)) in
          gg_eq #(r-1) (dim - 1) (tail d) j t2)

let rec roundtrip (#r : nat) (dim : natlt r) (d : shape r) (idx : conc d)
  : Lemma (ensures (let (j, m) = c_bring_forward_ff dim d idx in
                    c_bring_forward_gg dim d j m == idx))
          (decreases dim)
  = if dim = 0 then ()
    else (let (h, t) = idx <: szlt (d @! 0) & conc (tail d) in
          roundtrip #(r-1) (dim - 1) (tail d) t)

(* Rebuild a concrete index of `d` from `(j, m)`. `j` is taken as a bare size_t
   with its bound supplied separately, so callers can narrow a coordinate of a
   wider shape (the output) without a rigid refinement-type conversion. *)
inline_for_extraction noextract
let conc_unsplit (#r : nat) (dim : natlt r) (d : shape r)
  (j : sz) (_ : squash (SZ.v j < d @! dim)) (m : conc (modulo_i dim d))
  : (c : conc d { up c == (abs_bring_forward_bij dim d).gg (SZ.v j, up m) })
  = let jj : szlt (d @! dim) = j in
    gg_eq dim d jj m;
    bring_forward_commute2 dim d jj m;
    c_bring_forward_gg dim d jj m

(* From a concrete index `i : conc d`, peel the coordinate `j` along `dim` and
   the rest `m`, with `up i == (abs_bring_forward_bij dim d).gg (v j, up m)`. *)
inline_for_extraction noextract
let conc_split (#r : nat) (dim : natlt r) (d : shape r) (i : conc d)
  : (res : (szlt (d @! dim) & conc (modulo_i dim d))
        { let (j, m) = res in up i == (abs_bring_forward_bij dim d).gg (SZ.v j, up m) })
  = let res = c_bring_forward_ff dim d i in
    let (j, m) = res in
    roundtrip dim d i;
    let _ = conc_unsplit dim d j () m in
    res

(* ----------------------------------------------------------------------- *)
(* The per-element value relation (links abs_cat to the map's contract). *)

let vcat (#et: Type0) (#r : nat)
  (dim : natlt r) (dA dB dout : shape r)
  (eA : chest dA et) (eB : chest dB et)
  (na : nat { na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (x : abs dout) (_old : et) (o : et)
  : prop
  = o == abs_cat dim dA dB dout eA eB na pf_sz pfA pfB x

(* ----------------------------------------------------------------------- *)
(* The two read-only inputs, held with split fractions of `fr` (so the frame is
   shareable across threads), exactly as gather holds its input + index. *)

unfold
let cat_frame (#et : Type0) (#r : erased nat) (dA dB : shape r)
  (#lA : tlayout dA) (#lB : tlayout dB) {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (eA : chest dA et)
  (eB : chest dB et)
  (fA fB : perm) (fr : perm) : slprop =
    (tensor_pts_to gA #((fA /. (fA +. fB)) *. fr) eA) **
    (tensor_pts_to gB #((fB /. (fA +. fB)) *. fr) eB)

instance cat_frame_shareable
  (#et : Type0) (#r : erased nat) (dA dB : shape r)
  (#lA : tlayout dA) (#lB : tlayout dB) {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (eA : chest dA et)
  (eB : chest dB et)
  (fA fB : perm) :
  shareable (cat_frame dA dB #lA #lB gA gB eA eB fA fB) =
    double_shareable
      (fun fr -> tensor_pts_to gA #fr eA)
      (fun fr -> tensor_pts_to gB #fr eB)
      (fA /. (fA +. fB)) (fB /. (fA +. fB))

(* ----------------------------------------------------------------------- *)
(* Per-element function: read the right input for one output coordinate. *)

inline_for_extraction noextract
fn fcat
  (#et : Type0) (#r : nat)
  (dim : natlt r) (dA dB dout : shape r)
  (#lA : tlayout dA) (#lB : tlayout dB) {| ctlayout lA, ctlayout lB |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (na : sz { SZ.v na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (eA : chest dA et)
  (eB : chest dB et)
  (#fA #fB : perm)
  (#fr : perm) (i : conc dout) (x : et)
norewrite
preserves
  (cat_frame dA dB #lA #lB gA gB eA eB fA fB fr)
returns
  res : et
ensures
  pure (vcat dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB (up i) x res)
{
  let jm = conc_split dim dout i;
  let (j, m) = jm;
  let b = j <^ na;
  if b {
    let ia = conc_unsplit dim dA j () m;
    let v = tensor_read gA ia;
    v
  } else {
    let ib = conc_unsplit dim dB (j -^ na) () m;
    let v = tensor_read gB ib;
    v
  }
}

(* ----------------------------------------------------------------------- *)
(* The kernel: map over the output, reading from the two inputs. *)

inline_for_extraction noextract
fn cat_gpu
  (#et : Type0) (#r : nat)
  (dA dB dout : shape r)
  (cdA : cshape dA) (cdB : cshape dB) (cdout : cshape dout)
  (dim : natlt r)
  (na : sz { SZ.v na == dA @! dim })
  (pf_sz : squash ((dout @! dim) == (dA @! dim) + (dB @! dim)))
  (pfA : squash (modulo_i dim dA == modulo_i dim dout))
  (pfB : squash (modulo_i dim dB == modulo_i dim dout))
  (#lA : tlayout dA) (#lB : tlayout dB) (#lOut : tlayout dout)
  {| ctlayout lA, ctlayout lB, ctlayout lOut |}
  (gA : tensor et lA {is_global gA})
  (gB : tensor et lB {is_global gB})
  (gOut : tensor et lOut {is_global gOut})
  (n : sz {SZ.v n == sizeof dout /\ n <= max_blocks * max_threads /\ n > 0})
  (eA : chest dA et)
  (eB : chest dB et)
  (#fA #fB : perm)
  preserves cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)
  requires on gpu_loc (live gOut)
  ensures
    on gpu_loc (gOut |-> cat_chest dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB)
{
  with eOut. assert on gpu_loc (gOut |-> eOut);
  let kfun = (kmap cdout
      (cat_frame dA dB #lA #lB gA gB eA eB fA fB)
      #(cat_frame_shareable dA dB #lA #lB gA gB eA eB fA fB)
      (vcat dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB)
      (fcat dim dA dB dout #lA #lB gA gB na pf_sz pfA pfB eA eB)
      n gOut #eOut #_ #(fA +. fB));
  launch_sync
    kfun;
  with eOut'. assert on gpu_loc (gOut |-> eOut');
  assert pure (Kuiper.Chest.equal eOut'
                 (cat_chest dim dA dB dout eA eB (SZ.v na) pf_sz pfA pfB));
}
