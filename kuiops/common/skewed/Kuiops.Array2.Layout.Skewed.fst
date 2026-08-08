module Kuiops.Array2.Layout.Skewed

#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
open Kuiper.ForEvery
open Kuiper.Bijection

module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ---------------------------------------------------------------- layout *)

let skew_f (rows cols pad : nat) (idx : abs (rows @| cols @| INil))
  : GTot (natlt (rows * (cols + pad)))
= let (i, (j, ())) = idx in
  ML.lemma_mult_le_right (cols + pad) (i + 1) rows;
  i * (cols + pad) + j

let skew_inj (rows cols pad : nat)
  (x : abs (rows @| cols @| INil))
  (y : abs (rows @| cols @| INil) { skew_f rows cols pad x == skew_f rows cols pad y })
  : squash (x == y)
= let (i1, (j1, ())) = x in
  let (i2, (j2, ())) = y in
  let ld : pos = cols + pad in
  ML.lemma_mod_plus j1 i1 ld;
  ML.lemma_mod_plus j2 i2 ld;
  ML.lemma_div_plus j1 i1 ld;
  ML.lemma_div_plus j2 i2 ld;
  ML.small_mod j1 ld;
  ML.small_mod j2 ld;
  ML.small_div j1 ld;
  ML.small_div j2 ld

let l2_skewed_row_major (rows cols pad : nat)
  : l : tlayout (rows @| cols @| INil) { l.ulen == rows * (cols + pad) }
= { ulen = rows * (cols + pad);
    imap = mk_injection (skew_f rows cols pad) (skew_inj rows cols pad) }

let lemma_skewed_imap (rows cols pad : nat) (i : natlt rows) (j : natlt cols)
  : Lemma ((l2_skewed_row_major rows cols pad).imap.f (idx2 i j)
             == i * (cols + pad) + j)
          [SMTPat ((l2_skewed_row_major rows cols pad).imap.f (idx2 i j))]
= ()

inline_for_extraction noextract
let c_l2_skewed_row_major
  (#rows : erased nat) (#cols #pad : erased nat)
  (ld : SZ.t { SZ.v ld == cols + pad /\ SZ.fits rows /\ SZ.fits (rows * (cols + pad)) })
  : ctlayout (l2_skewed_row_major rows cols pad)
= { ulen_fits = ();
    all_fit   = ();
    cimap     = (fun (idx : conc (rows @| cols @| INil)) ->
                   match idx with
                   | (i, (j, ())) ->
                     ML.lemma_mult_le_right (cols + pad) (SZ.v i + 1) rows;
                     lemma_skewed_imap rows cols pad (SZ.v i) (SZ.v j);
                     SZ.add (SZ.mul i ld) j) }

inline_for_extraction noextract
let srm_l2_skewed_row_major
  (#rows : erased nat) (#cols #pad : erased nat)
  (ld : SZ.t { SZ.v ld == cols + pad })
  : strided_row_major (vtlayout_of_tlayout (l2_skewed_row_major rows cols pad))
= { offset = 0sz;
    stride = ld;
    pf     = (fun i j -> ()) }

let lemma_aligned_srm_l2_skewed_row_major
  (#rows : erased nat) (#cols #pad : erased nat)
  (ld : SZ.t { SZ.v ld == cols + pad })
  (n : pos)
  : Lemma (requires n /?+ (cols + pad))
          (ensures aligned_strided_row_major n
                     (srm_l2_skewed_row_major #rows #cols #pad ld))
= ()

(* ------------------------------------------------------- split and join *)

let skew_pad_cell
  (#et : Type0) (p : array et) (rows cols pad : nat)
  (rp : (natlt rows & natlt pad))
  : slprop
= exists* (w : et). pts_to_cell p (rp._1 * (cols + pad) + cols + rp._2) w

let skew_residual
  (#et : Type0)
  (p : array et)
  (rows cols pad : nat)
  : slprop
= forall+ (rp : (natlt rows & natlt pad)). skew_pad_cell p rows cols pad rp

(* [natlt (rows * ld)] and [natlt rows & natlt ld] name the same cells. *)
let widen (cols pad : nat) (c : natlt cols) : natlt (cols + pad) = c
let shift (cols pad : nat) (c : natlt pad) : natlt (cols + pad) = cols + c

let skew_idx (rows cols pad : nat) (r : natlt rows) (c : natlt (cols + pad))
  : natlt (rows * (cols + pad))
= ML.lemma_mult_le_right (cols + pad) (r + 1) rows;
  r * (cols + pad) + c

let lemma_flat_div (rows : nat) (ld : pos) (i : natlt (rows * ld))
  : Lemma (i / ld < rows /\ i % ld < ld)
= ML.euclidean_division_definition i ld;
  ML.lemma_div_le i (rows * ld) ld;
  ML.cancel_mul_div rows ld

let skew_bij (rows : nat) (cols : pos) (pad : nat)
  : (natlt (rows * (cols + pad)) =~ (natlt rows & natlt (cols + pad)))
= mk_bijection #(natlt (rows * (cols + pad)))
               #(natlt rows & natlt (cols + pad))
    (fun i -> lemma_flat_div rows (cols + pad) i;
              ((i / (cols + pad), i % (cols + pad))
                 <: (natlt rows & natlt (cols + pad))))
    (fun rc -> skew_idx rows cols pad rc._1 rc._2)
    (fun (r, c) -> ML.lemma_mod_plus c r (cols + pad);
                   ML.lemma_div_plus c r (cols + pad);
                   ML.small_mod c (cols + pad);
                   ML.small_div c (cols + pad))
    (fun i -> lemma_flat_div rows (cols + pad) i;
              ML.euclidean_division_definition i (cols + pad))

(* The mapped cells of a skewed tile: the first [cols] columns of each row. *)
unfold let in_tile (cols : nat) (rc : (natlt 'r & natlt 'l)) : prop = rc._2 < cols

let tile_bij (rows cols pad : nat)
  : ((natlt rows & natlt cols)
       =~ (rc : (natlt rows & natlt (cols + pad)) { in_tile cols rc }))
= mk_bijection #(natlt rows & natlt cols)
               #(rc : (natlt rows & natlt (cols + pad)) { in_tile cols rc })
    (fun (r, c) -> ((r, (c <: natlt (cols + pad)))
                      <: (rc : (natlt rows & natlt (cols + pad)) { in_tile cols rc })))
    (fun (r, c) -> (r, (c <: natlt cols)))
    (fun _ -> ())
    (fun _ -> ())

let pad_bij (rows cols pad : nat)
  : ((natlt rows & natlt pad)
       =~ (rc : (natlt rows & natlt (cols + pad)) { ~ (in_tile cols rc) }))
= mk_bijection #(natlt rows & natlt pad)
               #(rc : (natlt rows & natlt (cols + pad)) { ~ (in_tile cols rc) })
    (fun (r, c) -> ((r, (cols + c <: natlt (cols + pad)))
                      <: (rc : (natlt rows & natlt (cols + pad)) { ~ (in_tile cols rc) })))
    (fun (r, c) -> (r, (c - cols <: natlt pad)))
    (fun _ -> ())
    (fun _ -> ())

(* Cell-level view of the backing array, indexed by (row, physical column). *)
let skew_val (et : Type0) (rows cols pad : nat) =
  natlt rows -> natlt (cols + pad) -> GTot et

let skew_cell
  (#et : Type0) (p : array et) (rows cols pad : nat)
  (g : skew_val et rows cols pad)
  (rc : (natlt rows & natlt (cols + pad)))
  : slprop
= pts_to_cell p (skew_idx rows cols pad rc._1 rc._2) (g rc._1 rc._2)

let skew_chest
  (#et : Type0) (rows cols pad : nat) (g : skew_val et rows cols pad)
  : GTot (chest2 et rows cols)
= mk2 (fun (r : natlt rows) (c : natlt cols) -> g r c)

let lemma_skew_cell_eq
  (#et : Type0) (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  (g : skew_val et rows cols pad)
  (x : (natlt rows & natlt cols))
  : Lemma (skew_cell p rows cols pad g ((tile_bij rows cols pad).ff x)
           == (Cell (from_array (l2_skewed_row_major rows cols pad) p)
                    (idx2 x._1 x._2)
                 |-> Frac 1.0R (acc (skew_chest rows cols pad g)
                                    (idx2 x._1 x._2))))
          [SMTPat (skew_cell p rows cols pad g ((tile_bij rows cols pad).ff x))]
= let a = from_array (l2_skewed_row_major rows cols pad) p in
  tensor_pts_to_cell_eq a (idx2 x._1 x._2) 1.0R
    (acc (skew_chest rows cols pad g) (idx2 x._1 x._2))

let skew_ecell
  (#et : Type0) (p : array et) (rows cols pad : nat)
  (rc : (natlt rows & natlt (cols + pad)))
  : slprop
= exists* (w : et). pts_to_cell p (skew_idx rows cols pad rc._1 rc._2) w

ghost
fn skew_pad_hide
  (#et : Type0) (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  (g : skew_val et rows cols pad)
  (rp : (natlt rows & natlt pad))
  requires skew_cell p rows cols pad g ((pad_bij rows cols pad).ff rp)
  ensures skew_pad_cell p rows cols pad rp
{
  rewrite (skew_cell p rows cols pad g ((pad_bij rows cols pad).ff rp))
       as (pts_to_cell p (rp._1 * (cols + pad) + cols + rp._2)
                         (g rp._1 (cols + rp._2)));
  fold (skew_pad_cell p rows cols pad rp);
}

ghost
fn skew_split
  (#et : Type0)
  (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  requires
    (exists* (v : Seq.seq et). pts_to p v) **
    pure (SZ.fits (rows * (cols + pad)))
  ensures
    (exists* (em : chest2 et rows cols).
       from_array (l2_skewed_row_major rows cols pad) p |-> em) **
    skew_residual p rows cols pad
{
  array_to_slice p;
  slice_to_array p;
  with v. assert (pts_to p v);
  array_slice_1 p;
  forevery_iso (skew_bij rows cols pad)
    (fun (i : natlt (rows * (cols + pad))) -> pts_to_cell p i (Seq.index v i));
  let g : skew_val et rows cols pad =
    (fun (r : natlt rows) (c : natlt (cols + pad)) ->
       Seq.index v (skew_idx rows cols pad r c));
  forevery_ext
    (fun (rc : (natlt rows & natlt (cols + pad))) ->
       pts_to_cell p ((skew_bij rows cols pad).gg rc)
                     (Seq.index v ((skew_bij rows cols pad).gg rc)))
    (skew_cell p rows cols pad g);
  forevery_refine_split (skew_cell p rows cols pad g) (in_tile cols);
  (* mapped cells -> the tensor *)
  forevery_iso (bij_sym (tile_bij rows cols pad))
    (fun (rc : (rc : (natlt rows & natlt (cols + pad)) { in_tile cols rc })) ->
       skew_cell p rows cols pad g rc);
  forevery_ext
    (fun (rc : (natlt rows & natlt cols)) ->
       skew_cell p rows cols pad g ((tile_bij rows cols pad).ff rc))
    (fun (rc : (natlt rows & natlt cols)) ->
       Cell (from_array (l2_skewed_row_major rows cols pad) p) (idx2 rc._1 rc._2)
         |-> Frac 1.0R (acc (skew_chest rows cols pad g) (idx2 rc._1 rc._2)));
  tensor_implode2 (from_array (l2_skewed_row_major rows cols pad) p)
                  #1.0R #(skew_chest rows cols pad g);
  (* unmapped cells -> the residual *)
  forevery_iso (bij_sym (pad_bij rows cols pad))
    (fun (rc : (rc : (natlt rows & natlt (cols + pad)) { ~ (in_tile cols rc) })) ->
       skew_cell p rows cols pad g rc);
  forevery_map
    (fun (rp : (natlt rows & natlt pad)) ->
       skew_cell p rows cols pad g ((pad_bij rows cols pad).ff rp))
    (skew_pad_cell p rows cols pad)
    (skew_pad_hide rows cols pad p g);
  fold (skew_residual p rows cols pad);
}

ghost
fn skew_tile_hide
  (#et : Type0) (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  (em : chest2 et rows cols)
  (ij : (natlt rows & natlt cols))
  requires
    Cell (from_array (l2_skewed_row_major rows cols pad) p) (idx2 ij._1 ij._2)
      |-> Frac 1.0R (acc em (idx2 ij._1 ij._2))
  ensures skew_ecell p rows cols pad ((tile_bij rows cols pad).ff ij)
{
  tensor_pts_to_cell_eq (from_array (l2_skewed_row_major rows cols pad) p)
                        (idx2 ij._1 ij._2) 1.0R (acc em (idx2 ij._1 ij._2));
  rewrite (Cell (from_array (l2_skewed_row_major rows cols pad) p)
                (idx2 ij._1 ij._2)
             |-> Frac 1.0R (acc em (idx2 ij._1 ij._2)))
       as (pts_to_cell p (skew_idx rows cols pad ((tile_bij rows cols pad).ff ij)._1
                                    ((tile_bij rows cols pad).ff ij)._2)
                         (acc em (idx2 ij._1 ij._2)));
  fold (skew_ecell p rows cols pad ((tile_bij rows cols pad).ff ij));
}

ghost
fn skew_pad_show
  (#et : Type0) (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  (rp : (natlt rows & natlt pad))
  requires skew_pad_cell p rows cols pad rp
  ensures skew_ecell p rows cols pad ((pad_bij rows cols pad).ff rp)
{
  unfold (skew_pad_cell p rows cols pad rp);
  with w. assert (pts_to_cell p (rp._1 * (cols + pad) + cols + rp._2) w);
  rewrite (pts_to_cell p (rp._1 * (cols + pad) + cols + rp._2) w)
       as (pts_to_cell p (skew_idx rows cols pad ((pad_bij rows cols pad).ff rp)._1
                                    ((pad_bij rows cols pad).ff rp)._2) w);
  fold (skew_ecell p rows cols pad ((pad_bij rows cols pad).ff rp));
}

ghost
fn skew_join
  (#et : Type0)
  (rows : nat) (cols : pos) (pad : nat)
  (p : larray et (rows * (cols + pad)))
  requires
    (exists* (em : chest2 et rows cols).
       from_array (l2_skewed_row_major rows cols pad) p |-> em) **
    skew_residual p rows cols pad **
    pure (SZ.fits (rows * (cols + pad)))
  ensures exists* (v : Seq.seq et). pts_to p v
{
  with em. assert (from_array (l2_skewed_row_major rows cols pad) p |-> em);
  tensor_explode2 (from_array (l2_skewed_row_major rows cols pad) p);
  forevery_map
    (fun (ij : (natlt rows & natlt cols)) ->
       Cell (from_array (l2_skewed_row_major rows cols pad) p)
            (idx2 ij._1 ij._2)
         |-> Frac 1.0R (acc em (idx2 ij._1 ij._2)))
    (fun (ij : (natlt rows & natlt cols)) ->
       skew_ecell p rows cols pad ((tile_bij rows cols pad).ff ij))
    (skew_tile_hide rows cols pad p em);
  forevery_iso_back (bij_sym (tile_bij rows cols pad))
    (skew_ecell p rows cols pad);
  unfold (skew_residual p rows cols pad);
  forevery_map
    (skew_pad_cell p rows cols pad)
    (fun (rp : (natlt rows & natlt pad)) ->
       skew_ecell p rows cols pad ((pad_bij rows cols pad).ff rp))
    (skew_pad_show rows cols pad p);
  forevery_iso_back (bij_sym (pad_bij rows cols pad))
    (skew_ecell p rows cols pad);
  forevery_refine_join (skew_ecell p rows cols pad)
    (in_tile cols) (fun rc -> ~ (in_tile cols rc));
  forevery_unrefine (skew_ecell p rows cols pad);
  forevery_iso_back (skew_bij rows cols pad)
    (fun (i : natlt (rows * (cols + pad))) -> exists* (w : et). pts_to_cell p i w);
  gpu_array_unslice_1' p;
}
