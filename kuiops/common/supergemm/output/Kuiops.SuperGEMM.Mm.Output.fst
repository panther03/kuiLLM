module Kuiops.SuperGEMM.Mm.Output

(* Layout-generic output tiling for the software-pipelined tensor-core GEMM.

   [Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc]'s [output_fragment] and
   [output_lane_live] hardcode the output operand to row-major
   ([gD : array2 et (rm m n)]).  SuperGEMM requires A, B *and* D to carry a
   [strided_row_major] typeclass witness rather than being pinned to row-major,
   so this module generalises the two definitions to an arbitrary
   [lD : layout2 m n].  Everything underneath ([block_tile], [warp_tile],
   [array2_subtile], [live_lane_cells]) is already layout-generic, so the change
   is purely mechanical.

   TODO(upstream): fold these back into [...To.KernelDesc] as the primary
   definitions, with the row-major versions recovered at [lD := rm m n]. *)

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile, warp_tile }
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc { output_fragment, output_lane_live,
                                                     live_lane_cells, own_lane_cells,
                                                     in_lane }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor
module VG = Kuiper.Array2.Vectorized.Group

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

(* ---- lane-cell split/join helpers (local copies of the private upstream
        helpers in [...To.KernelDesc] / [...To.KernelDesc.Teardown]) ---- *)

let in_lane_covers_all
  (w : pos)
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  : Lemma (exists lane. in_lane w rows cols lane ij)
= let lane : natlt warp_size =
    VG.group_of w cols ij._1 ij._2 % warp_size in
  assert (in_lane w rows cols lane ij);
  ()

let in_lane_no_overlap
  (w : pos)
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  (lane1 lane2 : natlt warp_size)
  : Lemma
      (requires in_lane w rows cols lane1 ij /\ in_lane w rows cols lane2 ij)
      (ensures lane1 == lane2)
= ()

ghost
fn split_array2_into_lane_cells
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#em : chest2 et rows cols)
  requires m |-> em
  ensures forall+ (lane : natlt warp_size). own_lane_cells m em lane
{
  tensor_ilower2 m;
  forevery_flatten _;
  Classical.forall_intro (in_lane_covers_all (chunk et #_ #hvc) rows cols);
  forevery_refine_ext #_ #(fun _ -> True)
    (fun (ij : natlt rows & natlt cols) ->
       exists lane. in_lane (chunk et #_ #hvc) rows cols lane ij) _;
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap (chunk et #_ #hvc) rows cols ij lane1));
  forevery_split_or_n _ _;
  forevery_map
    (fun lane ->
      forall+ (ij : (natlt rows & natlt cols){
        in_lane (chunk et #_ #hvc) rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2)
          (acc2 em ij._1 ij._2))
    (fun lane -> own_lane_cells m em lane)
    fn lane { fold own_lane_cells m em lane };
}

(* ==== Helper 1: layout-generic live split (upstream [split_output_to_lanes]) ==== *)

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1 --split_queries always"
ghost
fn split_output_to_lanes'
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| T.ctlayout lD |}
  (gD : array2 et lD)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires live gD
  ensures
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_live' gD bm bn tm tn wm wn bid tid
{
  with (eD : chest2 _ _ _). assert gD |-> eD;
  array2_tile gD (SZ.v bm) (SZ.v bn) #eD #1.0R;
  forevery_unfactor' nblk (m / bm) (n / bn) _;

  forevery_map
    (fun (bid : natlt nblk) ->
      array2_subtile gD (SZ.v bm) (SZ.v bn)
        (bid / (n / bn)) (bid % (n / bn))
        |-> ematrix_subtile eD bm bn
              (bid / (n / bn)) (bid % (n / bn)))
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_live' gD bm bn tm tn wm wn bid tid)
    fn bid {
      rewrite each
        array2_subtile gD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn))
      as block_tile gD (SZ.v bm) (SZ.v bn) bid;
      let dBlock = block_tile gD (SZ.v bm) (SZ.v bn) bid;
      let eBlock =
        ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn));
      array2_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn)
        #(ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn)))
        #1.0R;
      rewrite each
        block_tile gD (SZ.v bm) (SZ.v bn) bid
        as dBlock;
      rewrite each
        ematrix_subtile eD (SZ.v bm) (SZ.v bn)
          (bid / (n / bn)) (bid % (n / bn))
        as eBlock;
      forevery_unfactor'
        (bm / (wm * tm) * (bn / (wn * tn)))
        (bm / (wm * tm))
        (bn / (wn * tn))
        _;
      forevery_map
        (fun (wid : natlt (
          bm / (wm * tm) * (bn / (wn * tn)))) ->
          array2_subtile dBlock
            (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))
          |-> ematrix_subtile eBlock (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn))))
        (fun (wid : natlt (
          bm / (wm * tm) * (bn / (wn * tn)))) ->
          forall+ (lane : natlt warp_size)
                   (mi : natlt wm)
                   (nj : natlt wn).
            live_lane_cells
              (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
              lane)
        fn wid {
          rewrite each
            array2_subtile dBlock
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))
          as warp_tile dBlock
            (wm * tm) (wn * tn) wid;
          let dWarp = warp_tile dBlock
            (wm * tm) (wn * tn) wid;
          let eWarp =
            ematrix_subtile eBlock
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)));
          array2_tile
            (warp_tile dBlock (wm * tm) (wn * tn) wid)
            (SZ.v tm) (SZ.v tn)
            #(ematrix_subtile eBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn))))
            #1.0R;
          rewrite each
            warp_tile dBlock (wm * tm) (wn * tn) wid
            as dWarp;
          rewrite each
            ematrix_subtile eBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))
            as eWarp;
          FStar.Math.Lemmas.cancel_mul_div (SZ.v wm) (SZ.v tm);
          FStar.Math.Lemmas.cancel_mul_div (SZ.v wn) (SZ.v tn);
          assert pure ((wm * tm) / tm == wm);
          assert pure ((wn * tn) / tn == wn);
          forevery_rw_size2
            ((wm * tm) / tm) wm
            ((wn * tn) / tn) wn
            #(fun (mi : natlt ((wm * tm) / tm))
                  (nj : natlt ((wn * tn) / tn)) ->
              array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                |-> ematrix_subtile eWarp tm tn mi nj);

          forevery_map_2
            (fun (mi : natlt wm) (nj : natlt wn) ->
              array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                |-> ematrix_subtile eWarp tm tn mi nj)
            (fun (mi : natlt wm) (nj : natlt wn) ->
              forall+ (lane : natlt warp_size).
                live_lane_cells
                  (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            fn mi nj {
              rewrite each
                array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj
                as output_fragment' gD bm bn tm tn wm wn bid wid mi nj;
              split_array2_into_lane_cells
                (output_fragment' gD bm bn tm tn wm wn bid wid mi nj);
              forevery_map
                (fun lane ->
                  own_lane_cells
                    (output_fragment' gD bm bn tm tn wm wn
                      bid wid mi nj)
                    (ematrix_subtile eWarp tm tn mi nj)
                    lane)
                (fun lane ->
                  live_lane_cells
                    (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
                    lane)
                fn lane {
                  fold live_lane_cells
                    (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
                    lane;
                };
            };

          forevery_map
            (fun (mi : natlt wm) ->
              forall+ (nj : natlt wn) (lane : natlt warp_size).
                live_lane_cells
                  (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            (fun (mi : natlt wm) ->
              forall+ (lane : natlt warp_size) (nj : natlt wn).
                live_lane_cells
                  (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
                  lane)
            fn mi { forevery_commute _ };
          forevery_commute
            (fun (mi : natlt wm) (lane : natlt warp_size) ->
              forall+ (nj : natlt wn).
                live_lane_cells
                  (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
                  lane);
        };

      forevery_unfactor' nthr
        (bm / (wm * tm) * (bn / (wn * tn))) warp_size _;
      forevery_map
        (fun (tid : natlt nthr) ->
          forall+ (mi : natlt wm) (nj : natlt wn).
            live_lane_cells
              (output_fragment' gD bm bn tm tn wm wn
                bid (tid / warp_size) mi nj)
              (tid % warp_size))
        (fun tid ->
          output_lane_live' gD bm bn tm tn wm wn bid tid)
        fn tid {
          fold output_lane_live' gD bm bn tm tn wm wn bid tid;
        };
    };
}
#pop-options

(* ---- lane-cell joins (local copies of the private upstream helpers) ---- *)

ghost
fn join_array2_from_lane_cells
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (#em : chest2 et rows cols)
  requires forall+ (lane : natlt warp_size). own_lane_cells m em lane
  ensures m |-> em
{
  forevery_map
    (fun lane -> own_lane_cells m em lane)
    (fun lane ->
      forall+ (ij : (natlt rows & natlt cols){
        in_lane (chunk et #_ #hvc) rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    fn lane { unfold own_lane_cells m em lane };
  forevery_join_or_n
    (fun (lane : natlt warp_size) ij ->
       in_lane (chunk et #_ #hvc) rows cols lane ij)
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  Classical.forall_intro (in_lane_covers_all (chunk et #_ #hvc) rows cols);
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap (chunk et #_ #hvc) rows cols ij lane1));
  forevery_refine_ext #_
    #(fun (ij : natlt rows & natlt cols) ->
      exists lane. in_lane (chunk et #_ #hvc) rows cols lane ij)
    (fun _ -> True) _;
  forevery_unflatten' _;
  tensor_iraise2 m;
}

(* Live variant of [join_lane_cells_approximates]: joins each lane's cells back
   into a live matrix, with the contents existentially quantified (no [%~]). *)
ghost
fn join_array2_from_live_lane_cells
  (#et : Type0) {| scalar et, hvc : has_vec_cpy et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  requires forall+ (lane : natlt warp_size). live_lane_cells m lane
  ensures exists* (em : chest2 et rows cols). m |-> em
{
  forevery_map
    (fun (lane : natlt warp_size) -> live_lane_cells m lane)
    (fun (lane : natlt warp_size) ->
      exists* (em : chest2 et rows cols). own_lane_cells m em lane)
    fn lane { unfold live_lane_cells m lane };
  let ff = forevery_exists #(natlt warp_size)
    (fun lane em -> own_lane_cells m em lane);
  let em' : chest2 et rows cols =
    mk2 (fun i j ->
      let lane : natlt warp_size =
        VG.group_of (chunk et #_ #hvc) cols i j % warp_size in
      acc2 (ff lane) i j);
  forevery_map
    (fun (lane : natlt warp_size) -> own_lane_cells m (ff lane) lane)
    (fun (lane : natlt warp_size) -> own_lane_cells m em' lane)
    fn lane {
      unfold own_lane_cells m (ff lane) lane;
      forevery_map
        #(ij : (natlt rows & natlt cols){
           in_lane (chunk et #_ #hvc) rows cols lane ij})
        (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 (ff lane) ij._1 ij._2))
        (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em' ij._1 ij._2))
        fn ij { () };
      fold own_lane_cells m em' lane;
    };
  join_array2_from_lane_cells m;
}

(* ==== Helper 2: layout-generic live gather (inverse of the split) ==== *)

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1 --split_queries always"
ghost
fn gather_warp_live'
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| cD : T.ctlayout lD |}
  (gD : array2 et lD)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (bid : natlt (SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)))
  (wid : natlt (SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn))))
  ()
  requires
    forall+ (lane : natlt warp_size) (mi : natlt wm) (nj : natlt wn).
      live_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid wid mi nj) lane
  ensures
    exists* (eWarp : chest2 et (SZ.v wm * SZ.v tm) (SZ.v wn * SZ.v tn)).
      warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn) wid |-> eWarp
{
  let _ : squash (SZ.fits lD.ulen) = cD.ulen_fits;
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wm) (SZ.v tm);
  FStar.Math.Lemmas.cancel_mul_div (SZ.v wn) (SZ.v tn);
  assert pure ((wm * tm) / tm == wm);
  assert pure ((wn * tn) / tn == wn);
  forevery_commute
    (fun (lane : natlt warp_size) (mi : natlt wm) ->
      forall+ (nj : natlt wn).
        live_lane_cells
          (output_fragment' gD bm bn tm tn wm wn bid wid mi nj) lane);
  forevery_mid_flip
    (fun (mi : natlt wm) (lane : natlt warp_size) (nj : natlt wn) ->
      live_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid wid mi nj) lane);
  forevery_map_2
    (fun (mi : natlt wm) (nj : natlt wn) ->
      forall+ (lane : natlt warp_size).
        live_lane_cells
          (output_fragment' gD bm bn tm tn wm wn bid wid mi nj) lane)
    (fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eFrag : chest2 et tm tn).
        output_fragment' gD bm bn tm tn wm wn bid wid mi nj |-> eFrag)
    fn mi nj {
      join_array2_from_live_lane_cells
        (output_fragment' gD bm bn tm tn wm wn bid wid mi nj);
    };
  let dWarp =
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
  forevery_map_2
    (fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eFrag : chest2 et tm tn).
        output_fragment' gD bm bn tm tn wm wn bid wid mi nj |-> eFrag)
    (fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eFrag : chest2 et tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag)
    fn mi nj {
      rewrite each output_fragment' gD bm bn tm tn wm wn bid wid mi nj
        as array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj;
    };
  forevery_rw_size2 wm ((wm * tm) / tm) wn ((wn * tn) / tn)
    #(fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eFrag : chest2 et tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag);
  array2_untile_underspec dWarp (SZ.v tm) (SZ.v tn);
  rewrite each dWarp as
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
}
#pop-options

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1 --split_queries always"
ghost
fn gather_block_live'
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| cD : T.ctlayout lD |}
  (gD : array2 et lD)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (bid : natlt (SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)))
  ()
  requires
    forall+ (wid : natlt (SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)))).
      exists* (eWarp : chest2 et (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn) wid |-> eWarp
  ensures
    exists* (eBlock : chest2 et bm bn).
      block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock
{
  let _ : squash (SZ.fits lD.ulen) = cD.ulen_fits;
  forevery_map
    #(natlt (bm / (wm * tm) * (bn / (wn * tn))))
    (fun wid ->
      exists* (eWarp : chest2 et (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn) wid |-> eWarp)
    (fun wid ->
      exists* (eWarp : chest2 et (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn)
          ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
            wid % (bn / (wn * tn))) |-> eWarp)
    fn wid {
      FStar.Math.Lemmas.euclidean_division_definition wid (bn / (wn * tn));
      rewrite each wid as
        ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
          wid % (bn / (wn * tn)));
    };
  forevery_factor'
    (bm / (wm * tm) * (bn / (wn * tn)))
    (bm / (wm * tm))
    (bn / (wn * tn))
    (fun wr wc ->
      exists* (eWarp : chest2 et (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc) |-> eWarp);
  let dBlock = block_tile gD (SZ.v bm) (SZ.v bn) bid;
  rewrite each block_tile gD (SZ.v bm) (SZ.v bn) bid as dBlock;
  forevery_map_2
    #(natlt (bm / (wm * tm)))
    #(natlt (bn / (wn * tn)))
    (fun wr wc ->
      exists* (eWarp : chest2 et (wm * tm) (wn * tn)).
        warp_tile dBlock (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc) |-> eWarp)
    (fun wr wc ->
      exists* (eWarp : chest2 et (wm * tm) (wn * tn)).
        array2_subtile dBlock (wm * tm) (wn * tn) wr wc |-> eWarp)
    fn wr wc {
      assert pure (
        (wr * (bn / (wn * tn)) + wc) / (bn / (wn * tn)) == wr);
      assert pure (
        (wr * (bn / (wn * tn)) + wc) % (bn / (wn * tn)) == wc);
      rewrite each
        warp_tile dBlock (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc)
      as array2_subtile dBlock (wm * tm) (wn * tn) wr wc;
    };
  array2_untile_underspec dBlock (wm * tm) (wn * tn);
  rewrite each dBlock as block_tile gD (SZ.v bm) (SZ.v bn) bid;
}
#pop-options

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1 --split_queries always"
ghost
fn gather_output_live'
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (#m #n : szp)
  (#lD : layout2 (SZ.v m) (SZ.v n)) {| cD : T.ctlayout lD |}
  (gD : array2 et lD)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ m /\ bn /?+ n /\
                wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (#_ : squash (SZ.fits (m * n)))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_live' gD bm bn tm tn wm wn bid tid
  ensures exists* (eD : chest2 et (SZ.v m) (SZ.v n)). gD |-> eD
{
  let _ : squash (SZ.fits lD.ulen) = cD.ulen_fits;
  forevery_map
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_live' gD bm bn tm tn wm wn bid tid)
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock)
    fn bid {
      forevery_map
        (fun (tid : natlt nthr) ->
          output_lane_live' gD bm bn tm tn wm wn bid tid)
        (fun (tid : natlt nthr) ->
          forall+ (mi : natlt wm) (nj : natlt wn).
            live_lane_cells
              (output_fragment' gD bm bn tm tn wm wn
                bid (tid / warp_size) mi nj)
              (tid % warp_size))
        fn tid {
          unfold output_lane_live' gD bm bn tm tn wm wn bid tid;
        };
      forevery_factor' nthr
        (bm / (wm * tm) * (bn / (wn * tn))) warp_size
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
             (lane : natlt warp_size) ->
          forall+ (mi : natlt wm) (nj : natlt wn).
            live_lane_cells
              (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
              lane);
      forevery_map
        #(natlt (bm / (wm * tm) * (bn / (wn * tn))))
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          forall+ (lane : natlt warp_size)
                   (mi : natlt wm) (nj : natlt wn).
            live_lane_cells
              (output_fragment' gD bm bn tm tn wm wn bid wid mi nj)
              lane)
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          exists* (eWarp : chest2 et (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid |-> eWarp)
        fn wid {
          gather_warp_live' gD bm bn tm tn wm wn bid wid ();
        };
      gather_block_live' gD bm bn tm tn wm wn bid ();
    };
  forevery_map
    #(natlt nblk)
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock)
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          ((bid / (n / bn)) * (n / bn) + bid % (n / bn)) |-> eBlock)
    fn bid {
      FStar.Math.Lemmas.euclidean_division_definition bid (n / bn);
      rewrite each bid as
        ((bid / (n / bn)) * (n / bn) + bid % (n / bn));
    };
  forevery_factor' nblk (m / bm) (n / bn)
    (fun br bc ->
      exists* (eBlock : chest2 et bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock);
  forevery_map_2
    #(natlt (m / bm)) #(natlt (n / bn))
    (fun br bc ->
      exists* (eBlock : chest2 et bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock)
    (fun br bc ->
      exists* (eBlock : chest2 et bm bn).
        array2_subtile gD (SZ.v bm) (SZ.v bn) br bc |-> eBlock)
    fn br bc {
      assert pure ((br * (n / bn) + bc) / (n / bn) == br);
      assert pure ((br * (n / bn) + bc) % (n / bn) == bc);
      rewrite each block_tile gD (SZ.v bm) (SZ.v bn)
        (br * (n / bn) + bc)
      as array2_subtile gD (SZ.v bm) (SZ.v bn) br bc;
    };
  array2_untile_underspec gD (SZ.v bm) (SZ.v bn);
}
#pop-options
