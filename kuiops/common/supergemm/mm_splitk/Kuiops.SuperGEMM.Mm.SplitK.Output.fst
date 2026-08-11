module Kuiops.SuperGEMM.Mm.SplitK.Output

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Kernel.GEMM.Tiled.Common.Vec { block_tile, warp_tile }
open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { warp_tile_pts_to, warp_tile_approximates }
open Kuiper.Kernel.GEMM.TensorCore2D.To.EpilogueState { epilogue_warp_input }
open Kuiops.SuperGEMM.Mm.SplitK.Gather
open Kuiper.EMatrix.Tiling { ematrix_subtile }

module SZ = Kuiper.SizeT
module T = Kuiper.Tensor

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1 --split_queries always"
ghost
fn split_ws_to_warps
  (#et : Type0) {| scalar et |}
  (#mm #nn : szp)
  (#lW : layout2 (SZ.v mm) (SZ.v nn)) {| T.ctlayout lW |}
  (gW : array2 et lW)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ mm /\ bn /?+ nn /\
                wm * tm /?+ bm /\ wn * tn /?+ bn /\
                tm /?+ bm /\ tn /?+ bn))
  (nblk : szp{SZ.v nblk == SZ.v mm / SZ.v bm * (SZ.v nn / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires live gW
  ensures
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      ws_warp_live gW bm bn tm tn wm wn bid tid
{
  with (eW : chest2 _ _ _). assert gW |-> eW;
  array2_tile gW (SZ.v bm) (SZ.v bn) #eW #1.0R;
  forevery_unfactor' nblk (mm / bm) (nn / bn) _;

  forevery_map
    (fun (bid : natlt nblk) ->
      array2_subtile gW (SZ.v bm) (SZ.v bn)
        (bid / (nn / bn)) (bid % (nn / bn))
        |-> ematrix_subtile eW bm bn
              (bid / (nn / bn)) (bid % (nn / bn)))
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        ws_warp_live gW bm bn tm tn wm wn bid tid)
    fn bid {
      rewrite each
        array2_subtile gW (SZ.v bm) (SZ.v bn)
          (bid / (nn / bn)) (bid % (nn / bn))
      as block_tile gW (SZ.v bm) (SZ.v bn) bid;
      let wBlock = block_tile gW (SZ.v bm) (SZ.v bn) bid;
      let eBlock =
        ematrix_subtile eW (SZ.v bm) (SZ.v bn)
          (bid / (nn / bn)) (bid % (nn / bn));
      array2_tile (block_tile gW (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn)
        #(ematrix_subtile eW (SZ.v bm) (SZ.v bn)
          (bid / (nn / bn)) (bid % (nn / bn)))
        #1.0R;
      rewrite each block_tile gW (SZ.v bm) (SZ.v bn) bid as wBlock;
      rewrite each
        ematrix_subtile eW (SZ.v bm) (SZ.v bn)
          (bid / (nn / bn)) (bid % (nn / bn))
        as eBlock;
      forevery_unfactor'
        (bm / (wm * tm) * (bn / (wn * tn)))
        (bm / (wm * tm))
        (bn / (wn * tn))
        _;
      forevery_map
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          array2_subtile wBlock
            (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))
          |-> ematrix_subtile eBlock (wm * tm) (wn * tn)
                (wid / (bn / (wn * tn)))
                (wid % (bn / (wn * tn))))
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          forall+ (lane : natlt warp_size).
            exists* (em : chest2 et (wm * tm) (wn * tn)).
              warp_tile_pts_to gW bm bn tm tn wm wn bid wid em)
        fn wid {
          rewrite each
            array2_subtile wBlock
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn)))
              (wid % (bn / (wn * tn)))
          as warp_tile wBlock (wm * tm) (wn * tn) wid;
          rewrite each
            warp_tile wBlock (wm * tm) (wn * tn) wid
          as warp_tile (block_tile gW (SZ.v bm) (SZ.v bn) bid) (wm * tm) (wn * tn) wid;
          tensor_share_n
            (warp_tile (block_tile gW (SZ.v bm) (SZ.v bn) bid) (wm * tm) (wn * tn) wid)
            warp_size;
          forevery_map
            (fun (lane : natlt warp_size) ->
              warp_tile (block_tile gW (SZ.v bm) (SZ.v bn) bid) (wm * tm) (wn * tn) wid
                |-> Frac (1.0R /. warp_size)
                  (ematrix_subtile eBlock (wm * tm) (wn * tn)
                    (wid / (bn / (wn * tn)))
                    (wid % (bn / (wn * tn)))))
            (fun (lane : natlt warp_size) ->
              exists* (em : chest2 et (wm * tm) (wn * tn)).
                warp_tile_pts_to gW bm bn tm tn wm wn bid wid em)
            fn lane {
              fold (warp_tile_pts_to gW bm bn tm tn wm wn bid wid
                (ematrix_subtile eBlock (wm * tm) (wn * tn)
                  (wid / (bn / (wn * tn)))
                  (wid % (bn / (wn * tn)))));
            };
        };
      forevery_unfactor' nthr
        (bm / (wm * tm) * (bn / (wn * tn))) warp_size _;
      forevery_map
        (fun (tid : natlt nthr) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            warp_tile_pts_to gW bm bn tm tn wm wn bid (tid / warp_size) em)
        (fun (tid : natlt nthr) ->
          ws_warp_live gW bm bn tm tn wm wn bid tid)
        fn tid {
          fold (ws_warp_live gW bm bn tm tn wm wn bid tid);
        };
    };
}
#pop-options

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1 --split_queries always"
ghost
fn gather_ws_warps
  (#et : Type0) {| scalar et |}
  (#mm #nn : szp)
  (#lW : layout2 (SZ.v mm) (SZ.v nn)) {| T.ctlayout lW |}
  (gW : array2 et lW)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ mm /\ bn /?+ nn /\
                wm * tm /?+ bm /\ wn * tn /?+ bn /\
                tm /?+ bm /\ tn /?+ bn))
  (#_ : squash (SZ.fits lW.ulen))
  (nblk : szp{SZ.v nblk == SZ.v mm / SZ.v bm * (SZ.v nn / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  ()
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      ws_warp_live gW bm bn tm tn wm wn bid tid
  ensures live gW
{
  forevery_map
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        ws_warp_live gW bm bn tm tn wm wn bid tid)
    (fun (bid : natlt nblk) ->
      exists* (em : chest2 et (SZ.v bm) (SZ.v bn)).
        block_tile gW (SZ.v bm) (SZ.v bn) bid |-> em)
    fn bid {
      let wBlock = block_tile gW (SZ.v bm) (SZ.v bn) bid;
      forevery_map
        (fun (tid : natlt nthr) ->
          ws_warp_live gW bm bn tm tn wm wn bid tid)
        (fun (tid : natlt nthr) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            warp_tile wBlock (wm * tm) (wn * tn) (tid / warp_size)
              |-> Frac (1.0R /. warp_size) em)
        fn tid {
          unfold (ws_warp_live gW bm bn tm tn wm wn bid tid);
          with em. assert (warp_tile_pts_to gW bm bn tm tn wm wn bid (tid / warp_size) em);
          unfold (warp_tile_pts_to gW bm bn tm tn wm wn bid (tid / warp_size) em);
          rewrite each
            warp_tile (block_tile gW (SZ.v bm) (SZ.v bn) bid) (wm * tm) (wn * tn) (tid / warp_size)
          as warp_tile wBlock (wm * tm) (wn * tn) (tid / warp_size);
        };
      forevery_factor' nthr
        (bm / (wm * tm) * (bn / (wn * tn))) warp_size
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) (lane : natlt warp_size) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            warp_tile wBlock (wm * tm) (wn * tn) wid
              |-> Frac (1.0R /. warp_size) em);
      forevery_map
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          forall+ (lane : natlt warp_size).
            exists* (em : chest2 et (wm * tm) (wn * tn)).
              warp_tile wBlock (wm * tm) (wn * tn) wid
                |-> Frac (1.0R /. warp_size) em)
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            array2_subtile wBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn)))
              |-> em)
        fn wid {
          tensor_gather_n_underspec (warp_tile wBlock (wm * tm) (wn * tn) wid) warp_size;
          rewrite each warp_tile wBlock (wm * tm) (wn * tn) wid
          as array2_subtile wBlock (wm * tm) (wn * tn)
               (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn)));
        };
      forevery_factor'
        (bm / (wm * tm) * (bn / (wn * tn)))
        (bm / (wm * tm))
        (bn / (wn * tn))
        (fun (wr : natlt (bm / (wm * tm))) (wc : natlt (bn / (wn * tn))) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            array2_subtile wBlock (wm * tm) (wn * tn) wr wc |-> em);
      array2_untile_underspec wBlock (wm * tm) (wn * tn) #1.0R;
      rewrite each wBlock as
        array2_subtile gW (SZ.v bm) (SZ.v bn) (bid / (nn / bn)) (bid % (nn / bn));
    };
  forevery_factor' nblk (mm / bm) (nn / bn)
    (fun (br : natlt (mm / bm)) (bc : natlt (nn / bn)) ->
      exists* (em : chest2 et (SZ.v bm) (SZ.v bn)).
        array2_subtile gW (SZ.v bm) (SZ.v bn) br bc |-> em);
  array2_untile_underspec gW (SZ.v bm) (SZ.v bn) #1.0R;
}
#pop-options

#push-options "--z3rlimit 15 --fuel 1 --ifuel 1 --split_queries always"
ghost
fn gather_ws_approximates
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#mm #nn : szp)
  (#lW : layout2 (SZ.v mm) (SZ.v nn)) {| T.ctlayout lW |}
  (gW : array2 et lW)
  (bm bn tm tn wm wn : szp)
  (#_ : squash (bm /?+ mm /\ bn /?+ nn /\
                wm * tm /?+ bm /\ wn * tn /?+ bn /\
                tm /?+ bm /\ tn /?+ bn))
  (#_ : squash (SZ.fits lW.ulen))
  (nblk : szp{SZ.v nblk == SZ.v mm / SZ.v bm * (SZ.v nn / SZ.v bn)})
  (nthr : szp{SZ.v nthr == SZ.v bm / (SZ.v wm * SZ.v tm) * (SZ.v bn / (SZ.v wn * SZ.v tn)) * warp_size})
  (rW : chest2 real (SZ.v mm) (SZ.v nn))
  ()
  requires
    forall+ (bid : natlt nblk) (tid : natlt nthr).
      ws_warp_approximates gW bm bn tm tn wm wn bid tid rW
  ensures
    exists* (eW : chest2 et (SZ.v mm) (SZ.v nn)).
      gW |-> eW ** pure (eW %~ rW)
{
  forevery_map
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        ws_warp_approximates gW bm bn tm tn wm wn bid tid rW)
    (fun (bid : natlt nblk) ->
      exists* (em : chest2 et (SZ.v bm) (SZ.v bn)).
        block_tile gW (SZ.v bm) (SZ.v bn) bid |-> em **
        pure (em %~ ematrix_subtile rW (SZ.v bm) (SZ.v bn)
                      (bid / (nn / bn)) (bid % (nn / bn))))
    fn bid {
      let wBlock = block_tile gW (SZ.v bm) (SZ.v bn) bid;
      forevery_map
        (fun (tid : natlt nthr) ->
          ws_warp_approximates gW bm bn tm tn wm wn bid tid rW)
        (fun (tid : natlt nthr) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            warp_tile wBlock (wm * tm) (wn * tn) (tid / warp_size)
              |-> Frac (1.0R /. warp_size) em **
            pure (em %~ ematrix_subtile
                          (ematrix_subtile rW (SZ.v bm) (SZ.v bn)
                            (bid / (nn / bn)) (bid % (nn / bn)))
                          (wm * tm) (wn * tn)
                          ((tid / warp_size) / (bn / (wn * tn)))
                          ((tid / warp_size) % (bn / (wn * tn)))))
        fn tid {
          unfold (ws_warp_approximates gW bm bn tm tn wm wn bid tid rW);
          unfold (warp_tile_approximates gW bm bn tm tn wm wn bid (tid / warp_size)
                   (epilogue_warp_input rW bm bn tm tn wm wn bid tid));
          with em. assert (warp_tile_pts_to gW bm bn tm tn wm wn bid (tid / warp_size) em);
          unfold (warp_tile_pts_to gW bm bn tm tn wm wn bid (tid / warp_size) em);
          rewrite each
            warp_tile (block_tile gW (SZ.v bm) (SZ.v bn) bid) (wm * tm) (wn * tn) (tid / warp_size)
          as warp_tile wBlock (wm * tm) (wn * tn) (tid / warp_size);
        };
      forevery_factor' nthr
        (bm / (wm * tm) * (bn / (wn * tn))) warp_size
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) (lane : natlt warp_size) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            warp_tile wBlock (wm * tm) (wn * tn) wid
              |-> Frac (1.0R /. warp_size) em **
            pure (em %~ ematrix_subtile
                          (ematrix_subtile rW (SZ.v bm) (SZ.v bn)
                            (bid / (nn / bn)) (bid % (nn / bn)))
                          (wm * tm) (wn * tn)
                          (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn)))));
      forevery_map
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          forall+ (lane : natlt warp_size).
            exists* (em : chest2 et (wm * tm) (wn * tn)).
              warp_tile wBlock (wm * tm) (wn * tn) wid
                |-> Frac (1.0R /. warp_size) em **
              pure (em %~ ematrix_subtile
                            (ematrix_subtile rW (SZ.v bm) (SZ.v bn)
                              (bid / (nn / bn)) (bid % (nn / bn)))
                            (wm * tm) (wn * tn)
                            (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn)))))
        (fun (wid : natlt (bm / (wm * tm) * (bn / (wn * tn)))) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            array2_subtile wBlock (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn)))
              |-> em **
            pure (em %~ ematrix_subtile
                          (ematrix_subtile rW (SZ.v bm) (SZ.v bn)
                            (bid / (nn / bn)) (bid % (nn / bn)))
                          (wm * tm) (wn * tn)
                          (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn)))))
        fn wid {
          array2_gather_n_approximates (warp_tile wBlock (wm * tm) (wn * tn) wid)
            warp_size #1.0R
            (ematrix_subtile
              (ematrix_subtile rW (SZ.v bm) (SZ.v bn)
                (bid / (nn / bn)) (bid % (nn / bn)))
              (wm * tm) (wn * tn)
              (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn))));
          rewrite each warp_tile wBlock (wm * tm) (wn * tn) wid
          as array2_subtile wBlock (wm * tm) (wn * tn)
               (wid / (bn / (wn * tn))) (wid % (bn / (wn * tn)));
        };
      forevery_factor'
        (bm / (wm * tm) * (bn / (wn * tn)))
        (bm / (wm * tm))
        (bn / (wn * tn))
        (fun (wr : natlt (bm / (wm * tm))) (wc : natlt (bn / (wn * tn))) ->
          exists* (em : chest2 et (wm * tm) (wn * tn)).
            array2_subtile wBlock (wm * tm) (wn * tn) wr wc |-> em **
            pure (em %~ ematrix_subtile
                          (ematrix_subtile rW (SZ.v bm) (SZ.v bn)
                            (bid / (nn / bn)) (bid % (nn / bn)))
                          (wm * tm) (wn * tn) wr wc));
      array2_untile_approximates wBlock (wm * tm) (wn * tn)
        (ematrix_subtile rW (SZ.v bm) (SZ.v bn)
          (bid / (nn / bn)) (bid % (nn / bn)));
      rewrite each wBlock as
        array2_subtile gW (SZ.v bm) (SZ.v bn) (bid / (nn / bn)) (bid % (nn / bn));
    };
  forevery_factor' nblk (mm / bm) (nn / bn)
    (fun (br : natlt (mm / bm)) (bc : natlt (nn / bn)) ->
      exists* (em : chest2 et (SZ.v bm) (SZ.v bn)).
        array2_subtile gW (SZ.v bm) (SZ.v bn) br bc |-> em **
        pure (em %~ ematrix_subtile rW (SZ.v bm) (SZ.v bn) br bc));
  array2_untile_approximates gW (SZ.v bm) (SZ.v bn) rW;
}
#pop-options
