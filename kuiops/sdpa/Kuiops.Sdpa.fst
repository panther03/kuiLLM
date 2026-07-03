module Kuiops.SDPA

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Bijection
open Kuiper.Float.Casts

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Spec.Attention
open Kuiper.Kernel.BatchedGEMM
open Kuiper.Kernel.RowSoftmax

let fold_unfold_chest_id (#et:Type0) (#r:nat{r>1}) (#d:shape r) (m : chest d et)
  : Lemma (unfold_chest #et #r #d (fold_chest #et #r #d m) == m)
  = let ICons h1 (ICons h2 ts) = d in
    introduce forall (i : abs d).
      acc (unfold_chest #et #r #d (fold_chest #et #r #d m)) i == acc m i
    with (
      let (i1,(i2,it)) : (natlt h1 & (natlt h2 & abs ts)) = i in
      FStar.Math.Lemmas.lemma_div_plus i2 i1 h2;
      FStar.Math.Lemmas.lemma_mod_plus i2 i1 h2;
      FStar.Math.Lemmas.small_div i2 h2;
      FStar.Math.Lemmas.small_mod i2 h2;
      assert (unfold_index #r #d (fold_index #r #d i) == i)
    );
    Kuiper.Chest.lemma_equal_intro (unfold_chest #et #r #d (fold_chest #et #r #d m)) m;
    Kuiper.Chest.ext (unfold_chest #et #r #d (fold_chest #et #r #d m)) m

let sdpa_scores_spec 
  (#n #h : szp)
  (#l #s : szp)
  (#e: szp)
  (rQ : chest4 real n h l e)
  (rK : chest4 real n h e s)
  (rS : chest4 real n h l s)
  (rscale : real) = 
  mk4 (fun i j -> acc2 (attn_scores
          (slice_page4 (rQ) i j)
          (slice_page4 (rK) i j)
          (slice_page4 (rS) i j)
          rscale))

let sdpa_naive_aux
  (#n #h : szp { SZ.fits (n * h) })
  (#l #s : szp)
  (#e: szp)
  (rQ : chest4 real n h l e)
  (rK : chest4 real n h e s)
  (rS : chest4 real n h l s)
  (rscale : real):
  Lemma (ensures 
    (sdpa_scores_spec rQ rK rS rscale)
    == 
    (unfold_chest (MS.bmmcomb (fun bias_qk score -> bias_qk +. (score *. rscale)) 
      #(SZ.v (n *^ h)) #l #e #s (fold_chest rS) (fold_chest rQ) (fold_chest rK)))) = admit ()

inline_for_extraction noextract
fn sdpa_naive_scores
  (#et: Type0) 
  {| floating et, floating et |}
  {| real_like et, real_like et |}
  {| floating_real_like et, floating_real_like et |}
  {| float_cast et et, float_cast et et |}
  (n h : szp)
  (l s : szp)
  (e: szp)
  (#lQ: tlayout    (n @| h @| l @| e @| INil)  { is_full lQ }) // needed for tlayout_bij for now.
  (#lK: tlayout    (n @| h @| e @| s @| INil)  { is_full lK })
  (#lS: tlayout    (n @| h @| l @| s @| INil)  { is_full lS })
  {| ctlayout lQ, ctlayout lK, ctlayout lS |}
  (gQ    : tensor et lQ    { is_global gQ    })
  (gK    : tensor et lK    { is_global gK    })
  (gS    : tensor et lS    { is_global gS })
  (scale : et)
  (#eQ : erased    (chest4 et n h l e))
  (#eK : erased    (chest4 et n h e s))
  (#eS : erased    (chest4 et n h l s))
  (#fQ #fK : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gQ    |-> Frac fQ eQ) **
    on gpu_loc (gK    |-> Frac fK eK)
  requires
    on gpu_loc (gS |-> eS) **
    pure (
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
      SZ.fits (h * l) /\
      l * s <= max_blocks * max_threads /\
      n * h * l <= max_blocks /\
      n * h * l * s <= max_blocks * max_threads
    )
  ensures
    (exists* (eS': chest4 et n h l s). 
      on gpu_loc (gS |-> eS') ** 
      pure (
        is_global gS /\
        eS' %~ sdpa_scores_spec 
          (to_real_chest eQ)
          (to_real_chest eK)
          (to_real_chest eS)
          (to_real scale)))
{
  let rQ    = to_real_chest eQ;
  let rK    = to_real_chest eK;
  let rS    = to_real_chest eS;

  // I guess there should just be a function that _returns_ the new tensor,
  // similar to how we have for tiles and slices
  let gQf = from_array (tlayout_fold_outer lQ) (core gQ);
  let gKf = from_array (tlayout_fold_outer lK) (core gK);
  let gSf = from_array (tlayout_fold_outer lS) (core gS);
  assert rewrites_to gQf (from_array (tlayout_fold_outer lQ) (core gQ));
  assert rewrites_to gKf (from_array (tlayout_fold_outer lK) (core gK));
  assert rewrites_to gSf (from_array (tlayout_fold_outer lS) (core gS));
  map_loc gpu_loc (fun () -> tensor_fold_outer gQ #fQ);
  map_loc gpu_loc (fun () -> tensor_fold_outer gK #fK);
  map_loc gpu_loc (fun () -> tensor_fold_outer gS);
  let eQf = fold_chest eQ; let rQf = fold_chest rQ;
  let eKf = fold_chest eK; let rKf = fold_chest rK;
  let eSf = fold_chest eS; let rSf = fold_chest rS;

  assert on gpu_loc (gQf |-> Frac fQ eQf);
  assert on gpu_loc (gKf |-> Frac fK eKf);

  //let comb = (fun bias_qk score -> bias_qk `add` (score `mul` scale));
  bmmcomb_gpu_exact #et (fun bias_qk score -> bias_qk `add` (score `mul` scale))
    (n*^h) l e s #_ #_ #_ 
    #(ctlayout_fold_outer lQ) #(ctlayout_fold_outer lK) #(ctlayout_fold_outer lS)
    gQf gKf gSf;

  // Unneeded for below??  i dont remember what i was doing
  // let lSf: tlayout ((SZ.v (n *^ h)) @| l @| s @| INil) = tlayout_fold_outer lS;
  // let gSf': tensor et lSf = gSf;
  // assert rewrites_to gSf' gSf;
  // assert on gpu_loc (gSf' |-> (MS.bmmcomb comb #(SZ.v (n *^ h)) #l #e #s (fold_chest eS) eQf eKf));
  with eSf'. assert on gpu_loc (gSf |-> eSf');

  map_loc gpu_loc (fun () -> tensor_unfold_outer gQf #fQ);
  map_loc gpu_loc (fun () -> tensor_unfold_outer gKf #fK);
  map_loc gpu_loc (fun () -> tensor_unfold_outer gSf);
  
  // This is stupid and should just be a trade; fold and then restore when done.
  fold_unfold_chest_id eQ;
  rewrite (on gpu_loc ((from_array lQ (core gQf)) |-> Frac fQ (unfold_chest eQf))) 
    as (on gpu_loc (gQ |-> Frac fQ eQ));
  fold_unfold_chest_id eK;
  rewrite (on gpu_loc ((from_array lK (core gKf)) |-> Frac fK (unfold_chest eKf))) 
    as (on gpu_loc (gK |-> Frac fK eK));

  // This should also be a trade, similar to the ghosts that we have for tiling and slices etc.
  rewrite (on gpu_loc ((from_array lS (core gSf)) |-> unfold_chest eSf')) 
    as (on gpu_loc (gS |-> unfold_chest eSf'));
  assume pure (unfold_chest eSf' %~ 
    (unfold_chest (MS.bmmcomb (fun bias_qk score -> bias_qk +. (score *. (to_real scale))) 
      #(SZ.v (n *^ h)) #l #e #s (fold_chest rS) (fold_chest rQ) (fold_chest rK))));
  sdpa_naive_aux rQ rK rS (to_real scale);
  assert pure (unfold_chest eSf' %~ sdpa_scores_spec rQ rK rS (to_real scale));

  ()
}

// TODO: this shouldn't require so much boilerplate...

#push-options "--split_queries always"
let transpose4_2 (#d0 #d1 #d2 #d3 : nat) : 
  (abs (d0 @| d1 @| d2 @| d3 @| INil) =~ abs (d0 @| d1 @| d3 @| d2 @| INil)) =
{
  ff = (fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,())))));
  gg = (fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,())))));
  // weird that ez doesn't take care of it...
  ff_gg = (fun x -> (
    let (i,(j,(k,(l,())))) = x in
    assert ((fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,()))))) x) == (i,(j,(l,(k,()))))
  ));
  gg_ff = (fun x -> (
    let (i,(j,(k,(l,())))) = x in
    assert ((fun (i,(j,(k,(l,())))) -> (i,(j,(l,(k,()))))) x) == (i,(j,(l,(k,()))))
  ));
}

/// Concrete index mapping for [transpose4_2], mirroring its ghost inverse [gg]
/// (which swaps the last two dimensions). Carries no proof obligations beyond the
/// pointwise swap, so [transpose4_2_conc_correct] is definitional.
inline_for_extraction noextract
let transpose4_2_conc (#d0 #d1 #d2 #d3 : nat)
  (x : conc (d0 @| d1 @| d3 @| d2 @| INil))
  : conc (d0 @| d1 @| d2 @| d3 @| INil)
  = let (i,(j,(k,(l,())))) = x in (i,(j,(l,(k,()))))

let transpose4_2_conc_correct (#d0 #d1 #d2 #d3 : nat)
  (x : conc (d0 @| d1 @| d3 @| d2 @| INil))
  : (up (transpose4_2_conc #d0 #d1 #d2 #d3 x) == (transpose4_2 #d0 #d1 #d2 #d3).gg (up x))
  = ()

/// Extractable [ctlayout] for the K-transpose relayout: instantiates [ctlayout_bij]
/// with the concrete swap above.
inline_for_extraction noextract
let ctlayout_bij_transpose
  (#d0 #d1 #d2 #d3 : szp)
  (lin : tlayout (d0 @| d1 @| d2 @| d3 @| INil)) {| c : ctlayout lin |}
  : ctlayout (tlayout_bij (transpose4_2 #(SZ.v d0) #(SZ.v d1) #(SZ.v d2) #(SZ.v d3)) lin)
  = ctlayout_bij (transpose4_2 #(SZ.v d0) #(SZ.v d1) #(SZ.v d2) #(SZ.v d3))
      (transpose4_2_conc #(SZ.v d0) #(SZ.v d1) #(SZ.v d2) #(SZ.v d3))
      (transpose4_2_conc_correct #(SZ.v d0) #(SZ.v d1) #(SZ.v d2) #(SZ.v d3))
      lin

inline_for_extraction noextract
fn sdpa_naive
  (#et: Type0) 
  {| floating et, floating et |}
  {| real_like et, real_like et |}
  {| floating_real_like et, floating_real_like et |}
  {| float_cast et et, float_cast et et |}
  (n h : szp)
  (l s : szp)
  (e ev : szp)
  (#lQ: tlayout    (n @| h @| l @| e @| INil)  { is_full lQ }) // needed for tlayout_bij for now.
  (#lK: tlayout    (n @| h @| s @| e @| INil)  { is_full lK })
  (#lV: tlayout    (n @| h @| s @| ev @| INil) { is_full lV })
  (#lbias: tlayout (n @| h @| l @| s @| INil)  { is_full lbias })
  {| ctlayout lQ, ctlayout lK, ctlayout lV, ctlayout lbias |}
  (gQ    : tensor et lQ    { is_global gQ    })
  (gK    : tensor et lK    { is_global gK    })
  (gV    : tensor et lV    { is_global gV    })
  (gbias : tensor et lbias { is_global gbias })
  (scale : et)
  (#eQ : erased    (chest4 et n h l e))
  (#eK : erased    (chest4 et n h s e))
  (#eV : erased    (chest4 et n h s ev))
  (#ebias : erased (chest4 et n h l s))
  (#rKT : erased   (chest4 real n h e s))
  (#fQ #fK #fV : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gQ    |-> Frac fQ eQ) **
    on gpu_loc (gK    |-> Frac fK eK) **
    on gpu_loc (gV    |-> Frac fV eV)
  requires
    on gpu_loc (gbias |-> ebias) **
    pure (
      SZ.fits (l * ev) /\
      SZ.fits (n * h * l * e) /\
      SZ.fits (n * h * s * e)  /\
      SZ.fits (n * h * s * ev)  /\
      SZ.fits (n * h * l * ev)  /\ 
      SZ.fits (n * h * l * s)  /\
      SZ.fits (n * h * l) /\
      SZ.fits (h * l) /\
      (mk4 (fun i j k l -> acc4 eK i j l k)) %~ rKT /\
      l * s <= max_blocks * max_threads /\
      l * ev <= max_blocks * max_threads /\
      n * h * l <= max_blocks /\
      n * h * l * s <= max_blocks * max_threads
    )
  returns
    // TODO: polymorphic out layout
    out : tensor et (l4_batched_row_major n h l ev)
  ensures
    // For simplicity, bias is used to hold the scores.
    live gbias **
    (exists* (eO : chest4 et n h l ev).
      on gpu_loc (out |-> eO) **
      pure (
        is_global out /\
        eO %~ attention_real_batched
          (to_real_chest eQ)
          rKT
          (to_real_chest eV)
          (to_real_chest ebias)
          (to_real scale))) 
{
  
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gQ);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gK);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gV);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gbias);

  let rQ    = to_real_chest eQ;
  let rV    = to_real_chest eV;
  let rbias = to_real_chest ebias;
  
  // Transpose K via ghost
  let f_transpose = transpose4_2 #n #h #s #e; 
  let gKT: tensor et (tlayout_bij f_transpose lK) = from_array (tlayout_bij f_transpose lK) (core gK);
  assert rewrites_to gKT (from_array (tlayout_bij f_transpose lK) (core gK));
  map_loc gpu_loc (fun () -> tensor_apply_bij f_transpose gK #fK);
  let eKT = mk (n @| h @| e @| s @| INil) (fun i -> acc eK (i <~| f_transpose));
  assert on gpu_loc (gKT |-> Frac fK eKT);
  assert pure (eKT %~ rKT);

  admit ();
}