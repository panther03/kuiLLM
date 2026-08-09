module Kuiops.Sdpa.Flash.Approx

(* The top of the functional-correctness tower: under
   [Kuiops.Sdpa.Flash.Spec.sdpa_flash_finite], the chest the kernel leaves in
   the output tensor approximates [Kuiops.Sdpa.Flash.Spec.sdpa_flash_real]. *)

open Kuiper
open Kuiper.Common
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Floating
open Kuiper.Scalars.Base
open Kuiper.Approximates.Base
open Kuiper.EMatrix.Tiling
open Kuiper.Kernel.FlashAttention.KernelDesc
open Kuiops.Sdpa.Flash.Types

module SZ = Kuiper.SizeT
module FC = Kuiper.Float.Casts
module SF = Kuiops.Sdpa.Flash.Spec.Float
module SS = Kuiops.Sdpa.Flash.Spec.Step
module SV = Kuiops.Sdpa.Flash.Vals
module SD = Kuiops.Sdpa.Flash.Denom
module FSpec = Kuiops.Sdpa.Flash.Spec
module FSp = Kuiops.Sdpa.Flash.Split

(* ------------------------------------------------------------------ *)
(* The key / value page a block reads.                                 *)
(* ------------------------------------------------------------------ *)

(* The kernel narrows K and V by slicing the batch axis and then the head
   axis; the spec slices the head axis first.  Both land on the same page. *)
unfold let page_kv
  (#et : Type0) (#b #hkv #sk #d : nat)
  (eK : chest (b @| hkv @| sk @| d @| INil) et)
  (bi : natlt b) (kvh : natlt hkv)
  : chest2 et sk d
  = FSpec.page_kv eK bi kvh

let page_kv_real
  (#et : Type0) {| scalar et |} {| real_like et |}
  (#b #hkv #sk #d : nat)
  (eK : chest (b @| hkv @| sk @| d @| INil) et)
  (bi : natlt b) (kvh : natlt hkv)
  : Lemma (to_real_chest (page_kv eK bi kvh)
           == slice_page4 (to_real_chest eK) bi kvh)
  = assert (equal (to_real_chest (page_kv eK bi kvh))
                  (slice_page4 (to_real_chest eK) bi kvh))

(* ------------------------------------------------------------------ *)
(* The published block values feed the epilogue lemma.                 *)
(* ------------------------------------------------------------------ *)

#push-options "--z3rlimit 30 --fuel 1 --ifuel 2"

let escale_at_cell
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  (i : natlt 16) (w : natlt (SZ.v nw))
  : Lemma
      (acc2 (SV.flash_escale_at nw d b hq sq rows sk eQ eKg emask has_mask
               causal scale bi kvh group r0) w i
       == SF.gscale
            (SV.flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal
               scale bi kvh group r0)
            (SF.gmax
              (SV.flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal
                 scale bi kvh group r0) i (SZ.v nw)) w i)
  = let eM = SV.flash_eM_at nw d b hq sq rows sk eQ eKg emask has_mask causal
               scale bi kvh group r0 in
    let escale = SV.flash_escale_at nw d b hq sq rows sk eQ eKg emask has_mask
                   causal scale bi kvh group r0 in
    SV.flash_escale_egl_at_def nw d b hq sq rows sk eQ eKg emask has_mask
      causal scale bi kvh group r0 i;
    assert (acc2 (ematrix_stride_subtile escale 1 16 0 i) w 0
            == acc2 escale w i);
    assert (acc2 (SF.gscale_col eM i) w 0
            == SF.gscale eM (SF.gmax eM i (SZ.v nw)) w i)

let eO_at_cell
  (#et_ab #et_acc : Type0)
  {| floating et_acc |} {| real_like et_acc |}
  {| scalar et_ab |} {| real_like et_ab |}
  {| FC.float_cast et_ab et_acc |} {| FC.float_cast et_acc et_ab |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg eVg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat)
  (i : natlt 16) (c : natlt (SZ.v d)) (w : natlt (SZ.v nw))
  : Lemma
      (SS.ocomb_val
         (SV.flash_eO_at nw d b hq sq rows sk eQ eKg eVg emask has_mask causal
            scale bi kvh group r0) i c w
       == acc2 (SS.block_O emask has_mask causal bi kvh group (SZ.v rows) r0
                  scale (SF.q_tile 16 (SZ.v rows) group eQ bi kvh r0) eKg eVg
                  (SZ.v nw)
                  (SF.key_tiles 16 16 (SZ.v sq) (SZ.v sk) (SZ.v rows) r0 causal)
                  w) i c)
  = let dv : pos = SZ.v d in
    let eO = SV.flash_eO_at nw d b hq sq rows sk eQ eKg eVg emask has_mask
               causal scale bi kvh group r0 in
    SF.ocomb_row_lt (SZ.v nw) 16 w i;
    SV.flash_eO_at_def nw d b hq sq rows sk eQ eKg eVg emask has_mask causal
      scale bi kvh group r0 w;
    assert (acc2 (ematrix_subtile eO 16 dv w 0) i c
            == acc2 eO (SF.ocomb_row 16 w i) c)

#pop-options

(* ------------------------------------------------------------------ *)
(* The overflow side conditions, per output row.                       *)
(* ------------------------------------------------------------------ *)

(* [Spec.sdpa_flash_finite] specialised to one query row: every score the
   lane computes for a key it attends to is finite.  Everything else the
   float level needs -- finiteness of the correction weights and positivity
   of the epilogue denominator -- follows from this ([Flash.Denom]). *)
let row_no_overflow
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq sq rows sk : szp)
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eKg : chest2 et_ab (SZ.v sk) (SZ.v d))
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh group r0 : nat) (i : natlt 16)
  : prop
  = let hqv : pos = SZ.v hq in
    let sqv : pos = SZ.v sq in
    let skv : pos = SZ.v sk in
    let rowsv : pos = SZ.v rows in
    let qh = SF.lane_qh hqv sqv kvh group rowsv r0 i in
    let qpos = SF.lane_qpos sqv rowsv r0 i in
    let cbound = SF.lane_cbound sqv skv rowsv r0 i in
    let eQt = SF.q_tile 16 rowsv group eQ bi kvh r0 in
    SS.all_finite emask has_mask true causal bi qh qpos cbound scale
      eQt eKg i

#push-options "--z3rlimit 40 --fuel 1 --ifuel 2 --split_queries always"

(* One logical output cell of the kernel approximates the spec's attention
   output for the query row it belongs to. *)
let out_vfun_cell
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _fr : floating_real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq hkv group sq rows sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq })
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eK eV : chest (SZ.v b @| SZ.v hkv @| SZ.v sk @| SZ.v d @| INil) et_ab)
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh : natlt (SZ.v hkv))
  (r : natlt (SZ.v rows)) (dd : natlt (SZ.v d))
  : Lemma
      (requires
        row_no_overflow nw d b hq sq rows sk eQ (page_kv eK bi kvh) emask
          has_mask causal scale bi kvh (SZ.v group) (r / 16 * 16) (r % 16))
      (ensures
        (let qh = out_qh (SZ.v hq) (SZ.v sq) kvh (SZ.v group) r in
         let qpos = out_qpos (SZ.v sq) r in
         flash_out_vfun #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
           nw d b hq hkv group sq rows sk eQ eK eV emask
           has_mask causal scale (bi, kvh, r, dd)
         %~ acc2 (FSpec.attention_page_real
                    (slice_page4 (to_real_chest eQ) bi qh)
                    (slice_page4 (to_real_chest eK) bi kvh)
                    (slice_page4 (to_real_chest eV) bi kvh)
                    (slice_page4 (FSpec.attn_bias has_mask
                                    (to_real_chest emask)) bi qh)
                    (to_real scale) causal)
                 qpos dd))
  = let hqv : pos = SZ.v hq in
    let sqv : pos = SZ.v sq in
    let skv : pos = SZ.v sk in
    let rowsv : pos = SZ.v rows in
    let nwv : pos = SZ.v nw in
    let dv : pos = SZ.v d in
    let g : nat = SZ.v group in
    let r0 : nat = r / 16 * 16 in
    let i : natlt 16 = r % 16 in
    FStar.Math.Lemmas.euclidean_division_definition r 16;
    assert (r0 + i == r);
    assert (SF.lane_active_row rowsv r0 i);
    let eKg = page_kv eK bi kvh in
    let eVg = page_kv eV bi kvh in
    let qh = SF.lane_qh hqv sqv kvh g rowsv r0 i in
    let qpos = SF.lane_qpos sqv rowsv r0 i in
    let cbound = SF.lane_cbound sqv skv rowsv r0 i in
    let nkt = SF.key_tiles 16 16 sqv skv rowsv r0 causal in
    let escale = SV.flash_escale_at nw d b hq sq rows sk eQ eKg emask has_mask
                   causal scale bi kvh g r0 in
    let eO = SV.flash_eO_at nw d b hq sq rows sk eQ eKg eVg emask has_mask
               causal scale bi kvh g r0 in
    let egl = SV.flash_egl_at nw d b hq sq rows sk eQ eKg emask has_mask
                causal scale bi kvh g r0 in
    introduce forall (w : natlt nwv).
      acc2 escale w i
      == SF.gscale (SV.flash_eM_at nw d b hq sq rows sk #() eQ eKg emask has_mask
                      causal scale bi kvh g r0)
           (SF.gmax (SV.flash_eM_at nw d b hq sq rows sk #() eQ eKg emask has_mask
                       causal scale bi kvh g r0) i nwv) w i
    with escale_at_cell nw d b hq sq rows sk #() eQ eKg emask has_mask causal
           scale bi kvh g r0 i w;
    introduce forall (w : natlt nwv).
      SS.ocomb_val eO i dd w
      == acc2 (SS.block_O #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
                 #(SZ.v b) #hqv #sqv #skv #dv #()
                 emask has_mask causal bi kvh g rowsv r0 scale
                 (SF.q_tile 16 rowsv g eQ bi kvh r0) eKg eVg nwv nkt w) i dd
    with eO_at_cell nw d b hq sq rows sk #() eQ eKg eVg emask has_mask causal
           scale bi kvh g r0 i dd w;
    SV.flash_eM_at_def nw d b hq sq rows sk #() eQ eKg emask has_mask
      causal scale bi kvh g r0;
    SV.flash_eL_at_def nw d b hq sq rows sk #() eQ eKg emask has_mask
      causal scale bi kvh g r0;
    SV.flash_escale_egl_at_def nw d b hq sq rows sk eQ eKg emask has_mask
      causal scale bi kvh g r0 i;
    assert (SF.lane_params_ok hqv sqv skv kvh g rowsv r0 i true qh qpos cbound);
    assert (SF.key_ok true causal skv cbound 0);
    introduce forall (w : natlt nwv).
      SS.cw_upto #et_ab #et_acc #_f #_r #_s #_rb #_c1
        #(SZ.v b) #hqv #sqv #skv #dv #()
        emask has_mask true causal bi qh qpos cbound scale
        (SF.q_tile 16 rowsv g eQ bi kvh r0) eKg i nwv w
        (SF.warp_iters nwv nkt w)
    with SS.cw_upto_holds #et_ab #et_acc #_f #_r #_s #_rb #_c1
           #(SZ.v b) #hqv #sqv #skv #dv #()
           emask has_mask true causal bi qh qpos cbound scale
           (SF.q_tile 16 rowsv g eQ bi kvh r0) eKg i nwv w
           (SF.warp_iters nwv nkt w);
    SD.flash_denom_pos nw d b hq sq rows sk #() eQ eKg emask has_mask causal
      scale bi kvh g r0 i;
    SS.page_out_cell #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
      #(SZ.v b) #hqv #sqv #skv #dv #()
      eQ emask has_mask causal bi qh qpos kvh g r0 cbound rowsv scale
      eKg eVg i dd nwv nkt escale eO egl 0;
    page_kv_real eK bi kvh;
    page_kv_real eV bi kvh

#pop-options

(* ------------------------------------------------------------------ *)
(* Chest-level assembly.                                               *)
(* ------------------------------------------------------------------ *)

#push-options "--z3rlimit 30 --fuel 1 --ifuel 2"

(* The input-level precondition gives [row_no_overflow] for every row of
   every block the kernel launches. *)
let pre_row
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq hkv group sq rows sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq })
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eK : chest (SZ.v b @| SZ.v hkv @| SZ.v sk @| SZ.v d @| INil) et_ab)
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (kvh : natlt (SZ.v hkv)) (r : natlt (SZ.v rows))
  : Lemma
      (requires
        FSpec.sdpa_flash_finite #et_ab #et_acc (SZ.v group) #(SZ.v sq)
          #(SZ.v sk) #(SZ.v d) #() (SZ.v rows) eQ eK emask has_mask causal
          scale)
      (ensures
        row_no_overflow nw d b hq sq rows sk #() eQ (page_kv eK bi kvh) emask
          has_mask causal scale bi kvh (SZ.v group) (r / 16 * 16) (r % 16))
  = let hqv : pos = SZ.v hq in
    let sqv : pos = SZ.v sq in
    let skv : pos = SZ.v sk in
    let rowsv : pos = SZ.v rows in
    let dv : pos = SZ.v d in
    let g : pos = SZ.v group in
    let r0 : nat = r / 16 * 16 in
    let i : natlt 16 = r % 16 in
    FStar.Math.Lemmas.euclidean_division_definition r 16;
    assert (SF.lane_rr rowsv r0 i == r);
    let qh = SF.lane_qh hqv sqv kvh g rowsv r0 i in
    let qpos = SF.lane_qpos sqv rowsv r0 i in
    let cbound = SF.lane_cbound sqv skv rowsv r0 i in
    let eQt = SF.q_tile 16 rowsv g eQ bi kvh r0 in
    let eKg = page_kv eK bi kvh in
    assert (qh == FSpec.query_head #hqv #(SZ.v hkv) #sqv g rowsv kvh r);
    assert (qpos == r % sqv /\ cbound == qpos + (skv - sqv));
    introduce forall (k0 : nat) (t : natlt 16).
      k0 <= skv /\ SF.key_ok true causal skv cbound (k0 + t) ==>
      Finite? (kind (acc1 (SF.tile_scores #et_acc #et_ab #_f #_r #_s #_rb #_c1
                             emask has_mask true causal bi qh qpos k0 cbound
                             scale (SS.tile_score_row #et_ab #et_acc #_s #_
                                      #skv #dv #() eQt eKg k0 i)) t))
    with introduce _ ==> _
    with _. assert (acc1 (SF.tile_scores #et_acc #et_ab #_f #_r #_s #_rb #_c1
                            emask has_mask true causal bi qh qpos k0 cbound
                            scale (SS.tile_score_row #et_ab #et_acc #_s #_
                                     #skv #dv #() eQt eKg k0 i)) t
                    == FSpec.flash_score #et_ab #et_acc #_f #_r #_s #_rb #_c1
                         #(SZ.v b) #hqv #sqv #skv #dv #() emask has_mask bi
                         qh qpos scale eQt eKg i k0 t)

#pop-options

#push-options "--z3rlimit 60 --fuel 1 --ifuel 1 --split_queries always"

(* The kernel's output chest and the spec agree at one index. *)
let out_idx_cell
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _fr : floating_real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq hkv group sq rows sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq })
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eK eV : chest (SZ.v b @| SZ.v hkv @| SZ.v sk @| SZ.v d @| INil) et_ab)
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  (bi : natlt (SZ.v b)) (qh : natlt (SZ.v hq))
  (qpos : natlt (SZ.v sq)) (dd : natlt (SZ.v d))
  : Lemma
      (requires
        FSpec.sdpa_flash_finite #et_ab #et_acc (SZ.v group) #(SZ.v sq)
          #(SZ.v sk) #(SZ.v d) #() (SZ.v rows) eQ eK emask has_mask causal
          scale)
      (ensures
        acc (FSp.flash_out_chest b hq hkv group sq rows d
               (flash_out_vfun #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
                  nw d b hq hkv group sq rows sk #() eQ eK eV emask
                  has_mask causal scale))
            (bi, (qh, (qpos, (dd, ()))))
        %~ acc (FSpec.sdpa_flash_real #(SZ.v b) #(SZ.v hq) #(SZ.v hkv)
                  (SZ.v group) #(SZ.v sq) #(SZ.v sk) #(SZ.v d)
                  (to_real_chest eQ) (to_real_chest eK) (to_real_chest eV)
                  (to_real_chest emask) (to_real scale) causal has_mask)
               (bi, (qh, (qpos, (dd, ())))))
  = let hqv : pos = SZ.v hq in
    let hkvv : pos = SZ.v hkv in
    let sqv : pos = SZ.v sq in
    let rowsv : pos = SZ.v rows in
    let g : pos = SZ.v group in
    let bij = flash_output_logical_bij (SZ.v b) hkvv g sqv hqv rowsv
                (SZ.v d) in
    let idx : Kuiper.Shape.abs (SZ.v b @| hqv @| sqv @| SZ.v d @| INil) =
      (bi, (qh, (qpos, (dd, ())))) in
    let y = bij.ff idx in
    FSp.flash_out_chest_acc b hq hkv group sq rows d
      (flash_out_vfun #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
         nw d b hq hkv group sq rows sk #() eQ eK eV emask
         has_mask causal scale);
    assert (y._1 == bi /\ y._4 == dd);
    assert (y._2 == qh / g);
    assert (y._3 == (qh % g) * sqv + qpos);
    FStar.Math.Lemmas.lemma_div_mod qh g;
    FStar.Math.Lemmas.small_div qpos sqv;
    FStar.Math.Lemmas.small_mod qpos sqv;
    FStar.Math.Lemmas.lemma_div_plus qpos (qh % g) sqv;
    FStar.Math.Lemmas.lemma_mod_plus qpos (qh % g) sqv;
    assert (y._3 / sqv == qh % g);
    assert (y._3 % sqv == qpos);
    assert (y._2 * g + y._3 / sqv == qh);
    assert (out_qh hqv sqv y._2 g y._3 == qh);
    assert (out_qpos sqv y._3 == qpos);
    assert (FSpec.kv_head #hkvv g qh == y._2);
    pre_row #et_ab #et_acc #_f #_r #_s #_rb #_c1
      nw d b hq hkv group sq rows sk #() eQ eK emask
      has_mask causal scale bi y._2 y._3;
    out_vfun_cell #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
      nw d b hq hkv group sq rows sk #() eQ eK eV emask
      has_mask causal scale bi y._2 y._3 dd

#pop-options

#push-options "--z3rlimit 30 --fuel 1 --ifuel 1"

(* The chest the kernel leaves in the output tensor approximates the
   real-valued flash-attention specification. *)
let flash_out_approx
  (#et_ab #et_acc : Type0)
  {| _f : floating et_acc |} {| _r : real_like et_acc |}
  {| _fr : floating_real_like et_acc |}
  {| _s : scalar et_ab |} {| _rb : real_like et_ab |}
  {| _c1 : FC.float_cast et_ab et_acc |} {| _c2 : FC.float_cast et_acc et_ab |}
  (nw : szp) (d : szp { 16 /?+ SZ.v d })
  (b hq hkv group sq rows sk : szp {
    SZ.v hq == SZ.v hkv * SZ.v group /\
    SZ.v rows == SZ.v group * SZ.v sq })
  (#_ : squash (SZ.v sq <= SZ.v sk))
  (eQ : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil) et_ab)
  (eK eV : chest (SZ.v b @| SZ.v hkv @| SZ.v sk @| SZ.v d @| INil) et_ab)
  (emask : chest (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v sk @| INil) et_ab)
  (has_mask causal : bool) (scale : et_acc)
  : Lemma
      (requires
        FSpec.sdpa_flash_finite #et_ab #et_acc (SZ.v group) #(SZ.v sq)
          #(SZ.v sk) #(SZ.v d) #() (SZ.v rows) eQ eK emask has_mask causal
          scale)
      (ensures
        FSp.flash_out_chest b hq hkv group sq rows d
          (flash_out_vfun #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
             nw d b hq hkv group sq rows sk #() eQ eK eV emask
             has_mask causal scale)
        %~ FSpec.sdpa_flash_real #(SZ.v b) #(SZ.v hq) #(SZ.v hkv)
             (SZ.v group) #(SZ.v sq) #(SZ.v sk) #(SZ.v d)
             (to_real_chest eQ) (to_real_chest eK) (to_real_chest eV)
             (to_real_chest emask) (to_real scale) causal has_mask)
  = introduce forall
      (idx : Kuiper.Shape.abs
               (SZ.v b @| SZ.v hq @| SZ.v sq @| SZ.v d @| INil)).
      acc (FSp.flash_out_chest b hq hkv group sq rows d
             (flash_out_vfun #et_ab #et_acc #_f #_r #_s #_rb #_c1 #_c2
                nw d b hq hkv group sq rows sk #() eQ eK eV emask
                has_mask causal scale)) idx
      %~ acc (FSpec.sdpa_flash_real #(SZ.v b) #(SZ.v hq) #(SZ.v hkv)
                (SZ.v group) #(SZ.v sq) #(SZ.v sk) #(SZ.v d)
                (to_real_chest eQ) (to_real_chest eK) (to_real_chest eV)
                (to_real_chest emask) (to_real scale) causal has_mask) idx
    with (let (bi, (qh, (qpos, (dd, ())))) = idx in
          out_idx_cell #et_ab #et_acc #_f #_r #_fr #_s #_rb #_c1 #_c2
            nw d b hq hkv group sq rows sk #() eQ eK eV emask
            has_mask causal scale bi qh qpos dd)

#pop-options
