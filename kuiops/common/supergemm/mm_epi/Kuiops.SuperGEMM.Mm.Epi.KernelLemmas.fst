module Kuiops.SuperGEMM.Mm.Epi.KernelLemmas

(* Pure chest/subtile algebra supporting the functional [teardown] of
   [Kuiops.SuperGEMM.Mm.Epi.Kernel]: the C-combining counterpart of
   [Kuiops.SuperGEMM.Mm.KernelLemmas], whose [map_subtile_commute],
   [subtile_subtile_compose], [coerce_subtile_col] and [td_bounds] are reused
   as-is.

   TODO(upstream): [comb_subtile_commute] and [cband_is_subtile] are generic
   chest-subtile algebra and belong next to [map_subtile_commute] in a
   [Kuiper.EMatrix.Tiling]-adjacent module. *)

open Kuiper
open Kuiper.Chest
open Kuiper.TensorCore
open Kuiops.SuperGEMM.Mm.Params
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul, warp_matmul_is_subtile }
open Kuiops.SuperGEMM.Mm.Epi.EpilogueLemmas { cband, window_bound }
open Kuiops.SuperGEMM.Mm.KernelLemmas { subtile_subtile_compose, coerce_subtile_col }

module SZ = Kuiper.SizeT
module P = Kuiops.SuperGEMM.Mm.Params
module SH = Kuiops.SuperGEMM.Mm.Shared
module ES = Kuiops.SuperGEMM.Mm.Epi.Shared
module EL = Kuiops.SuperGEMM.Mm.Epi.EpilogueLemmas
module MS = Kuiper.Spec.GEMM
module ML = FStar.Math.Lemmas

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

(* Binary counterpart of [KernelLemmas.map_subtile_commute]. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"
let comb_subtile_commute
  (#et : Type) (#rows #cols : nat)
  (f : et -> et -> et) (em1 em2 : chest2 et rows cols)
  (tr : pos{tr /? rows}) (tc : pos{tc /? cols})
  (r : natlt (rows / tr)) (c : natlt (cols / tc))
  : Lemma (ematrix_subtile (chest_comb f em1 em2) tr tc r c
           == chest_comb f (ematrix_subtile em1 tr tc r c) (ematrix_subtile em2 tr tc r c))
= assert (Kuiper.Chest.equal (ematrix_subtile (chest_comb f em1 em2) tr tc r c)
            (chest_comb f (ematrix_subtile em1 tr tc r c) (ematrix_subtile em2 tr tc r c)))
#pop-options

(* A C window whose origin is tile-aligned IS the corresponding subtile. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"
let cband_is_subtile
  (#et : Type) (#rows #cols : nat)
  (em : chest2 et rows cols)
  (tr : pos{tr /? rows}) (tc : pos{tc /? cols})
  (r : natlt (rows / tr)) (c : natlt (cols / tc))
  (sq : squash (r * tr + tr <= rows /\ c * tc + tc <= cols))
  : Lemma (cband em tr tc (r * tr) (c * tc) sq == ematrix_subtile em tr tc r c)
= assert (Kuiper.Chest.equal (cband em tr tc (r * tr) (c * tc) sq)
            (ematrix_subtile em tr tc r c))
#pop-options

(* The warp's C window origin, expressed through the global warp index. *)
let origin_eq (bm wm block_row warp_m : nat)
  : Lemma (requires wm > 0 /\ bm % wm == 0)
          (ensures (block_row * (bm / wm) + warp_m) * wm
                   == block_row * bm + warp_m * wm)
= ML.lemma_div_exact bm wm;
  ML.distributivity_add_left (block_row * (bm / wm)) warp_m wm;
  ML.paren_mul_right block_row (bm / wm) wm

(* C-combining counterpart of [KernelLemmas.coerce_wm_nested]. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let coerce_wm_nested_c
  (#mm #nn : nat) (#kk : pos)
  (rC : chest2 real mm nn)
  (rA : chest2 real mm kk) (rB : chest2 real nn kk)
  (comb_r : real -> real -> real)
  (bm bn wm_c wn_c wn_1 : pos)
  (block_row block_col : nat)
  (grow : natlt (mm / wm_c)) (gcol : natlt (nn / wn_1))
  (idx_m : natlt (bm / wm_c)) (idx_n : natlt (bn / wn_1))
  (sqc : squash (grow * wm_c + wm_c <= mm /\ gcol * wn_c + wn_c <= nn))
  (_ : squash (
    bm /? mm /\ bn /? nn /\ wm_c /? bm /\ wn_1 /? bn /\ wn_c == wn_1 /\
    wm_c /? mm /\ wn_1 /? nn /\
    block_row < mm / bm /\ block_col < nn / bn /\
    grow == block_row * (bm / wm_c) + idx_m /\
    gcol == block_col * (bn / wn_1) + idx_n))
  : Lemma
    (requires
      grow < mm / wm_c /\ gcol < nn / wn_1 /\
      block_row * (bm / wm_c) + idx_m < mm / wm_c /\
      block_col * (bn / wn_1) + idx_n < nn / wn_1)
    (ensures
      coerce_eq #(chest2 real wm_c wn_c) #(chest2 real wm_c wn_1) ()
        (chest_comb comb_r
          (cband rC wm_c wn_c (grow * wm_c) (gcol * wn_c) sqc)
          (warp_matmul rA rB wm_c wn_c grow gcol))
      == ematrix_subtile
           (ematrix_subtile (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
             bm bn block_row block_col)
           wm_c wn_1 idx_m idx_n)
= let rD_whole = MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB) in
  calc (==) {
    coerce_eq #(chest2 real wm_c wn_c) #(chest2 real wm_c wn_1) ()
      (chest_comb comb_r
        (cband rC wm_c wn_c (grow * wm_c) (gcol * wn_c) sqc)
        (warp_matmul rA rB wm_c wn_c grow gcol));
    == { cband_is_subtile rC wm_c wn_c grow gcol sqc;
         warp_matmul_is_subtile rA rB wm_c wn_c grow gcol;
         comb_subtile_commute comb_r rC
           (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)) wm_c wn_c grow gcol }
    coerce_eq #(chest2 real wm_c wn_c) #(chest2 real wm_c wn_1) ()
      (ematrix_subtile rD_whole wm_c wn_c grow gcol);
    == { coerce_subtile_col rD_whole wm_c wn_c wn_1 grow gcol gcol () }
    ematrix_subtile rD_whole wm_c wn_1 grow gcol;
    == { subtile_subtile_compose rD_whole bm bn wm_c wn_1
           block_row block_col idx_m idx_n () grow gcol () ();
         Kuiper.Chest.lemma_equal_intro
           (ematrix_subtile (ematrix_subtile rD_whole bm bn block_row block_col)
             wm_c wn_1 idx_m idx_n)
           (ematrix_subtile rD_whole wm_c wn_1 grow gcol);
         Kuiper.Chest.ext
           (ematrix_subtile (ematrix_subtile rD_whole bm bn block_row block_col)
             wm_c wn_1 idx_m idx_n)
           (ematrix_subtile rD_whole wm_c wn_1 grow gcol) }
    ematrix_subtile (ematrix_subtile rD_whole bm bn block_row block_col)
      wm_c wn_1 idx_m idx_n;
  }
#pop-options

(* C-combining counterpart of [KernelLemmas.lane_target_is_subtile]. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 30"
let lane_target_c_is_subtile
  (#m #n #k : szp)
  (rC : chest2 real (SZ.v m) (SZ.v n))
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (comb_r : real -> real -> real)
  (bm bn wm wn : szp)
  (sq : squash (SZ.v bm /?+ SZ.v m /\ SZ.v bn /?+ SZ.v n /\
                SZ.v wm /?+ SZ.v bm /\ SZ.v wn /?+ SZ.v bn /\
                frag /?+ SZ.v wm /\ frag /?+ SZ.v wn))
  (nblk : szp{SZ.v nblk == SZ.v m / SZ.v bm * (SZ.v n / SZ.v bn)})
  (nthr : szp{SZ.v nthr == P.nthr bm bn wm wn})
  (bid : natlt nblk) (tid : natlt nthr)
  (brow : natlt (SZ.v m / SZ.v bm))
  (bcol : natlt (SZ.v n / SZ.v bn))
  (wrow : natlt (SZ.v bm / (mfrag wm * frag)))
  (wcol : natlt (SZ.v bn / (1 * SZ.v wn)))
  : Lemma
    (requires
      SZ.v n / SZ.v bn > 0 /\
      SZ.v bn / (1 * SZ.v wn) > 0 /\
      brow == bid / (SZ.v n / SZ.v bn) /\
      bcol == bid % (SZ.v n / SZ.v bn) /\
      wrow == (tid / warp_size) / (SZ.v bn / (1 * SZ.v wn)) /\
      wcol == (tid / warp_size) % (SZ.v bn / (1 * SZ.v wn)))
    (ensures
      ES.lane_target_c rC rA rB comb_r bm bn wm wn #sq nblk nthr bid tid
      == ematrix_subtile
           (ematrix_subtile
             (MS.mmcomb comb_r rC rA (Kuiper.EMatrix.mtranspose rB))
             (SZ.v bm) (SZ.v bn) brow bcol)
           (mfrag wm * frag) (1 * SZ.v wn)
           wrow wcol)
= let num_n = SZ.v n / SZ.v bn in
  let block_row = bid / num_n in
  let block_col = bid % num_n in
  let wnn = SZ.v bn / SZ.v wn in
  let wid = tid / warp_size in
  let warp_m = wid / wnn in
  let warp_n = wid % wnn in
  ML.lemma_div_exact (SZ.v wm) frag;
  ML.lemma_div_exact (SZ.v wn) frag;
  assert (mfrag wm * frag == SZ.v wm);
  assert (nfrag wn * frag == SZ.v wn);
  SH.div_ub bid (SZ.v m / SZ.v bm) num_n;
  SH.div_ub wid (SZ.v bm / SZ.v wm) wnn;
  SH.grow_bound (SZ.v m) (SZ.v bm) (SZ.v wm) block_row warp_m;
  SH.grow_bound (SZ.v n) (SZ.v bn) (SZ.v wn) block_col warp_n;
  window_bound (SZ.v m) (SZ.v bm) (SZ.v wm) block_row warp_m;
  window_bound (SZ.v n) (SZ.v bn) (SZ.v wn) block_col warp_n;
  Kuiper.Divides.lemma_divides_trans (SZ.v wm) (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_divides_trans (SZ.v wn) (SZ.v bn) (SZ.v n);
  let grow : natlt (SZ.v m / SZ.v wm) = block_row * (SZ.v bm / SZ.v wm) + warp_m in
  let gcol : natlt (SZ.v n / SZ.v wn) = block_col * (SZ.v bn / SZ.v wn) + warp_n in
  origin_eq (SZ.v bm) (SZ.v wm) block_row warp_m;
  origin_eq (SZ.v bn) (SZ.v wn) block_col warp_n;
  let wm_c : (x:pos{x /?+ SZ.v m /\ x /?+ SZ.v bm}) = mfrag wm * frag in
  let wn_c : (x:pos{x /?+ SZ.v n}) = nfrag wn * frag in
  let wn_1 : (x:pos{x /?+ SZ.v n /\ x /?+ SZ.v bn}) = 1 * SZ.v wn in
  assert (1 * SZ.v wn == SZ.v wn);
  assert (SZ.v bm / wm_c == SZ.v bm / SZ.v wm);
  assert (SZ.v bn / wn_1 == SZ.v bn / SZ.v wn);
  assert (SZ.v m / wm_c == SZ.v m / SZ.v wm);
  assert (SZ.v n / wn_1 == SZ.v n / SZ.v wn);
  assert (SZ.v n / wn_c == SZ.v n / SZ.v wn);
  assert (SZ.v bn / (1 * SZ.v wn) == wnn);
  let grow_c : natlt (SZ.v m / wm_c) = grow in
  let gcol_1 : natlt (SZ.v n / wn_c) = gcol in
  let idx_m : natlt (SZ.v bm / wm_c) = wid / (SZ.v bn / (1 * SZ.v wn)) in
  let idx_n : natlt (SZ.v bn / wn_1) = wid % (SZ.v bn / (1 * SZ.v wn)) in
  assert (grow_c == block_row * (SZ.v bm / wm_c) + idx_m);
  assert (gcol_1 == block_col * (SZ.v bn / wn_1) + idx_n);
  assert (grow_c * wm_c + wm_c <= SZ.v m);
  assert (gcol_1 * wn_c + wn_c <= SZ.v n);
  assert (EL.lane_c_target rC bm bn wm wn #sq nblk nthr bid tid
          == cband rC wm_c wn_c (grow_c * wm_c) (gcol_1 * wn_c) ());
  coerce_wm_nested_c #(SZ.v m) #(SZ.v n) #(SZ.v k) rC rA rB comb_r
    (SZ.v bm) (SZ.v bn) wm_c wn_c wn_1 block_row block_col grow_c gcol_1 idx_m idx_n () ();
  assert (ES.lane_target_c rC rA rB comb_r bm bn wm wn #sq nblk nthr bid tid
          == coerce_eq #(chest2 real wm_c wn_c) #(chest2 real wm_c wn_1) ()
               (chest_comb comb_r
                 (cband rC wm_c wn_c (grow_c * wm_c) (gcol_1 * wn_c) ())
                 (warp_matmul rA rB wm_c wn_c grow_c gcol_1)))
#pop-options
