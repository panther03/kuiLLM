module Kuiops.SuperGEMM.Mm.KernelLemmas

(* Pure chest/subtile algebra supporting the functional [teardown] of
   [Kuiops.SuperGEMM.Mm.Kernel].

   Split out of [Kernel] so the SMT-heavy pure lemmas iterate independently of
   the Pulse kernel body: this module does not depend on [Epilogue], [KLoop],
   [Barrier], [Stage] or [Output].

   TODO(upstream): [map_subtile_commute], [subtile_subtile_compose],
   [coerce_subtile_col] and [coerce_wm_nested] are generic chest-subtile
   algebra and belong in a [Kuiper.EMatrix.Tiling]-adjacent module. *)

open Kuiper
open Kuiper.Chest
open Kuiper.TensorCore
open Kuiops.SuperGEMM.Mm.Params
open Kuiper.EMatrix.Tiling { ematrix_subtile }
open Kuiops.SuperGEMM.Mm.Spec { warp_matmul, warp_matmul_is_subtile }

module SZ = Kuiper.SizeT
module P = Kuiops.SuperGEMM.Mm.Params
module SH = Kuiops.SuperGEMM.Mm.Shared
module MS = Kuiper.Spec.GEMM
module ML = FStar.Math.Lemmas

#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

(* ---------------------------------------------------------------------- *)
(* Bridge: [SH.lane_target] equals the doubly-nested subtile of the whole   *)
(* real target that [gather_output_approximates'] requires per (bid,tid).   *)
(* Ported here for the functional [teardown]; the pure chest algebra        *)
(* ([map_subtile_commute], [subtile_subtile_compose], [coerce_subtile_col], *)
(* [coerce_wm_nested]) is reusable and belongs upstream.                    *)
(* TODO(upstream): move the four pure chest-subtile lemmas into a shared     *)
(* [Kuiper.EMatrix.Tiling]-adjacent module.                                 *)
(* ---------------------------------------------------------------------- *)

#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"
let map_subtile_commute
  (#et1 #et2 : Type) (#rows #cols : nat)
  (f : et1 -> et2) (em : chest2 et1 rows cols)
  (tr : pos{tr /? rows}) (tc : pos{tc /? cols})
  (r : natlt (rows / tr)) (c : natlt (cols / tc))
  : Lemma (ematrix_subtile (chest_map f em) tr tc r c
           == chest_map f (ematrix_subtile em tr tc r c))
= assert (Kuiper.Chest.equal (ematrix_subtile (chest_map f em) tr tc r c)
                (chest_map f (ematrix_subtile em tr tc r c)))
#pop-options

(* Associating an inner tile offset with its enclosing tile is a small ring
   fact, but proving it separately avoids feeding the whole chest context to
   the nonlinear solver. *)
let nested_tile_offset
  (outer : nat) (tile : pos) (block inner i : nat)
  : Lemma (requires outer == tile * (outer / tile))
          (ensures block * outer + (inner * tile + i)
                   == (block * (outer / tile) + inner) * tile + i)
= ()

let tile_cell_index
  (extent : nat) (tile : pos{tile /? extent})
  (block : natlt (extent / tile)) (i : natlt tile)
  : natlt extent
= ML.lemma_div_exact extent tile;
  block * tile + i

(* Expose one cell of a subtile without asking SMT to unfold nested chest
   constructors in the caller. *)
let subtile_cell
  (#et : Type) (#rows #cols : nat)
  (em : chest2 et rows cols)
  (tr : pos{tr /? rows}) (tc : pos{tc /? cols})
  (r : natlt (rows / tr)) (c : natlt (cols / tc))
  (i : natlt tr) (j : natlt tc)
  (oi : natlt rows) (oj : natlt cols)
  (_ : squash (oi == r * tr + i /\ oj == c * tc + j))
  : Lemma
      (acc2 (ematrix_subtile em tr tc r c) i j
       == acc2 em oi oj)
= ()

#push-options "--fuel 1 --ifuel 1 --z3rlimit 3"
let subtile_subtile_compose
  (#et : Type) (#rows #cols : nat)
  (em : chest2 et rows cols)
  (br0 bc0 : pos) (tr tc : pos)
  (br : natlt (rows / br0)) (bc : natlt (cols / bc0))
  (r : natlt (br0 / tr)) (c : natlt (bc0 / tc))
  (_ : squash (br0 /? rows /\ bc0 /? cols /\ tr /? br0 /\ tc /? bc0))
  (rr : natlt (rows / tr)) (cc : natlt (cols / tc))
  (_ : squash (rr == br * (br0 / tr) + r /\
                cc == bc * (bc0 / tc) + c))
  (_ : squash (tr /? rows /\ tc /? cols))
  : Lemma
    (ensures
      forall (ij : Kuiper.Shape.abs
        (Kuiper.Shape.ICons tr
          (Kuiper.Shape.ICons tc Kuiper.Shape.INil))).
        acc (ematrix_subtile (ematrix_subtile em br0 bc0 br bc) tr tc r c) ij
        == acc (ematrix_subtile em tr tc rr cc) ij)
= ML.lemma_div_exact br0 tr;
  ML.lemma_div_exact bc0 tc;
  introduce forall
    (ij : Kuiper.Shape.abs
      (Kuiper.Shape.ICons tr
        (Kuiper.Shape.ICons tc Kuiper.Shape.INil))).
    acc (ematrix_subtile (ematrix_subtile em br0 bc0 br bc) tr tc r c) ij
    == acc (ematrix_subtile em tr tc rr cc) ij
  with (
    let (i, (j, ())) = ij in
    let ri = tile_cell_index br0 tr r i in
    let cj = tile_cell_index bc0 tc c j in
    let outer_i = tile_cell_index rows br0 br ri in
    let outer_j = tile_cell_index cols bc0 bc cj in
    let fine_i = tile_cell_index rows tr rr i in
    let fine_j = tile_cell_index cols tc cc j in
    calc (==) {
      acc (ematrix_subtile (ematrix_subtile em br0 bc0 br bc) tr tc r c)
        (i, (j, ()));
      == { subtile_cell (ematrix_subtile em br0 bc0 br bc)
             tr tc r c i j ri cj () }
      acc2 (ematrix_subtile em br0 bc0 br bc)
        ri cj;
      == { subtile_cell em br0 bc0 br bc
             ri cj outer_i outer_j () }
      acc2 em outer_i outer_j;
      == { nested_tile_offset br0 tr br r i;
           nested_tile_offset bc0 tc bc c j }
      acc2 em fine_i fine_j;
      == { subtile_cell em tr tc rr cc i j fine_i fine_j () }
      acc (ematrix_subtile em tr tc rr cc)
        (i, (j, ())); }
  )
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let coerce_subtile_col
  (#et : Type) (#rows #cols : nat)
  (em : chest2 et rows cols)
  (tr : pos{tr /? rows})
  (c1 c2 : pos{c1 /? cols /\ c2 /? cols})
  (r : natlt (rows / tr))
  (cc1 : natlt (cols / c1)) (cc2 : natlt (cols / c2))
  (_ : squash (c1 == c2 /\ (cc1 <: nat) == (cc2 <: nat)))
  : Lemma (coerce_eq #(chest2 et tr c1) #(chest2 et tr c2) () (ematrix_subtile em tr c1 r cc1)
           == ematrix_subtile em tr c2 r cc2)
= assert (Kuiper.Chest.equal (coerce_eq #(chest2 et tr c1) #(chest2 et tr c2) () (ematrix_subtile em tr c1 r cc1))
                (ematrix_subtile em tr c2 r cc2))
#pop-options

(* pure subtile algebra: coerce(map(warp_matmul)) == doubly-nested subtile,
   given the decoded indices and bounds as explicit hypotheses. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let coerce_wm_nested
  (#mm #nn : nat) (#kk : pos)
  (rA : chest2 real mm kk) (rB : chest2 real nn kk)
  (post_map_r : real -> real)
  (bm bn wm_c wn_c wn_1 : pos)
  (block_row block_col : nat)
  (grow : natlt (mm / wm_c)) (gcol : natlt (nn / wn_1))
  (idx_m : natlt (bm / wm_c)) (idx_n : natlt (bn / wn_1))
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
        (chest_map post_map_r (warp_matmul rA rB wm_c wn_c grow gcol))
      == ematrix_subtile
           (ematrix_subtile (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))
             bm bn block_row block_col)
           wm_c wn_1 idx_m idx_n)
= let rD_whole = chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)) in
  calc (==) {
    coerce_eq #(chest2 real wm_c wn_c) #(chest2 real wm_c wn_1) ()
      (chest_map post_map_r (warp_matmul rA rB wm_c wn_c grow gcol));
    == { warp_matmul_is_subtile rA rB wm_c wn_c grow gcol;
         map_subtile_commute post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)) wm_c wn_c grow gcol }
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

let mfrag_frag_eq (w : szp)
  : Lemma (requires frag /?+ SZ.v w)
          (ensures mfrag w * frag == SZ.v w)
          [SMTPat (mfrag w * frag)]
= ML.lemma_div_exact (SZ.v w) frag

#push-options "--fuel 2 --ifuel 1 --z3rlimit 30"
let lane_target_is_subtile
  (#m #n #k : szp)
  (rA : chest2 real (SZ.v m) (SZ.v k))
  (rB : chest2 real (SZ.v n) (SZ.v k))
  (post_map_r : real -> real)
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
      SH.lane_target rA rB post_map_r bm bn wm wn #sq nblk nthr bid tid
      == ematrix_subtile
           (ematrix_subtile
             (chest_map post_map_r (MS.matmul rA (Kuiper.EMatrix.mtranspose rB)))
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
  Kuiper.Divides.lemma_divides_trans (SZ.v wm) (SZ.v bm) (SZ.v m);
  Kuiper.Divides.lemma_divides_trans (SZ.v wn) (SZ.v bn) (SZ.v n);
  let grow : natlt (SZ.v m / SZ.v wm) = block_row * (SZ.v bm / SZ.v wm) + warp_m in
  let gcol : natlt (SZ.v n / SZ.v wn) = block_col * (SZ.v bn / SZ.v wn) + warp_n in
  let wm_c : (x:pos{x /?+ SZ.v m /\ x /?+ SZ.v bm}) = mfrag wm * frag in
  let wn_c : (x:pos{x /?+ SZ.v n}) = nfrag wn * frag in
  let wn_1 : (x:pos{x /?+ SZ.v n /\ x /?+ SZ.v bn}) = 1 * SZ.v wn in
  assert (1 * SZ.v wn == SZ.v wn);
  assert (SZ.v bm / wm_c == SZ.v bm / SZ.v wm);
  assert (SZ.v bn / wn_1 == SZ.v bn / SZ.v wn);
  assert (SZ.v m / wm_c == SZ.v m / SZ.v wm);
  assert (SZ.v n / wn_1 == SZ.v n / SZ.v wn);
  assert (SZ.v n / wn_c == SZ.v n / SZ.v wn);
  assert (SZ.v n / wn_c == SZ.v n / wn_1);
  assert (wm_c /?+ SZ.v m);
  assert (wn_c /?+ SZ.v n);
  assert (SZ.v bn / (1 * SZ.v wn) == wnn);
  let grow_c : natlt (SZ.v m / wm_c) = grow in
  let gcol_1 : natlt (SZ.v n / wn_c) = gcol in
  let idx_m : natlt (SZ.v bm / wm_c) = wid / (SZ.v bn / (1 * SZ.v wn)) in
  let idx_n : natlt (SZ.v bn / wn_1) = wid % (SZ.v bn / (1 * SZ.v wn)) in
  assert (grow_c == block_row * (SZ.v bm / wm_c) + idx_m);
  assert (gcol_1 == block_col * (SZ.v bn / wn_1) + idx_n);
  coerce_wm_nested #(SZ.v m) #(SZ.v n) #(SZ.v k) rA rB post_map_r
    (SZ.v bm) (SZ.v bn) wm_c wn_c wn_1 block_row block_col grow_c gcol_1 idx_m idx_n ();
  assert (SH.lane_target rA rB post_map_r bm bn wm wn #sq nblk nthr bid tid
          == coerce_eq #(chest2 real wm_c wn_c) #(chest2 real wm_c wn_1) ()
               (chest_map post_map_r (warp_matmul rA rB wm_c wn_c grow_c gcol_1)))
#pop-options

(* All block/warp decode bounds needed to build the doubly-nested-subtile
   target [gather_output_approximates'] requires, in one pure lemma so the
   [teardown] map body stays free of the nonlinear div/mod discharges. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 30"
let td_bounds (m n bm bn wm wn nblk nthr bid tid : nat)
  : Lemma
    (requires
       m > 0 /\ n > 0 /\ bm > 0 /\ bn > 0 /\ wm > 0 /\ wn > 0 /\
       m % bm == 0 /\ n % bn == 0 /\ bm % wm == 0 /\ bn % wn == 0 /\
       wm % frag == 0 /\
       nblk == (m / bm) * (n / bn) /\
       nthr == (bm / wm) * (bn / wn) * warp_size /\
       bid < nblk /\ tid < nthr)
    (ensures
       n / bn > 0 /\ m / bm > 0 /\ bn / (1 * wn) > 0 /\
       bid / (n / bn) < m / bm /\ bid % (n / bn) < n / bn /\
       (tid / warp_size) / (bn / (1 * wn)) < bm / (wm / frag * frag) /\
       (tid / warp_size) % (bn / (1 * wn)) < bn / (1 * wn))
= ML.lemma_div_exact n bn;
  ML.lemma_div_exact m bm;
  ML.lemma_div_exact bn wn;
  ML.lemma_div_exact wm frag;
  assert (1 * wn == wn);
  assert (wm / frag * frag == wm);
  SH.div_ub bid (m / bm) (n / bn);
  ML.lemma_mod_lt bid (n / bn);
  let wid = tid / warp_size in
  SH.div_ub tid ((bm / wm) * (bn / wn)) warp_size;
  SH.div_ub wid (bm / wm) (bn / wn);
  ML.lemma_mod_lt wid (bn / wn)
#pop-options
