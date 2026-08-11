"""Focused, interleaved re-measure of the uniform-splits cost.

``sweep_splitk_params`` timed each point once in a long serial run, which drifts
badly on a shared GPU. This times only the points that decide the answer -- the
best port-legal (bk, splits) against the best point overall -- interleaved
round-robin over several rounds, reporting the minimum per point so a burst of
contention inflates a round rather than a contender.

Result (A6000, bf16, bm128 bn128 wm32 wn64 skew8):

    down_proj_sp4       256x4864x896   best bk=64 sp=6  39.6us   port 48.99  23.6%
    down_proj_m128_sp8  128x4864x896   best bk=64 sp=7  31.8us   port 36.12  13.8%
    qkv_proj_sp2        256x896x1152   best bk=64 sp=3  18.5us   port 19.03   3.1%
    mlp_up_sp2          256x896x4864   best bk=32 sp=2  59.4us   port 59.36   0.0%

The cost is structural, not a gap in the candidate list: K = 4864 = 2^8 * 19, so
at bk=64 the tile count is 76 and the legal split counts are its divisors,
{1, 2, 4, 19, 38, 76} -- a gap from 4 straight to 19 that straddles the optimum
at 5-7. Widening the candidate list cannot help; the lattice itself is sparse.
This is exactly the case the reference's proportional division exists for.

TODO(upstream): closing it needs a general (offset, length) column-band view --
``band_layout`` plus its ``ctlayout`` and ``strided_row_major`` instances and a
``factored`` read-only extraction. Kuiper has no such primitive: every slicing
form in Kuiper.Tensor.Tiling and Kuiper.Kernel.FlashAttention.KernelDesc
requires uniform divisibility.

Run with:  micromamba run -n kuillm env PYTHONPATH=. python etc/sweep_splitk_focus.py
"""
import torch

from kuipy.benchmarking import warm_up, rel_fro, time_call
from etc.ref_flat_gemm_splitk import load as load_ref_splitk

# (case, M, K, N, [(bk, splits), ...]) -- the top of each case's sweep table
# plus every port-legal point that could plausibly win.
CASES = [
    ("down_proj_sp4", 256, 4864, 896,
     [(64, 5), (64, 6), (64, 4), (64, 19), (32, 4), (32, 6), (32, 8), (32, 19)]),
    ("down_proj_m128_sp8", 128, 4864, 896,
     [(64, 6), (64, 7), (64, 8), (64, 4), (64, 19), (32, 8), (32, 12), (32, 19)]),
    ("qkv_proj_sp2", 256, 896, 1152,
     [(64, 2), (64, 3), (64, 7), (32, 3), (32, 4), (32, 7), (32, 2)]),
    ("mlp_up_sp2", 256, 896, 4864,
     [(32, 2), (32, 3), (32, 4), (64, 2), (64, 3), (16, 2)]),
]

ROUNDS = 5
BM = BN = 128
WM, WN, SKEW = 32, 64, 8


def port_legal(K, bk, splits):
    return K % bk == 0 and (K // bk) % splits == 0


def main():
    torch.manual_seed(0)
    warm_up()
    results = {}
    for name, M, K, N, pts in CASES:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        bt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t()
        ref = a.float() @ bt.float()
        fns = {}
        for bk, sp in pts:
            fn = load_ref_splitk(splits=sp, bm=BM, bn=BN, bk=bk, wm=WM, wn=WN,
                                 skew=SKEW, group=1)
            assert rel_fro(fn(a, bt).float(), ref) < 5e-3
            fns[(bk, sp)] = fn
            results[(name, bk, sp)] = []
        for _ in range(ROUNDS):
            for bk, sp in pts:
                fn = fns[(bk, sp)]
                ms, _ = time_call(lambda: fn(a, bt), iters=200, warmup=50)
                results[(name, bk, sp)].append(ms * 1e3)

    print()
    print(f"{'case':20s} {'bk':>3s} {'sp':>3s} {'port?':>5s} {'min us':>8s} {'med us':>8s}")
    for name, M, K, N, pts in CASES:
        for bk, sp in pts:
            v = sorted(results[(name, bk, sp)])
            ok = "yes" if port_legal(K, bk, sp) else "NO"
            print(f"{name:20s} {bk:3d} {sp:3d} {ok:>5s} {v[0]:8.2f} "
                  f"{v[len(v)//2]:8.2f}")

    print()
    print(f"{'case':20s} {'best-any':>24s} {'best-port':>24s} {'cost':>7s}")
    for name, M, K, N, pts in CASES:
        best = {p: min(results[(name,) + p]) for p in pts}
        ba = min(best, key=best.get)
        legal = [p for p in pts if port_legal(K, p[0], p[1])]
        bp = min(legal, key=best.get)
        cost = (best[bp] / best[ba] - 1.0) * 100.0
        print(f"{name:20s} {f'bk={ba[0]} sp={ba[1]} {best[ba]:.2f}us':>24s} "
              f"{f'bk={bp[0]} sp={bp[1]} {best[bp]:.2f}us':>24s} {cost:6.1f}%")


if __name__ == "__main__":
    main()
