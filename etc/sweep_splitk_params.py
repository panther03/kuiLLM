"""What does the verified kernel's ``splits | ktiles`` restriction cost?

The reference divides ``[0, ktiles)`` proportionally, so it accepts any
``splits <= ktiles``. The verified port divides uniformly, so it accepts only
``splits`` dividing ``ktiles = K/bk``. This sweeps the FULL reference grid on
the Qwen shapes and marks which points the port cannot express, which isolates
the question from the port: the same CUDA runs at every point, only the
parameterization varies.

Run with:  micromamba run -n kuillm env PYTHONPATH=. python etc/sweep_splitk_params.py
"""
import torch

from kuipy.benchmarking import warm_up, rel_fro, time_call
from etc.ref_flat_gemm_splitk import load as load_ref_splitk

CASES = [
    ("down_proj_sp4", 256, 4864, 896),
    ("down_proj_m128_sp8", 128, 4864, 896),
    ("qkv_proj_sp2", 256, 896, 1152),
    ("mlp_up_sp2", 256, 896, 4864),
]

BKS = (16, 32, 64)
SPLITS = (2, 3, 4, 6, 8, 12, 16)
FRAG, VEC, THREADS = 16, 8, 256
BM = BN = 128
WM, WN, SKEW = 32, 64, 8


def legal_for_ref(K, bk, splits):
    if bk % FRAG or THREADS * VEC % bk:
        return False
    return K % bk == 0 and splits <= K // bk


def legal_for_port(K, bk, splits):
    return legal_for_ref(K, bk, splits) and (K // bk) % splits == 0


def main():
    torch.manual_seed(0)
    warm_up()
    rows = []
    for name, M, K, N in CASES:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        bt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).t()
        ref = (a.float() @ bt.float())
        for bk in BKS:
            for splits in SPLITS:
                if not legal_for_ref(K, bk, splits):
                    continue
                try:
                    fn = load_ref_splitk(splits=splits, bm=BM, bn=BN, bk=bk,
                                         wm=WM, wn=WN, skew=SKEW, group=1)
                    d = fn(a, bt)
                except Exception as e:  # nvcc rejected this parameterization
                    print(f"  skip {name} bk={bk} sp={splits}: {e}"[:140])
                    continue
                err = rel_fro(d.float(), ref)
                ms, _ = time_call(lambda: fn(a, bt), iters=300, warmup=100)
                us = ms * 1e3
                rows.append((name, M, K, N, bk, splits,
                             legal_for_port(K, bk, splits), us, err))

    print()
    hdr = f"{'case':20s} {'M':>4s} {'K':>5s} {'N':>5s} {'bk':>3s} {'sp':>3s} " \
          f"{'port?':>5s} {'us':>8s} {'rel-err':>9s}"
    print(hdr)
    for r in rows:
        name, M, K, N, bk, sp, ok, us, err = r
        print(f"{name:20s} {M:4d} {K:5d} {N:5d} {bk:3d} {sp:3d} "
              f"{('yes' if ok else 'NO'):>5s} {us:8.2f} {err:9.2e}")

    print()
    print(f"{'case':20s} {'best-any':>28s} {'best-port':>28s} {'cost':>7s}")
    for name, M, K, N in CASES:
        sub = [r for r in rows if r[0] == name]
        if not sub:
            continue
        ba = min(sub, key=lambda r: r[7])
        bp = min([r for r in sub if r[6]], key=lambda r: r[7])
        cost = (bp[7] / ba[7] - 1.0) * 100.0
        print(f"{name:20s} {f'bk={ba[4]} sp={ba[5]} {ba[7]:.2f}us':>28s} "
              f"{f'bk={bp[4]} sp={bp[5]} {bp[7]:.2f}us':>28s} {cost:6.2f}%")


if __name__ == "__main__":
    main()
