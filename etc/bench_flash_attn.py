#!/usr/bin/env python
"""Benchmark the plain-CUDA FlashAttention in etc/ against the `flash_attn`
package on Qwen2.5-0.5B attention parameters (14 query heads, 2 KV heads,
head_dim 64 -> GQA group 7, causal).

Run with the project venv:
    .venv/bin/python etc/bench_flash_attn.py

Covers both regimes seen in kernel_call.log:
  * decode  (seqlen_q == 1 per sequence, ragged KV context) -- the hot path
  * prefill (seqlen_q == seqlen_k, uniform)

For each scenario it reports the max relative-Frobenius output divergence vs
flash_attn and the mean per-call latency of each implementation.
"""
import math
import os
import sys

# torch.utils.cpp_extension shells out to `ninja`, which lives in the venv's
# bin dir but may not be on PATH when invoked as `.venv/bin/python ...`.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

import torch
from torch.utils.cpp_extension import load
from flash_attn import flash_attn_varlen_func

# --- Qwen2.5-0.5B attention shape -------------------------------------------
H_Q, H_KV, HEAD_DIM = 14, 2, 64
SCALE = 1.0 / math.sqrt(HEAD_DIM)
DEV = "cuda"
DTYPE = torch.bfloat16

HERE = os.path.dirname(os.path.abspath(__file__))


def build_kernel():
    return load(
        name="fa_etc_bench",
        sources=[os.path.join(HERE, "flash_attn_wrapper.cu"),
                 os.path.join(HERE, "flash_attn_kernel.cu")],
        extra_include_paths=[os.path.join(HERE, os.pardir, "include"), HERE],
        verbose=False,
    )


def cu_seqlens(lengths):
    out = [0]
    for l in lengths:
        out.append(out[-1] + l)
    return torch.tensor(out, dtype=torch.int32, device=DEV)


def make_inputs(lens_q, lens_k, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    cuq, cuk = cu_seqlens(lens_q), cu_seqlens(lens_k)
    tq, tk = int(cuq[-1]), int(cuk[-1])
    q = torch.randn(tq, H_Q, HEAD_DIM, device=DEV, dtype=DTYPE, generator=g)
    k = torch.randn(tk, H_KV, HEAD_DIM, device=DEV, dtype=DTYPE, generator=g)
    v = torch.randn(tk, H_KV, HEAD_DIM, device=DEV, dtype=DTYPE, generator=g)
    return q, k, v, cuq, cuk, max(lens_q), max(lens_k)


def time_call(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/call


def rel_fro(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-9)).item()


def run_scenario(name, lens_q, lens_k, mod, causal=True):
    q, k, v, cuq, cuk, msq, msk = make_inputs(lens_q, lens_k)

    def ours():
        return mod.run(q, k, v, cuq, cuk, msq, msk, 0.0, SCALE, causal)

    def theirs():
        return flash_attn_varlen_func(q, k, v, cuq, cuk, msq, msk,
                                      dropout_p=0.0, softmax_scale=SCALE,
                                      causal=causal)

    out_ours = ours()[0]
    out_fa = theirs()
    err = rel_fro(out_ours, out_fa)

    t_ours = time_call(ours)
    t_fa = time_call(theirs)

    tq, tk = int(cuq[-1]), int(cuk[-1])
    print(f"{name:<28} b={len(lens_q):>3} tq={tq:>6} tk={tk:>6} "
          f"| rel-err {err:.2e} | ours {t_ours*1e3:8.1f}us  "
          f"flash_attn {t_fa*1e3:8.1f}us  | speedup {t_fa/t_ours:5.2f}x")


def main():
    print("Building etc/ FlashAttention extension...", file=sys.stderr)
    mod = build_kernel()
    print(f"Qwen2.5-0.5B attn: H_Q={H_Q} H_KV={H_KV} head_dim={HEAD_DIM} "
          f"scale={SCALE:.4f} causal=True\n")

    torch.manual_seed(0)

    # Decode: one query token per sequence, ragged KV context (the hot path).
    for ctx in (128, 512, 2048):
        lens_q = [1] * 64
        lens_k = [ctx] * 64
        run_scenario(f"decode  b64 ctx={ctx}", lens_q, lens_k, mod)

    # Decode with ragged contexts (closer to kernel_call.log).
    glen = torch.Generator().manual_seed(7)
    lens_k = [int(x) for x in torch.randint(40, 110, (64,), generator=glen)]
    run_scenario("decode  b64 ragged-ctx", [1] * 64, lens_k, mod)

    # Prefill: full self-attention over a prompt, uniform length.
    for (b, s) in ((8, 128), (4, 512), (2, 1024)):
        run_scenario(f"prefill b{b} seq={s}", [s] * b, [s] * b, mod)


if __name__ == "__main__":
    main()
