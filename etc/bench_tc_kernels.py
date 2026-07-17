#!/usr/bin/env python3
"""Validate + benchmark the etc/ tensor-core reference kernels (tc_linear,
tc_flash_attn) as drop-in replacements for the exact stock kernels they target
in the infer_golden.py trace:

  * linear -> cuBLAS bf16 tensor-core GEMM (F.linear)
  * sdpa   -> cuDNN flash_fprop SDPA (F.scaled_dot_product_attention)

For every case it reports the max relative-Frobenius divergence vs the stock
kernel and the mean per-call latency of each. Shapes are Qwen2.5-0.5B at the
production batch of 256, bf16.

    python3 etc/bench_tc_kernels.py
"""
import argparse
import math
import os
import sys

os.environ.setdefault("PATH", "")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

DEV = "cuda"
DT = torch.bfloat16
HERE = os.path.dirname(os.path.abspath(__file__))

# Qwen2.5-0.5B-Instruct config.
HID, NH, NKV, HEAD_DIM = 896, 14, 2, 64
QKV_OUT = NH * HEAD_DIM + 2 * NKV * HEAD_DIM   # fused q/k/v projection
INTER = 4864
VOCAB = 151936
SCALE = HEAD_DIM ** -0.5
BATCH = 256


def build():
    return load(
        name="tc_kernels",
        sources=[os.path.join(HERE, "tc_kernels_wrapper.cu"),
                 os.path.join(HERE, "tc_linear.cu"),
                 os.path.join(HERE, "tc_flash_attn.cu")],
        extra_include_paths=[HERE],
        verbose=False,
    )


def rel_fro(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-9)).item()


def time_call(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms/call


def bench_linear(mod, no_splitk=False):
    print("=== linear: C = A @ W^T (+bias)   vs F.linear (cuBLAS bf16) ==="
          + ("   [split-K disabled]" if no_splitk else ""))
    g = torch.Generator(device=DEV).manual_seed(0)
    # (name, M, K, N, bias) covering every GEMM in the decode step (batch 256).
    cases = [
        ("qkv_proj (bias)", BATCH, HID, QKV_OUT, True),
        ("o_proj",          BATCH, HID, HID,     False),
        ("gate_proj",       BATCH, HID, INTER,   False),
        ("up_proj",         BATCH, HID, INTER,   False),
        ("down_proj",       BATCH, INTER, HID,   False),
        ("lm_head",         BATCH, HID, VOCAB,   False),
    ]
    ok = True
    for name, M, K, N, use_bias in cases:
        A = torch.randn(M, K, device=DEV, dtype=DT, generator=g) * 0.1
        W = torch.randn(N, K, device=DEV, dtype=DT, generator=g) * 0.1
        b = torch.randn(N, device=DEV, dtype=DT, generator=g) * 0.1 if use_bias else None
        ref = F.linear(A, W, b)
        got = mod.linear(A, W, b, no_splitk)
        err = rel_fro(got, ref)
        t_ours = time_call(lambda: mod.linear(A, W, b, no_splitk))
        t_ref = time_call(lambda: F.linear(A, W, b))
        flag = "ok " if err < 5e-2 else "BAD"
        ok &= err < 5e-2
        print(f"  [{flag}] {name:<16} M={M:>4} K={K:>5} N={N:>6} "
              f"| rel-err {err:.2e} | ours {t_ours*1e3:8.1f}us  ref {t_ref*1e3:8.1f}us"
              f"  {t_ours/t_ref:5.2f}x  {100*t_ref/t_ours:5.1f}%")
    return ok


def ref_sdpa(q, k, v, mask, causal):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=SCALE, is_causal=causal, enable_gqa=True)


def bench_sdpa(mod, force_decode_kernel=False):
    print("=== sdpa: flash attention   vs F.scaled_dot_product_attention (cuDNN) ==="
          + ("   [prefill uses decode kernel]" if force_decode_kernel else ""))
    g = torch.Generator(device=DEV).manual_seed(1)
    ok = True

    def rand(*shape):
        return torch.randn(*shape, device=DEV, dtype=DT, generator=g)

    # Decode: one query token, full KV context, explicit additive causal mask
    # ([1,1,1,Sk]) -- exactly the golden hot path.
    for ctx in (128, 512, 1024):
        q = rand(BATCH, NH, 1, HEAD_DIM)
        kk = rand(BATCH, NKV, ctx, HEAD_DIM)
        vv = rand(BATCH, NKV, ctx, HEAD_DIM)
        mask = torch.zeros(1, 1, 1, ctx, device=DEV, dtype=DT)
        ref = ref_sdpa(q, kk, vv, mask, False)
        got = mod.sdpa(q, kk, vv, mask, SCALE, False, force_decode_kernel)
        err = rel_fro(got, ref)
        t_ours = time_call(lambda: mod.sdpa(q, kk, vv, mask, SCALE, False, force_decode_kernel))
        t_ref = time_call(lambda: ref_sdpa(q, kk, vv, mask, False))
        flag = "ok " if err < 5e-2 else "BAD"
        ok &= err < 5e-2
        print(f"  [{flag}] decode  ctx={ctx:<5} | rel-err {err:.2e} "
              f"| ours {t_ours*1e3:8.1f}us  ref {t_ref*1e3:8.1f}us  {t_ours/t_ref:5.2f}x  {100*t_ref/t_ours:5.1f}%")

    # Prefill: full self-attention, is_causal=True, no explicit mask.
    for s in (128, 512):
        q = rand(BATCH, NH, s, HEAD_DIM)
        kk = rand(BATCH, NKV, s, HEAD_DIM)
        vv = rand(BATCH, NKV, s, HEAD_DIM)
        ref = ref_sdpa(q, kk, vv, None, True)
        got = mod.sdpa(q, kk, vv, None, SCALE, True, force_decode_kernel)
        err = rel_fro(got, ref)
        t_ours = time_call(lambda: mod.sdpa(q, kk, vv, None, SCALE, True, force_decode_kernel))
        t_ref = time_call(lambda: ref_sdpa(q, kk, vv, None, True))
        flag = "ok " if err < 5e-2 else "BAD"
        ok &= err < 5e-2
        print(f"  [{flag}] prefill seq={s:<5} | rel-err {err:.2e} "
              f"| ours {t_ours*1e3:8.1f}us  ref {t_ref*1e3:8.1f}us  {t_ours/t_ref:5.2f}x  {100*t_ref/t_ours:5.1f}%")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-splitk", action="store_true",
                    help="force the single-pass GEMM (never use the split-K path)")
    ap.add_argument("--force-decode-kernel", action="store_true",
                    help="always use tc_flash_attn_kernel (the decode/GQA kernel), "
                         "never the specialized prefill kernel")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("Building etc/ tensor-core kernels...", file=sys.stderr)
    mod = build()
    print(f"device: {torch.cuda.get_device_name(0)}   dtype: bf16   batch: {BATCH}\n")
    ok = bench_linear(mod, no_splitk=args.no_splitk)
    print()
    ok &= bench_sdpa(mod, force_decode_kernel=args.force_decode_kernel)
    print("\n" + ("ALL PASSED" if ok else "SOME FAILED (rel-err >= 5e-2)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
