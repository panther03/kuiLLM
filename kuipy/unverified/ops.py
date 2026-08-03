## TODO: this will become refactored into 
# - this file: unverified operator definitions
# - benchmarking.py: generic utilities for benchmarking code
# - bench.ipynb: Jupyter notebook with the specific benchmarks this file handles.


#!/usr/bin/env python3
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kuipy import kuiops

aten = torch.ops.aten

DEV = "cuda"
DT = torch.bfloat16
LINEAR_DT = torch.bfloat16
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Kuiper headers: prefer the project's installed copy (make install-kuiper-*),
# fall back to the Kuiper source tree ($KUIPER_HOME/include).
_INST_INC = os.path.join(ROOT, "inst", "include", "kuiper")
_KH = os.environ.get("KUIPER_HOME", os.path.expanduser("~/work/kuiper"))
KUIPER_INC = _INST_INC if os.path.exists(os.path.join(_INST_INC, "kuiper.h")) \
    else os.path.join(_KH, "include")

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
                 os.path.join(HERE, "tc2d_linear_manual.cu"),
                 os.path.join(HERE, "tc_flash_attn.cu"),
                 os.path.join(HERE, "flash_attn_fa1.cu"),
                 os.path.join(HERE, "Kuiops_Sdpa_Flash_Inst.cu")],
        extra_include_paths=[HERE, KUIPER_INC, os.path.join(ROOT, "include")],
        # Kuiper headers rely on implicit int->half conversion (fragment `{0}`
        # init); undo the torch defaults that would disable it.
        extra_cuda_cflags=["-U__CUDA_NO_HALF_OPERATORS__",
                           "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_HALF2_OPERATORS__"],
        verbose=False,
    )


def kui_run(impl, func, args, kwargs=None):
    """Dispatch through the JIT stack exactly like KuiperMode does."""
    kwargs = kwargs or {}
    spec = impl.supported(func, args, kwargs)
    assert spec is not None, f"{func} not supported for {args[0].shape} {kwargs}"
    return impl.run(spec, args, kwargs)


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


# (name, M, K, N) covering every bias-free GEMM in the decode step (batch 256).
# qkv_proj is excluded because its vector bias would require broadcasting, while
# the kernels expect a full (M, N) C matrix.
LINEAR_CASES = [
    ("o_proj",          BATCH, HID, HID),
    ("gate_proj",       BATCH, HID, INTER),
    ("up_proj",         BATCH, HID, INTER),
    ("down_proj",       BATCH, INTER, HID),
    ("lm_head",         BATCH, HID, VOCAB),
    ("square_4096",     4096,  4096, 4096),
]

GEMM_CASES = [
    ("o_proj",      BATCH, HID,  HID),
    ("down_proj",   BATCH, INTER, HID),
    ("square_4096", 4096,  4096, 4096),
]


def report(name, M, K, N, err, t_ours, t_ref, extra=""):
    flag = "ok " if err < 5e-2 else "BAD"
    flops = 2.0 * M * N * K  # multiply-add per output element
    gf_ours = flops / (t_ours * 1e-3) / 1e9
    gf_ref = flops / (t_ref * 1e-3) / 1e9
    print(f"  [{flag}] {name:<16} M={M:>4} K={K:>5} N={N:>6} "
          f"| {extra}rel-err {err:.2e} "
          f"| ours {t_ours*1e3:8.1f}us  ref {t_ref*1e3:8.1f}us"
          f"  {t_ours/t_ref:5.2f}x  {100*t_ref/t_ours:5.1f}%"
          f"  | ours {gf_ours:8.1f} GFLOP/s  ref {gf_ref:8.1f} GFLOP/s")
    return err < 5e-2


def bench_mm(dtype, out_dtype, implementation=None):
    """aten.mm through kuiops.MmImpl vs F.linear."""
    implementation_label = implementation or "default"
    print(f"=== mm: C = A @ W^T   vs F.linear   "
          f"[kuipy MmImpl {implementation_label}, "
          f"in {dtype} -> out {out_dtype}] ===")
    impl = kuiops.MmImpl()
    g = torch.Generator(device=DEV).manual_seed(0)
    kwargs = {"out_dtype": out_dtype}
    if implementation is not None:
        kwargs["impl"] = implementation
    ok = True
    for name, M, K, N in LINEAR_CASES:
        A = torch.randn(M, K, device=DEV, dtype=dtype, generator=g) * 0.1
        W = torch.randn(N, K, device=DEV, dtype=dtype, generator=g) * 0.1
        Wt = W.t().contiguous()  # (K, N): the mm RHS, prepared outside timing
        ref = F.linear(A, W)
        got = kui_run(impl, aten.mm.default, (A, Wt), kwargs)
        err = rel_fro(got, ref)
        spec = impl.supported(aten.mm.default, (A, Wt), kwargs)
        t_ours = time_call(lambda: impl.run(spec, (A, Wt), kwargs))
        t_ref = time_call(lambda: F.linear(A, W))
        ok &= report(name, M, K, N, err, t_ours, t_ref, extra=f"{spec['impl']:<8} | ")
    return ok


def bench_addmm(dtype):
    """Full epilogue D = alpha*(A@B) + beta*C (beta != 0) through kuiops.AddmmImpl,
    which the bias-free mm path never hits. Validated against a fp32 reference."""
    print("=== addmm epilogue: D = alpha*(A@B) + beta*C   vs fp32 reference "
          f"(perf vs {dtype} torch.addmm)   [kuipy AddmmImpl] ===")
    impl = kuiops.AddmmImpl()
    g = torch.Generator(device=DEV).manual_seed(2)
    alpha, beta = 0.75, 1.5
    kw = dict(alpha=alpha, beta=beta)
    ok = True
    for name, M, K, N in GEMM_CASES:
        A = torch.randn(M, K, device=DEV, dtype=dtype, generator=g) * 0.1
        B = torch.randn(K, N, device=DEV, dtype=dtype, generator=g) * 0.1
        C = torch.randn(M, N, device=DEV, dtype=dtype, generator=g) * 0.1
        ref = alpha * (A.float() @ B.float()) + beta * C.float()
        got = kui_run(impl, aten.addmm.default, (C, A, B), kw)
        err = rel_fro(got, ref)
        spec = impl.supported(aten.addmm.default, (C, A, B), kw)
        t_ours = time_call(lambda: impl.run(spec, (C, A, B), kw))
        t_ref = time_call(lambda: torch.addmm(C, A, B, beta=beta, alpha=alpha))
        ok &= report(name, M, K, N, err, t_ours, t_ref,
                     extra=f"{spec['impl']:<8} | alpha={alpha} beta={beta} | ")
    return ok


def bench_gemm_manual(mod):
    """The hand-written etc/tc2d_linear_manual.cu GEMM (bf16 in/out, fp32
    accumulate, in-place epilogue), same cases as bench_addmm."""
    print("=== manual gemm epilogue: D = alpha*(A@B) + beta*C   vs fp32 reference "
          "(perf vs bf16 torch.addmm)   [etc/tc2d_linear_manual.cu] ===")
    g = torch.Generator(device=DEV).manual_seed(2)
    alpha, beta = 0.75, 1.5
    ok = True
    for name, M, K, N in GEMM_CASES:
        A = torch.randn(M, K, device=DEV, dtype=LINEAR_DT, generator=g) * 0.1
        B = torch.randn(K, N, device=DEV, dtype=LINEAR_DT, generator=g) * 0.1
        C = torch.randn(M, N, device=DEV, dtype=LINEAR_DT, generator=g) * 0.1
        ref = alpha * (A.float() @ B.float()) + beta * C.float()
        got = mod.gemm_manual(A, B, C, alpha, beta)
        err = rel_fro(got, ref)
        t_ours = time_call(lambda: mod.gemm_manual(A, B, C, alpha, beta))
        t_ref = time_call(lambda: torch.addmm(C, A, B, beta=beta, alpha=alpha))
        ok &= report(name, M, K, N, err, t_ours, t_ref,
                     extra=f"alpha={alpha} beta={beta} | ")
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
    for ctx in (128, 512, 1024, 16384):
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


def bench_sdpa_kuiper(mod):
    """Verified Kuiper decode kernel vs the FA1 reference it ports and cuDNN.

    The Kuiper kernel's mask is a dense (B,Hq,Sq,Sk) tensor -- a Kuiper tlayout
    is an injection, so there is no broadcast layout to instantiate it with --
    so the mask is materialised and handed to every contender for fairness.
    """
    print("=== sdpa: VERIFIED Kuiper flash attention (decode)   vs flash_attn_fa1.cu and cuDNN ===")
    g = torch.Generator(device=DEV).manual_seed(1)
    ok = True

    def rand(*shape):
        return torch.randn(*shape, device=DEV, dtype=DT, generator=g)

    for ctx in (128, 512, 1024, 16384):
        q = rand(BATCH, NH, 1, HEAD_DIM)
        kk = rand(BATCH, NKV, ctx, HEAD_DIM)
        vv = rand(BATCH, NKV, ctx, HEAD_DIM)
        mask = torch.zeros(BATCH, NH, 1, ctx, device=DEV, dtype=DT)
        ref = ref_sdpa(q, kk, vv, mask, False)
        got = mod.sdpa_kuiper(q, kk, vv, mask, SCALE, False)
        err = rel_fro(got, ref)
        t_ours = time_call(lambda: mod.sdpa_kuiper(q, kk, vv, mask, SCALE, False))
        t_fa1 = time_call(lambda: mod.sdpa_fa1(q, kk, vv, mask, SCALE, False, True))
        t_ref = time_call(lambda: ref_sdpa(q, kk, vv, mask, False))
        flag = "ok " if err < 5e-2 else "BAD"
        ok &= err < 5e-2
        print(f"  [{flag}] decode  ctx={ctx:<5} | rel-err {err:.2e} "
              f"| kuiper {t_ours*1e3:8.1f}us  fa1 {t_fa1*1e3:8.1f}us  cudnn {t_ref*1e3:8.1f}us"
              f"  | kuiper {100*t_ref/t_ours:5.1f}% of cudnn, fa1 {100*t_ref/t_fa1:5.1f}%")

    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdpa-only", action="store_true",
                    help="skip the matmul benchmarks")
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
    print(f"device: {torch.cuda.get_device_name(0)}   "
          f"dtype: matmul bf16 / sdpa bf16   batch: {BATCH}\n")
    ok = True
    if not args.sdpa_only:
        # ok &= bench_mm(torch.bfloat16, torch.bfloat16, "tc2d")
        # print()
        ok &= bench_mm(torch.bfloat16, torch.bfloat16, "tc2d_to")
        print()
        ok &= bench_mm(torch.float16, torch.float16)
        print()
        ok &= bench_addmm(torch.bfloat16)
        print()
        ok &= bench_addmm(torch.float16)
        print()
        ok &= bench_gemm_manual(mod)
        print()
    ok &= bench_sdpa(mod, force_decode_kernel=args.force_decode_kernel)
    print()
    ok &= bench_sdpa_kuiper(mod)
    print("\n" + ("ALL PASSED" if ok else "SOME FAILED (rel-err >= 5e-2)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
