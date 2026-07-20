"""Compiled+fused counterpart of infer_golden.py.

SAME hand-written lean forward (fused QKV GEMM, cuDNN GQA, precomputed RoPE,
static KV cache) as infer_golden.py -- the ONLY difference is the decode
mechanism: here the step is ``torch.compile(mode="reduce-overhead")`` (Inductor
Triton fusion + CUDA-graph trees) instead of a hand-captured CUDA graph.

Purpose: isolate the effect of JIT fusion on the ~14% pointwise tail (RMSNorm,
RoPE, SiLU-GLU, residuals, copies) while holding the GEMMs (cuBLAS) and attention
(cuDNN) fixed. If fusion is pure upside, this should land at or below the manual
graph; if the fully-folded manual replay wins, we learn the bottleneck is host
overhead / folding, not fusion.

The KV cache is made cudagraph-safe for Inductor via ``mark_static_address`` (the
naive mutation is what made a prior quick test silently fall out of cudagraphs).
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_golden as ig
from infer_golden import DEVICE, DTYPE


def _force_cudnn_sdpa():
    """Pick cuDNN for SDPA globally (no context manager -> fullgraph-friendly).

    cuDNN is the only fused backend that accepts GQA + an additive mask; math is
    left on as a never-chosen fallback.
    """
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def block(m, h, cos, sin, kc, vc, lyr, mask, is_causal, write_pos):
    """ig.Qwen2Fast._block, minus the sdpa_kernel context manager."""
    b, s, _ = h.shape
    r = h
    hn = F.rms_norm(h, (m.HID,), lyr.in_ln, m.EPS)
    q, k, v = F.linear(hn, lyr.qkv_w, lyr.qkv_b).split([m.QO, m.KO, m.VO], dim=-1)
    q = q.view(b, s, m.NH, m.D).transpose(1, 2)
    k = k.view(b, s, m.NKV, m.D).transpose(1, 2)
    v = v.view(b, s, m.NKV, m.D).transpose(1, 2)
    q = q * cos + m._rot_half(q) * sin
    k = k * cos + m._rot_half(k) * sin
    if write_pos is None:
        kc[:, :, :s, :] = k
        vc[:, :, :s, :] = v
        ka, va = k, v
    else:
        kc.index_copy_(2, write_pos, k)
        vc.index_copy_(2, write_pos, v)
        ka, va = kc, vc
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        ao = F.scaled_dot_product_attention(
            q, ka, va, attn_mask=mask, scale=m.SCALE, is_causal=is_causal, enable_gqa=True)
    ao = ao.transpose(1, 2).reshape(b, s, m.HID)
    h = r + F.linear(ao, lyr.o_w)
    r = h
    hn = F.rms_norm(h, (m.HID,), lyr.post_ln, m.EPS)
    g = F.silu(F.linear(hn, lyr.gate_w)) * F.linear(hn, lyr.up_w)
    return r + F.linear(g, lyr.down_w)


@torch.inference_mode()
def generate_compiled(m, ids, n, warmup=3):
    b, plen = ids.shape
    total = plen + n + 8
    kc = [torch.zeros(b, m.NKV, total, m.D, device=DEVICE, dtype=DTYPE) for _ in range(m.NL)]
    vc = [torch.zeros(b, m.NKV, total, m.D, device=DEVICE, dtype=DTYPE) for _ in range(m.NL)]
    for t in kc + vc:
        torch._dynamo.mark_static_address(t)
    cos_t, sin_t = m._rope_tables(total)
    kp = torch.arange(total, device=DEVICE)
    neg = torch.finfo(DTYPE).min
    mask_rows = torch.where(
        kp[None, :] <= kp[:, None],
        torch.zeros((), dtype=DTYPE, device=DEVICE),
        torch.full((), neg, dtype=DTYPE, device=DEVICE),
    )

    def prefill():
        h = F.embedding(ids, m.embed)
        cos = cos_t[:plen].view(1, 1, plen, m.D)
        sin = sin_t[:plen].view(1, 1, plen, m.D)
        for li in range(m.NL):
            h = block(m, h, cos, sin, kc[li], vc[li], m.layers[li], None, True, None)
        h = F.rms_norm(h, (m.HID,), m.final_norm, m.EPS)
        return F.linear(h[:, -1:], m.lm_head).argmax(-1)

    prefill()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first = prefill()
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    tok_in = first.clone()
    pos = torch.tensor([plen], device=DEVICE)
    step = torch.zeros(1, dtype=torch.long, device=DEVICE)
    out_buf = torch.zeros(b, n, dtype=torch.long, device=DEVICE)
    for t in (tok_in, pos, step, out_buf):
        torch._dynamo.mark_static_address(t)

    # Fully folded: the compiled step also writes the output slot, feeds the next
    # token back, and advances pos/step -- all in place on static tensors -- so
    # the decode loop is just the compiled call (no per-token eager ops).
    def decode_step():
        cos = cos_t.index_select(0, pos).view(1, 1, 1, m.D)
        sin = sin_t.index_select(0, pos).view(1, 1, 1, m.D)
        mask = mask_rows.index_select(0, pos).view(1, 1, 1, total)
        h = F.embedding(tok_in, m.embed)
        for li in range(m.NL):
            h = block(m, h, cos, sin, kc[li], vc[li], m.layers[li], mask, False, pos)
        h = F.rms_norm(h, (m.HID,), m.final_norm, m.EPS)
        nxt = F.linear(h, m.lm_head).argmax(-1)
        out_buf.index_copy_(1, step, nxt)
        tok_in.copy_(nxt)
        pos.add_(1)
        step.add_(1)

    compiled = torch.compile(decode_step, mode="reduce-overhead", fullgraph=True)

    for _ in range(warmup):
        torch.compiler.cudagraph_mark_step_begin()
        compiled()
    torch.cuda.synchronize()
    pos.fill_(plen)
    step.zero_()
    tok_in.copy_(first)

    t0 = time.perf_counter()
    for _ in range(n - 1):
        torch.compiler.cudagraph_mark_step_begin()
        compiled()
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    out_ids = torch.cat([ids, first, out_buf[:, :n - 1]], dim=1)
    return out_ids, prefill_s, decode_s


def main():
    ap = argparse.ArgumentParser(description="Compiled+fused Qwen2.5 steelman (torch.compile).")
    ap.add_argument("prompt", nargs="?", default=ig.DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=ig.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch", type=int, default=ig.DEFAULT_BATCH)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ig._tune_backend()
    _force_cudnn_sdpa()
    tok, model = ig.load()

    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt").to(DEVICE)
    ids = enc["input_ids"]
    if args.batch > 1:
        ids = ids.repeat(args.batch, 1)
    batch, prompt_tokens = ids.shape

    print("Compiling decode step (torch.compile reduce-overhead)...")
    out, prefill_s, decode_s = generate_compiled(model, ids, args.max_new_tokens, args.warmup)
    gen_tokens = out.shape[-1] - prompt_tokens
    decode_tokens = max(gen_tokens - 1, 1)

    response = tok.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    print("\n" + "─" * 70)
    print(f"Prompt:   {args.prompt}")
    print(f"Response: {response.strip()}")
    print("─" * 70)
    print(f"dtype: bf16   batch: {batch}")
    print(f"Prompt tokens/seq:    {prompt_tokens}")
    print(f"Generated tokens/seq: {gen_tokens}")
    print(f"Prompt   tps: {batch * prompt_tokens / prefill_s:8.1f} tok/s "
          f"(prefill {prefill_s * 1e3:.1f} ms)")
    print(f"Generate tps: {batch * decode_tokens / decode_s:8.1f} tok/s "
          f"(decode {decode_s * 1e3:.1f} ms, "
          f"{decode_tokens / decode_s:.1f} tok/s/seq)")


if __name__ == "__main__":
    main()
