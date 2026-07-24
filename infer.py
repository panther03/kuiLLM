"""Compiled Qwen2.5-0.5B inference with verified Kuiper GPU kernels.

This is ``etc/infer_golden_compiled.py`` (a hand-written lean Qwen2 forward run
under ``torch.compile(mode="reduce-overhead")``) with the Kuiper Inductor backend
hooked in by default: an Inductor post-grad pass rewrites the supported ATen ops
(the GEMM family + sdpa) to verified ``kuiperjit::*`` kernels. ``--no-kuiper``
installs no pass and is therefore functionally identical to
``infer_golden_compiled.py`` (stock Inductor / Triton / cuBLAS / cuDNN).

Flags of note:
  * ``--no-kuiper``     run stock torch.compile (the golden compiled baseline).
  * ``--verify``        run each Kuiper op alongside stock PyTorch and report the
                        relative-Frobenius divergence (forces an eager, non-CUDA-
                        graph compile so the host-side compare can sync).
  * ``--batch-compile`` extract every matched kernel during warm-up and build
                        them in one combined compilation.
  * ``--batch N``       inference batch size.
  * ``--nsys``          bracket only the measured decode in the CUDA profiler API
                        (for ``nsys --capture-range=cudaProfilerApi``).
  * ``--dump-kernels``  trace the compiled graph and (re)write ``KERNELS.md``
                        with a dependency visualization and operator inventory.
"""

import argparse
import contextlib
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "etc"))
import infer_golden as ig
from infer_golden import DEVICE, DTYPE
from infer_golden_compiled import block, _force_cudnn_sdpa

import kuipy
from kuipy import verify as V
from kuipy.inductor import tracing


@torch.inference_mode()
def generate(m, ids, n, warmup=3, temperature=0.0, profile=False,
             batch_compile=False, verify=False):
    """Compiled decode mirroring ``infer_golden_compiled.generate_compiled``.

    When the Kuiper backend is enabled (``kuipy.enable()`` called by the caller)
    the compiled graph's supported ops run on Kuiper kernels. ``verify`` forces a
    plain (non-CUDA-graph) compile so the per-op host-side compare can sync."""
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

    greedy = not (temperature and temperature > 0.0)

    def decode_step():
        cos = cos_t.index_select(0, pos).view(1, 1, 1, m.D)
        sin = sin_t.index_select(0, pos).view(1, 1, 1, m.D)
        mask = mask_rows.index_select(0, pos).view(1, 1, 1, total)
        h = F.embedding(tok_in, m.embed)
        for li in range(m.NL):
            h = block(m, h, cos, sin, kc[li], vc[li], m.layers[li], mask, False, pos)
        h = F.rms_norm(h, (m.HID,), m.final_norm, m.EPS)
        logits = F.linear(h, m.lm_head)
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        out_buf.index_copy_(1, step, nxt)
        tok_in.copy_(nxt)
        pos.add_(1)
        step.add_(1)

    # Verify needs an eager (non-cudagraph) pass so the host-side norm compare can
    # sync; otherwise use reduce-overhead (Inductor + CUDA-graph trees).
    if verify:
        compiled = torch.compile(decode_step, fullgraph=True)
    else:
        compiled = torch.compile(decode_step, mode="reduce-overhead", fullgraph=True)

    cap = kuipy.batch_capture() if batch_compile else contextlib.nullcontext()
    with cap:
        for _ in range(warmup):
            torch.compiler.cudagraph_mark_step_begin()
            compiled()
    torch.cuda.synchronize()

    pos.fill_(plen)
    step.zero_()
    tok_in.copy_(first)

    if verify:
        V.set_enabled(True, V.tol)

    t0 = time.perf_counter()
    with ig.profile_region(profile, "decode"):
        for _ in range(n - 1):
            torch.compiler.cudagraph_mark_step_begin()
            compiled()
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    if verify:
        V.set_enabled(False)

    out_ids = torch.cat([ids, first, out_buf[:, :n - 1]], dim=1)
    return out_ids, prefill_s, decode_s


def _load_prompts(args):
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        if not prompts:
            raise SystemExit(f"no prompts found in {args.prompts}")
        return prompts
    return [args.prompt]


def main():
    ap = argparse.ArgumentParser(
        description="Compiled Qwen2.5 inference with verified Kuiper kernels.")
    ap.add_argument("prompt", nargs="?", default=ig.DEFAULT_PROMPT)
    ap.add_argument("--prompts", help="file with one prompt per line (overrides positional)")
    ap.add_argument("--max-new-tokens", type=int, default=ig.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch", type=int, default=ig.DEFAULT_BATCH)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (matches the golden baseline).")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-kuiper", action="store_true",
                    help="run stock torch.compile (identical to infer_golden_compiled.py).")
    ap.add_argument("--verify", action="store_true",
                    help="check every Kuiper op against stock PyTorch (forces non-cudagraph compile).")
    ap.add_argument("--verify-tol", type=float, default=2e-2)
    ap.add_argument("--batch-compile", action="store_true",
                    help="build every matched kernel in one combined compilation.")
    ap.add_argument("--nsys", action="store_true",
                    help="bracket only the measured decode in the CUDA profiler API.")
    ap.add_argument("--dump-kernels", nargs="?", const="KERNELS.md", default=None,
                    help="trace the compiled graph and write a Mermaid visualization "
                         "and operator table to KERNELS.md.")
    args = ap.parse_args()

    use_kuiper = not args.no_kuiper

    ig._tune_backend()
    _force_cudnn_sdpa()

    if args.verify and not use_kuiper:
        raise SystemExit("--verify requires the Kuiper backend (drop --no-kuiper).")

    if args.dump_kernels is not None:
        tracing.set_enabled(True)
    if use_kuiper:
        kuipy.enable()
        if args.verify:
            V.set_enabled(False, args.verify_tol)  # tol now; flipped on for decode
            V.reset()
    elif args.dump_kernels is not None:
        # Trace the stock graph without replacing any ops.
        from kuipy import inductor
        inductor.enable_tracing()

    tok, model = ig.load()

    prompts = _load_prompts(args)
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    ids = enc["input_ids"]
    if ids.shape[0] == 1 and args.batch > 1:
        ids = ids.repeat(args.batch, 1)
    batch, prompt_tokens = ids.shape

    label = "Kuiper" if use_kuiper else "stock torch.compile"
    print(f"Compiling decode step ({label}, reduce-overhead)...")
    out, prefill_s, decode_s = generate(
        model, ids, args.max_new_tokens, warmup=args.warmup,
        temperature=args.temperature, profile=args.nsys,
        batch_compile=args.batch_compile, verify=args.verify)
    gen_tokens = out.shape[-1] - prompt_tokens
    decode_tokens = max(gen_tokens - 1, 1)

    response = tok.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    print("\n" + "─" * 70)
    print(f"Prompt:   {prompts[0]}")
    print(f"Response: {response.strip()}")
    print("─" * 70)
    print(f"dtype: bf16   batch: {batch}   kuiper: {use_kuiper}")
    print(f"Prompt tokens/seq:    {prompt_tokens}")
    print(f"Generated tokens/seq: {gen_tokens}")
    print(f"Prompt   tps: {batch * prompt_tokens / prefill_s:8.1f} tok/s "
          f"(prefill {prefill_s * 1e3:.1f} ms)")
    print(f"Generate tps: {batch * decode_tokens / decode_s:8.1f} tok/s "
          f"(decode {decode_s * 1e3:.1f} ms, "
          f"{decode_tokens / decode_s:.1f} tok/s/seq)")

    if args.verify:
        V.print_report(report_tol=args.verify_tol)

    if args.dump_kernels is not None:
        total, claimed = tracing.dump_markdown(args.dump_kernels)
        print(f"[trace] wrote {args.dump_kernels}: {total} ops, {claimed} on Kuiper kernels")


if __name__ == "__main__":
    main()
