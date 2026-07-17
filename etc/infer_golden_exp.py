# Steelman PyTorch speed test for Qwen2.5-0.5B. (EXPERIMENTS)
#
# This is the *pure PyTorch* baseline: no Kuiper, no kuipy dispatch. The point
# is a fair, hyper-optimized comparison of "we ran Qwen2.5 in PyTorch with its
# own kernels". Concretely that means letting Inductor do everything it can:
#
#   * torch.compile with the default (Inductor) backend, so RMSNorm, RoPE,
#     SiLU/GLU, residual adds, etc. get fused into a handful of kernels instead
#     of the dozens of tiny reciprocal/sqrt/mean/mul launches you see in eager
#     or in the Kuiper graph-mode backend (which runs the ATen graph via
#     boxed_nop and deliberately does NO fusion so ops stay interceptable).
#   * A static KV cache so the decode step has a fixed shape and the whole
#     forward can be captured once and replayed.
#   * mode="reduce-overhead" (CUDA graphs) so per-token decode -- which is
#     memory/launch-overhead bound for a 0.5B model -- pays essentially zero
#     kernel-launch cost.
#   * TF32 matmuls, cuDNN autotuning, SDPA (flash) attention, inference_mode.

import argparse
import contextlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig


@contextlib.contextmanager
def profile_region(active, label="measured"):
    """Bracket only the measured work for nsys ``--capture-range=cudaProfilerApi``.

    Warmup, torch.compile autotuning and CUDA-graph capture (which in Triton
    mode launch a large number of throwaway kernels) run outside this region and
    are excluded from the capture, so Triton vs no-Triton traces cover the same
    steady-state workload."""
    if not active:
        yield
        return
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PROMPT = "Summarize the plot of Hamlet in 1000 characters."
DEFAULT_MAX_NEW_TOKENS = 256

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _use_fused_rmsnorm():
    """Route Qwen2RMSNorm through the fused ``F.rms_norm`` ATen kernel.

    HF writes RMSNorm out of primitive ops (pow/mean/rsqrt/mul -> ~8 tiny
    kernels). Without Triton those stay unfused. Stock PyTorch ships a single
    fused ``rms_norm`` CUDA kernel, so this recovers the fusion using only
    interceptable ATen ops -- no Triton codegen required."""
    import torch.nn.functional as Fnn
    from transformers.models.qwen2 import modeling_qwen2

    def forward(self, hidden_states):
        return Fnn.rms_norm(hidden_states, (hidden_states.shape[-1],),
                            self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward


def _use_cudnn_gqa_attention():
    """Grouped-query attention with **no ``repeat_kv`` copy** and a fused backend.

    HF only takes SDPA's native GQA when ``attention_mask is None``; with a mask it
    materializes the *full static-cache-length* KV via a strided ``expand``+
    ``reshape`` copy every layer every step. That copy is pure bandwidth, zero
    compute -- the giant ``direct_copy`` kernel (eager) / the bulk of the fused
    ``triton_poi_fused_*`` prologue (Inductor).

    Passing ``enable_gqa=True`` removes the copy, but PyTorch's dispatcher then
    *always* routes ``enable_gqa`` + a dense mask to the slow **math** backend
    (flash/mem-efficient reject GQA-with-mask), which is even slower than the copy.
    The only fused backend that accepts GQA broadcast *and* an additive mask is
    **cuDNN**, and it is not auto-selected -- we must force it. Doing so does the
    GQA broadcast inside the kernel and streams KV once: ~8x faster per attention
    call than repeat_kv+default at batch 256, with no full-length copy. cuDNN SDPA
    is a stock ATen op (``_scaled_dot_product_cudnn_attention``), so it stays
    interceptable -- unlike a Triton fusion."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.integrations import sdpa_attention
    from transformers import modeling_utils

    torch.backends.cuda.enable_cudnn_sdp(True)
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: True

    _orig = sdpa_attention.sdpa_attention_forward

    def forward(*args, **kwargs):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return _orig(*args, **kwargs)

    sdpa_attention.sdpa_attention_forward = forward
    modeling_utils.ALL_ATTENTION_FUNCTIONS["sdpa"] = forward


def _tune_backend():
    """Global knobs that let PyTorch pick its fastest kernels."""
    if DEVICE != "cuda":
        return
    # TF32 for fp32 matmul/conv paths (RoPE, some reductions).
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Prefer the flash / memory-efficient SDPA backends over the math fallback.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def load(dtype, compile_mode, no_triton, manual, cudnn_gqa):
    print(f"Loading {MODEL_ID} in {dtype} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=_DTYPES[dtype],
        device_map=DEVICE,
        attn_implementation="sdpa",
    ).eval()

    # Static cache -> fixed decode shape. transformers then auto-compiles the
    # decode step itself (it leaves prefill eager and manages CUDA-graph step
    # marking correctly, which is why we must NOT hand-compile model.forward).
    model.generation_config.cache_implementation = "static"

    if manual:
        # Manual CUDA-graph mode drives its own decode loop (see
        # manual_cudagraph_generate); Triton-free by construction, so also take
        # the fused-ATen RMSNorm and cuDNN GQA attention (no repeat_kv copy).
        _use_fused_rmsnorm()
        if cudnn_gqa:
            _use_cudnn_gqa_attention()
        model.generation_config.disable_compile = True
    elif DEVICE != "cuda" or compile_mode == "none":
        model.generation_config.disable_compile = True
    elif no_triton:
        # No Triton codegen: only stock ATen kernels (the ones Kuiper can
        # intercept). Recover RMSNorm fusion via the fused ATen op, then use the
        # `cudagraphs` dynamo backend to kill per-kernel launch overhead without
        # generating any Triton. HF's Inductor auto-compile is disabled so we
        # own the (Triton-free) compilation.
        _use_fused_rmsnorm()
        if cudnn_gqa:
            _use_cudnn_gqa_attention()
        model.generation_config.disable_compile = True
        model.forward = torch.compile(
            model.forward, backend="cudagraphs", fullgraph=True, dynamic=False,
        )
    else:
        # fullgraph=True is safe with the sdpa attention path (no graph breaks).
        model.generation_config.compile_config = CompileConfig(
            mode=compile_mode, fullgraph=True, dynamic=False,
        )
    return tok, model


def _gen(model, tok, inputs, n):
    with torch.inference_mode():
        return model.generate(
            **inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )


def _timed_gen(model, tok, inputs, n):
    _sync()
    t = time.perf_counter()
    out = _gen(model, tok, inputs, n)
    _sync()
    return out, time.perf_counter() - t


def _warm(model, tok, inputs, n, rounds=3):
    """torch.compile + the static cache + CUDA-graph capture need a few passes
    per generate shape before timings stabilize."""
    for _ in range(rounds):
        _gen(model, tok, inputs, n)
    _sync()


@torch.inference_mode()
def manual_cudagraph_generate(model, inputs, n, warmup=3, profile=False):
    """No-Triton steelman: greedy decode with a hand-captured CUDA graph.

    The static KV cache fixes every decode-step shape, so we capture the
    single-token forward once and ``replay()`` it per token -- collapsing the
    ~200 stock ATen kernels/token into one graph launch, without any Triton
    codegen. This mirrors what Inductor's ``cudagraph_trees`` does for
    ``reduce-overhead``, minus the Triton kernels.

    Returns ``(out_ids, prefill_s, decode_s)`` where ``out_ids`` is the full
    prompt+generation sequence (so callers can decode it like generate output).

    Note: the prompt is assumed uniform-length across the batch (true for the
    replicated-prompt benchmark), so a single scalar position/mask serves every
    row. Ragged batches would need per-row positions.
    """
    from transformers import StaticCache

    ids = inputs["input_ids"]
    batch, plen = ids.shape
    total_len = plen + n + 8
    cache = StaticCache(config=model.config, max_cache_len=total_len)

    neg = torch.finfo(model.dtype).min
    key_pos = torch.arange(total_len, device=DEVICE)

    def make_mask(pos):
        allow = (key_pos <= pos).view(1, 1, 1, total_len)
        return torch.where(allow, torch.zeros((), dtype=model.dtype, device=DEVICE),
                           torch.full((), neg, dtype=model.dtype, device=DEVICE))

    def reset_cumulative_length(val):
        for layer in cache.layers:
            layer.cumulative_length.fill_(val)

    # --- Prefill (eager, dynamic shape; not captured) ---
    # A throwaway prefill first, so the timed one excludes one-time costs
    # (kernel loads, cuDNN autotune, allocator warmup).
    prefill_pos = torch.arange(plen, device=DEVICE)
    model(input_ids=ids, cache_position=prefill_pos,
          past_key_values=StaticCache(config=model.config, max_cache_len=total_len),
          use_cache=True)
    _sync()
    t0 = time.perf_counter()
    out = model(input_ids=ids, cache_position=prefill_pos,
                past_key_values=cache, use_cache=True)
    cur = out.logits[:, -1].argmax(-1, keepdim=True)
    _sync()
    prefill_s = time.perf_counter() - t0

    # --- Capture the single-token decode step ---
    s_in = cur.clone()
    s_pos = torch.tensor([plen], device=DEVICE)
    s_posid = torch.full((batch, 1), plen, device=DEVICE)
    s_mask = make_mask(plen)

    def step_forward():
        return model(input_ids=s_in, cache_position=s_pos, position_ids=s_posid,
                     attention_mask=s_mask, past_key_values=cache, use_cache=True)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(warmup):
            step_forward()
    torch.cuda.current_stream().wait_stream(warm_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s_out = step_forward()

    # StaticLayer.update derives its write slot from cumulative_length and
    # auto-advances it on every replay; undo the warmup/capture increments so
    # the first replay writes slot `plen`.
    reset_cumulative_length(plen)

    # --- Replay loop ---
    gen = [cur]
    _sync()
    t0 = time.perf_counter()
    with profile_region(profile, "decode"):
        for i in range(n - 1):
            pos = plen + i
            s_in.copy_(cur)
            s_pos.fill_(pos)
            s_posid.fill_(pos)
            s_mask.copy_(make_mask(pos))
            graph.replay()
            cur = s_out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(cur.clone())
    _sync()
    decode_s = time.perf_counter() - t0

    out_ids = torch.cat([ids] + gen, dim=1)
    return out_ids, prefill_s, decode_s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--dtype", choices=list(_DTYPES), default="bf16")
    ap.add_argument(
        "--compile", dest="compile_mode", default="reduce-overhead",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode. reduce-overhead = CUDA graphs (best decode); "
             "max-autotune also autotunes matmuls (slow to compile).",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1,
                    help="number of sequences to generate in parallel "
                         "(the prompt is replicated across the batch).")
    ap.add_argument("--no-triton", action="store_true",
                    help="disable Triton codegen: use the `cudagraphs` backend "
                         "(no Inductor) plus fused ATen ops (F.rms_norm), so only "
                         "interceptable stock kernels run. Overrides --compile.")
    ap.add_argument("--manual-cudagraph", dest="manual", action="store_true",
                    help="Triton-free steelman: hand-capture the decode step with "
                         "torch.cuda.CUDAGraph and replay it per token. Full "
                         "launch-overhead elimination over stock ATen kernels. "
                         "Overrides --compile / --no-triton.")
    ap.add_argument("--no-cudnn-gqa", dest="cudnn_gqa", action="store_false",
                    help="disable the cuDNN grouped-query attention path (used by "
                         "--manual-cudagraph / --no-triton by default). With it ON "
                         "(default) SDPA runs enable_gqa=True on the forced cuDNN "
                         "backend, doing the GQA broadcast inside the kernel with no "
                         "repeat_kv full-length copy (~8x faster attention at batch "
                         "256). Pass this flag to fall back to repeat_kv for A/B.")
    ap.add_argument("--nsys", action="store_true",
                    help="bracket only the final measured generation in the CUDA "
                         "profiler API (torch.cuda.profiler start/stop) so nsys "
                         "`--capture-range=cudaProfilerApi` skips warmup/compile.")
    args = ap.parse_args()

    _tune_backend()
    tok, model = load(args.dtype, args.compile_mode, args.no_triton, args.manual,
                      args.cudnn_gqa)

    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    if args.batch > 1:
        inputs = {k: v.repeat(args.batch, 1) for k, v in inputs.items()}
    batch = inputs["input_ids"].shape[0]
    prompt_tokens = inputs["input_ids"].shape[-1]

    if args.manual:
        print("Capturing decode CUDA graph...")
        out, prefill_s, decode_s = manual_cudagraph_generate(
            model, inputs, args.max_new_tokens, args.warmup, profile=args.nsys)
        gen_tokens = out.shape[-1] - prompt_tokens
        decode_tokens = max(gen_tokens - 1, 1)
    else:
        print("Warming up (compiling graphs)...")
        _warm(model, tok, inputs, 1, args.warmup)                    # prefill graph
        _warm(model, tok, inputs, args.max_new_tokens, args.warmup)  # decode graph

        _, prefill_s = _timed_gen(model, tok, inputs, 1)
        with profile_region(args.nsys, "generate"):
            out, total_s = _timed_gen(model, tok, inputs, args.max_new_tokens)

        gen_tokens = out.shape[-1] - prompt_tokens
        decode_s = max(total_s - prefill_s, 1e-9)
        decode_tokens = max(gen_tokens - 1, 1)

    response = tok.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    print("\n" + "─" * 70)
    print(f"Prompt:   {args.prompt}")
    print(f"Response: {response.strip()}")
    print("─" * 70)
    if args.manual:
        mode_label = "manual-cudagraph(no-triton)"
    elif args.no_triton:
        mode_label = "cudagraphs(no-triton)"
    else:
        mode_label = args.compile_mode
    print(f"dtype: {args.dtype}   compile: {mode_label}   batch: {batch}")
    print(f"Prompt tokens/seq:    {prompt_tokens}")
    print(f"Generated tokens/seq: {gen_tokens}")
    print(f"Prompt   tps: {batch * prompt_tokens / prefill_s:8.1f} tok/s "
          f"(prefill {prefill_s * 1e3:.1f} ms)")
    print(f"Generate tps: {batch * decode_tokens / decode_s:8.1f} tok/s "
          f"(decode {decode_s * 1e3:.1f} ms, "
          f"{decode_tokens / decode_s:.1f} tok/s/seq)")


if __name__ == "__main__":
    main()
