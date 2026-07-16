"""Hyper-optimized *pure PyTorch* Qwen2.5-0.5B inference (steelman baseline).

The fastest honest "we ran Qwen2.5 in PyTorch with its own kernels" run we can
build. No Kuiper, no kuipy dispatch, no Triton codegen. transformers is used only
to load the checkpoint + tokenizer; the forward pass is a hand-written PyTorch
model that calls stock ATen kernels directly (``F.linear`` -> cuBLAS,
``F.scaled_dot_product_attention`` -> cuDNN, ``F.rms_norm``, ``F.silu``, ...).
Hardcoded to the production config (batch 256, bf16); single code path, no flags.

Why hand-written instead of transformers eager/compile? The GEMMs (~72% of decode)
and the cuDNN attention (~11%) are already optimal via stock kernels -- the win is
in the other ~17%, which HF spends on avoidable work. Writing the loop ourselves
lets us:

  * **Fuse q/k/v into one GEMM** (the tiny k/v projections become one large
    tensor-core GEMM). gate/up are deliberately *not* fused: the strided SiLU/mul
    on a split view falls back to a slow non-vectorized kernel that costs more
    than the GEMM launch it saves (measured).
  * **cuDNN grouped-query attention, no ``repeat_kv``.** enable_gqa=True on a
    forced cuDNN SDPA backend does the 7x KV broadcast inside the kernel; cuDNN is
    the only fused backend that accepts GQA + an additive mask, and it must be
    forced (the dispatcher would silently fall back to the slow math backend).
  * **Precomputed RoPE tables.** cos/sin for every position are built once and
    indexed, instead of recomputing inv_freq @ pos + cos + sin every step/layer.
  * **A tight static KV cache** written in place, and one attention-output copy
    instead of HF's two ``.contiguous()`` calls.
  * **A fully self-contained decode CUDA graph.** The captured step advances the
    position, writes the KV slot, runs the argmax, feeds the next token back, and
    appends to the output buffer -- so the decode loop is literally ``replay()``
    with zero per-token host/launch overhead.
"""

import argparse
import contextlib
import time
import types

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PROMPT = "Summarize the plot of Hamlet in 1000 characters."
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH = 256
DTYPE = torch.bfloat16


@contextlib.contextmanager
def profile_region(active, label="decode"):
    """Bracket only the measured decode for nsys ``--capture-range=cudaProfilerApi``."""
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


def _tune_backend():
    if DEVICE != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_cudnn_sdp(True)


class Qwen2Fast:
    """Hand-written Qwen2 forward over the HF-loaded weights (stock ATen kernels)."""

    def __init__(self, hf):
        cfg = hf.config
        self.NL = cfg.num_hidden_layers
        self.NH = cfg.num_attention_heads
        self.NKV = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.HID = cfg.hidden_size
        self.EPS = cfg.rms_norm_eps
        self.SCALE = self.D ** -0.5
        self.QO, self.KO, self.VO = self.NH * self.D, self.NKV * self.D, self.NKV * self.D
        theta = cfg.rope_parameters["rope_theta"]

        m = hf.model
        self.embed = m.embed_tokens.weight
        self.lm_head = hf.lm_head.weight
        self.final_norm = m.norm.weight
        self.layers = []
        for l in m.layers:
            a = l.self_attn
            self.layers.append(types.SimpleNamespace(
                in_ln=l.input_layernorm.weight,
                # q/k/v fused into one GEMM (k/v alone are tiny and inefficient).
                qkv_w=torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous(),
                qkv_b=torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], 0).contiguous(),
                o_w=a.o_proj.weight,
                post_ln=l.post_attention_layernorm.weight,
                gate_w=l.mlp.gate_proj.weight, up_w=l.mlp.up_proj.weight,
                down_w=l.mlp.down_proj.weight,
            ))
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.D, 2, device=DEVICE, dtype=torch.float) / self.D))

    def _rope_tables(self, n):
        t = torch.arange(n, device=DEVICE, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(DTYPE), emb.sin().to(DTYPE)

    def _rot_half(self, x):
        d = self.D // 2
        return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

    def _block(self, h, cos, sin, kc, vc, lyr, mask, is_causal, write_pos):
        b, s, _ = h.shape
        r = h
        hn = F.rms_norm(h, (self.HID,), lyr.in_ln, self.EPS)
        q, k, v = F.linear(hn, lyr.qkv_w, lyr.qkv_b).split([self.QO, self.KO, self.VO], dim=-1)
        q = q.view(b, s, self.NH, self.D).transpose(1, 2)
        k = k.view(b, s, self.NKV, self.D).transpose(1, 2)
        v = v.view(b, s, self.NKV, self.D).transpose(1, 2)
        q = q * cos + self._rot_half(q) * sin
        k = k * cos + self._rot_half(k) * sin
        if write_pos is None:                       # prefill: fresh k/v are the cache
            kc[:, :, :s, :] = k
            vc[:, :, :s, :] = v
            ka, va = k, v
        else:                                       # decode: write one slot, attend all
            kc.index_copy_(2, write_pos, k)
            vc.index_copy_(2, write_pos, v)
            ka, va = kc, vc
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            ao = F.scaled_dot_product_attention(
                q, ka, va, attn_mask=mask, scale=self.SCALE,
                is_causal=is_causal, enable_gqa=True)
        ao = ao.transpose(1, 2).reshape(b, s, self.HID)
        h = r + F.linear(ao, lyr.o_w)
        r = h
        hn = F.rms_norm(h, (self.HID,), lyr.post_ln, self.EPS)
        g = F.silu(F.linear(hn, lyr.gate_w)) * F.linear(hn, lyr.up_w)
        return r + F.linear(g, lyr.down_w)

    @torch.inference_mode()
    def generate(self, ids, n, warmup=3, profile=False):
        b, plen = ids.shape
        total = plen + n + 8
        kc = [torch.zeros(b, self.NKV, total, self.D, device=DEVICE, dtype=DTYPE) for _ in range(self.NL)]
        vc = [torch.zeros(b, self.NKV, total, self.D, device=DEVICE, dtype=DTYPE) for _ in range(self.NL)]
        cos_t, sin_t = self._rope_tables(total)
        kp = torch.arange(total, device=DEVICE)
        neg = torch.finfo(DTYPE).min
        # mask_rows[p] is the decode additive mask at position p (allow keys 0..p).
        mask_rows = torch.where(
            kp[None, :] <= kp[:, None],
            torch.zeros((), dtype=DTYPE, device=DEVICE),
            torch.full((), neg, dtype=DTYPE, device=DEVICE),
        )

        # --- Prefill (throwaway then timed; excludes one-time kernel load/autotune) ---
        def prefill():
            h = F.embedding(ids, self.embed)
            cos = cos_t[:plen].view(1, 1, plen, self.D)
            sin = sin_t[:plen].view(1, 1, plen, self.D)
            for li in range(self.NL):
                h = self._block(h, cos, sin, kc[li], vc[li], self.layers[li], None, True, None)
            h = F.rms_norm(h, (self.HID,), self.final_norm, self.EPS)
            return F.linear(h[:, -1:], self.lm_head).argmax(-1)  # [b, 1]

        prefill()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first = prefill()
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        # --- Fully self-contained decode step (captured once, replayed) ---
        tok_in = first.clone()
        pos = torch.tensor([plen], device=DEVICE)
        step = torch.zeros(1, dtype=torch.long, device=DEVICE)
        out_buf = torch.zeros(b, n, dtype=torch.long, device=DEVICE)

        def step_body():
            cos = cos_t.index_select(0, pos).view(1, 1, 1, self.D)
            sin = sin_t.index_select(0, pos).view(1, 1, 1, self.D)
            mask = mask_rows.index_select(0, pos).view(1, 1, 1, total)
            h = F.embedding(tok_in, self.embed)
            for li in range(self.NL):
                h = self._block(h, cos, sin, kc[li], vc[li], self.layers[li], mask, False, pos)
            h = F.rms_norm(h, (self.HID,), self.final_norm, self.EPS)
            nxt = F.linear(h, self.lm_head).argmax(-1)  # [b, 1]
            out_buf.index_copy_(1, step, nxt)
            tok_in.copy_(nxt)
            pos.add_(1)
            step.add_(1)

        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(warmup):
                step_body()
        torch.cuda.current_stream().wait_stream(warm_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step_body()

        # Undo the warmup/capture advances (their KV writes are at slots >= plen,
        # which real replays overwrite before ever attending to them).
        pos.fill_(plen)
        step.zero_()
        tok_in.copy_(first)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile_region(profile, "decode"):
            for _ in range(n - 1):
                graph.replay()
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0

        out_ids = torch.cat([ids, first, out_buf[:, :n - 1]], dim=1)
        return out_ids, prefill_s, decode_s


def load():
    print(f"Loading {MODEL_ID} in bf16 on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE,
    ).eval()
    return tok, Qwen2Fast(hf)


def main():
    ap = argparse.ArgumentParser(description="Hyper-optimized PyTorch Qwen2.5 steelman.")
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--nsys", action="store_true",
                    help="bracket only the measured decode in the CUDA profiler API "
                         "so nsys --capture-range=cudaProfilerApi skips warmup/capture.")
    args = ap.parse_args()

    _tune_backend()
    tok, model = load()

    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt").to(DEVICE)
    ids = enc["input_ids"]
    if args.batch > 1:
        ids = ids.repeat(args.batch, 1)
    batch, prompt_tokens = ids.shape

    print("Capturing decode CUDA graph...")
    out, prefill_s, decode_s = model.generate(
        ids, args.max_new_tokens, args.warmup, profile=args.nsys)
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
