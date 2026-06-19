import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import kuipy

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_BATCH = 64

print(f"Loading {MODEL_ID} in bf16 on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Batched generation requires left padding so all sequences end at the same
# position and decode reads from each row's last index uniformly.
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token    = tokenizer.eos_token

use_flash_attn = os.getenv("USE_FLASH_ATTN", "0") == "1"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
)
model.eval()

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
if DEVICE == "cuda":
    print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated\n")

def _pad_batch_to_multiple(prompts: Sequence[str], multiple: int) -> List[str]:
    """Pad the prompt list with a dummy prompt so len(prompts) is a multiple
    of `multiple`. The padding rows are discarded after generation.
    """
    if multiple <= 1:
        return list(prompts)
    n   = len(prompts)
    pad = (-n) % multiple
    if pad == 0:
        return list(prompts)
    return list(prompts) + ["."] * pad


@torch.no_grad()
def generate_batch(
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    pad_to_multiple: int = DEFAULT_BATCH,
    use_kuiper: bool = True,
    timing: bool = False,
) -> List[str]:
    """Generate responses for many prompts in one batched forward pass.

    The batch is padded up to a multiple of ``pad_to_multiple``. (default 64)

    When ``use_kuiper`` is true (default), the forward pass runs under a
    ``kuiper_ext.KuiperMode`` context, which re-routes eligible aten matmul
    calls (mm / addmm / bmm) to the verified Kuiper CUDA kernels. Tensors
    remain on the regular CUDA device — only the kernel implementation
    changes.

    When ``timing`` is true, the forward pass is wrapped in a torch
    profiler so we can report total time spent inside CUDA kernels as
    well as wall-clock time. Adds a small amount of overhead.
    """
    real_n = len(prompts)
    if real_n == 0:
        return []

    padded_prompts = _pad_batch_to_multiple(prompts, pad_to_multiple)
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in padded_prompts
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,           # left-pad to the longest row
    ).to(DEVICE)

    if kuipy.is_available() and ((use_kuiper and DEVICE == "cuda") or (kuipy.ENABLE_PRINT_PROFILING)):
        kernel_ctx = kuipy.KuiperMode()
        if kuipy.ENABLE_PRINT_PROFILING:
            kernel_ctx.dummy_print_mode = True
            kernel_tag = "torch"
        else:
            kernel_tag = "kuiper"
    else:
        from contextlib import nullcontext
        kernel_ctx = nullcontext()
        kernel_tag = "torch"

    if timing and DEVICE == "cuda":
        from torch.profiler import profile, ProfilerActivity
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        )
    else:
        from contextlib import nullcontext
        prof_ctx = nullcontext()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with prof_ctx as prof, kernel_ctx:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Slice off the prompt portion (same length for every row because of
    # left-padding) and decode each row.
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Drop the synthetic padding rows.
    decoded = decoded[:real_n]

    new_token_count = new_tokens.shape[-1] * len(padded_prompts)
    real_token_count = new_tokens.shape[-1] * real_n
    print(
        f"[batched generate / {kernel_tag}] {real_n} prompts (padded to {len(padded_prompts)}), "
        f"{new_tokens.shape[-1]} new tokens each → "
        f"{elapsed:.2f}s ({real_token_count/elapsed:.1f} real tok/s, "
        f"{new_token_count/elapsed:.1f} kernel tok/s)"
    )

    if timing and DEVICE == "cuda" and prof is not None:
        _report_timing(prof, elapsed)

    return decoded


def _report_timing(prof, wall_s: float) -> None:
    """Print a wall-clock vs. CUDA-kernel timing summary from a profiler run.

    The profiler reports two kinds of rows in `key_averages()`:
    rows with ``device_type == DeviceType.CPU`` are host-side op records
    (e.g. ``aten::mm``) and rows with ``device_type == DeviceType.CUDA``
    are the actual GPU kernel runs. We sum device time only over the CUDA
    rows so we don't double-count a kernel against its launching op.
    """
    from torch.autograd import DeviceType

    cuda_kernel_us = 0
    cuda_memcpy_us = 0
    cpu_op_us = 0

    events = prof.key_averages()
    for ev in events:
        # `self_device_time_total` was renamed from `self_cuda_time_total` in
        # newer torch versions; tolerate both.
        dev_us = getattr(ev, "self_device_time_total", None)
        if dev_us is None:
            dev_us = getattr(ev, "self_cuda_time_total", 0)

        if ev.device_type == DeviceType.CUDA:
            name = (ev.key or "").lower()
            if "memcpy" in name or "memset" in name:
                cuda_memcpy_us += dev_us
            else:
                cuda_kernel_us += dev_us
        else:  # CPU-side op record
            cpu_op_us += ev.self_cpu_time_total

    wall_us = wall_s * 1e6

    def _fmt(us):
        return f"{us/1e3:8.1f} ms" if us < 1e6 else f"{us/1e6:8.3f} s "

    print("[timing]")
    print(f"  wall clock           : {_fmt(wall_us)}")
    print(f"  CUDA kernel time     : {_fmt(cuda_kernel_us)}  "
          f"({100*cuda_kernel_us/max(wall_us,1):5.1f}% of wall)")
    print(f"  CUDA memcpy/memset   : {_fmt(cuda_memcpy_us)}  "
          f"({100*cuda_memcpy_us/max(wall_us,1):5.1f}% of wall)")
    print(f"  CPU op time (sum)    : {_fmt(cpu_op_us)}     (incl. kernel launch overhead)")

    # Top-5 hottest CUDA kernels for a quick sense of where time goes.
    cuda_events = [ev for ev in events if ev.device_type == DeviceType.CUDA]
    hottest = sorted(
        cuda_events,
        key=lambda e: getattr(e, "self_device_time_total",
                              getattr(e, "self_cuda_time_total", 0)),
        reverse=True,
    )[:5]
    if hottest:
        print("  top kernels by CUDA time:")
        for ev in hottest:
            t = getattr(ev, "self_device_time_total",
                        getattr(ev, "self_cuda_time_total", 0))
            name = ev.key
            if len(name) > 80:
                name = name[:77] + "..."
            print(f"    {_fmt(t)}  {name}")


def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
             use_kuiper: bool = True, timing: bool = False) -> str:
    """Single-prompt convenience wrapper."""
    return generate_batch([prompt], max_new_tokens=max_new_tokens,
                          temperature=temperature, pad_to_multiple=1,
                          use_kuiper=use_kuiper, timing=timing)[0]


def _read_prompts(path: Path) -> List[str]:
    """Read one prompt per non-empty line. Lines starting with '#' are skipped."""
    out: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


# A small built-in pool of diverse prompts so a default `python infer.py`
# fills a batch of 64 even with no input file.
_DEFAULT_PROMPT_POOL = [
    "Explain what a transformer neural network is in two sentences.",
    "What is the difference between supervised and unsupervised learning?",
    "Write a one-line haiku about GPUs.",
    "Translate 'good morning' into French and Japanese.",
    "Summarize the plot of Hamlet in three sentences.",
    "What does the Python `yield` keyword do?",
    "Name three planets in our solar system.",
    "What is the capital of Iceland?",
    "Give one tip for writing readable code.",
    "Why is the sky blue? (one sentence)",
    "What is overfitting in machine learning?",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "What does 'CUDA' stand for?",
    "Write a SQL query selecting all rows from a table called 'users'.",
    "Name two famous works by Shakespeare.",
    "What is the speed of light in km/s (approximate)?",
]


def _fill_pool(n: int) -> List[str]:
    pool = _DEFAULT_PROMPT_POOL
    out  = [pool[i % len(pool)] for i in range(n)]
    return out


def _main() -> int:
    ap = argparse.ArgumentParser(description="Batched Qwen2.5 inference")
    ap.add_argument("prompt", nargs="?", default=None,
                    help="Single prompt to run (single-prompt mode). Mutually exclusive with --prompts.")
    ap.add_argument("--prompts", type=Path, default=None,
                    help="File with one prompt per line.")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"Batch size to pad up to (default {DEFAULT_BATCH})")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--print", dest="show", action="store_true",
                    help="Print every batched response (default: only show the first few).")
    ap.add_argument("--no-kuiper", dest="use_kuiper", action="store_false",
                    help="Disable Kuiper kernels and run with stock PyTorch (for A/B comparison).")
    ap.set_defaults(use_kuiper=True)
    ap.add_argument("--timing", action="store_true",
                    help="Print wall-clock and CUDA-kernel timing breakdown "
                         "(uses torch.profiler; small overhead).")
    args = ap.parse_args()

    if args.prompt and args.prompts:
        ap.error("pass either a positional prompt or --prompts FILE, not both")

    if args.prompts is not None:
        prompts = _read_prompts(args.prompts)
        if not prompts:
            print(f"[infer] no prompts found in {args.prompts}", file=sys.stderr)
            return 1
    elif args.prompt is not None:
        # Single-prompt mode: replicate the user's prompt to fill the batch and expose more parallelism to GPU kernels
        prompts = [args.prompt]
    else:
        prompts = _fill_pool(args.batch)
        print(f"[infer] no prompts given — using {len(prompts)} demo prompts.\n")

    pad_to = args.batch if len(prompts) == 1 else args.batch

    responses = generate_batch(
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        pad_to_multiple=pad_to,
        use_kuiper=args.use_kuiper,
        timing=args.timing,
    )

    show_n = len(responses) if args.show else min(3, len(responses))
    for i in range(show_n):
        print("\n" + "─" * 70)
        print(f"Prompt {i}:  {prompts[i]}")
        print(f"Response:  {responses[i].strip()}")
    if show_n < len(responses):
        print(f"\n… ({len(responses) - show_n} more, pass --print to see all)")

    if kuipy.is_available() and kuipy.ENABLE_PRINT_PROFILING:
        with open("kernel_call.log", "w") as f:
            kuipy.print_profile_data(f)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())