import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
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
) -> List[str]:
    """Generate responses for many prompts in one batched forward pass.

    The batch is padded up to a multiple of ``pad_to_multiple``. (default 64)
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

    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

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
        f"[batched generate] {real_n} prompts (padded to {len(padded_prompts)}), "
        f"{new_tokens.shape[-1]} new tokens each → "
        f"{elapsed:.2f}s ({real_token_count/elapsed:.1f} real tok/s, "
        f"{new_token_count/elapsed:.1f} kernel tok/s)"
    )

    return decoded


def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    """Single-prompt convenience wrapper."""
    return generate_batch([prompt], max_new_tokens=max_new_tokens,
                          temperature=temperature, pad_to_multiple=1)[0]


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
    )

    show_n = len(responses) if args.show else min(3, len(responses))
    for i in range(show_n):
        print("\n" + "─" * 70)
        print(f"Prompt {i}:  {prompts[i]}")
        print(f"Response:  {responses[i].strip()}")
    if show_n < len(responses):
        print(f"\n… ({len(responses) - show_n} more, pass --print to see all)")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
