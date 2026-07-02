# pytorch qwen speed test 

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "Summarize the plot of Hamlet in 1000 characters."
MAX_NEW_TOKENS = 256

print(f"Loading {MODEL_ID} in bf16 on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=DEVICE,
).eval()

# Compile out of eager mode. A static KV cache keeps the decode step at a
# fixed shape, so torch.compile can capture one graph and reuse it every step.
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, fullgraph=True)


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _gen(inputs, n):
    return model.generate(**inputs, max_new_tokens=n, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)


def _timed_gen(inputs, n):
    _sync()
    t = time.perf_counter()
    out = _gen(inputs, n)
    _sync()
    return out, time.perf_counter() - t


def _warm(inputs, n, rounds=3):
    """torch.compile + the static cache need a couple of passes per generate
    shape before timings stabilize, so run each measured shape a few times."""
    for _ in range(rounds):
        _gen(inputs, n)
    _sync()


@torch.no_grad()
def main():
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    prompt_tokens = inputs["input_ids"].shape[-1]

    print("Warming up (compiling graphs)...")
    _warm(inputs, 1)                 # prefill-only graph
    _warm(inputs, MAX_NEW_TOKENS)    # decode graph

    # Prefill: time to produce the first token.
    _, prefill_s = _timed_gen(inputs, 1)

    # Full generation; decode time is the remainder.
    out, total_s = _timed_gen(inputs, MAX_NEW_TOKENS)

    gen_tokens = out.shape[-1] - prompt_tokens
    decode_s = max(total_s - prefill_s, 1e-9)
    decode_tokens = max(gen_tokens - 1, 1)

    response = tokenizer.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    print("\n" + "─" * 70)
    print(f"Prompt:   {PROMPT}")
    print(f"Response: {response.strip()}")
    print("─" * 70)
    print(f"Prompt tokens:     {prompt_tokens}")
    print(f"Generated tokens:  {gen_tokens}")
    print(f"Prompt   tps: {prompt_tokens / prefill_s:8.1f} tok/s "
          f"(prefill {prefill_s*1e3:.1f} ms)")
    print(f"Generate tps: {decode_tokens / decode_s:8.1f} tok/s "
          f"(decode {decode_s*1e3:.1f} ms)")


if __name__ == "__main__":
    main()
