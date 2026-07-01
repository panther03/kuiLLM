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


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    prompt_tokens = inputs["input_ids"].shape[-1]

    # Warmup so CUDA lazy init / caching doesn't skew the timings.
    model.generate(**inputs, max_new_tokens=4, do_sample=False,
                   pad_token_id=tokenizer.eos_token_id)
    _sync()

    # Prefill: time to produce the first token.
    _sync()
    t0 = time.perf_counter()
    first = model.generate(
        **inputs, max_new_tokens=1, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    _sync()
    prefill_s = time.perf_counter() - t0

    # Full generation; decode time is the remainder.
    _sync()
    t1 = time.perf_counter()
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    _sync()
    total_s = time.perf_counter() - t1

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
