import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Route compatible nn.Linear layers through verified Kuiper GEMM kernels.
# Set KUIPER=0 to disable (use stock PyTorch). See KUIPER_INTEGRATION.md.
USE_KUIPER = DEVICE == "cuda" and os.environ.get("KUIPER", "1") != "0"

print(f"Loading {MODEL_ID} in bf16 on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
model.eval()

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated\n" if DEVICE == "cuda" else "")

if USE_KUIPER:
    # Build the extension (JIT-compiled on first import) and swap in
    # KuiperLinear for every nn.Linear except the LM head. The LM head
    # output dim (vocab=151936) isn't a multiple of 64, so it would
    # always fall back anyway — skip it to avoid the wasted swap.
    from kuiper_ext.integration import enable_kuiper

    print("Enabling Kuiper-verified GEMM for nn.Linear layers...")
    info = enable_kuiper(
        model,
        cast_bf16_to_f16=False,                      # use verified bf16 GEMM (no input cast)
        linear_filter=lambda qname: "lm_head" not in qname,
    )
    print(f"  → replaced {info['linears_replaced']} nn.Linear modules with KuiperLinear\n")


def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    # Quick smoke test
    prompt = "Explain what a transformer neural network is in two sentences."
    print(f"Prompt: {prompt}\n")
    response = generate(prompt)
    print(f"Response: {response}")
