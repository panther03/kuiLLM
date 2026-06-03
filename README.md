# kuiLLM × Kuiper

PyTorch inference for Qwen2.5-0.5B with selected CUDA kernels routed
through the verified-by-construction kernels in
[Kuiper](https://github.com/FStarLang/kuiper).

```
kuiLLM/
├── infer.py                  # original PyTorch inference script
├── profile_ops.py            # records CUDA kernel calls to trace.json
├── trace.json                # one captured forward pass (1055 kernel calls)
├── kuiper_ext/               # PyTorch CUDA extension wrapping Kuiper kernels
│   ├── __init__.py           # JIT build entry point
│   ├── integration.py        # KuiperLinear + enable_kuiper(model, ...)
│   └── csrc/ops.cu           # C++/CUDA wrappers + PYBIND module
├── tests/test_kuiper_ops.py  # correctness tests (kuiper vs torch reference)
├── KUIPER_INTEGRATION.md     # per-trace-kernel mapping → hooked / fallback
├── MISSING_KERNELS.md        # what Kuiper still needs (priority-ordered)
└── API_MISMATCHES.md         # every place the wrapper bends reality
```

## Quick start

```bash
# 1. Make sure ninja and python3-dev are available (one-time):
.venv/bin/pip install ninja
sudo apt-get install -y python3.12-dev   # for Python.h

# 2. Run the correctness tests (this also triggers the JIT build):
PYTHONPATH=$PWD .venv/bin/python tests/test_kuiper_ops.py

# 3. Run Qwen2.5 inference with KuiperLinear routing:
PYTHONPATH=$PWD .venv/bin/python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kuiper_ext.integration import enable_kuiper

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.bfloat16, device_map='cuda')
model.eval()

info = enable_kuiper(model, cast_bf16_to_f16=True,
                    linear_filter=lambda n: 'lm_head' not in n)
print(info)
"
```

## What's hooked, what's not

See `KUIPER_INTEGRATION.md` for the per-kernel breakdown. TL;DR:

- ✅ Every `nn.Linear` (168/169) routes through verified
      `Klas_GEMM_TensorCore_g_gemm_f16_f16_64x64x64_16x16x16` when the
      flattened input M divides 64. Falls back otherwise.
- ❌ Flash attention, RMSNorm (fused), RoPE, SwiGLU, KV-cache concat,
      and all bf16 elementwise ops stay on PyTorch. See
      `MISSING_KERNELS.md`.

The bf16 → f16 cast on the matmul inputs costs ~0.05 % of logit
magnitude on average; top-1 next-token agreement is preserved on the
prompts we tested. A verified bf16 GEMM would remove this trade-off and
is the highest-impact addition to Kuiper.
