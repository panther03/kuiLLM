1. GEMM supporting non-divisibility by 64
    - there is a big matmul at the end (LM head) that we are not currently able to run
2. bf16xbf16 -> bf16 gemm ; requires cast for C elements inside kernel, because MMAs dont support f32
3. flashattention

minor ones:
- RoPE
- RMSNorm
- SwiGLU
- Concat
- Lookup