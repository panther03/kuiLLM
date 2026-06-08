"""Correctness tests for kuiper_ext wrappers and the KuiperLinear module.

Run with:  cd /home/julien/work/kuiLLM && .venv/bin/python -m pytest tests/

(Or directly:  PATH=$PWD/.venv/bin:$PATH .venv/bin/python tests/test_kuiper_ops.py)
"""

import math
import sys

import torch
import torch.nn as nn

import kuiper_ext

_DEVICE = "cuda" if torch.cuda.is_available() else None


def _need_cuda():
    if _DEVICE is None:
        print("CUDA not available, skipping")
        sys.exit(0)


def _assert_close(name, got, ref, atol, rtol):
    err = (got - ref).abs().max().item()
    rerr = (got - ref).abs().div(ref.abs().clamp(min=1)).max().item()
    print(f"  {name:30s}  max_abs={err:.3e}  max_rel={rerr:.3e}")
    assert err <= atol or rerr <= rtol, f"{name}: error too large ({err}, {rerr})"


def test_mm_bf16():
    print("[test_mm_bf16]")
    torch.manual_seed(0)
    for M, K, N in [(64, 64, 64), (64, 128, 128), (192, 256, 64)]:
        Ar = torch.randn(M, K, device="cpu", dtype=torch.bfloat16)
        A =  Ar.to(device=_DEVICE)
        Br = torch.randn(K, N, device="cpu", dtype=torch.bfloat16)
        B =  Br.to(device=_DEVICE)

        res = kuiper_ext.mm_bf16xbf16_bf16(A, B)
        _assert_close(f"mm_bf16 {M}x{K}x{N}",
                      res.to(device="cpu"), Ar @ Br,
                      atol=1e-2, rtol=1e-3)

def test_bmm_f32():
    print("[test_bmm_f32]")
    torch.manual_seed(0)
    for batch, M, K, N in [(3, 64, 64, 64), (3, 431, 69, 50), (6, 100, 128, 100)]:
        Ar = torch.randn(batch, M, K, device="cpu", dtype=torch.float32)
        A =  Ar.to(device=_DEVICE)
        Br = torch.randn(batch, K, N, device="cpu", dtype=torch.float32)
        B =  Br.to(device=_DEVICE)

        res = kuiper_ext.bmm_f32xf32_f32(A, B)
        _assert_close(f"bmm_f32 {batch}x{M}x{K}x{N}",
                      res.to(device="cpu"), torch.bmm(Ar, Br),
                      atol=1e-3, rtol=1e-4)


def test_addmm_bf16():
    """Exercise the addmm wrapper directly (raw pybind entry point) and via
    the KuiperMode dispatcher (aten::addmm interception, incl. 1D-bias
    expansion as ships from nn.Linear)."""
    print("[test_addmm_bf16]")
    torch.manual_seed(0)

    # addmm kernel accumulates in pure bf16 (vs `mm` which uses f32), so
    # per-element error grows with K and can be sizable at small ref values.
    # Bound on max-abs / max-magnitude-of-ref (matrix-norm relative error).
    REL_TOL = 0.05  # 5% — matches what we see at K=256

    def _check(name, got, ref):
        diff = (got.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        rel = diff / max(scale, 1e-6)
        print(f"  {name:36s}  max_abs={diff:.3e}  scale={scale:.2e}  rel={rel:.3e}")
        assert rel <= REL_TOL, f"{name}: rel error too large ({rel:.3f})"

    for M, K, N in [(32, 64, 64), (64, 128, 128), (192, 256, 64)]:
        Ar = torch.randn(M, K, device="cpu", dtype=torch.bfloat16)
        Br = torch.randn(K, N, device="cpu", dtype=torch.bfloat16)
        A = Ar.to(device=_DEVICE)
        B = Br.to(device=_DEVICE)

        # --- 2D bias, default beta=alpha=1 -----------------------------------
        Cr_2d = torch.randn(M, N, device="cpu", dtype=torch.bfloat16)
        C_2d = Cr_2d.to(device=_DEVICE)
        ref_2d = torch.addmm(Cr_2d, Ar, Br)
        res_2d = kuiper_ext.addmm_bf16xbf16xbf16_bf16(A, B, C_2d, 1.0, 1.0)
        _check(f"addmm_bf16 2D-bias {M}x{K}x{N}", res_2d.cpu(), ref_2d)

        # --- 2D bias with explicit beta / alpha ------------------------------
        beta, alpha = 0.5, 2.0
        ref_ba = torch.addmm(Cr_2d, Ar, Br, beta=beta, alpha=alpha)
        res_ba = kuiper_ext.addmm_bf16xbf16xbf16_bf16(A, B, C_2d, beta, alpha)
        _check(f"addmm_bf16 b=.5 a=2 {M}x{K}x{N}", res_ba.cpu(), ref_ba)

        # --- 1D bias [N] via KuiperMode (the nn.Linear path) -----------------
        # nn.Linear ships a 1D bias relying on broadcasting; the dispatcher
        # expand()s it to (M, N) before forwarding. Verify that flow here.
        bias1d_r = torch.randn(N, device="cpu", dtype=torch.bfloat16)
        bias1d = bias1d_r.to(device=_DEVICE)
        ref_1d = torch.addmm(bias1d_r, Ar, Br)
        with kuiper_ext.KuiperMode():
            res_1d = torch.addmm(bias1d, A, B)
        _check(f"addmm_bf16 1D-bias mode {M}x{K}x{N}", res_1d.cpu(), ref_1d)

        # --- bias must not be stomped (wrapper clones internally) ------------
        C_keep = C_2d.clone()
        _ = kuiper_ext.addmm_bf16xbf16xbf16_bf16(A, B, C_2d, 1.0, 1.0)
        assert torch.equal(C_2d, C_keep), \
            f"addmm wrapper mutated the bias tensor at {M}x{K}x{N}"


def main():
    _need_cuda()
    print(f"Using device: {_DEVICE}")
    print(f"kuiper_ext available: {kuiper_ext.is_available()}")
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print("ALL OK")


if __name__ == "__main__":
    main()
