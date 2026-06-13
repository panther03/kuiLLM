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


def test_elementwise_direct_wrappers():
    print("[test_elementwise_direct_wrappers]")
    torch.manual_seed(0)
    X_bf = torch.randn(513, 7, device=_DEVICE, dtype=torch.bfloat16)
    Y_bf = torch.randn(513, 7, device=_DEVICE, dtype=torch.bfloat16)
    X_f = torch.rand(513, 7, device=_DEVICE, dtype=torch.float32) + 0.25
    Y_f = torch.randn(513, 7, device=_DEVICE, dtype=torch.float32)

    _assert_close("silu_bf16", kuiper_ext.silu_bf16(X_bf).float().cpu(),
                  torch.nn.functional.silu(X_bf).float().cpu(), atol=2e-2, rtol=2e-2)
    _assert_close("neg_bf16", kuiper_ext.neg_bf16(X_bf).float().cpu(),
                  (-X_bf).float().cpu(), atol=0, rtol=0)
    _assert_close("add_bf16", kuiper_ext.add_bf16(X_bf, Y_bf).float().cpu(),
                  (X_bf + Y_bf).float().cpu(), atol=0, rtol=0)
    _assert_close("mul_bf16", kuiper_ext.mul_bf16(X_bf, Y_bf).float().cpu(),
                  (X_bf * Y_bf).float().cpu(), atol=0, rtol=0)
    _assert_close("rsqrt_f32", kuiper_ext.rsqrt_f32(X_f).cpu(), torch.rsqrt(X_f).cpu(),
                  atol=1e-6, rtol=1e-6)
    _assert_close("square_f32", kuiper_ext.square_f32(Y_f).cpu(), (Y_f * Y_f).cpu(),
                  atol=1e-6, rtol=1e-6)
    _assert_close("sin_f32", kuiper_ext.sin_f32(Y_f).cpu(), torch.sin(Y_f).cpu(),
                  atol=1e-6, rtol=1e-6)
    _assert_close("cos_f32", kuiper_ext.cos_f32(Y_f).cpu(), torch.cos(Y_f).cpu(),
                  atol=1e-6, rtol=1e-6)
    _assert_close("mul_f32", kuiper_ext.mul_f32(X_f, Y_f).cpu(), (X_f * Y_f).cpu(),
                  atol=1e-6, rtol=1e-6)
    _assert_close("add_const_f32", kuiper_ext.add_const_f32(X_f, 1e-5).cpu(),
                  (X_f + 1e-5).cpu(), atol=1e-6, rtol=1e-6)
    _assert_close("mul_const_f32", kuiper_ext.mul_const_f32(X_f, 0.5).cpu(),
                  (X_f * 0.5).cpu(), atol=1e-6, rtol=1e-6)

    X_keep = X_bf.clone()
    _ = kuiper_ext.silu_bf16(X_bf)
    assert torch.equal(X_bf, X_keep), "elementwise wrapper mutated input"


def test_elementwise_dispatch_mode():
    print("[test_elementwise_dispatch_mode]")
    torch.manual_seed(1)
    X_bf = torch.randn(37, 19, device=_DEVICE, dtype=torch.bfloat16)
    Y_bf = torch.randn(37, 19, device=_DEVICE, dtype=torch.bfloat16)
    X_f = torch.rand(37, 19, device=_DEVICE, dtype=torch.float32) + 0.25
    Y_f = torch.randn(37, 19, device=_DEVICE, dtype=torch.float32)
    eps = torch.tensor(1e-5, device=_DEVICE, dtype=torch.float64)

    with kuiper_ext.KuiperMode():
        got = [
            torch.nn.functional.silu(X_bf),
            torch.neg(X_bf),
            torch.add(X_bf, Y_bf),
            torch.mul(X_bf, Y_bf),
            torch.rsqrt(X_f),
            torch.pow(Y_f, 2),
            torch.cos(Y_f),
            torch.sin(Y_f),
            torch.mul(X_f, Y_f),
            torch.add(X_f, eps),
            torch.mul(X_f, eps),
        ]

    ref = [
        torch.nn.functional.silu(X_bf),
        torch.neg(X_bf),
        torch.add(X_bf, Y_bf),
        torch.mul(X_bf, Y_bf),
        torch.rsqrt(X_f),
        torch.pow(Y_f, 2),
        torch.cos(Y_f),
        torch.sin(Y_f),
        torch.mul(X_f, Y_f),
        torch.add(X_f, eps),
        torch.mul(X_f, eps),
    ]
    names = ["silu", "neg", "add_bf16", "mul_bf16", "rsqrt", "pow2",
             "cos", "sin", "mul_f32", "add_const", "mul_const"]
    for name, g, r in zip(names, got, ref):
        _assert_close(f"dispatch {name}", g.float().cpu(), r.float().cpu(),
                      atol=2e-2 if g.dtype == torch.bfloat16 else 1e-6,
                      rtol=2e-2 if g.dtype == torch.bfloat16 else 1e-6)


def test_mean_f32_lastdim():
    print("[test_mean_f32_lastdim]")
    torch.manual_seed(0)
    for shape in [(4, 16), (3, 5, 3584), (2, 7, 32)]:
        Xr = torch.randn(*shape, device="cpu", dtype=torch.float32)
        X = Xr.to(device=_DEVICE).contiguous()
        ref = Xr.mean(dim=-1, keepdim=True)

        direct = kuiper_ext.mean_f32_lastdim(X.reshape(-1, shape[-1])).reshape(*shape[:-1], 1)
        _assert_close(f"mean_f32 direct {shape}", direct.cpu(), ref, atol=1e-5, rtol=1e-5)

        with kuiper_ext.KuiperMode():
            res = X.mean(dim=-1, keepdim=True)
        _assert_close(f"mean_f32 mode {shape}", res.cpu(), ref, atol=1e-5, rtol=1e-5)
def test_cat_cast_kernels():
    print("[test_cat_cast_kernels]")
    torch.manual_seed(0)

    a = torch.randn(48, device=_DEVICE, dtype=torch.bfloat16).contiguous()
    b = torch.randn(48, device=_DEVICE, dtype=torch.bfloat16).contiguous()
    got = kuiper_ext.cat2_bf16(a, b)
    assert torch.equal(got.cpu(), torch.cat([a.cpu(), b.cpu()], dim=0))

    x_bf16 = torch.randn(4, 17, device=_DEVICE, dtype=torch.bfloat16).contiguous()
    got_f32 = kuiper_ext.cast_bf16_to_f32(x_bf16)
    assert got_f32.dtype == torch.float32
    assert torch.equal(got_f32.cpu(), x_bf16.float().cpu())

    x_f32 = torch.randn(4, 17, device=_DEVICE, dtype=torch.float32).contiguous()
    got_bf16 = kuiper_ext.cast_f32_to_bf16(x_f32)
    assert got_bf16.dtype == torch.bfloat16
    assert torch.equal(got_bf16.cpu(), x_f32.to(torch.bfloat16).cpu())

    copied = kuiper_ext.cast_bf16_to_bf16(x_bf16)
    assert copied.dtype == torch.bfloat16
    assert copied.data_ptr() != x_bf16.data_ptr()
    assert torch.equal(copied.cpu(), x_bf16.cpu())

    with kuiper_ext.KuiperMode():
        mode_cat = torch.cat([a, b], dim=0)
        mode_f32 = x_bf16.to(torch.float32)
        mode_bf16 = x_f32.to(torch.bfloat16)
    assert torch.equal(mode_cat.cpu(), torch.cat([a.cpu(), b.cpu()], dim=0))
    assert torch.equal(mode_f32.cpu(), x_bf16.float().cpu())
    assert torch.equal(mode_bf16.cpu(), x_f32.cpu().to(torch.bfloat16))
def test_misc_arange_gather():
    print("[test_misc_arange_gather]")
    for n in [1, 17, 4096]:
        got = kuiper_ext.arange_i64(n, 0, 1).cpu()
        ref = torch.arange(n, dtype=torch.int64)
        assert torch.equal(got, ref)

    # arange with non-zero start and step
    got = kuiper_ext.arange_i64(8, 5, 3).cpu()
    ref = torch.arange(5, 5 + 8 * 3, 3, dtype=torch.int64)
    assert torch.equal(got, ref)

    src_r = torch.randn(1024, device="cpu", dtype=torch.bfloat16)
    idx_r = torch.tensor([0, 7, 13, 1023, 511, 4, 4, 900], dtype=torch.int64)
    src = src_r.to(device=_DEVICE)
    idx = idx_r.to(device=_DEVICE)
    got = kuiper_ext.gather_bf16(src, idx).cpu()
    assert torch.equal(got, src_r[idx_r])

    with kuiper_ext.KuiperMode():
        got_mode = torch.gather(src, 0, idx).cpu()
    assert torch.equal(got_mode, src_r[idx_r])


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
