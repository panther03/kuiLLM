"""Correctness tests for the split-K + epilogue SuperGEMM addmm backend.

This backend launches two kernels with a stream synchronization between them,
so it is illegal under CUDA graph capture and must only ever be reached through
an explicit ``impl="supergemm_splitk_epi"``. Both halves of that are checked
here, along with numeric agreement with ``torch.addmm`` at the shapes split-K
exists for (skinny M, large K).

The epilogue lives in pass 2 because ``lincomb`` is affine in the accumulated
value: applying it per split would add ``beta*C`` once per split. The
``beta``-sensitive cases below are what would catch that.

The first run of each new (tiling, split count) compiles a kernel (F* + nvcc,
tens of seconds); later runs hit the cache.
"""
import pytest
import torch

from kuipy import kuiops

aten = torch.ops.aten

_TILE = dict(bm=128, bn=128, bk=32, wm=32, wn=64, skew=8, group=1)


def _need_bf16_tensor_cores():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 tensor cores require sm_80+")


def _rel_fro(out, ref):
    o, r = out.float(), ref.float()
    return ((o - r).norm() / (r.norm() + 1e-6)).item()


def _run(m, k, n, splits, cbcast, alpha, beta):
    impl = kuiops.AddmmImpl()
    torch.manual_seed(0)
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
    C = (torch.randn(n, device="cuda", dtype=torch.bfloat16) if cbcast
         else torch.randn(m, n, device="cuda", dtype=torch.bfloat16))
    spec = impl._spec("supergemm_splitk_epi", dict(**_TILE, splits=splits),
                      torch.bfloat16, torch.float32, torch.bfloat16,
                      cbcast=cbcast)
    out = impl.run(spec, (C, A, B), {"alpha": alpha, "beta": beta})
    ref = torch.addmm(C, A, B, alpha=alpha, beta=beta)
    return out, ref


@pytest.mark.parametrize("m,k,n,splits", [
    (256, 4864, 896, 4),
    (128, 4864, 896, 8),
    (256, 896, 1152, 2),
])
@pytest.mark.parametrize("cbcast", [False, True])
def test_addmm_splitk_epi(m, k, n, splits, cbcast):
    _need_bf16_tensor_cores()
    out, ref = _run(m, k, n, splits, cbcast, 1.0, 1.0)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert _rel_fro(out, ref) < 3e-2


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 2.0), (1.0, 0.0),
                                        (0.0, 1.0)])
def test_addmm_splitk_epi_alpha_beta(alpha, beta):
    """``beta*C`` must be added exactly ONCE, not once per split. With
    ``alpha=0`` the whole answer is ``beta*C``, so a per-split epilogue would
    be off by a factor of ``splits``."""
    _need_bf16_tensor_cores()
    out, ref = _run(256, 4864, 896, 4, False, alpha, beta)
    assert _rel_fro(out, ref) < 3e-2


def test_addmm_splitk_epi_matches_nonsplit():
    """Split-K is an implementation strategy, so it must land on the same
    answer as the non-split epilogue kernel up to fp32 summation order."""
    _need_bf16_tensor_cores()
    impl = kuiops.AddmmImpl()
    torch.manual_seed(0)
    A = torch.randn(256, 4864, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(896, 4864, device="cuda", dtype=torch.bfloat16).t()
    C = torch.randn(256, 896, device="cuda", dtype=torch.bfloat16)
    kw = {"alpha": 0.5, "beta": 2.0}
    split = impl.run(impl._spec("supergemm_splitk_epi",
                                dict(**_TILE, splits=4), torch.bfloat16,
                                torch.float32, torch.bfloat16), (C, A, B), kw)
    plain = impl.run(impl._spec("supergemm", dict(**_TILE), torch.bfloat16,
                                torch.float32, torch.bfloat16), (C, A, B), kw)
    assert _rel_fro(split, plain) < 1e-3


def test_addmm_splitk_epi_is_explicit_only():
    """The stream sync makes this backend illegal under graph capture, so it
    must never be selected unless it was asked for by name."""
    _need_bf16_tensor_cores()
    impl = kuiops.AddmmImpl()
    A = torch.randn(256, 4864, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(896, 4864, device="cuda", dtype=torch.bfloat16).t()
    C = torch.randn(896, device="cuda", dtype=torch.bfloat16)

    spec = impl.supported(aten.addmm.default, (C, A, B), {})
    assert spec is not None and spec["backend"] != "supergemm_splitk_epi"

    spec = impl.supported(aten.addmm.default, (C, A, B),
                          {"impl": "supergemm_splitk_epi"})
    assert spec is not None and spec["backend"] == "supergemm_splitk_epi"
    assert spec["tile"]["splits"] in kuiops._SPLITK_SPLITS
