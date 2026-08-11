"""Correctness tests for the split-K SuperGEMM backend.

Split-K launches two kernels with a stream synchronization between them, so it
is illegal under CUDA graph capture and must only ever be reached through an
explicit ``impl="supergemm_splitk"``. Both halves of that are checked here,
along with numeric agreement with ``torch.mm`` at the shapes split-K exists for
(skinny M, large K).

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


@pytest.mark.parametrize("m,k,n,splits", [
    (256, 4864, 896, 4),
    (128, 4864, 896, 8),
    (256, 896, 1152, 2),
])
def test_mm_supergemm_splitk(m, k, n, splits):
    _need_bf16_tensor_cores()
    impl = kuiops.MmImpl()
    torch.manual_seed(0)
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
    spec = impl._spec("supergemm_splitk", dict(**_TILE, splits=splits),
                      torch.bfloat16, torch.float32, torch.bfloat16)
    out = impl.run(spec, (A, B), {})
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert _rel_fro(out, torch.mm(A, B)) < 3e-2


def test_mm_splitk_matches_nonsplit():
    """Split-K is an implementation strategy, so it must land on the same
    answer as the non-split kernel up to fp32 summation order."""
    _need_bf16_tensor_cores()
    impl = kuiops.MmImpl()
    torch.manual_seed(0)
    A = torch.randn(256, 4864, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(896, 4864, device="cuda", dtype=torch.bfloat16).t()
    split = impl.run(impl._spec("supergemm_splitk", dict(**_TILE, splits=4),
                                torch.bfloat16, torch.float32,
                                torch.bfloat16), (A, B), {})
    plain = impl.run(impl._spec("supergemm", dict(**_TILE),
                                torch.bfloat16, torch.float32,
                                torch.bfloat16), (A, B), {})
    assert _rel_fro(split, plain) < 1e-3


def test_mm_splitk_is_explicit_only():
    """The stream sync makes split-K illegal under graph capture, so it must
    never be selected unless it was asked for by name."""
    _need_bf16_tensor_cores()
    impl = kuiops.MmImpl()
    A = torch.randn(256, 4864, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(896, 4864, device="cuda", dtype=torch.bfloat16).t()

    spec = impl.supported(aten.mm.default, (A, B), {})
    assert spec is not None and spec["backend"] != "supergemm_splitk"

    spec = impl.supported(aten.mm.default, (A, B),
                          {"impl": "supergemm_splitk"})
    assert spec is not None and spec["backend"] == "supergemm_splitk"
    assert spec["tile"]["splits"] in kuiops._SPLITK_SPLITS


def test_mm_splitk_rejects_indivisible_k():
    """``splits * bk`` must divide K: the verified kernel gives every split the
    same k range, and a short split would leave the reduction reading a
    workspace nobody wrote."""
    _need_bf16_tensor_cores()
    impl = kuiops.MmImpl()
    for tile in kuiops._tiles("supergemm_splitk", torch.bfloat16, 1,
                              256, 896, 4864,
                              acc_dtype=torch.float32,
                              out_dtype=torch.bfloat16):
        assert 4864 % (tile["splits"] * tile["bk"]) == 0
    # K = 4800 is divisible by 2 and 4 but 4800/8 = 600 is not a multiple of
    # any legal bk, so the 8-way split must not be offered.
    tiles = kuiops._tiles("supergemm_splitk", torch.bfloat16, 1,
                          256, 896, 4800, acc_dtype=torch.float32,
                          out_dtype=torch.bfloat16)
    for tile in tiles:
        assert 4800 % (tile["splits"] * tile["bk"]) == 0
