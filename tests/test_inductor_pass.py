"""FX-level tests for the Kuiper Inductor post-grad pass.

These build small post-grad-style FX graphs with FAKE tensors and run
``KuiperPostGradPass`` / ``claim`` directly — no kernel compilation, no CUDA
execution — so they are fast and don't need the F* toolchain. They assert that
supported ops are rewritten to ``kuiperjit::*`` custom ops and that the tracer
records the inventory.
"""
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from kuipy.inductor import passes, custom_ops, tracing

_CUDA = torch.cuda.is_available()


def _need_cuda():
    if not _CUDA:
        pytest.skip("CUDA not available (Kuiper kernels are CUDA-only)")


def _fake_graph(fn, *example_inputs):
    gm = make_fx(fn, tracing_mode="fake")(*example_inputs)
    return gm.graph


def _targets(graph):
    return [str(n.target) for n in graph.nodes if n.op == "call_function"]


def test_mm_and_bmm_replaced_on_cuda():
    _need_cuda()
    dev, dt = "cuda", torch.bfloat16

    def fn(a, b, q, k):
        return torch.mm(a, b), torch.bmm(q, k)

    a = torch.randn(256, 896, device=dev, dtype=dt)
    b = torch.randn(896, 896, device=dev, dtype=dt)
    q = torch.randn(4, 64, 64, device=dev, dtype=dt)
    k = torch.randn(4, 64, 64, device=dev, dtype=dt)
    graph = _fake_graph(fn, a, b, q, k)

    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)
    tgts = _targets(graph)
    assert "kuiperjit.mm.default" in tgts
    assert "kuiperjit.bmm.default" in tgts
    assert "aten.mm.default" not in tgts
    assert "aten.bmm.default" not in tgts


def test_cpu_ops_not_replaced():
    def fn(a, b):
        return torch.mm(a, b)

    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    graph = _fake_graph(fn, a, b)
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)
    assert "kuiperjit.mm.default" not in _targets(graph)
    assert "aten.mm.default" in _targets(graph)


def test_broadcast_addmm_not_claimed():
    """F.linear-style addmm has a 1-D bias (broadcast); the Kuiper addmm kernel
    does not broadcast, so it must be left on cuBLAS."""
    _need_cuda()
    dev, dt = "cuda", torch.bfloat16

    def fn(bias, a, b):
        return torch.addmm(bias, a, b)

    bias = torch.randn(896, device=dev, dtype=dt)  # 1-D -> broadcast
    a = torch.randn(256, 896, device=dev, dtype=dt)
    b = torch.randn(896, 896, device=dev, dtype=dt)
    graph = _fake_graph(fn, bias, a, b)
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)
    assert "kuiperjit.addmm.default" not in _targets(graph)


def test_tracer_records_inventory(tmp_path):
    _need_cuda()
    dev, dt = "cuda", torch.bfloat16

    def fn(a, b):
        return torch.mm(a, b)

    a = torch.randn(256, 896, device=dev, dtype=dt)
    b = torch.randn(896, 896, device=dev, dtype=dt)
    graph = _fake_graph(fn, a, b)

    tracing.reset()
    tracing.set_enabled(True)
    try:
        passes.KuiperPostGradPass()(graph)
    finally:
        tracing.set_enabled(False)

    out = tmp_path / "K.md"
    total, claimed = tracing.dump_markdown(str(out))
    assert total >= 1 and claimed >= 1
    text = out.read_text()
    assert "aten::mm" in text and "yes" in text


def test_tracer_records_dependency_graph(tmp_path):
    def fn(a, b):
        return torch.sin(torch.mm(a, b))

    graph = _fake_graph(fn, torch.randn(8, 8), torch.randn(8, 8))
    tracing.reset()
    tracing.set_enabled(True)
    try:
        passes.KuiperPostGradPass(replace=False)(graph)
    finally:
        tracing.set_enabled(False)

    out = tmp_path / "K.md"
    tracing.dump_markdown(str(out))
    text = out.read_text()
    assert "## Kernel dependency graph" in text
    assert "```mermaid" in text
    assert "aten::mm" in text and "aten::sin" in text
    assert " --> " in text
    assert "classDef kuiper" in text and "classDef fallback" in text


def test_custom_fusion_rule_runs():
    calls = []

    def rule(graph):
        calls.append(len(list(graph.nodes)))
        return False

    passes.clear_fusion_rules()
    passes.register_fusion_rule(rule)

    def fn(a, b):
        return torch.mm(a, b)

    graph = _fake_graph(fn, torch.randn(8, 8), torch.randn(8, 8))
    passes.KuiperPostGradPass()(graph)
    passes.clear_fusion_rules()
    assert calls, "registered fusion rule was not invoked"
