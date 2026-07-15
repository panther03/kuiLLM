"""Tests for torch.compile execution with Kuiper runtime dispatch."""

import io

import pytest
import torch

import kuipy
import kuipy.config as config
from kuipy.graph import KuiperBackend, fuse_supported_patterns


aten = torch.ops.aten


def _manual_mm_add_graph(bias, A, B):
    graph = torch.fx.Graph()
    c_node = graph.placeholder("C")
    a_node = graph.placeholder("A")
    b_node = graph.placeholder("B")
    mm = graph.call_function(aten.mm.default, (a_node, b_node))
    add = graph.call_function(aten.add.Tensor, (mm, c_node))
    graph.output(add)
    for node, value in zip((c_node, a_node, b_node, mm, add),
                           (bias, A, B, A @ B, A @ B + bias)):
        node.meta["val"] = value
    return torch.fx.GraphModule({}, graph)


def test_kuiper_mode_does_not_disable_dynamo():
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def fn(x):
        return x.sin()

    with kuipy.KuiperMode(use_kuiper=False):
        fn(torch.randn(8))
        fn(torch.randn(8))

    assert counter.frame_count == 1


def test_compiled_graph_reaches_kuiper_dispatch(monkeypatch):
    seen = []

    def dispatch(func, args, kwargs):
        seen.append(func)
        return func(*args, **kwargs) if func is aten.mm.default else None

    monkeypatch.setattr(kuipy, "_jit_dispatch", dispatch)
    monkeypatch.setattr(config, "ENABLE_PRINT_PROFILING", True)
    kuipy.profile_data.clear()
    backend = KuiperBackend()

    @torch.no_grad()
    def fn(A, B):
        return torch.mm(A, B)

    compiled = torch.compile(fn, backend=backend, fullgraph=True)
    A, B = torch.randn(4, 4), torch.randn(4, 4)
    with kuipy.KuiperMode():
        out = compiled(A, B)

    assert torch.allclose(out, torch.mm(A, B))
    assert aten.mm.default in seen
    assert backend.compile_count == 1
    report = io.StringIO()
    kuipy.print_profile_data(report)
    assert "aten.mm.default [kuiper]" in report.getvalue()
    kuipy.profile_data.clear()


def test_supported_mm_add_fuses_to_addmm():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        A = torch.empty(64, 64, device="cuda")
        B = torch.empty(64, 64, device="cuda")
        C = torch.empty(64, 64, device="cuda")
        gm = _manual_mm_add_graph(C, A, B)

    assert fuse_supported_patterns(gm) == 1
    targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    assert targets == [aten.addmm.default]


def test_broadcast_mm_add_is_not_fused():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        A = torch.empty(64, 64, device="cuda")
        B = torch.empty(64, 64, device="cuda")
        C = torch.empty(64, device="cuda")
        gm = _manual_mm_add_graph(C, A, B)

    assert fuse_supported_patterns(gm) == 0


def test_compile_graph_stock_fallback():
    compiled = kuipy.compile_graph(lambda x: torch.cos(x) + 1)
    x = torch.randn(8)
    assert torch.allclose(compiled(x), torch.cos(x) + 1)


def test_tiny_qwen_forward_compiles():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        expected = model(input_ids, use_cache=False).logits
        compiled = kuipy.compile_graph(model.forward, fullgraph=True)
        with kuipy.KuiperMode(use_kuiper=False):
            actual = compiled(input_ids, use_cache=False).logits

    assert torch.allclose(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compiled_kuiper_call_is_profiled(monkeypatch):
    old_strictness = config.JIT_STRICTNESS
    old_profiling = config.ENABLE_PRINT_PROFILING
    config.JIT_STRICTNESS = 1
    config.ENABLE_PRINT_PROFILING = True
    kuipy.profile_data.clear()
    try:
        compiled = kuipy.compile_graph(lambda A, B: torch.mm(A, B))
        A = torch.randn(64, 64, device="cuda")
        B = torch.randn(64, 64, device="cuda")
        with kuipy.KuiperMode(trace=True):
            out = compiled(A, B)
        assert torch.allclose(out, torch.mm(A, B), atol=1e-3, rtol=1e-3)

        report = io.StringIO()
        kuipy.print_profile_data(report)
        assert "aten.mm.default [kuiper]" in report.getvalue()
    finally:
        config.JIT_STRICTNESS = old_strictness
        config.ENABLE_PRINT_PROFILING = old_profiling
        kuipy.profile_data.clear()
