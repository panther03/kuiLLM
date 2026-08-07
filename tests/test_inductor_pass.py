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

from kuipy.inductor import passes, custom_ops, fusion, tracing

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


def test_broadcast_addmm_claimed():
    """nn.Linear emits addmm with a 1-D (broadcast) bias. C is read through a
    virtual layout, which need not be injective, so a stride-0 row axis is
    expressible and the kernel serves it."""
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
    assert "kuiperjit.addmm.default" in _targets(graph)


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


# ---------------------------------------------------------------------------
# Elementwise-into-reduction fusion
# ---------------------------------------------------------------------------

def _fused(graph):
    return [n for n in graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.kuiperjit.hreduce_poly.default]


def test_reduction_absorbs_pre_and_post_maps():
    _need_cuda()

    def fn(x):
        return torch.relu(x).pow(2).sum(dim=-1) / 64

    graph = _fake_graph(fn, torch.randn(4, 64, device="cuda"))
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)

    tgts = _targets(graph)
    assert tgts == ["kuiperjit.hreduce_poly.default"], tgts
    node = _fused(graph)[0]
    assert node.args[0].op == "placeholder"
    assert fusion.decode_maps(node.args[5]) == [
        torch.ops.aten.relu.default, (torch.ops.aten.pow.Tensor_Scalar, 2)]
    assert fusion.decode_maps(node.args[6]) == [(torch.ops.aten.div.Tensor, 64)]


def test_reduction_absorbs_pre_map_only():
    _need_cuda()

    def fn(x):
        return torch.nn.functional.silu(x).sum(dim=1)

    graph = _fake_graph(fn, torch.randn(4, 64, device="cuda"))
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)

    assert _targets(graph) == ["kuiperjit.hreduce_poly.default"]
    node = _fused(graph)[0]
    assert fusion.decode_maps(node.args[5]) == [torch.ops.aten.silu.default]
    assert fusion.decode_maps(node.args[6]) == []


def test_unsupported_map_is_not_fused():
    """``sin`` has no real counterpart, so the approximate reduce kernel cannot
    prove it; the anchor must decline rather than fuse partially."""
    _need_cuda()

    def fn(x):
        return torch.sin(x).sum(dim=-1)

    graph = _fake_graph(fn, torch.randn(4, 64, device="cuda"))
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)
    assert not _fused(graph)
    assert "aten.sin.default" in _targets(graph)


def test_multi_use_map_is_not_absorbed():
    """A node read elsewhere in the graph must stay materialised."""
    _need_cuda()

    def fn(x):
        y = torch.relu(x)
        return y.sum(dim=-1), y

    graph = _fake_graph(fn, torch.randn(4, 64, device="cuda"))
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)
    assert not _fused(graph)
    assert "aten.relu.default" in _targets(graph)


def test_dtype_changing_op_is_not_a_map():
    """Maps run at the tensor's own element type, so a promoting cast is not
    fusable and the reduce is claimed unfused."""
    _need_cuda()

    def fn(x):
        return x.to(torch.float32).sum(dim=-1)

    graph = _fake_graph(fn, torch.randn(4, 64, device="cuda", dtype=torch.bfloat16))
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)
    assert not _fused(graph)


def test_fused_reduction_matches_eager():
    _need_cuda()
    x = torch.randn(8, 128, device="cuda")

    def fn(t):
        return torch.relu(t).pow(2).sum(dim=-1) / 128

    out = torch.ops.kuiperjit.hreduce_poly.default(
        x, "aten.sum.dim_IntList", 1, False, None,
        fusion.encode_maps([torch.ops.aten.relu.default,
                            (torch.ops.aten.pow.Tensor_Scalar, 2)]),
        fusion.encode_maps([(torch.ops.aten.div.Tensor, 128)]))
    torch.testing.assert_close(out, fn(x), rtol=1e-5, atol=1e-5)


def test_rmsnorm_collapses_to_one_reduction():
    """The whole RMSNorm reduction -- square, mean, add epsilon, rsqrt -- is one
    kernel; only the final broadcast multiply is left."""
    _need_cuda()

    def fn(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    graph = _fake_graph(fn, torch.randn(4, 128, device="cuda"))
    passes.clear_fusion_rules()
    passes.KuiperPostGradPass()(graph)

    tgts = _targets(graph)
    assert tgts == ["kuiperjit.hreduce_poly.default", "aten.mul.Tensor"], tgts
    node = _fused(graph)[0]
    assert node.args[1] == "aten.mean.dim" and node.args[3] is True
    assert fusion.decode_maps(node.args[5]) == [
        (torch.ops.aten.pow.Tensor_Scalar, 2)]
    assert fusion.decode_maps(node.args[6]) == [
        (torch.ops.aten.add.Tensor, 1e-6), torch.ops.aten.rsqrt.default]


def test_fused_rmsnorm_matches_eager():
    _need_cuda()
    torch.manual_seed(0)
    x = torch.randn(8, 256, device="cuda")
    out = torch.ops.kuiperjit.hreduce_poly.default(
        x, "aten.mean.dim", 1, True, None,
        fusion.encode_maps([(torch.ops.aten.pow.Tensor_Scalar, 2)]),
        fusion.encode_maps([(torch.ops.aten.add.Tensor, 1e-6),
                            torch.ops.aten.rsqrt.default]))
    ref = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
