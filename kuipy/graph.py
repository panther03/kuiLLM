"""torch.compile backend for Kuiper-dispatched FX graphs."""

import torch

from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.debugging import boxed_nop
from torch.fx import Node

from .kuiops import AddmmImpl


_ADDMM = AddmmImpl({})


def _value(arg):
    if isinstance(arg, Node):
        return arg.meta.get("val")
    return arg


def fuse_supported_patterns(gm):
    """Apply graph rewrites whose fused operator has a Kuiper implementation."""
    aten = torch.ops.aten
    count = 0

    for add in list(gm.graph.nodes):
        if add.op != "call_function" or add.target is not aten.add.Tensor:
            continue
        if len(add.args) != 2 or add.kwargs.get("alpha", 1) != 1:
            continue

        lhs, rhs = add.args
        mm, bias = (lhs, rhs) if isinstance(lhs, Node) and lhs.target is aten.mm.default else (rhs, lhs)
        if not isinstance(mm, Node) or mm.op != "call_function" or mm.target is not aten.mm.default:
            continue
        if len(mm.users) != 1 or len(mm.args) != 2:
            continue

        A, B = mm.args
        fake_args = (_value(bias), _value(A), _value(B))
        if any(x is None for x in fake_args):
            continue
        try:
            supported = _ADDMM.supported(aten.addmm.default, fake_args, {})
        except (TypeError, ValueError, RuntimeError):
            supported = None
        if supported is None:
            continue

        with gm.graph.inserting_before(add):
            fused = gm.graph.call_function(aten.addmm.default, (bias, A, B))
        fused.meta = add.meta.copy()
        add.replace_all_uses_with(fused)
        gm.graph.erase_node(add)
        gm.graph.erase_node(mm)
        count += 1

    if count:
        gm.graph.lint()
        gm.recompile()
    return count


class KuiperBackend:
    """AOT backend that preserves ATen calls for runtime Kuiper dispatch."""

    def __init__(self):
        self.compile_count = 0
        self.fusion_count = 0
        self.graphs = []
        self._aot = aot_autograd(
            fw_compiler=self._compile_forward,
            bw_compiler=boxed_nop,
            decompositions=None,
            keep_inference_input_mutations=True,
        )

    def _compile_forward(self, gm, example_inputs):
        self.compile_count += 1
        self.fusion_count += fuse_supported_patterns(gm)
        self.graphs.append(gm)
        return boxed_nop(gm, example_inputs)

    def __call__(self, gm, example_inputs):
        return self._aot(gm, example_inputs)


@register_backend(name="kuiper")
def kuiper_backend(gm, example_inputs):
    return KuiperBackend()(gm, example_inputs)


def compile_graph(fn, *, fullgraph=True, dynamic=None, options=None):
    """Compile ``fn`` without Inductor/Triton while retaining ATen dispatch."""
    backend = KuiperBackend()
    compiled = torch.compile(
        fn,
        backend=backend,
        fullgraph=fullgraph,
        dynamic=dynamic,
        options=options,
    )
    compiled._kuiper_backend = backend
    return compiled
