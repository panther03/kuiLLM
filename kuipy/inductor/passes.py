"""Inductor post-grad pass that swaps supported ATen ops for Kuiper kernels.

Registered as ``torch._inductor.config.post_grad_custom_post_pass``. Inductor
calls it once per graph (after decompositions / functionalization) with the
post-grad ``torch.fx.Graph``; we mutate it in place, replacing every node a
Kuiper kernel can serve (see ``custom_ops.claim``) with the matching
``kuiperjit::*`` custom op. User-registered fusion rules run first, so they can
rewrite patterns (e.g. fold an elementwise map into a reduction's ``pre`` arg)
before the built-in one-to-one GEMM replacement.
"""
import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from . import custom_ops
from . import tracing

# User-registered fusion rules: fn(graph) -> bool (True if it changed anything).
_fusion_rules = []


def register_fusion_rule(fn):
    """Register a custom fusion rule ``fn(graph: torch.fx.Graph) -> bool``.

    Called on every post-grad graph before the built-in GEMM replacement. The
    rule mutates the graph in place and returns whether it changed anything.
    This is the hook for pattern fusions such as folding elementwise ops into a
    reduction's ``pre`` argument."""
    _fusion_rules.append(fn)
    return fn


def clear_fusion_rules():
    _fusion_rules.clear()


def _replace(graph, node, new_target, new_args):
    with graph.inserting_before(node):
        new_node = graph.call_function(new_target, tuple(new_args))
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)
    return new_node


class KuiperPostGradPass(CustomGraphPass):
    """Post-grad pass. With ``replace=True`` (default) it swaps supported ops for
    Kuiper kernels and applies fusion rules. With ``replace=False`` it only walks
    the graph to feed the tracer (marking which ops a Kuiper kernel *could* serve)
    and leaves it untouched — used to build KERNELS.md under ``--no-kuiper``."""

    def __init__(self, replace=True):
        self.replace = replace

    def __call__(self, graph):
        changed = False
        if self.replace:
            for rule in _fusion_rules:
                changed = bool(rule(graph)) or changed

        graph_trace = tracing.begin_graph(graph)
        for node in list(graph.nodes):
            claimed = custom_ops.claim(node)
            tracing.record_node(node, claimed is not None, graph_trace)
            if not self.replace or claimed is None:
                continue
            new_target, new_args = claimed
            _replace(graph, node, new_target, new_args)
            changed = True
        tracing.finish_graph(graph_trace)

        if changed:
            graph.lint()

    def uuid(self):
        files = (custom_ops.__file__, tracing.__file__, __file__)
        return get_hash_for_files(files) + (b":trace" if not self.replace else b"")
