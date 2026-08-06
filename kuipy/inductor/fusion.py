"""Greedy fusion of elementwise maps into the ops that can absorb them.

Several Kuiper kernels take a ``pre_map`` applied to each input element and a
``post_map`` applied to the result, at no extra memory traffic. This pass walks
the post-grad graph and, for every *anchor* (an op whose ``kuipy.kuiops`` family
accepts ``pre_map`` / ``post_map`` kwargs), greedily absorbs the elementwise
chain feeding it and the one it feeds.

An elementwise node is fusable when it is unary, or binary with a constant
second operand resolved at compile time, and it neither changes dtype nor
shape. Chains are only followed through single-use nodes, so nothing observable
elsewhere in the graph disappears.

The candidate maps are offered to the family's ``supported()`` as a whole; if it
declines, that anchor is left alone -- no other configuration is tried.

Reductions are the only anchor today; ``register_anchor`` is the hook for the
GEMM and binary-elementwise families.
"""
import json

import torch
from torch.fx import Node

from .. import kuiops
from .custom_ops import _supported

aten = torch.ops.aten

# Elementwise ops a map can be built from, by qualified name (the wire format of
# the fused custom op).
_ELEM_BY_NAME = {str(op): op for op in kuiops.ElementwiseImpl._IMPL}


def _fake(x):
    return x.meta.get("val") if isinstance(x, Node) else x


def encode_maps(entries):
    return json.dumps([[str(e[0]), e[1]] if isinstance(e, tuple) else [str(e), None]
                       for e in entries])


def decode_maps(text):
    return [_ELEM_BY_NAME[name] if c is None else (_ELEM_BY_NAME[name], c)
            for name, c in json.loads(text)]


def _map_of(node):
    """The map list entry (``op`` or ``(op, const)``) for ``node``, if it is
    fusable as a unary map."""
    if node.op != "call_function" or node.target not in kuiops.ElementwiseImpl._IMPL:
        return None
    if node.kwargs or not node.args or not isinstance(node.args[0], Node):
        return None
    if len(node.args) == 1:
        const = None
    elif len(node.args) == 2:
        const = node.args[1]
        if isinstance(const, bool) or not isinstance(const, (int, float)):
            return None
    else:
        return None
    # A map runs inside the kernel at the tensor's own element type, so any op
    # that promotes or reshapes is not one.
    src, out = _fake(node.args[0]), _fake(node)
    if not (isinstance(src, torch.Tensor) and isinstance(out, torch.Tensor)):
        return None
    if src.dtype != out.dtype or src.shape != out.shape:
        return None
    return node.target if const is None else (node.target, const)


def _collect_pre(src, consumed):
    """Walk back from ``src`` through single-use elementwise nodes. Returns the
    first node that is not one, the maps in application order, and the nodes to
    erase."""
    maps, nodes, cur = [], [], src
    while (isinstance(cur, Node) and cur not in consumed and len(cur.users) == 1):
        m = _map_of(cur)
        if m is None:
            break
        maps.append(m)
        nodes.append(cur)
        cur = cur.args[0]
    maps.reverse()
    return cur, maps, nodes


def _collect_post(node, consumed):
    """Walk forward from ``node`` through single-use elementwise consumers."""
    maps, nodes, cur = [], [], node
    while len(cur.users) == 1:
        nxt = next(iter(cur.users))
        if nxt in consumed or not nxt.args or nxt.args[0] is not cur:
            break
        m = _map_of(nxt)
        if m is None:
            break
        maps.append(m)
        nodes.append(nxt)
        cur = nxt
    return cur, maps, nodes


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

_ANCHORS = {}


def register_anchor(anchor):
    """Register a fusion anchor: an object with an ``ops`` tuple of aten
    overloads, an ``input_index`` naming the tensor argument the pre-map chain
    feeds, and ``fuse(node, root, pre, post) -> (target, args) | None``."""
    for op in anchor.ops:
        _ANCHORS[op] = anchor
    return anchor


_REDUCE_OPS = (aten.sum.dim_IntList, aten.prod.dim_int,
               aten.all.dim, aten.any.dim)
REDUCE_BY_NAME = {str(op): op for op in _REDUCE_OPS}


def _reduce_dim(node, rank):
    """The single reduced axis, or ``None`` if this is not a one-axis reduce."""
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dim is None:
        return 0 if rank == 1 else None
    if isinstance(dim, (list, tuple)):
        if len(dim) != 1:
            return None
        dim = dim[0]
    if not isinstance(dim, int):
        return None
    return dim + rank if dim < 0 else dim


@register_anchor
class _ReduceAnchor:
    ops = _REDUCE_OPS
    input_index = 0

    @staticmethod
    def fuse(node, root, pre, post):
        src = _fake(root)
        if not isinstance(src, torch.Tensor):
            return None
        dim = _reduce_dim(node, src.dim())
        if dim is None:
            return None
        keepdim = (node.args[2] if len(node.args) > 2
                   else node.kwargs.get("keepdim", False))
        dtype = node.kwargs.get("dtype")
        dim_arg = [dim] if node.target is aten.sum.dim_IntList else dim
        kwargs = {"pre_map": pre, "post_map": post}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if not _supported(node.target, (src, dim_arg, bool(keepdim)), kwargs):
            return None
        return (torch.ops.kuiperjit.hreduce_poly.default,
                (root, str(node.target), dim, bool(keepdim), dtype,
                 encode_maps(pre), encode_maps(post)))


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

def apply(graph):
    """Fuse elementwise chains into every anchor that accepts them. Returns
    whether the graph changed."""
    changed = False
    consumed = set()
    for node in list(graph.nodes):
        if (node.op != "call_function" or node.target not in _ANCHORS
                or node in consumed):
            continue
        anchor = _ANCHORS[node.target]
        src = node.args[anchor.input_index]
        if not isinstance(src, Node):
            continue
        root, pre, pre_nodes = _collect_pre(src, consumed)
        tail, post, post_nodes = _collect_post(node, consumed)
        if not pre and not post:
            continue
        built = anchor.fuse(node, root, pre, post)
        if built is None:
            continue
        target, args = built
        with graph.inserting_before(node):
            fused = graph.call_function(target, tuple(args))
        fused.meta.update(tail.meta)
        tail.replace_all_uses_with(fused)
        for dead in list(reversed(post_nodes)) + [node] + pre_nodes:
            graph.erase_node(dead)
            consumed.add(dead)
        changed = True
    return changed
