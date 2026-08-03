"""Kuiper JIT kernels exposed as opaque ``torch.library`` custom ops.

Each supported ATen op (currently the GEMM family ``mm`` / ``addmm`` / ``bmm``
and efficient ``sdpa``) has a ``kuiperjit::*`` custom op that runs the verified
Kuiper kernel via the corresponding ``kuipy.kuiops`` Impl. Inductor treats these
as opaque externs: it schedules them, plans their outputs, and captures them into
``reduce-overhead`` CUDA graphs (the kernels launch on the current stream, so they
are capture-safe).

The ``claim`` helper decides, for a post-grad FX node, whether a Kuiper kernel
can serve it — shared by the replacement pass (``passes.py``) and the tracer
(``tracing.py``) so both agree on coverage.
"""
import math
from typing import Tuple

import torch
from torch import Tensor

from .. import registry
from .. import verify as V
from .. import compile as _compile

aten = torch.ops.aten

_EFFICIENT_SDPA = aten._scaled_dot_product_efficient_attention.default
_CUDNN_SDPA = aten._scaled_dot_product_cudnn_attention.default


def _run_impl(func, args, kwargs):
    """Run a kuiops Impl for a node the pass already claimed at compile time.

    A batch capture (``kuipy.batch_capture``) extracts + queues the kernel and
    signals ``CaptureDeferred``; we then run the reference ATen op so the warm-up
    pass still produces correct values while every kernel is collected for one
    combined build. When ``verify`` is active the reference is compared to the
    Kuiper output (host-side norm — eager, non-captured pass only)."""
    impl = registry.impl_for(func)
    spec = impl.supported(func, args, kwargs)
    if spec is None:
        return func(*args, **kwargs)
    try:
        out = impl.run(spec, args, kwargs)
    except _compile.CaptureDeferred:
        return func(*args, **kwargs)
    if V.enabled:
        V.compare(str(func), out, func(*args, **kwargs))
    return out


# ---------------------------------------------------------------------------
# Custom op definitions
# ---------------------------------------------------------------------------

@torch.library.custom_op("kuiperjit::mm", mutates_args=())
def mm(a: Tensor, b: Tensor) -> Tensor:
    return _run_impl(aten.mm.default, (a, b), {})


@mm.register_fake
def _mm_fake(a, b):
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op("kuiperjit::addmm", mutates_args=())
def addmm(bias: Tensor, a: Tensor, b: Tensor, beta: float, alpha: float) -> Tensor:
    return _run_impl(aten.addmm.default, (bias, a, b),
                     {"beta": beta, "alpha": alpha})


@addmm.register_fake
def _addmm_fake(bias, a, b, beta, alpha):
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op("kuiperjit::bmm", mutates_args=())
def bmm(a: Tensor, b: Tensor) -> Tensor:
    return _run_impl(aten.bmm.default, (a, b), {})


@bmm.register_fake
def _bmm_fake(a, b):
    return a.new_empty((a.shape[0], a.shape[1], b.shape[2]))


@torch.library.custom_op("kuiperjit::sdpa", mutates_args=())
def sdpa(q: Tensor, k: Tensor, v: Tensor, bias: Tensor,
         scale: float, causal: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    args = (q, k, v, bias, False, 0.0, causal)
    out = _run_impl(_EFFICIENT_SDPA, args, {"scale": scale})
    return tuple(out)


@sdpa.register_fake
def _sdpa_fake(q, k, v, bias, scale, causal):
    N, H, L, _ = q.shape
    Ev = v.shape[3]
    out = q.new_empty((N, H, L, Ev))
    lse = q.new_empty((N, H, 0), dtype=torch.float32)
    empty = q.new_empty((0,), dtype=torch.int64)
    return out, lse, empty, empty


@torch.library.custom_op("kuiperjit::sdpa_cudnn", mutates_args=())
def sdpa_cudnn(q: Tensor, k: Tensor, v: Tensor, bias: Tensor, scale: float,
               causal: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int,
                                      Tensor, Tensor, Tensor]:
    args = (q, k, v, bias, False, 0.0, causal, False)
    return tuple(_run_impl(_CUDNN_SDPA, args, {"scale": scale}))


@sdpa_cudnn.register_fake
def _sdpa_cudnn_fake(q, k, v, bias, scale, causal):
    N, H, L, D = q.shape
    e0 = q.new_empty((0,), dtype=torch.int64)
    empty = q.new_empty((), dtype=torch.int64)
    return (q.new_empty((N, H, L, D)),
            q.new_empty((N, H, 0), dtype=torch.float32),
            e0, e0, 0, 0, empty, empty, q.new_empty((0,)))


# ---------------------------------------------------------------------------
# Claim decision (shared by the pass and the tracer)
# ---------------------------------------------------------------------------

def _fake(x):
    from torch.fx import Node
    return x.meta.get("val") if isinstance(x, Node) else x


def _supported(func, args, kwargs):
    impl = registry.impl_for(func)
    # Non-graph-safe families sync inside the kernel, which is illegal under CUDA
    # graph capture; never claim them from the compiled graph (they stay callable
    # directly, e.g. from the unit tests).
    if not impl.graph_safe:
        return False
    try:
        return impl.supported(func, args, kwargs) is not None
    except (TypeError, ValueError, RuntimeError, KeyError):
        return False


def claim(node):
    """If ``node`` (a post-grad FX ``call_function``) can be served by a Kuiper
    kernel, return ``(custom_op_overload, new_args)`` to replace it with; else
    ``None``. Pure inspection — no graph mutation."""
    if node.op != "call_function":
        return None
    t = node.target

    if t is aten.mm.default:
        a, b = node.args
        if _supported(aten.mm.default, (_fake(a), _fake(b)), {}):
            return (torch.ops.kuiperjit.mm.default, (a, b))

    elif t is aten.bmm.default:
        a, b = node.args
        if _supported(aten.bmm.default, (_fake(a), _fake(b)), {}):
            return (torch.ops.kuiperjit.bmm.default, (a, b))

    elif t is aten.addmm.default:
        bias, a, b = node.args[:3]
        beta = float(node.kwargs.get("beta", 1))
        alpha = float(node.kwargs.get("alpha", 1))
        kw = {"beta": beta, "alpha": alpha}
        if _supported(aten.addmm.default, (_fake(bias), _fake(a), _fake(b)), kw):
            return (torch.ops.kuiperjit.addmm.default, (bias, a, b, beta, alpha))

    elif t is _EFFICIENT_SDPA or t is _CUDNN_SDPA:
        q, k, v, bias = node.args[:4]
        compute_lse = node.args[4] if len(node.args) > 4 else False
        causal = bool(node.args[6]) if len(node.args) > 6 else False
        scale = node.kwargs.get("scale")
        sargs = (_fake(q), _fake(k), _fake(v), _fake(bias),
                 compute_lse, 0.0, causal, False)
        if _supported(t, sargs, {"scale": scale}):
            fq = _fake(q)
            if scale is None and fq is not None:
                scale = 1.0 / math.sqrt(int(fq.shape[-1]))
            op = (torch.ops.kuiperjit.sdpa.default if t is _EFFICIENT_SDPA
                  else torch.ops.kuiperjit.sdpa_cudnn.default)
            return (op, (q, k, v, bias, float(scale), causal))

    return None
