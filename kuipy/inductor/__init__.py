"""Kuiper Inductor backend: hook verified GPU kernels into ``torch.compile``.

``enable()`` installs a post-grad custom pass (``passes.KuiperPostGradPass``)
that rewrites supported ATen ops to opaque ``kuiperjit::*`` custom ops backed by
verified Kuiper kernels. ``disable()`` removes it. Importing this package
registers the custom ops (via ``custom_ops``) but installs no pass until
``enable()`` is called, so ``--no-kuiper`` runs are byte-identical to stock
``torch.compile``.
"""
import os

import torch

from . import custom_ops  # noqa: F401  (registers kuiperjit::* ops on import)
from . import fusion, passes, tracing
from .fusion import register_anchor
from .passes import register_fusion_rule, clear_fusion_rules

_MODE_ENV = "KUIPY_INDUCTOR_MODE"
_pass = None


def mode():
    """Measurement mode (``KUIPY_INDUCTOR_MODE``): ``all_kuiper`` (default),
    ``fusion_only``, or ``autotune``. Currently informational — the GEMM MVP
    always replaces every supported op."""
    return os.environ.get(_MODE_ENV, "all_kuiper").lower()


def enable(replace=True):
    """Install the post-grad pass. ``replace=True`` rewrites supported ops to
    Kuiper kernels; ``replace=False`` installs a trace-only pass that leaves the
    graph untouched (so ``--no-kuiper`` output is unchanged) but still feeds the
    tracer for ``--dump-kernels``."""
    global _pass
    if _pass is None or _pass.replace != replace:
        _pass = passes.KuiperPostGradPass(replace=replace)
    torch._inductor.config.post_grad_custom_post_pass = _pass
    return _pass


def enable_tracing():
    """Install a trace-only pass (no op replacement) to populate KERNELS.md."""
    return enable(replace=False)


def disable():
    global _pass
    torch._inductor.config.post_grad_custom_post_pass = None
    _pass = None


def is_enabled():
    return torch._inductor.config.post_grad_custom_post_pass is not None
