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
from . import passes, tracing
from .passes import register_fusion_rule, clear_fusion_rules

_MODE_ENV = "KUIPY_INDUCTOR_MODE"
_pass = None


def mode():
    """Measurement mode (``KUIPY_INDUCTOR_MODE``): ``all_kuiper`` (default),
    ``fusion_only``, or ``autotune``. Currently informational — the GEMM MVP
    always replaces every supported op."""
    return os.environ.get(_MODE_ENV, "all_kuiper").lower()


def enable():
    global _pass
    if _pass is None:
        _pass = passes.KuiperPostGradPass()
    torch._inductor.config.post_grad_custom_post_pass = _pass
    return _pass


def disable():
    global _pass
    torch._inductor.config.post_grad_custom_post_pass = None
    _pass = None


def is_enabled():
    return torch._inductor.config.post_grad_custom_post_pass is not None
