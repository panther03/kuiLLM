"""Registry mapping aten ops to JIT Kuiper implementations + dispatch entry point.

The hot path is intentionally tiny: a dict lookup, a cheap ``supported()`` check,
and (on a hit) a memoised compiled-kernel call. On any miss it returns ``None`` so
the caller falls through to stock PyTorch.
"""
from .gemm import GemmImpl

_GEMM = GemmImpl()

# Lazily populated on first use (needs torch.ops.aten).
_REGISTRY = None


def _build_registry():
    import torch
    aten = torch.ops.aten
    return {
        aten.mm.default: _GEMM,
        aten.addmm.default: _GEMM,
    }


def try_dispatch(func, args, kwargs):
    """Return a result Tensor if a JIT Kuiper kernel handles ``func``, else None."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    impl = _REGISTRY.get(func)
    if impl is None:
        return None
    spec = impl.supported(func, args, kwargs)
    if spec is None:
        return None
    return impl.run(spec, args, kwargs)
