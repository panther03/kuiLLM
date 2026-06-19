"""Registry mapping aten ops to JIT Kuiper implementations + dispatch entry point.

The hot path is intentionally tiny: a dict lookup, a cheap ``supported()`` check,
and (on a hit) a memoised compiled-kernel call. On any miss it returns ``None`` so
the caller falls through to stock PyTorch.
"""
from .gemm import GemmImpl
from .kuiops import (ElementwiseImpl, ReduceImpl, CatCastImpl,
                       ArangeImpl, GatherImpl, BmmImpl)

_GEMM = GemmImpl()
_ELEM = ElementwiseImpl()
_REDUCE = ReduceImpl()
_CATCAST = CatCastImpl()
_ARANGE = ArangeImpl()
_GATHER = GatherImpl()
_BMM = BmmImpl()

# Lazily populated on first use (needs torch.ops.aten).
_REGISTRY = None


def _build_registry():
    import torch
    aten = torch.ops.aten
    return {
        aten.mm.default: _GEMM,
        aten.addmm.default: _GEMM,
        aten.bmm.default: _BMM,
        aten.silu.default: _ELEM,
        aten.neg.default: _ELEM,
        aten.rsqrt.default: _ELEM,
        aten.cos.default: _ELEM,
        aten.sin.default: _ELEM,
        aten.pow.Tensor_Scalar: _ELEM,
        aten.add.Tensor: _ELEM,
        aten.add.Scalar: _ELEM,
        aten.mul.Tensor: _ELEM,
        aten.mul.Scalar: _ELEM,
        aten.mean.dim: _REDUCE,
        aten.cat.default: _CATCAST,
        aten._to_copy.default: _CATCAST,
        aten.arange.default: _ARANGE,
        aten.gather.default: _GATHER,
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
