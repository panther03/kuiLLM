"""Registry mapping aten ops to JIT Kuiper implementations + dispatch entry point.

The hot path is intentionally tiny: a dict lookup, a cheap ``supported()`` check,
and (on a hit) a memoised compiled-kernel call. On any miss it returns ``None`` so
the caller falls through to stock PyTorch.
"""
from .kuiops import (ElementwiseImpl, MmImpl, BmmImpl, AddmmImpl, SoftmaxImpl,
                     SdpaImpl, GatherImpl, ScatterImpl, CatImpl, MeanImpl)
from . import config as C

# Lazily populated on first use (needs torch.ops.aten).
_REGISTRY = None


def _build_registry(tune_params):
    import torch
    aten = torch.ops.aten
    _ELEM = ElementwiseImpl(tune_params)
    _MM = MmImpl(tune_params)
    _BMM = BmmImpl(tune_params)
    _ADDMM = AddmmImpl(tune_params)
    _SOFTMAX = SoftmaxImpl(tune_params)
    _GATHER = GatherImpl(tune_params)
    _SCATTER = ScatterImpl(tune_params)
    _CAT = CatImpl(tune_params)
    _MEAN = MeanImpl(tune_params)
    # NOTE: currently disconnected
    # _SDPA = SdpaImpl(tune_params)
    return {
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
        aten.mm.default: _MM,
        aten.bmm.default: _BMM,
        aten.addmm.default: _ADDMM,
        aten._softmax.default: _SOFTMAX,
        aten.gather.default: _GATHER,
        aten.scatter.src: _SCATTER,
        aten.cat.default: _CAT,
        aten.mean.dim: _MEAN,
        #aten._scaled_dot_product_efficient_attention.default: _SDPA,
    }

def _tune_impls(tune_params):
    for impl in set(_REGISTRY.values()):
        tune_params = impl.tune(tune_params)
    return tune_params

def try_dispatch(func, args, kwargs):
    """Return a result Tensor if a JIT Kuiper kernel handles ``func``, else None."""
    global _REGISTRY
    if _REGISTRY is None:
        # LATER: load these from json
        tune_params = {}
        _REGISTRY = _build_registry(tune_params)
        if C.RE_TUNE:
            tune_params = _tune_impls(tune_params)
            # LATER: write it back out

    impl = _REGISTRY.get(func)
    if impl is None:
        return None
    spec = impl.supported(func, args, kwargs)
    if spec is None:
        return None
    return impl.run(spec, args, kwargs)
