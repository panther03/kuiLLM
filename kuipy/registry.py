"""Mapping from ATen ops to the ``kuipy.kuiops`` family implementing them, plus
the ``run`` dispatch entry point."""
from . import kuiops

import torch
aten = torch.ops.aten


REGISTRY = {
    aten.silu.default:          kuiops.ElementwiseImpl,
    aten.relu.default:          kuiops.ElementwiseImpl,
    aten.neg.default:           kuiops.ElementwiseImpl,
    aten.rsqrt.default:         kuiops.ElementwiseImpl,
    aten.cos.default:           kuiops.ElementwiseImpl,
    aten.sin.default:           kuiops.ElementwiseImpl,
    aten.pow.Tensor_Scalar:     kuiops.ElementwiseImpl,
    aten.add.Tensor:            kuiops.ElementwiseImpl,
    aten.add.Scalar:            kuiops.ElementwiseImpl,
    aten.mul.Tensor:            kuiops.ElementwiseImpl,
    aten.mul.Scalar:            kuiops.ElementwiseImpl,
    aten.sub.Tensor:            kuiops.ElementwiseImpl,
    aten.sub.Scalar:            kuiops.ElementwiseImpl,
    aten.div.Tensor:            kuiops.ElementwiseImpl,
    aten.div.Scalar:            kuiops.ElementwiseImpl,
    aten.bitwise_not.default:   kuiops.ElementwiseImpl,
    aten.bitwise_and.Tensor:    kuiops.ElementwiseImpl,
    aten.bitwise_or.Tensor:     kuiops.ElementwiseImpl,
    aten.eq.Scalar:             kuiops.ElementwiseImpl,
    aten.le.Scalar:             kuiops.ElementwiseImpl,
    aten.lt.Scalar:             kuiops.ElementwiseImpl,
    aten.where.self:            kuiops.ElementwiseImpl,
    aten.mm.default:            kuiops.MmImpl,
    aten.bmm.default:           kuiops.BmmImpl,
    aten.addmm.default:         kuiops.AddmmImpl,
    aten._softmax.default:      kuiops.SoftmaxImpl,
    aten.gather.default:        kuiops.GatherImpl,
    aten.scatter.src:           kuiops.ScatterImpl,
    aten.cat.default:           kuiops.CatImpl,
    aten.sum.dim_IntList:       kuiops.HReducePolyImpl,
    aten.prod.dim_int:          kuiops.HReducePolyImpl,
    aten.all.dim:               kuiops.HReducePolyImpl,
    aten.any.dim:               kuiops.HReducePolyImpl,
    aten.mean.dim:              kuiops.MeanImpl,
    aten._scaled_dot_product_efficient_attention.default: kuiops.SdpaImpl,
    aten._scaled_dot_product_cudnn_attention.default:     kuiops.SdpaImpl,
}

_impls = {}


def impl_for(op):
    """The (memoised) family instance serving ``op``."""
    cls = REGISTRY.get(op)
    if cls is None:
        raise KeyError(f"no Kuiper implementation registered for {op}")
    inst = _impls.get(cls)
    if inst is None:
        inst = _impls[cls] = cls()
    return inst


def _key_part(v):
    """Hashable stand-in for one argument, covering everything ``supported()``
    inspects: shape, stride, dtype and device. Raises for anything else, which
    ``_spec_key`` turns into "do not cache"."""
    if isinstance(v, torch.Tensor):
        return ("t", tuple(v.shape), tuple(v.stride()), v.dtype, str(v.device))
    if isinstance(v, (bool, int, float, str, torch.dtype, torch.device)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_key_part(x) for x in v)
    raise TypeError(v)


def _spec_key(args, kwargs):
    try:
        return (tuple(_key_part(a) for a in args),
                tuple((k, _key_part(v)) for k, v in sorted(kwargs.items())))
    except TypeError:
        return None


_SPEC_CACHE_MAX = 1024


def run(op, **preset_kwargs):
    """Return a drop-in replacement for ``op`` backed by its Kuiper kernel.

    ``preset_kwargs`` are merged into the kwargs of each call, and may include
    the family's kernel-selection extras (e.g. ``impl=``, ``out_dtype=``). The
    returned callable raises if the arguments are outside what the kernel
    supports -- there is no fallback to stock PyTorch.

    ``supported()`` enumerates and scores every legal kernel parameterisation,
    which costs far more than launching the kernel, so its result is memoised on
    the argument metadata it looks at. The Inductor path resolves the spec once
    at trace time; this keeps the eager path from paying it per call."""
    impl = impl_for(op)
    cache = {}

    def call(*args, **kwargs):
        kwargs = {**preset_kwargs, **kwargs}
        key = _spec_key(args, kwargs)
        spec = cache.get(key) if key is not None else None
        if spec is None:
            spec = impl.supported(op, args, kwargs)
            if spec is None:
                raise NotImplementedError(
                    f"{op} unsupported by {type(impl).__name__} for these arguments")
            if key is not None:
                if len(cache) >= _SPEC_CACHE_MAX:
                    cache.clear()
                cache[key] = spec
        return impl.run(spec, args, kwargs)

    return call
