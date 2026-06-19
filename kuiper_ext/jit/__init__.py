"""JIT extraction & compilation of verified Kuiper kernels, driven by the tensors
seen by PyTorch's dispatcher.

Public entry point: ``try_dispatch(func, args, kwargs)`` -> Tensor | None.
"""
from .registry import try_dispatch

__all__ = ["try_dispatch"]
