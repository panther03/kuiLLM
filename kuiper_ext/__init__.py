import os
import sys
from pathlib import Path

ENABLE_PRINT_PROFILING = os.environ.get("PRINT_PROFILING", "0") == "1"

# If set, JIT extraction/compilation errors propagate instead of falling back to
# stock PyTorch. Off by default so an unsupported shape never breaks a model.
JIT_STRICT = os.environ.get("KUIPER_JIT_STRICT", "0") == "1"

try:
    KUIPER_ROOT = Path(os.environ["KUIPER_HOME"])
except KeyError:
    KUIPER_ROOT = None
    print("$KUIPER_HOME must be defined and point to the root of the kuiper repo!")

_jit_dispatch = None
_jit_warned = False


def _jit_try(func, args, kwargs):
    """Attempt JIT Kuiper dispatch. Returns a Tensor or None (miss/disabled)."""
    global _jit_dispatch, _jit_warned
    if _jit_dispatch is None:
        from kuiper_ext.jit import try_dispatch as _jit_dispatch  # noqa: F811
    try:
        return _jit_dispatch(func, args, kwargs)
    except Exception as e:
        if JIT_STRICT:
            raise
        if not _jit_warned:
            _jit_warned = True
            print(f"[kuiper_ext] JIT dispatch failed, falling back to PyTorch: {e}",
                  file=sys.stderr)
        return None


def is_available() -> bool:
    """True if the Kuiper JIT toolchain (F* + the kuiper repo) is reachable."""
    if KUIPER_ROOT is None:
        return False
    return (KUIPER_ROOT / "inst" / "bin" / "fstar.exe").exists()


# Lazy attribute proxy so `kuiper_ext.KuiperMode` builds the class on first use
# (it needs torch, which we don't want to import at module import time).
def __getattr__(name):
    if name == "KuiperMode":
        return _KuiperMode_cls()
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------
#
# A TorchDispatchMode sees every aten call on the current thread. Eligible ops
# are served by JIT-extracted + compiled Kuiper kernels (cached, so a hot cache
# is just a dict lookup); everything else falls through to stock PyTorch.

_KuiperMode_class = None
profile_data = set()


def _tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for e in x:
            yield from _tensors(e)


def _launches_gpu_kernel(func, args, out):
    """False for pure view/metadata ops (output aliases an input, no kernel)."""
    if func._schema.is_mutable:
        return True
    ins = {t.untyped_storage().data_ptr() for t in _tensors(args)}
    outs = list(_tensors(out))
    return not (outs and all(t.untyped_storage().data_ptr() in ins for t in outs))


def _has_cuda_kernel(func):
    """True iff the op registers a direct CUDA kernel (vs a composite that
    decomposes/redispatches internally and whose GPU sub-ops we cannot intercept)."""
    import torch
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.CUDA)
    except Exception:
        return False


def _KuiperMode_cls():
    global _KuiperMode_class
    if _KuiperMode_class is not None:
        return _KuiperMode_class

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class KuiperMode(TorchDispatchMode):
        """Re-route eligible aten op calls to JIT Kuiper kernels.

        Use as:
            with kuiper_ext.KuiperMode():
                model(...)
        """
        dummy_print_mode = False  # only profile, never use Kuiper

        @classmethod
        def arg_data(_, a):
            if isinstance(a, torch.Tensor):
                return (len(a.shape), a.dtype, a.device)
            elif isinstance(a, (list, tuple)):
                return (0x67, tuple([KuiperMode.arg_data(ai) for ai in a]))
            elif isinstance(a, (torch.dtype, torch.device, bool)):
                return a
            else:
                return type(a).__name__

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}

            if self.dummy_print_mode:
                out = func(*args, **kwargs)
            else:
                out = _jit_try(func, args, kwargs)
                if out is None:
                    out = func(*args, **kwargs)

            if ENABLE_PRINT_PROFILING and _launches_gpu_kernel(func, args, out):
                profile_data.add((func,
                                  tuple([KuiperMode.arg_data(a) for a in args]),
                                  tuple(kwargs.keys()),
                                  tuple([KuiperMode.arg_data(v) for v in kwargs.values()]),
                                  KuiperMode.arg_data(out)))
            return out

    _KuiperMode_class = KuiperMode
    return _KuiperMode_class


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def print_profile_arg(a):
    try:
        if a[0] == 0x67:
            return "[" + ", ".join(print_profile_arg(ai) for ai in a[1]) + ", ]"
        elif not isinstance(a, str):
            (nshape, dtype, device) = a
            return f"Tensor(nshape={nshape}, dtype={dtype}, device={device})"
        else:
            return str(a)
    except Exception:
        return str(a)


def print_profile_data(out_dev=sys.stdout):
    global profile_data
    out = {}
    for func, args, kwargs_keys, kwargs_values, ret in profile_data:
        l = out.get(func, [])
        l.append(f"args={[print_profile_arg(a) for a in args]}, "
                 f"kwargs={{ {', '.join(f'{k}={print_profile_arg(v)}' for k, v in zip(kwargs_keys, kwargs_values))} }} "
                 f"-> {print_profile_arg(ret)}")
        out[func] = l
    for func, calls in out.items():
        tag = "" if _has_cuda_kernel(func) else "  [no CUDA kernel: composite — internal GPU sub-ops not interceptable]"
        print(f"Function {func}:{tag}", file=out_dev)
        for call in calls:
            print(f"  {call}", file=out_dev)
