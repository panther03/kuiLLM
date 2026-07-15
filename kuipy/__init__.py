import os
import sys
from pathlib import Path

from . import config as C

def arg_data(a):
    import torch
    if isinstance(a, torch.Tensor):
        shape = len(a.shape)
        if C.JIT_VERBOSITY > 1:
            shape = tuple(a.shape)
        return (shape, a.dtype, a.device)
    elif isinstance(a, (list, tuple)):
        return (0x67, tuple([arg_data(ai) for ai in a]))
    elif isinstance(a, (torch.dtype, torch.device, bool)):
        return a
    else:
        return type(a).__name__
    
def print_arg_data(a):
    try:
        if a[0] == 0x67:
            return "[" + ", ".join(print_arg_data(ai) for ai in a[1]) + ", ]"
        elif not isinstance(a, str):
            (nshape, dtype, device) = a
            return f"Tensor(nshape={nshape}, dtype={dtype}, device={device})"
        else:
            return str(a)
    except Exception:
        return str(a)
    
def print_call_data_aux(args, kwargs_keys, kwargs_values, ret):
    return (f"args={[print_arg_data(a) for a in args]}, "
            f"kwargs={{ {', '.join(f'{k}={print_arg_data(v)}' for k, v in zip(kwargs_keys, kwargs_values))} }} "
            f"-> {print_arg_data(ret)}")
    
def print_call_data(args, kwargs):
    return print_call_data_aux(tuple([arg_data(a) for a in args]),
                    tuple(kwargs.keys()),
                    tuple([arg_data(v) for v in kwargs.values()]),
                    "no return captured")

def is_available() -> bool:
    """True if the Kuiper JIT toolchain (F* + the kuiper repo) is reachable."""
    if C.KUIPER_INST is None:
        return False
    return (C.KUIPER_INST / "bin" / "fstar.exe").exists()


# ---------------------------------------------------------------------------
# Batch compile
# ---------------------------------------------------------------------------
#
# Normally each distinct kernel instantiation is compiled into its own torch
# extension the first time it is dispatched (an F* extraction + a full nvcc/g++
# build, dominated by parsing torch/extension.h once per kernel). Batch mode
# amortizes that: under ``batch_capture()`` matched ops are only *extracted* and
# their execution falls back to stock PyTorch, then on exit every captured
# wrapper is concatenated into one translation unit and compiled together into a
# single shared extension -- so torch/extension.h is parsed once for the whole
# run. Typical use: one warm-up pass under ``batch_capture()`` to discover and
# build every kernel, then the real workload runs on the compiled kernels.
#
#     with kuipy.KuiperMode():
#         with kuipy.batch_capture():
#             model(warmup_inputs)   # extracts everything, runs via PyTorch
#         model(real_inputs)         # runs the batch-compiled Kuiper kernels

def start_batch_capture():
    """Begin capturing matched kernels for a single combined build. While active,
    matched ops are extracted but deferred to stock PyTorch."""
    from . import compile as _c
    _c.start_capture()


def finalize_batch_capture():
    """Compile every captured kernel into one shared extension and return it
    (or None if nothing was captured)."""
    from . import compile as _c
    return _c.finalize_capture()


def is_batch_capturing() -> bool:
    from . import compile as _c
    return _c.is_capturing()


def batch_capture():
    """Context manager wrapping ``start_batch_capture`` / ``finalize_batch_capture``."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        start_batch_capture()
        try:
            yield
        finally:
            finalize_batch_capture()

    return _cm()


_jit_dispatch = None
_jit_warned = False

def _jit_try(func, args, kwargs, VERY_STRICT=False, trace=False):
    """Attempt JIT Kuiper dispatch. Returns None on failure."""
    global _jit_dispatch, _jit_warned
    if _jit_dispatch is None:
        from .registry import try_dispatch as _jit_dispatch  # noqa: F811
    ret = None
    try:
        if trace:
            ret = _jit_dispatch(func, args, kwargs, trace=True)
        else:
            ret = _jit_dispatch(func, args, kwargs)
    except Exception as e:
        if C.JIT_STRICTNESS > 0:
            raise
        if not _jit_warned:
            _jit_warned = True
            print(f"[kuipy] JIT dispatch failed, falling back to PyTorch: {e}",
                  file=sys.stderr)
        ret = None
    if ret is None:
        if C.JIT_VERBOSITY > 1:
            print(f"[kuipy] operator {func} ({print_call_data(args, kwargs)}) did not match", file=sys.stderr)
        if C.JIT_STRICTNESS > 1:
            raise
            
    return ret


# Lazy attribute proxy so `kuiper_ext.KuiperMode` builds the class on first use
# (it needs torch, which we don't want to import at module import time).
def __getattr__(name):
    if name == "KuiperMode":
        return _KuiperMode_cls()
    if name == "compile_graph":
        from .graph import compile_graph
        return compile_graph
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

        When ``verify`` is true, every op that Kuiper actually handles is ALSO
        run through stock PyTorch and the two results are compared (relative
        Frobenius norm). Divergences are accumulated in ``verify_stats`` and can
        be printed with ``print_verify_report()``. This roughly doubles the work
        for dispatched ops, so it is for correctness checking, not benchmarking.
        """
        dummy_print_mode = False  # only profile, never use Kuiper

        @classmethod
        def ignore_compile_internals(cls):
            return True

        def __init__(self, verify=False, verify_tol=2e-2, use_kuiper=True,
                     trace=False):
            super().__init__()
            self.verify = verify
            self.verify_tol = verify_tol
            self.use_kuiper = use_kuiper
            self.trace = trace

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}

            used_kuiper = False
            if self.dummy_print_mode or not self.use_kuiper:
                out = func(*args, **kwargs)
            else:
                out = _jit_try(func, args, kwargs, trace=self.trace)
                used_kuiper = out is not None
                if not used_kuiper:
                    out = func(*args, **kwargs)
                elif self.verify and not func._schema.is_mutable:
                    ref = func(*args, **kwargs)
                    _verify_compare(func, out, ref, self.verify_tol)

            if C.ENABLE_PRINT_PROFILING and _launches_gpu_kernel(func, args, out):
                profile_data.add((func,
                                  tuple([arg_data(a) for a in args]),
                                  tuple(kwargs.keys()),
                                  tuple([arg_data(v) for v in kwargs.values()]),
                                  arg_data(out),
                                  used_kuiper))
            return out

    _KuiperMode_class = KuiperMode
    return _KuiperMode_class


# ---------------------------------------------------------------------------
# Numerical verification (Kuiper vs. stock PyTorch)
# ---------------------------------------------------------------------------

# str(func) -> {n, fail, max_rel, worst}
verify_stats = {}


def reset_verify():
    verify_stats.clear()


def _verify_compare(func, out, ref, tol):
    """Compare matching floating tensors in ``out`` vs ``ref`` by relative
    Frobenius norm and fold the result into ``verify_stats[str(func)]``.

    Tensors are restricted to positions where the reference is finite: fully
    masked attention rows (all keys ``-inf``) yield implementation-defined
    NaN/0 outputs that are not meaningful to compare. Empty reference tensors
    (e.g. the LSE when ``compute_log_sumexp`` is false) are skipped."""
    import torch
    name = str(func)
    st = verify_stats.setdefault(name, {"n": 0, "fail": 0, "max_rel": 0.0, "worst": None})
    for o, r in zip(_tensors(out), _tensors(ref)):
        if not (o.is_floating_point() and r.is_floating_point()):
            continue
        if r.numel() == 0 or o.numel() == 0:
            continue
        if o.shape != r.shape:
            st["fail"] += 1
            st["worst"] = f"shape mismatch {tuple(o.shape)} vs {tuple(r.shape)}"
            continue
        of, rf = o.float(), r.float()
        finite = torch.isfinite(rf) & torch.isfinite(of)
        if not finite.any():
            continue
        of, rf = of[finite], rf[finite]
        rel = ((of - rf).norm() / (rf.norm() + 1e-12)).item()
        st["n"] += 1
        if rel > st["max_rel"]:
            st["max_rel"] = rel
        if rel > tol:
            st["fail"] += 1
            st["worst"] = f"rel={rel:.3e} (tol {tol:.1e})"


def print_verify_report(out_dev=sys.stdout, tol=2e-2):
    """Print a per-op pass/fail summary collected during a verify run."""
    if not verify_stats:
        print("[verify] no Kuiper-dispatched ops were checked.", file=out_dev)
        return
    total_fail = sum(s["fail"] for s in verify_stats.values())
    print("[verify] Kuiper vs stock PyTorch (relative Frobenius norm, "
          f"tol {tol:.1e}):", file=out_dev)
    for name in sorted(verify_stats):
        s = verify_stats[name]
        status = "FAIL" if s["fail"] else "ok"
        line = (f"  [{status:4}] {name}: {s['n']} checked, "
                f"{s['fail']} fail, max_rel={s['max_rel']:.3e}")
        if s["worst"]:
            line += f", worst: {s['worst']}"
        print(line, file=out_dev)
    verdict = "PASS" if total_fail == 0 else f"FAIL ({total_fail} divergences)"
    print(f"[verify] result: {verdict}", file=out_dev)
    return total_fail == 0


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def print_profile_data(out_dev=sys.stdout):
    global profile_data
    out = {}
    for func, args, kwargs_keys, kwargs_values, ret, used_kuiper in profile_data:
        l = out.get((func, used_kuiper), [])
        l.append(print_call_data_aux(args, kwargs_keys, kwargs_values, ret))
        out[(func, used_kuiper)] = l
    for (func, used_kuiper), calls in out.items():
        tag = "" if _has_cuda_kernel(func) else "  [no CUDA kernel: composite — internal GPU sub-ops not interceptable]"
        backend = "kuiper" if used_kuiper else "torch"
        print(f"Function {func} [{backend}]:{tag}", file=out_dev)
        for call in calls:
            print(f"  {call}", file=out_dev)
