"""kuipy: hook verified Kuiper GPU kernels into ``torch.compile``.

The pipeline is compiled with Inductor; an Inductor post-grad custom pass
(``kuipy.inductor``) rewrites supported ATen ops (the GEMM family + sdpa) to
opaque ``kuiperjit::*`` custom ops backed by verified Kuiper kernels. Nothing is
installed until ``kuipy.enable()`` is called, so a plain ``torch.compile`` run is
untouched.

Public surface:
  * ``enable()`` / ``disable()`` / ``is_enabled()`` — install/remove the backend.
  * ``register_fusion_rule(fn)`` — register a custom post-grad graph fusion rule.
  * ``batch_capture()`` — warm-up context that extracts every matched kernel and
    builds them in one combined compilation.
  * ``verify`` (``kuipy.verify``) — numerical check vs stock PyTorch.
  * ``tracing`` (``kuipy.inductor.tracing``) — op inventory + KERNELS.md dump.
  * ``is_available()`` — whether the Kuiper JIT toolchain is reachable.
"""
from . import config as C
from . import verify


def is_available() -> bool:
    """True if the Kuiper JIT toolchain (F* + the kuiper repo) is reachable."""
    if C.KUIPER_INST is None:
        return False
    return (C.KUIPER_INST / "bin" / "fstar.exe").exists()


# ---------------------------------------------------------------------------
# Inductor backend
# ---------------------------------------------------------------------------

def enable():
    """Install the Kuiper Inductor backend (post-grad custom pass)."""
    from . import inductor
    return inductor.enable()


def disable():
    from . import inductor
    inductor.disable()


def is_enabled() -> bool:
    from . import inductor
    return inductor.is_enabled()


def register_fusion_rule(fn):
    """Register a custom post-grad fusion rule ``fn(graph) -> bool``.

    Rules run on every compiled graph before the built-in GEMM replacement, so
    they can rewrite patterns first (e.g. fold elementwise ops into a reduction's
    ``pre`` argument)."""
    from . import inductor
    return inductor.register_fusion_rule(fn)


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
# single shared extension. Typical use: run the compile warm-up under
# ``batch_capture()`` to discover and build every kernel in one shot.

def start_batch_capture():
    from . import compile as _c
    _c.start_capture()


def finalize_batch_capture():
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
