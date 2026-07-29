"""Stage-level profiler for the JIT build pipeline.

Enabled with ``KUIPY_JIT_PROFILE=1`` (or ``=<path>`` to also dump a JSON trace).
Every expensive stage of a kernel build -- F* check, F* extract, karamel, the
sed/indent fixup, lock waits, and the nvcc/ninja compile -- is wrapped in
``stage()``; a summary is printed at interpreter exit.

The report has two views: per-stage aggregates (where the wall time goes across
a whole run) and the slowest individual kernel builds (which instantiation is
pathological).
"""
import atexit
import json
import os
import threading
import time
from contextlib import contextmanager

ENABLED = os.environ.get("KUIPY_JIT_PROFILE", "0") not in ("0", "", "no", "false")
_TRACE_PATH = os.environ.get("KUIPY_JIT_PROFILE", "")
if _TRACE_PATH in ("0", "1", "", "no", "false", "yes", "true"):
    _TRACE_PATH = None

_lock = threading.Lock()
_records = []          # list of (stage, kernel, seconds, t_start)
_t0 = time.perf_counter()


def record(stage_name: str, kernel: str | None, seconds: float, t_start: float = 0.0):
    if not ENABLED:
        return
    with _lock:
        _records.append((stage_name, kernel, seconds, t_start))


@contextmanager
def stage(stage_name: str, kernel: str | None = None):
    """Time a build stage. A no-op (no timing overhead beyond a branch) when
    profiling is disabled."""
    if not ENABLED:
        yield
        return
    t = time.perf_counter()
    try:
        yield
    finally:
        record(stage_name, kernel, time.perf_counter() - t, t - _t0)


def reset():
    global _t0
    with _lock:
        _records.clear()
        _t0 = time.perf_counter()


def _aggregate():
    per_stage = {}
    per_kernel = {}
    for name, kernel, secs, _ in _records:
        s = per_stage.setdefault(name, [0, 0.0, 0.0])
        s[0] += 1
        s[1] += secs
        s[2] = max(s[2], secs)
        if kernel is not None:
            per_kernel[kernel] = per_kernel.get(kernel, 0.0) + secs
    return per_stage, per_kernel


def report(out=None, top=15):
    """Print the profile summary. Returns the aggregated per-stage dict."""
    import sys
    out = out or sys.stderr
    with _lock:
        if not _records:
            return {}
        per_stage, per_kernel = _aggregate()
        wall = time.perf_counter() - _t0

    total = sum(v[1] for v in per_stage.values())
    print("\n[kuipy-jit profile] pipeline stages "
          f"(wall {wall:.1f}s, measured {total:.1f}s)", file=out)
    print(f"  {'stage':<22} {'n':>4} {'total':>9} {'mean':>8} {'max':>8} {'%':>6}",
          file=out)
    for name, (n, tot, mx) in sorted(per_stage.items(), key=lambda kv: -kv[1][1]):
        pct = 100.0 * tot / total if total else 0.0
        print(f"  {name:<22} {n:>4} {tot:>8.2f}s {tot / n:>7.2f}s {mx:>7.2f}s {pct:>5.1f}%",
              file=out)

    if per_kernel:
        print(f"\n[kuipy-jit profile] slowest kernels (top {top})", file=out)
        for kernel, tot in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:top]:
            print(f"  {tot:>8.2f}s  {kernel}", file=out)

    if _TRACE_PATH:
        with open(_TRACE_PATH, "w") as f:
            json.dump([{"stage": s, "kernel": k, "seconds": d, "start": t}
                       for s, k, d, t in _records], f, indent=1)
        print(f"[kuipy-jit profile] trace written to {_TRACE_PATH}", file=out)
    return per_stage


if ENABLED:
    atexit.register(report)
