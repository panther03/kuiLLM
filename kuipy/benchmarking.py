"""Generic benchmarking driver: run a set of implementations over a set of
shapes and collect timing, throughput and accuracy into a table.

The specific benchmarks live in ``bench_ops.ipynb``.
"""
import time

import torch


def warm_up(ms=300, device="cuda"):
    """Drive the GPU to a steady clock before any timing is taken.

    Idle SM clocks sit an order of magnitude below boost (210 vs 2100 MHz on an
    A6000) and need ~100ms of load to ramp -- far longer than a per-call warmup
    of a few hundred microseconds. Without this the first contender of the first
    case absorbs the ramp and reads 2-3x slow.
    """
    a = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    deadline = time.perf_counter() + ms * 1e-3
    while time.perf_counter() < deadline:
        for _ in range(20):
            torch.mm(a, b)
        torch.cuda.synchronize()


def rel_fro(a, b):
    """Relative Frobenius norm of the error -- the meaningful accuracy metric for
    low-precision accumulation, where elementwise tolerances blow up on
    near-zero reference entries."""
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-9)).item()


def time_call(fn, iters=50, warmup=10):
    """Mean ms/call over ``iters`` calls, plus the result of the last call."""
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, out


def bench_matrix(cases, make_inputs, case_columns, case_column_names, impls,
                 reference, flops=None, iters=50, warmup=10, warmup_ms=300,
                 settle_ms=50):
    """Benchmark ``impls`` against ``reference`` over ``cases``.

    ``cases`` are ``(name, *data)`` tuples; ``make_inputs``, ``case_columns`` and
    ``flops`` are all called with ``*data``. ``make_inputs`` returns the
    ``(args, kwargs)`` handed to every contender, ``case_columns`` the shape
    columns to display (labelled by ``case_column_names``) and ``flops`` the
    floating point operation count of the case, enabling the GFLOP/s columns.

    ``impls`` is a list of ``(name, callable)``. Each contender is timed once;
    the output of its last timed call is what its error is measured on, against
    the reference's. ``warmup_ms`` of dummy load precedes the whole run to pin
    the GPU clocks (see ``warm_up``), and ``settle_ms`` re-pins them before each
    individual measurement: a single up-front ramp leaves the first contender of
    the first case still climbing, which read 6x slow on a 35us kernel and made
    the first row of the table meaningless.

    Even with both of those, the *whole* first case still read slow -- the dummy
    ``torch.mm`` load ramps the clocks but does not prime the contenders' own
    code paths (module load, shared-memory carveout, first-launch JIT), and on a
    ~40us kernel that residue was a 25% error, large enough to look like a real
    regression. The first case is therefore run once and discarded before any
    timing is taken.

    Returns a ``pandas.DataFrame``, one row per case.
    """
    import pandas as pd

    warm_up(warmup_ms)
    contenders = list(impls) + [("ref", reference)]
    if cases:
        prime_args, prime_kwargs = make_inputs(*cases[0][1:])
        for _, fn in contenders:
            time_call(lambda: fn(*prime_args, **prime_kwargs),
                      iters=iters, warmup=warmup)
    rows = []
    for name, *data in cases:
        args, kwargs = make_inputs(*data)
        times, outs = {}, {}
        for label, fn in contenders:
            if settle_ms:
                warm_up(settle_ms)
            times[label], outs[label] = time_call(
                lambda: fn(*args, **kwargs), iters=iters, warmup=warmup)

        row = {"case": name}
        row.update(zip(case_column_names, case_columns(*data)))
        if flops is not None:
            f = flops(*data)
            for label, _ in contenders:
                row[f"{label} GFLOP/s"] = f / (times[label] * 1e-3) / 1e9
        for label, _ in contenders:
            row[f"{label} us"] = times[label] * 1e3
        for label, _ in impls:
            row[f"{label} rel-err"] = rel_fro(outs[label], outs["ref"])
        rows.append(row)

    return pd.DataFrame(rows)
