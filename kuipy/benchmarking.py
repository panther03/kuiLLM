"""Generic benchmarking driver: run a set of implementations over a set of
shapes and collect timing, throughput and accuracy into a table.

The specific benchmarks live in ``bench_ops.ipynb``.
"""
import torch


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
                 reference, flops=None, iters=50, warmup=10):
    """Benchmark ``impls`` against ``reference`` over ``cases``.

    ``cases`` are ``(name, *data)`` tuples; ``make_inputs``, ``case_columns`` and
    ``flops`` are all called with ``*data``. ``make_inputs`` returns the
    ``(args, kwargs)`` handed to every contender, ``case_columns`` the shape
    columns to display (labelled by ``case_column_names``) and ``flops`` the
    floating point operation count of the case, enabling the GFLOP/s columns.

    ``impls`` is a list of ``(name, callable)``. Each contender is timed once;
    the output of its last timed call is what its error is measured on, against
    the reference's.

    Returns a ``pandas.DataFrame``, one row per case.
    """
    import pandas as pd

    contenders = list(impls) + [("ref", reference)]
    rows = []
    for name, *data in cases:
        args, kwargs = make_inputs(*data)
        times, outs = {}, {}
        for label, fn in contenders:
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
