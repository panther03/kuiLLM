"""Unit tests for the numerical-verification machinery used by `infer.py --verify`.

These exercise `kuipy.verify.compare` / `print_report` directly with synthetic
tensors (no kernel compilation, CPU-only) so they run fast.
"""
import io

import torch

from kuipy import verify as V


def _fresh():
    V.reset()


def test_matching_tensors_pass():
    _fresh()
    a = torch.randn(8, 8)
    V.compare("aten.mm", a, a.clone(), 2e-2)
    s = V.verify_stats["aten.mm"]
    assert s["n"] == 1 and s["fail"] == 0 and s["max_rel"] < 1e-6


def test_divergent_tensors_fail():
    _fresh()
    a = torch.randn(8, 8)
    b = a + 0.5 * a.abs().mean()
    V.compare("aten.mm", a, b, 2e-2)
    s = V.verify_stats["aten.mm"]
    assert s["fail"] == 1 and s["max_rel"] > 2e-2 and "rel=" in s["worst"]


def test_empty_reference_is_skipped():
    _fresh()
    out = torch.randn(2, 3, 1)
    ref = torch.randn(2, 3, 0)
    V.compare("aten.sdpa", out, ref, 2e-2)
    assert "aten.sdpa" in V.verify_stats
    s = V.verify_stats["aten.sdpa"]
    assert s["n"] == 0 and s["fail"] == 0


def test_nonfinite_positions_are_ignored():
    _fresh()
    out = torch.randn(4, 8)
    ref = out.clone()
    out[0] = float("nan")
    ref[0] = 0.0
    V.compare("aten.sdpa", out, ref, 2e-2)
    s = V.verify_stats["aten.sdpa"]
    assert s["n"] == 1 and s["fail"] == 0


def test_tuple_outputs_compared_elementwise():
    _fresh()
    a, b = torch.randn(4, 4), torch.randn(4)
    V.compare("aten.sdpa", (a, b), (a.clone(), b.clone()), 2e-2)
    assert V.verify_stats["aten.sdpa"]["n"] == 2


def test_report_pass_and_fail():
    _fresh()
    a = torch.randn(8, 8)
    V.compare("aten.mm", a, a.clone(), 2e-2)
    buf = io.StringIO()
    ok = V.print_report(out_dev=buf, report_tol=2e-2)
    assert ok is True and "PASS" in buf.getvalue()

    V.compare("aten.bmm", a, a + 1.0, 2e-2)
    buf = io.StringIO()
    ok = V.print_report(out_dev=buf, report_tol=2e-2)
    assert ok is False and "FAIL" in buf.getvalue()
