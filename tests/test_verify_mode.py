"""Unit tests for the numerical-verification machinery used by `infer.py --verify`.

These exercise `kuipy._verify_compare` / `print_verify_report` directly with
synthetic tensors (no kernel compilation, CPU-only) so they run fast.
"""
import io

import torch

import kuipy


def _fresh():
    kuipy.reset_verify()


class _F:  # stand-in for an aten OpOverload (only str() is used as the key)
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def test_matching_tensors_pass():
    _fresh()
    a = torch.randn(8, 8)
    kuipy._verify_compare(_F("aten.mm"), a, a.clone(), tol=2e-2)
    s = kuipy.verify_stats["aten.mm"]
    assert s["n"] == 1 and s["fail"] == 0 and s["max_rel"] < 1e-6


def test_divergent_tensors_fail():
    _fresh()
    a = torch.randn(8, 8)
    b = a + 0.5 * a.abs().mean()
    kuipy._verify_compare(_F("aten.mm"), a, b, tol=2e-2)
    s = kuipy.verify_stats["aten.mm"]
    assert s["fail"] == 1 and s["max_rel"] > 2e-2 and "rel=" in s["worst"]


def test_empty_reference_is_skipped():
    _fresh()
    out = torch.randn(2, 3, 1)              # kuiper always computes the LSE
    ref = torch.randn(2, 3, 0)              # torch returns it empty
    kuipy._verify_compare(_F("aten.sdpa"), out, ref, tol=2e-2)
    # Nothing comparable -> no checks, no failures recorded.
    assert "aten.sdpa" in kuipy.verify_stats
    s = kuipy.verify_stats["aten.sdpa"]
    assert s["n"] == 0 and s["fail"] == 0


def test_nonfinite_positions_are_ignored():
    _fresh()
    # A fully-masked attention row: kuiper -> NaN, torch -> 0. Those positions
    # must be excluded so they don't masquerade as a divergence.
    out = torch.randn(4, 8)
    ref = out.clone()
    out[0] = float("nan")
    ref[0] = 0.0
    kuipy._verify_compare(_F("aten.sdpa"), out, ref, tol=2e-2)
    s = kuipy.verify_stats["aten.sdpa"]
    assert s["n"] == 1 and s["fail"] == 0


def test_tuple_outputs_compared_elementwise():
    _fresh()
    a, b = torch.randn(4, 4), torch.randn(4)
    kuipy._verify_compare(_F("aten.sdpa"), (a, b), (a.clone(), b.clone()), tol=2e-2)
    assert kuipy.verify_stats["aten.sdpa"]["n"] == 2


def test_report_pass_and_fail():
    _fresh()
    a = torch.randn(8, 8)
    kuipy._verify_compare(_F("aten.mm"), a, a.clone(), tol=2e-2)
    buf = io.StringIO()
    ok = kuipy.print_verify_report(out_dev=buf, tol=2e-2)
    assert ok is True and "PASS" in buf.getvalue()

    kuipy._verify_compare(_F("aten.bmm"), a, a + 1.0, tol=2e-2)
    buf = io.StringIO()
    ok = kuipy.print_verify_report(out_dev=buf, tol=2e-2)
    assert ok is False and "FAIL" in buf.getvalue()
