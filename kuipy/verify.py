"""Numerical verification of Kuiper kernels against stock PyTorch.

A Kuiper-dispatched op can be run alongside its stock-PyTorch reference and the
two results compared by relative Frobenius norm. Divergences accumulate in
``verify_stats`` (keyed by op name) and are printed by ``print_report``.

Under the Inductor backend the fused kernels bypass the ATen dispatcher, so this
is driven by the ``kuiperjit::*`` custom ops themselves (see
``kuipy.inductor.custom_ops``): when ``enabled`` is set they compute the
reference via the original ATen op and fold the comparison in here. This must
run in an eager, non-CUDA-graph pass (the host-side norm compare needs a sync).
"""
import sys

# Global switch flipped on for an eager verify pass.
enabled = False
tol = 2e-2

# op name -> {n, fail, max_rel, worst}
verify_stats = {}


def set_enabled(on: bool, verify_tol: float = 2e-2):
    global enabled, tol
    enabled = on
    tol = verify_tol


def reset():
    verify_stats.clear()


def _tensors(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for e in x:
            yield from _tensors(e)


def compare(name, out, ref, cmp_tol=None):
    """Compare matching floating tensors in ``out`` vs ``ref`` by relative
    Frobenius norm and fold the result into ``verify_stats[name]``.

    Tensors are restricted to positions where both are finite: fully masked
    attention rows (all keys ``-inf``) yield implementation-defined NaN/0
    outputs not meaningful to compare. Empty references are skipped."""
    import torch
    cmp_tol = tol if cmp_tol is None else cmp_tol
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
        if rel > cmp_tol:
            st["fail"] += 1
            st["worst"] = f"rel={rel:.3e} (tol {cmp_tol:.1e})"


def print_report(out_dev=sys.stdout, report_tol=None):
    """Print a per-op pass/fail summary collected during a verify run."""
    report_tol = tol if report_tol is None else report_tol
    if not verify_stats:
        print("[verify] no Kuiper-dispatched ops were checked.", file=out_dev)
        return True
    total_fail = sum(s["fail"] for s in verify_stats.values())
    print("[verify] Kuiper vs stock PyTorch (relative Frobenius norm, "
          f"tol {report_tol:.1e}):", file=out_dev)
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
