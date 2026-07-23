"""Kernel/operator tracing for the compiled pipeline.

The post-grad pass (``passes.py``) records every ATen ``call_function`` node it
sees together with whether a Kuiper kernel claimed it. ``dump_markdown`` renders
the collected inventory as a ``KERNELS.md``-style table — the integration
backlog, regenerated straight from what the compiled graph actually executes.
"""
import torch

# key (op, arg_sig, out_sig) -> {"count": int, "claimed": bool}
_records = {}
enabled = False

_DT = {
    torch.float16: "f16", torch.float32: "f32", torch.float64: "f64",
    torch.bfloat16: "bf16", torch.int64: "int64", torch.int32: "int32",
    torch.int16: "int16", torch.int8: "int8", torch.uint8: "uint8",
    torch.bool: "bool",
}


def set_enabled(on: bool):
    global enabled
    enabled = on


def reset():
    _records.clear()


def _dt(dt):
    return _DT.get(dt, str(dt).replace("torch.", ""))


def _render(val):
    """Render a fake value (tensor / list / scalar) the way KERNELS.md does."""
    from torch.fx import Node
    if isinstance(val, Node):
        val = val.meta.get("val")
    if isinstance(val, torch.Tensor):
        dev = "c" if val.is_cuda else ("cpu" if val.device.type == "cpu" else val.device.type)
        return f"T({val.dim()},{_dt(val.dtype)},{dev})"
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_render(v) for v in val) + "]"
    if isinstance(val, torch.dtype):
        return _dt(val)
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    return type(val).__name__


def record_node(node, claimed: bool):
    if not enabled:
        return
    if node.op != "call_function" or not hasattr(node.target, "name"):
        return
    name = getattr(node.target, "name", lambda: str(node.target))
    op = name() if callable(name) else str(node.target)
    args = ", ".join(_render(a) for a in node.args)
    kwargs = ", ".join(f"{k}={_render(v)}" for k, v in node.kwargs.items())
    if kwargs:
        args = f"{args} | {kwargs}" if args else kwargs
    out = _render(node.meta.get("val"))
    key = (op, args, out)
    rec = _records.setdefault(key, {"count": 0, "claimed": claimed})
    rec["count"] += 1
    rec["claimed"] = rec["claimed"] or claimed


def dump_markdown(path):
    """Write the collected op inventory as a KERNELS.md table."""
    rows = sorted(_records.items(), key=lambda kv: (not kv[1]["claimed"], kv[0][0]))
    lines = [
        "# Kernel implementation checklist",
        "",
        "Auto-generated from the compiled pipeline by `infer.py --dump-kernels`.",
        "Each row is an ATen op the Inductor graph executed; **Kuiper?** marks the",
        "ops served by a verified `kuiperjit::*` kernel (the rest fall back to",
        "Triton / cuBLAS / cuDNN).",
        "",
        "| Op | Args | Out | Kuiper? |",
        "| -- | ---- | --- | ------- |",
    ]
    for (op, args, out), rec in rows:
        mark = "yes" if rec["claimed"] else ""
        lines.append(f"| {op} | {args} | {out} | {mark} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    n_claimed = sum(1 for _, r in rows if r["claimed"])
    return len(rows), n_claimed
