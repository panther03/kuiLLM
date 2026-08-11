"""Kernel/operator tracing for the compiled pipeline.

The post-grad pass (``passes.py``) records every ATen ``call_function`` node it
sees together with whether a Kuiper kernel claimed it and the dependencies
between nodes. ``dump_markdown`` renders an aggregated Mermaid graph followed by
the ``KERNELS.md``-style integration backlog.
"""
from collections import Counter

import torch

# key (op, arg_sig, out_sig) -> {"count": int, "claimed": bool}
_records = {}
_edges = Counter()
enabled = False

_DT = {
    torch.float16: "f16", torch.float32: "f32", torch.float64: "f64",
    torch.bfloat16: "bf16", torch.int64: "int64", torch.int32: "int32",
    torch.int16: "int16", torch.int8: "int8", torch.uint8: "uint8",
    torch.bool: "bool",
}


def set_enabled(on: bool):
    """Enable/disable graph tracing.

    Tracing only sees graphs the post-grad pass actually walks, and Inductor
    replays cached FX graphs without re-running the pass -- a warm cache would
    silently produce an empty trace. Disable the caches while tracing.
    """
    global enabled
    enabled = on
    if on:
        import torch._inductor.config as _ind_config
        _ind_config.force_disable_caches = True


def reset():
    _records.clear()
    _edges.clear()


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


def begin_graph(graph):
    """Snapshot graph dependencies before the replacement pass mutates them."""
    if not enabled:
        return None
    return {
        "inputs": {node: tuple(node.all_input_nodes) for node in graph.nodes},
        "keys": {},
    }


def record_node(node, claimed: bool, graph_trace=None):
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
    if graph_trace is not None:
        graph_trace["keys"][node] = key


def finish_graph(graph_trace):
    """Aggregate dependencies between the recorded operator signatures."""
    if graph_trace is None:
        return
    inputs = graph_trace["inputs"]
    keys = graph_trace["keys"]
    memo = {}

    def upstream(node):
        if node in keys:
            return {keys[node]}
        if node in memo:
            return memo[node]
        sources = set()
        for parent in inputs.get(node, ()):
            sources.update(upstream(parent))
        memo[node] = sources
        return sources

    for node, target in keys.items():
        sources = set()
        for parent in inputs[node]:
            sources.update(upstream(parent))
        for source in sources:
            _edges[(source, target)] += 1


def _mermaid(text):
    return (str(text).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def dump_markdown(path):
    """Write the dependency graph and collected op inventory as Markdown."""
    rows = sorted(_records.items(), key=lambda kv: (not kv[1]["claimed"], kv[0][0]))
    lines = [
        "# Kernel implementation checklist",
        "",
        "Auto-generated from the compiled pipeline by `infer.py --dump-kernels`.",
        "Each row is an ATen op the Inductor graph executed; **Kuiper?** marks the",
        "ops served by a verified `kuiperjit::*` kernel (the rest fall back to",
        "Triton / cuBLAS / cuDNN).",
        "",
        "## Kernel dependency graph",
        "",
        "Each node is a unique operator signature; repeated calls are collapsed.",
        "Edges show observed data dependencies and are labelled when repeated.",
        "Green nodes use Kuiper kernels and gray nodes use the fallback backend.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    node_ids = {key: f"k{i}" for i, (key, _) in enumerate(rows)}
    for key, rec in rows:
        op, _, out = key
        label = f"{_mermaid(op)}<br/>{_mermaid(out)}<br/>calls: {rec['count']}"
        lines.append(f'  {node_ids[key]}["{label}"]')
    for (source, target), count in sorted(
            _edges.items(), key=lambda item: (
                node_ids[item[0][0]], node_ids[item[0][1]])):
        label = f"|{count}|" if count > 1 else ""
        lines.append(f"  {node_ids[source]} -->{label} {node_ids[target]}")
    lines.extend([
        "  classDef kuiper fill:#d5f5e3,stroke:#1e8449,color:#17202a",
        "  classDef fallback fill:#e5e7e9,stroke:#626567,color:#17202a",
    ])
    claimed_ids = [node_ids[key] for key, rec in rows if rec["claimed"]]
    fallback_ids = [node_ids[key] for key, rec in rows if not rec["claimed"]]
    if claimed_ids:
        lines.append(f"  class {','.join(claimed_ids)} kuiper")
    if fallback_ids:
        lines.append(f"  class {','.join(fallback_ids)} fallback")
    lines.extend([
        "```",
        "",
        "## Kernel inventory",
        "",
        "| Op | Args | Out | Kuiper? |",
        "| -- | ---- | --- | ------- |",
    ])
    for (op, args, out), rec in rows:
        mark = "yes" if rec["claimed"] else ""
        lines.append(f"| {op} | {args} | {out} | {mark} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    n_claimed = sum(1 for _, r in rows if r["claimed"])
    return len(rows), n_claimed
