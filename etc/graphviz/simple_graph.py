"""Render a tiny operator graph, for slides.

Self-contained and CPU-only: unlike ``dump_graph.py``, which snapshots the real
compiled Qwen graph, this traces a handful of direct operator calls so the
picture fits on one slide.

``make_fx`` records what PyTorch actually dispatches, so the figure is the ATen
graph a backend sees, not the Python that produced it.

    python3 etc/graphviz/simple_graph.py
"""
import os
import subprocess

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from render import ATEN_FILL, ATEN_LINE, SRC_FILL, SRC_LINE, HL_FILL, HL_LINE

HERE = os.path.dirname(os.path.abspath(__file__))

# The ops worth pointing at from a slide.
INTERESTING = ("mm", "matmul", "bmm", "addmm")


def net(x, w, b):
    """A linear layer and an activation, written as bare operator calls."""
    h = torch.mm(x, w)
    h = h + b
    return torch.relu(h)


def label_of(node, names):
    if node.op in ("placeholder", "get_attr"):
        return names.get(node.name, str(node.target))
    if node.op == "output":
        return "output"
    text = str(node.target).replace("aten.", "")
    for suffix in (".default", ".Tensor", ".Scalar", ".dim_IntList", ".int"):
        text = text.replace(suffix, "")
    return text


def shape_of(node):
    val = node.meta.get("val", node.meta.get("tensor_meta"))
    shape = getattr(val, "shape", None)
    return "" if shape is None else "x".join(str(int(s)) for s in shape)


def dot_source(graph, names, rankdir):
    lines = ['digraph G {',
             f'  graph [rankdir={rankdir}, dpi=96, nodesep=0.35,'
             ' ranksep=0.45, bgcolor="white"];',
             '  node [shape=box, style="rounded,filled",'
             ' fontname="Helvetica", fontsize=18, margin="0.20,0.11",'
             ' penwidth=2.0];',
             '  edge [color="#90a4ae", arrowsize=0.8, penwidth=1.6];']
    for node in graph.nodes:
        label = label_of(node, names)
        if node.op in ("placeholder", "get_attr"):
            fill, line = SRC_FILL, SRC_LINE
        elif label in INTERESTING:
            fill, line = HL_FILL, HL_LINE
        else:
            fill, line = ATEN_FILL, ATEN_LINE
        shape = shape_of(node)
        if shape:
            label += f"\\n{shape}"
        lines.append(f'  "{node.name}" [label="{label}", fillcolor="{fill}",'
                     f' color="{line}"];')
        for inp in node.all_input_nodes:
            lines.append(f'  "{inp.name}" -> "{node.name}";')
    lines.append("}")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tb", action="store_true",
                    help="lay out top-to-bottom instead of left-to-right")
    args = ap.parse_args()

    torch.manual_seed(0)
    inputs = (torch.randn(8, 64), torch.randn(64, 64), torch.randn(64))
    with torch.no_grad():
        traced = make_fx(net)(*inputs)

    names = dict(zip([n.name for n in traced.graph.nodes
                      if n.op == "placeholder"], ("x", "W", "b")))
    src = dot_source(traced.graph, names, "TB" if args.tb else "LR")

    dot = os.path.join(HERE, "simple_graph.dot")
    open(dot, "w").write(src)
    for fmt in ("png", "svg"):
        subprocess.run(["dot", f"-T{fmt}", dot, "-o",
                        os.path.join(HERE, f"simple_graph.{fmt}")], check=True)

    from PIL import Image
    w, h = Image.open(os.path.join(HERE, "simple_graph.png")).size
    n = len([x for x in traced.graph.nodes
             if x.op not in ("placeholder", "output")])
    print(f"simple_graph.png: {n} operators, {w}x{h}")


if __name__ == "__main__":
    main()
