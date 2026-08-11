"""Render cropped views of the real Inductor post-grad DAG.

Reads pre.json / post.json (dumped by dump_graph.py), drops the view-only nodes
(reshape / permute / getitem / ...), lays the remaining neighbourhood out with
Graphviz and crops a viewport around the anchor ops, so the surrounding graph is
visibly cut off at the frame edges.
"""
import json
import os
import subprocess
from collections import deque

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 64.0

KUIPER_FILL, KUIPER_LINE = "#c8e6c9", "#2e7d32"
ATEN_FILL, ATEN_LINE = "#eceff1", "#78909c"
HL_FILL, HL_LINE = "#ffe0b2", "#ef6c00"
SRC_FILL, SRC_LINE = "#e3f2fd", "#1565c0"

# Pure metadata ops: they rename axes or pick a tuple element, they do not move
# or compute data, so they are contracted away and their consumers are wired
# straight to their source.
VIEW_OPS = {
    "aten.reshape.default", "aten.view.default", "aten._unsafe_view.default",
    "aten.permute.default", "aten.t.default", "aten.expand.default",
    "aten.squeeze.dim", "aten.unsqueeze.default", "aten.detach.default",
    "aten.alias.default", "<built-in function getitem>",
}


def load(which):
    nodes = json.load(open(os.path.join(HERE, f"{which}.json")))
    return {n["name"]: n for n in nodes}, nodes


def short(target):
    t = target.replace("aten.", "").replace("prims.", "prims::")
    t = t.replace("kuiperjit.", "kuiperjit::")
    for suffix in (".default", ".Tensor", ".Scalar", ".dim_IntList", ".dim"):
        t = t.replace(suffix, "")
    return t


def contract(byname, nodelist):
    """Map every node to the nearest non-view ancestor, and return the graph
    with the view nodes removed."""
    resolved = {}

    def resolve(name):
        if name in resolved:
            return resolved[name]
        n = byname[name]
        if n["target"] not in VIEW_OPS or not n["inputs"]:
            resolved[name] = name
        else:
            resolved[name] = resolve(n["inputs"][0])
        return resolved[name]

    kept = []
    for n in nodelist:
        if n["target"] in VIEW_OPS:
            continue
        ins, seen = [], set()
        for i in n["inputs"]:
            r = resolve(i)
            if r != n["name"] and r not in seen:
                seen.add(r)
                ins.append(r)
        kept.append(dict(n, inputs=ins))
    return {n["name"]: n for n in kept}, kept


def window(nodelist, byname, anchors, before, after):
    """A contiguous topological slice around the anchors, keeping only nodes
    reachable from an anchor within the slice plus the source tensors they read.
    Edges leaving the slice (other layers) fall off the frame."""
    idx = {n["name"]: i for i, n in enumerate(nodelist)}
    lo = max(min(idx[a] for a in anchors) - before, 0)
    hi = max(idx[a] for a in anchors) + after
    cand = {n["name"] for n in nodelist[lo:hi] if n["op"] != "placeholder"}
    adj = {}
    for name in cand:
        for i in byname[name]["inputs"]:
            if i in cand:
                adj.setdefault(name, set()).add(i)
                adj.setdefault(i, set()).add(name)
    keep, q = set(anchors), deque(anchors)
    while q:
        for nb in adj.get(q.popleft(), ()):
            if nb not in keep:
                keep.add(nb)
                q.append(nb)
    for name in list(keep):
        for i in byname[name]["inputs"]:
            if byname[i]["op"] == "placeholder":
                keep.add(i)
    return keep


def dot_source(byname, keep, anchors, highlight=(), anchor_kuiper=False):
    lines = ['digraph G {',
             '  graph [rankdir=TB, dpi=%d, nodesep=0.28, ranksep=0.45,'
             ' bgcolor="white"];' % DPI,
             '  node [shape=box, style="rounded,filled", fontname="Helvetica",'
             ' fontsize=15, margin="0.16,0.09", penwidth=1.6];',
             '  edge [color="#90a4ae", arrowsize=0.7, penwidth=1.3];']
    for name in keep:
        n = byname[name]
        if n["op"] == "placeholder":
            label = n["src"] or name
            fill, line, shape = SRC_FILL, SRC_LINE, "box"
        else:
            label = short(n["target"])
            kuiper = "kuiperjit" in label
            fill, line = (KUIPER_FILL, KUIPER_LINE) if kuiper else (ATEN_FILL, ATEN_LINE)
            shape = "box"
        if n["val"]:
            label += f'\\n{n["val"]}'
        pen, fs = 1.6, 15
        if name in highlight:
            fill, line, pen = HL_FILL, HL_LINE, 3.2
        if name in anchors:
            if anchor_kuiper:
                fill, line = KUIPER_FILL, KUIPER_LINE
            else:
                line = KUIPER_LINE if "kuiperjit" in label else HL_LINE
            pen, fs = 4.0, 18
        lines.append(f'  "{name}" [label="{label}", shape={shape}, '
                     f'fillcolor="{fill}", color="{line}", penwidth={pen}, '
                     f'fontsize={fs}];')
    for name in keep:
        for i in byname[name]["inputs"]:
            if i in keep:
                lines.append(f'  "{i}" -> "{name}";')
    lines.append("}")
    return "\n".join(lines)


def legend(img, extra=()):
    """Key in the top-left corner of the cropped frame."""
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        f = ImageFont.load_default()
    rows = [(KUIPER_FILL, KUIPER_LINE, "verified Kuiper kernel"),
            (ATEN_FILL, ATEN_LINE, "stock ATen / Inductor op"),
            (SRC_FILL, SRC_LINE, "source tensor (weight / cache)")] + list(extra)
    x, y, bw, bh = 14, 14, 330, 26 + 24 * len(rows)
    d.rectangle([x, y, x + bw, y + bh], fill="#ffffff", outline="#b0bec5", width=1)
    for i, (fill, line, text) in enumerate(rows):
        ty = y + 14 + 24 * i
        d.rectangle([x + 12, ty, x + 34, ty + 14], fill=fill, outline=line, width=2)
        d.text((x + 44, ty - 1), text, fill="#37474f", font=f)


def render(src, out_png, anchors, size, extra_legend=()):
    dot = out_png.replace(".png", ".dot")
    open(dot, "w").write(src)
    subprocess.run(["dot", "-Tpng", dot, "-o", out_png], check=True)
    subprocess.run(["dot", "-Tsvg", dot, "-o", out_png.replace(".png", ".svg")],
                   check=True)
    plain = subprocess.run(["dot", "-Tplain", dot], check=True,
                           capture_output=True, text=True).stdout
    gh = float(plain.splitlines()[0].split()[3])
    ys = [(gh - float(f[3])) * DPI for f in (ln.split() for ln in plain.splitlines())
          if f and f[0] == "node" and f[1].strip('"') in anchors]
    img = Image.open(out_png)
    cx, cy = img.width / 2.0, sum(ys) / len(ys)
    w, h = size
    w, h = min(w, img.width), min(h, img.height)
    left = max(0, min(int(cx - w / 2), img.width - w))
    top = max(0, min(int(cy - h / 2), img.height - h))
    crop = img.crop((left, top, left + w, top + h)).convert("RGB")
    legend(crop, extra_legend)
    crop.save(out_png.replace(".png", "_crop.png"))
    print(f"{os.path.basename(out_png)}: full {img.size} -> crop {crop.size}")


def gemm_attention(byname, nodelist, sdpa_op, gemm_ops, out,
                   anchor_kuiper=False):
    """Slice around one attention block: sdpa plus the GEMMs closest to it."""
    sdpa = [n["name"] for n in nodelist if n["target"] == sdpa_op]
    pick = sdpa[len(sdpa) // 2]
    gemms = [n["name"] for n in nodelist if n["target"] in gemm_ops]
    idx = {n["name"]: i for i, n in enumerate(nodelist)}
    anchors = [pick] + sorted(gemms, key=lambda g: abs(idx[g] - idx[pick]))[:4]
    keep = window(nodelist, byname, anchors, 22, 22)
    render(dot_source(byname, keep, set(anchors), anchor_kuiper=anchor_kuiper),
           os.path.join(HERE, out), set(anchors), (2000, 1250))


def main():
    post, postlist = contract(*load("post"))
    pre, prelist = contract(*load("pre"))

    # --- 1. GEMM + FlashAttention (post-fusion graph) ---
    gemm_attention(post, postlist, "kuiperjit.sdpa_cudnn.default",
                   ("kuiperjit.mm.default", "kuiperjit.addmm.default"),
                   "dag_gemm_attention.png")

    # --- 2. RMSNorm, before fusion (plain ATen) ---
    means = [n["name"] for n in prelist if n["target"] == "aten.mean.dim"]
    mpick = means[len(means) // 2]
    chain = {mpick} | {i for i in pre[mpick]["inputs"]
                       if pre[i]["target"] == "aten.pow.Tensor_Scalar"}
    frontier = [mpick]
    for _ in range(3):
        nxt = [m["name"] for m in prelist
               if any(f in m["inputs"] for f in frontier)
               and m["target"] in ("aten.add.Scalar", "aten.rsqrt.default")]
        chain.update(nxt)
        frontier = nxt
    keep = window(prelist, pre, list(chain), 14, 14)
    render(dot_source(pre, keep, {mpick}, highlight=chain),
           os.path.join(HERE, "dag_rmsnorm_before.png"), chain, (1700, 1250),
           extra_legend=[(HL_FILL, HL_LINE, "fused into one Kuiper reduction")])

    # --- 3. RMSNorm, after fusion (single kuiperjit reduction) ---
    hr = [n["name"] for n in postlist if n["target"] == "kuiperjit.hreduce_poly.default"]
    hpick = hr[len(hr) // 2]
    keep = window(postlist, post, [hpick], 12, 12)
    render(dot_source(post, keep, {hpick}),
           os.path.join(HERE, "dag_rmsnorm_after.png"), {hpick}, (1700, 1250))

    # --- 4. The same attention block on stock torch.compile (--no-kuiper) ---
    if os.path.exists(os.path.join(HERE, "stock.json")):
        stock, stocklist = contract(*load("stock"))
        gemm_attention(stock, stocklist,
                       "aten._scaled_dot_product_cudnn_attention.default",
                       ("aten.mm.default", "aten.addmm.default"),
                       "dag_gemm_attention_stock.png", anchor_kuiper=True)


if __name__ == "__main__":
    main()
