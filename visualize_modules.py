"""
Visualise the nn.Module hierarchy of the loaded model — i.e. the LAYERS the
user wrote, not the CUDA kernels PyTorch chose to back them with.

Outputs:
  module_graph.png  — collapsed module tree (repeated nn.ModuleList items
                      shown once with a ×N badge).
  Plus a text-mode tree printed to stdout.

Run:
  KUIPER_HOME=/path/to/kuiper PYTHONPATH=. python visualize_modules.py
  KUIPER=0 KUIPER_HOME=... ...                  # skip Kuiper integration

Use the same KUIPER=0 env-var convention as infer.py / profile_ops.py.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR    = Path(__file__).parent
USE_KUIPER = DEVICE == "cuda" and os.environ.get("KUIPER", "1") != "0"


# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} on {DEVICE} …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
)
model.eval()

if USE_KUIPER:
    from kuiper_ext.integration import enable_kuiper
    info = enable_kuiper(
        model, cast_bf16_to_f16=False,
        linear_filter=lambda qname: "lm_head" not in qname,
    )
    print(f"Kuiper integration: replaced {info['linears_replaced']} nn.Linear modules.")


# ── Build a collapsed module graph ────────────────────────────────────────────
# Every node has attributes: cls (str), count (int >=1), depth (int).
# A nn.ModuleList collapses to a single representative child labelled ×N.

def build_module_graph(module: nn.Module, qname: str = "", depth: int = 0,
                       G: nx.DiGraph = None) -> nx.DiGraph:
    if G is None:
        G = nx.DiGraph()
        root = qname or type(module).__name__
        G.add_node(root, cls=type(module).__name__, count=1, depth=depth,
                   leaf=root)
        qname = root

    for child_name, child in module.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            # Skip the ModuleList wrapper; render the first element as the
            # representative and annotate with the count.
            rep = child[0]
            rep_qname = f"{qname}.{child_name}[0..{len(child) - 1}]"
            leaf = f"{child_name}[0..{len(child) - 1}]"
            G.add_node(rep_qname, cls=type(rep).__name__, count=len(child),
                       depth=depth + 1, leaf=leaf)
            G.add_edge(qname, rep_qname)
            build_module_graph(rep, rep_qname, depth + 1, G)
        else:
            child_qname = f"{qname}.{child_name}"
            G.add_node(child_qname, cls=type(child).__name__, count=1,
                       depth=depth + 1, leaf=child_name)
            G.add_edge(qname, child_qname)
            build_module_graph(child, child_qname, depth + 1, G)
    return G


G = build_module_graph(model)
root = next(n for n, d in G.in_degree() if d == 0)
print(f"\nModule graph: {G.number_of_nodes()} unique nodes, {G.number_of_edges()} edges")


# ── Text-mode tree (printed to stdout) ────────────────────────────────────────
def print_tree(G, node, prefix="", is_last=True):
    data = G.nodes[node]
    badge = f" ×{data['count']}" if data["count"] > 1 else ""
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{data['leaf']}  ({data['cls']}{badge})")
    children = list(G.successors(node))
    new_prefix = prefix + ("    " if is_last else "│   ")
    for i, c in enumerate(children):
        print_tree(G, c, new_prefix, i == len(children) - 1)


print("\nModule tree:")
print_tree(G, root)


# ── Hierarchical tree layout ──────────────────────────────────────────────────
def hierarchy_pos(G, root, width=1.0, vert_gap=1.0):
    """Place each leaf in an equal x-slot, centre internal nodes above their
    subtree. Returns {node: (x, y)} with y = -depth."""
    leaf_count_cache = {}
    def leaves(n):
        if n in leaf_count_cache:
            return leaf_count_cache[n]
        ch = list(G.successors(n))
        c = 1 if not ch else sum(leaves(c) for c in ch)
        leaf_count_cache[n] = c
        return c

    pos = {}
    def place(n, x_left, x_right, level):
        pos[n] = ((x_left + x_right) / 2, -level * vert_gap)
        ch = list(G.successors(n))
        if not ch:
            return
        total = sum(leaves(c) for c in ch)
        cum = 0
        for c in ch:
            w = leaves(c)
            cl = x_left + (cum / total) * (x_right - x_left)
            cr = x_left + ((cum + w) / total) * (x_right - x_left)
            place(c, cl, cr, level + 1)
            cum += w

    place(root, 0, width, 0)
    return pos


# ── Colour palette by module class ────────────────────────────────────────────
def _color_for(cls: str) -> str:
    c = cls.lower()
    if "kuiperlinear" in c:           return "#d62728"   # red — replaced!
    if "linear" in c:                  return "#e07b54"   # orange
    if "embedding" in c:               return "#fd8d3c"   # amber
    if "norm" in c:                    return "#74c476"   # green
    if "attention" in c or "attn" in c:return "#6baed6"   # blue
    if "mlp" in c:                     return "#9e9ac8"   # purple
    if "rotary" in c:                  return "#c5b0d5"   # light purple
    if "act" in c or "silu" in c or "gelu" in c:
        return "#dadaeb"
    if "decoderlayer" in c or "block" in c:
        return "#bdbdbd"               # grey
    return "#f7f7f7"                   # white-ish for the rest


# ── Render ────────────────────────────────────────────────────────────────────
pos = hierarchy_pos(G, root, width=1.0, vert_gap=1.0)
fig, ax = plt.subplots(figsize=(20, 11))

# Edges first so nodes draw on top.
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9c9c9c",
                       arrows=False, width=1.0)

# Per-node labels and colours.
labels = {}
colors = []
sizes = []
for n in G.nodes():
    d = G.nodes[n]
    badge = f"\n×{d['count']}" if d["count"] > 1 else ""
    labels[n] = f"{d['leaf']}\n[{d['cls']}]{badge}"
    colors.append(_color_for(d["cls"]))
    # Bigger nodes for repeated blocks so they stand out.
    sizes.append(900 + (d["count"] - 1) * 60)

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                       edgecolors="#444444", linewidths=0.6)
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

legend = [
    mpatches.Patch(color="#d62728", label="KuiperLinear (verified)"),
    mpatches.Patch(color="#e07b54", label="nn.Linear (unmodified)"),
    mpatches.Patch(color="#fd8d3c", label="Embedding"),
    mpatches.Patch(color="#74c476", label="RMSNorm / LayerNorm"),
    mpatches.Patch(color="#6baed6", label="Attention block"),
    mpatches.Patch(color="#9e9ac8", label="MLP block"),
    mpatches.Patch(color="#c5b0d5", label="Rotary embedding"),
    mpatches.Patch(color="#bdbdbd", label="Decoder layer / container"),
    mpatches.Patch(color="#dadaeb", label="Activation"),
    mpatches.Patch(color="#f7f7f7", label="Other"),
]
ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.9)
title = f"Module hierarchy — {MODEL_ID}" + (" (Kuiper enabled)" if USE_KUIPER else "")
ax.set_title(title, fontsize=12)
ax.axis("off")
plt.tight_layout()

out_path = OUT_DIR / "module_graph.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nModule graph saved → {out_path}")
