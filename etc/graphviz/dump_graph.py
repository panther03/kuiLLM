"""Dump the real Inductor post-grad FX graph of the compiled Qwen2.5 decode step.

Writes two JSON files: ``pre.json`` (graph as Inductor hands it to the Kuiper
pass, i.e. plain ATen) and ``post.json`` (after fusion + kuiperjit replacement).

With ``--no-kuiper`` the backend is not installed at all (a trace-only pass just
observes the graph) and the single stock graph is written to ``stock.json``.
"""
import json
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(OUT))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "etc"))

import torch
import torch.nn.functional as F

import kuipy
from kuipy.inductor import passes


class Done(Exception):
    pass


DYNAMO_INPUTS = []


def pretty(name):
    """``g_import_main_m_layers_12_qkv_w_`` -> ``layers[12].qkv_w``."""
    import re
    name = name.strip("_")
    name = re.sub(r"^[GL]_", "", name, flags=re.I)
    name = name.replace("import_main_", "")
    name = re.sub(r"^m_", "", name, flags=re.I)
    return re.sub(r"^layers_(\d+)_", r"layers[\1].", name)


def snap(graph):
    nodes = []
    ph = 0
    for n in graph.nodes:
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            desc = f"{tuple(val.shape)} {str(val.dtype).replace('torch.', '')}"
        elif isinstance(val, (list, tuple)):
            t = next((v for v in val if isinstance(v, torch.Tensor)), None)
            desc = ("tuple: " + f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')}"
                    if t is not None else "tuple")
        else:
            desc = ""
        src = ""
        if n.op == "placeholder":
            if ph < len(DYNAMO_INPUTS):
                src = pretty(DYNAMO_INPUTS[ph])
            ph += 1
        nodes.append({
            "name": n.name,
            "src": src,
            "op": n.op,
            "target": str(n.target),
            "inputs": [i.name for i in n.all_input_nodes],
            "val": desc,
        })
    return nodes


_orig = passes.KuiperPostGradPass.__call__


NO_KUIPER = "--no-kuiper" in sys.argv


def write(name, nodes):
    json.dump(nodes, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(f"[dump] {name}: {len(nodes)} nodes", flush=True)


def patched(self, graph):
    pre = snap(graph)
    _orig(self, graph)
    if len(pre) < 50:  # skip tiny helper graphs
        return
    if NO_KUIPER:
        write("stock", pre)
    else:
        write("pre", pre)
        write("post", snap(graph))
    raise Done()


passes.KuiperPostGradPass.__call__ = patched

import infer_golden as ig
from infer_golden_compiled import block, _force_cudnn_sdpa

ig._tune_backend()
_force_cudnn_sdpa()
if NO_KUIPER:
    from kuipy import inductor
    inductor.enable_tracing()  # observe the stock graph, replace nothing
else:
    kuipy.enable()
torch._inductor.config.force_disable_caches = True

tok, m = ig.load()
DEVICE, DTYPE = ig.DEVICE, ig.DTYPE

b, plen, total = 256, 8, 64
ids = torch.zeros(b, plen, dtype=torch.long, device=DEVICE)
kc = [torch.zeros(b, m.NKV, total, m.D, device=DEVICE, dtype=DTYPE) for _ in range(m.NL)]
vc = [torch.zeros(b, m.NKV, total, m.D, device=DEVICE, dtype=DTYPE) for _ in range(m.NL)]
cos_t, sin_t = m._rope_tables(total)
kp = torch.arange(total, device=DEVICE)
neg = torch.finfo(DTYPE).min
mask_rows = torch.where(kp[None, :] <= kp[:, None],
                        torch.zeros((), dtype=DTYPE, device=DEVICE),
                        torch.full((), neg, dtype=DTYPE, device=DEVICE))
tok_in = torch.zeros(b, 1, dtype=torch.long, device=DEVICE)
pos = torch.tensor([plen], device=DEVICE)


def decode_step():
    cos = cos_t.index_select(0, pos).view(1, 1, 1, m.D)
    sin = sin_t.index_select(0, pos).view(1, 1, 1, m.D)
    mask = mask_rows.index_select(0, pos).view(1, 1, 1, total)
    h = F.embedding(tok_in, m.embed)
    for li in range(m.NL):
        h = block(m, h, cos, sin, kc[li], vc[li], m.layers[li], mask, False, pos)
    h = F.rms_norm(h, (m.HID,), m.final_norm, m.EPS)
    return F.linear(h, m.lm_head).argmax(-1)


def backend(gm, example_inputs):
    """Capture the Dynamo-level input names (the post-grad placeholders are
    renamed to arg0_1... in the same order) before handing off to Inductor."""
    from torch._dynamo.backends.registry import lookup_backend
    DYNAMO_INPUTS[:] = [str(n.target) for n in gm.graph.nodes
                        if n.op == "placeholder"]
    return lookup_backend("inductor")(gm, example_inputs)


compiled = torch.compile(decode_step, backend=backend, fullgraph=True)
try:
    with torch.inference_mode():
        compiled()
except Done:
    print("[dump] done")
except Exception as e:  # torch wraps user exceptions
    if "Done" not in repr(e):
        raise
    print("[dump] done (wrapped)")
