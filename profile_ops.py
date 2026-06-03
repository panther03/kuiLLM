"""
Profile Qwen2.5-0.5B forward pass.

By default, routes compatible nn.Linear layers through Kuiper-verified GEMM
kernels (matches the configuration used by infer.py). Set the env var
KUIPER=0 to profile the unmodified PyTorch baseline for comparison.

Outputs:
  trace.json       — Chrome/Perfetto trace (open at https://ui.perfetto.dev)
  call_tree.txt    — ASCII call tree (parent/child, self+total times, call counts)
  call_tree.png    — Icicle/flame chart of the call tree
  op_graph.png     — Operator DAG from torch.fx symbolic trace
"""

import os
import re
import sys
import bisect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
import torch.fx
import torch.profiler
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR     = Path(__file__).parent
USE_KUIPER  = DEVICE == "cuda" and os.environ.get("KUIPER", "1") != "0"

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} in bf16 on {DEVICE} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model.eval()
print("Model loaded.\n")

# ── Kuiper integration (opt-out via KUIPER=0) ─────────────────────────────────
if USE_KUIPER:
    from kuiper_ext.integration import enable_kuiper

    print("Enabling Kuiper-verified GEMM for nn.Linear layers …")
    info = enable_kuiper(
        model,
        cast_bf16_to_f16=False,   # use the verified bf16 GEMM (TensorCore2D bf16->f32)
        linear_filter=lambda qname: "lm_head" not in qname,
    )
    print(f"  → replaced {info['linears_replaced']} nn.Linear modules\n")
else:
    print("KUIPER=0 in env → profiling unmodified PyTorch baseline.\n")

# ── Prepare input ─────────────────────────────────────────────────────────────
prompt   = "What is a transformer neural network?"
messages = [{"role": "user", "content": prompt}]
text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs   = tokenizer(text, return_tensors="pt").to(DEVICE)

# Kuiper's f16 TensorCore GEMM requires M (= prefill tokens) divisible by 64.
# Pad the prompt with `pad_token`s on the left so the profile actually
# exercises the verified kernel path instead of always falling back. The
# attention mask is extended with zeros for the pad positions.
if USE_KUIPER:
    pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    cur_len = inputs["input_ids"].shape[-1]
    tgt_len = ((cur_len + 63) // 64) * 64
    n_pad   = tgt_len - cur_len
    if n_pad > 0:
        pad_ids = torch.full((1, n_pad), pad_id, dtype=inputs["input_ids"].dtype, device=DEVICE)
        pad_msk = torch.zeros((1, n_pad), dtype=inputs["attention_mask"].dtype, device=DEVICE)
        inputs["input_ids"]      = torch.cat([pad_ids, inputs["input_ids"]],      dim=-1)
        inputs["attention_mask"] = torch.cat([pad_msk, inputs["attention_mask"]], dim=-1)
        print(f"Padded prefill {cur_len} → {tgt_len} tokens so Kuiper GEMM kicks in.\n")

# Warm-up (ensures CUDA kernels are compiled before profiling)
print("Warming up …")
with torch.no_grad():
    _ = model(**inputs)
if DEVICE == "cuda":
    torch.cuda.synchronize()
print("Warm-up done.\n")

# Reset the kuiper/fallback counters AFTER warm-up so they reflect the
# profiled forward pass only.
if USE_KUIPER:
    from kuiper_ext.integration import get_dispatch_stats, reset_dispatch_stats
    reset_dispatch_stats()

# ── Profile ───────────────────────────────────────────────────────────────────
print("Profiling forward pass …")
activities = [torch.profiler.ProfilerActivity.CPU]
if DEVICE == "cuda":
    activities.append(torch.profiler.ProfilerActivity.CUDA)

with torch.profiler.profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    with torch.no_grad():
        # Wrap the forward in a single record_function so the whole call tree
        # has a single synthetic root, instead of dozens of parallel aten:: roots.
        with torch.profiler.record_function("model.forward"):
            _ = model(**inputs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

# ── Kuiper dispatch summary ───────────────────────────────────────────────────
if USE_KUIPER:
    stats = get_dispatch_stats()
    total = stats["kuiper"] + stats["fallback"]
    pct   = 100.0 * stats["kuiper"] / total if total else 0.0
    print(f"\nKuiperLinear dispatch: {stats['kuiper']}/{total} calls used the "
          f"verified GEMM ({pct:.1f}%), {stats['fallback']} fell back to F.linear.")

# ── Chrome trace ──────────────────────────────────────────────────────────────
trace_path = OUT_DIR / "trace.json"
prof.export_chrome_trace(str(trace_path))
print(f"Chrome trace saved → {trace_path}")

# ── Call-tree aggregation (parent/child, no duplication) ──────────────────────
#
# The flat `key_averages()` view double-counts nested calls: a single GEMM
# shows up under aten::linear, aten::matmul, aten::mm, kuiper::matmul_bf16
# AND the CUDA kernel — five separate rows all with ~the same total time.
#
# Walk the profiler's parent/child forest instead. We aggregate sibling calls
# with the same name under the same parent into a single node, summing times
# and counting calls. The result is a clean tree where every micro-second is
# attributed to exactly one inclusive path and one self-time owner.

USE_DEVICE_TIME = DEVICE == "cuda" and any(
    (getattr(e, "device_time_total", 0) or 0) > 0 for e in prof.events()
)
SELF_KEY  = "self_us"
TOTAL_KEY = "total_us"
TIME_LABEL = "CUDA" if USE_DEVICE_TIME else "CPU"


def _aggregate_call_tree(events):
    """Build an aggregated parent/child tree from profiler events.

    Three structural facts about ``prof.events()`` that this function compensates
    for:

    1. **CUDA kernel activities are independent events with no ``cpu_parent``.**
       If we naively used every event, the kernels (e.g. ``cutlass::Kernel2``,
       ``__hoisted_g_gemm_bf16``) would show up as flat siblings of the tree
       root rather than as descendants of the CPU op that launched them.
       We therefore restrict the parent/child walk to ``DeviceType.CPU``
       events.

    2. **PyTorch emits a CUDA-side "flow link" event per** ``record_function``
       **span**, with the SAME name and ``is_user_annotation=True``, one per
       kernel inside the span. So a single ``with record_function("X")``
       block containing 169 kernels yields 1 CPU event + 169 CUDA flow-link
       events all named ``X``. We drop those too.

    3. **Kernels launched from non-aten C++ bindings (e.g. our cpp_extension)
       have empty ``ev.kernels`` lists on their CPU launcher**, because kineto
       only auto-correlates kernels launched from ``aten::*`` ops. Our
       ``record_function("kuiper::matmul_bf16_to_f32")`` is a CPU event but
       the GEMM kernel it launches is orphan. We rescue these by walking the
       CPU events sorted by start time, and attaching each orphan kernel to
       the *deepest* CPU event whose ``[start, end]`` interval contains the
       orphan's interval.
    """

    def new_node(name, is_kernel=False):
        return {
            "name": name, "count": 0, "is_kernel": is_kernel,
            "self_us": 0.0, "total_us": 0.0,
            "children": {},
        }

    total_attr = "device_time_total" if USE_DEVICE_TIME else "cpu_time_total"

    # ── Step 1: map orphan CUDA kernels → deepest containing CPU event ────
    parent_to_orphans = {}  # id(cpu_ev) -> [cuda_ev, ...]
    if USE_DEVICE_TIME:
        cpu_sorted = sorted(
            [e for e in events if str(e.device_type) == "DeviceType.CPU"],
            key=lambda e: e.time_range.start,
        )
        starts = [e.time_range.start for e in cpu_sorted]

        # A CUDA event with the same name AND duration as a kernel already
        # listed in some CPU op's .kernels (kineto-linked) is the SAME kernel
        # and must not be orphan-matched a second time. Track those as a
        # multiset so duplicates (e.g. 168 identical-duration kernels)
        # decrement individually.
        from collections import Counter
        already_linked = Counter()
        for ev in events:
            if str(ev.device_type) != "DeviceType.CPU":
                continue
            for ker in getattr(ev, "kernels", []) or []:
                already_linked[(ker.name, round(float(ker.duration), 3))] += 1

        for cu in events:
            if str(cu.device_type) != "DeviceType.CUDA":
                continue
            if cu.cpu_parent is not None:
                continue
            if getattr(cu, "is_user_annotation", False):
                continue   # CUDA-side flow link for a record_function span
            dur = round(float(cu.time_range.end - cu.time_range.start), 3)
            key = (cu.name, dur)
            if already_linked.get(key, 0) > 0:
                already_linked[key] -= 1
                continue   # same kernel already attached via kineto .kernels list

            cs, cend = cu.time_range.start, cu.time_range.end
            # CUDA kernels run asynchronously: a kernel launched by a CPU op
            # may finish AFTER that CPU op's record_function span ends. So
            # match by the launch *instant* (start) rather than by full
            # interval containment, picking the deepest (smallest) CPU
            # interval whose [start, end] straddles the kernel's start.
            idx = bisect.bisect_right(starts, cs) - 1
            best, best_len = None, float("inf")
            while idx >= 0:
                ce = cpu_sorted[idx]
                s, e = ce.time_range.start, ce.time_range.end
                if e >= cs:                       # CPU op still open at kernel start
                    L = e - s
                    if L < best_len:
                        best, best_len = ce, L
                idx -= 1
            if best is not None:
                parent_to_orphans.setdefault(id(best), []).append(cu)

    # ── Step 2: walk the CPU forest, aggregating by name under each parent ─
    forest = {}

    def visit(ev, sib):
        node = sib.get(ev.name) or sib.setdefault(ev.name, new_node(ev.name))
        node["count"]    += 1
        node["total_us"] += getattr(ev, total_attr, 0) or 0

        if USE_DEVICE_TIME:
            # Kernels kineto already linked to this CPU op (via aten dispatch).
            for ker in getattr(ev, "kernels", []) or []:
                kname = f"⟦cuda⟧ {ker.name}"
                kn = node["children"].get(kname) or node["children"].setdefault(
                    kname, new_node(kname, is_kernel=True))
                kn["count"]    += 1
                kn["self_us"]  += float(ker.duration)
                kn["total_us"] += float(ker.duration)

            # Orphan kernels we matched by timestamp containment.
            for cu in parent_to_orphans.get(id(ev), []):
                kname = f"⟦cuda⟧ {cu.name}"
                kn = node["children"].get(kname) or node["children"].setdefault(
                    kname, new_node(kname, is_kernel=True))
                d = float(cu.time_range.end - cu.time_range.start)
                kn["count"]    += 1
                kn["self_us"]  += d
                kn["total_us"] += d

        for ch in ev.cpu_children:
            if str(ch.device_type) == "DeviceType.CPU":
                visit(ch, node["children"])

    for ev in events:
        if ev.cpu_parent is None and str(ev.device_type) == "DeviceType.CPU":
            visit(ev, forest)

    # ── Step 3: bottom-up fix-up so total >= sum(children.total) and ──────
    # self = total - sum(children.total). Necessary because orphan kernels
    # we attached to a child don't show up in the kineto-reported device_time
    # of the parent — we bump parent totals up to cover them.
    def fixup(nodes):
        for n in nodes.values():
            fixup(n["children"])
            ct = sum(c["total_us"] for c in n["children"].values())
            if ct > n["total_us"]:
                n["total_us"] = ct
            if not n["is_kernel"]:
                n["self_us"] = max(0.0, n["total_us"] - ct)
    fixup(forest)

    return forest


def _fmt_us(us: float) -> str:
    if us >= 1000.0:
        return f"{us / 1000.0:8.3f} ms"
    return f"{us:8.1f} µs"


# Heuristics for shortening fully-mangled C++ kernel names. We strip template
# parameters and surrounding namespaces so the tree stays readable, but keep
# the bit of the name that identifies the actual kernel. Applied in order.
_KERNEL_SHORTEN_RULES = [
    # Kuiper extracted kernels: __hoisted_g_gemm_bf16_f32_64x64x64_..(args)
    (re.compile(r"^__hoisted_(.+?)\(.*$"),                  r"\1"),
    # cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_…>(::Params)
    (re.compile(r"cutlass::Kernel2<([^>]+?)>.*"),           r"cutlass[\1]"),
    # PyTorchMemEffAttention::AttentionKernel<…>::Params
    (re.compile(r"PyTorchMemEffAttention::AttentionKernel<[^>]*>(::\w+)?"),
                                                            "MemEffAttention"),
    # at::native::elementwise_kernel<…BinaryFunctor<…,MulFunctor<…>,…>…>
    # Pull out the operation name (Mul, Add, Sub, Div, …) and keep "elementwise<Mul>".
    (re.compile(r"\b(?:vectorized_)?elementwise_kernel<[^>]*?(\w+)Functor<[^>]*?>[^>]*>",
                ),                                          r"elementwise<\1>"),
    (re.compile(r"\belementwise_kernel<[^>]*?CUDAFunctor_(\w+)[^>]*>"),
                                                            r"elementwise<\1>"),
    # named cuda kernels with a recognisable suffix: foo_kernel_cuda / foo_kernel
    (re.compile(r"\b([a-zA-Z_][\w]*?_kernel)(?:_cuda)?<[^>]*>"),
                                                            r"\1"),
    (re.compile(r"\b([a-zA-Z_][\w]*?_kernel_cuda)\b"),       r"\1"),
    # reduce_kernel templated by op kind
    (re.compile(r"reduce_kernel<[^>]*?(\w+)Ops[^>]*>"),     r"reduce<\1>"),
    # Drop noisy namespaces / void prefix
    (re.compile(r"\bat::native::(\(anonymous namespace\)::)?"), ""),
    (re.compile(r"\bvoid\s+"),                              ""),
    # Drop trailing argument list (anything after first balanced "(" we couldn't parse)
    (re.compile(r"\([^()]{30,}\)"),                         "(…)"),
    # Collapse any remaining big template arg
    (re.compile(r"<[^<>]{40,}>"),                           "<…>"),
]


def _shorten_kernel(name: str) -> str:
    """Make a mangled CUDA kernel name readable. Idempotent."""
    if not name.startswith("⟦cuda⟧ "):
        return name
    inner = name[len("⟦cuda⟧ "):]
    for rx, repl in _KERNEL_SHORTEN_RULES:
        inner = rx.sub(repl, inner)
    # Collapse remaining whitespace runs.
    inner = re.sub(r"\s{2,}", " ", inner).strip()
    if len(inner) > 80:
        inner = inner[:77] + "…"
    return "⟦cuda⟧ " + inner


def _print_call_tree(forest, total_root_us, out, max_depth=12, min_pct=0.2):
    """Print the aggregated tree as an indented ASCII outline.

    Nodes below `min_pct` of root time are folded into a "…" summary row so the
    output stays readable.
    """
    print(
        f"{'NODE':<70} {'TOTAL':>13} {'SELF':>13} {'CALLS':>7}  {'%':>5}",
        file=out,
    )
    print("─" * 115, file=out)

    def walk(nodes, depth, prefix=""):
        if depth > max_depth:
            return
        items = sorted(nodes.values(), key=lambda n: -n["total_us"])
        # Split into "shown" (above threshold) and "folded" (below threshold).
        shown, folded = [], []
        for n in items:
            pct = (n["total_us"] / total_root_us * 100.0) if total_root_us else 0.0
            (shown if pct >= min_pct else folded).append((n, pct))
        for i, (n, pct) in enumerate(shown):
            is_last = (i == len(shown) - 1 and not folded)
            branch  = "└─ " if is_last else "├─ "
            count   = f"×{n['count']}" if n["count"] > 1 else ""
            disp    = _shorten_kernel(n["name"])
            label   = f"{prefix}{branch}{disp} {count}"
            print(
                f"{label:<70} {_fmt_us(n['total_us']):>13} {_fmt_us(n['self_us']):>13} "
                f"{n['count']:>7}  {pct:>4.1f}%",
                file=out,
            )
            if n["children"]:
                ext = "   " if is_last else "│  "
                walk(n["children"], depth + 1, prefix + ext)
        if folded:
            folded_total = sum(n["total_us"] for n, _ in folded)
            folded_self  = sum(n["self_us"]  for n, _ in folded)
            folded_pct   = (folded_total / total_root_us * 100.0) if total_root_us else 0.0
            label = f"{prefix}└─ … {len(folded)} small nodes (< {min_pct:g}% each)"
            print(
                f"{label:<70} {_fmt_us(folded_total):>13} {_fmt_us(folded_self):>13} "
                f"{'':>7}  {folded_pct:>4.1f}%",
                file=out,
            )

    walk(forest, 0)


# ── Icicle / flame chart renderer ─────────────────────────────────────────────

def _node_color(name: str) -> str:
    n = name.lower()
    if "hoisted_g_gemm" in n or "klas_gemm" in n:
        return "#d94545"   # red       – Kuiper verified GEMM kernel
    if n.startswith("kuiper::") or "kuiper_" in n:
        return "#e85a4f"   # red-orange – Kuiper CPU wrapper
    if "cutlass" in n or "tensorop" in n or "gemm" in n:
        return "#e07b54"   # orange    – cuBLAS/CUTLASS GEMM
    if "memeffattention" in n or "fmha_" in n:
        return "#3182bd"   # darker blue – fused attention kernel
    if "mm" in n or "matmul" in n or "linear" in n:
        return "#f4a582"   # peach     – matmul/linear (CPU side)
    if "softmax" in n:
        return "#6baed6"   # blue      – softmax / attention
    if "scaled_dot" in n or "attention" in n or "sdp" in n:
        return "#3182bd"   # darker blue
    if "norm" in n or "rms" in n:
        return "#74c476"   # green     – norm
    if "silu" in n or "gelu" in n or "act" in n or "sigmoid" in n:
        return "#9e9ac8"   # purple    – activation
    if "embed" in n or "gather" in n:
        return "#fdbe85"   # amber     – embedding
    if "copy" in n or "cast" in n or "to" == n or "::to" in n:
        return "#e9c46a"   # yellow    – copies / casts
    if "add" in n or "mul" in n or "sub" in n or "div" in n:
        return "#bdbdbd"   # grey      – elementwise
    if "cuda" in n or "cudnn" in n:
        return "#a8a8a8"
    return "#f0f0f0"


def _layout_icicle(nodes, x0, y0, width, denom_us):
    """Lay out children inside a parent strip of `width` units.

    Width of each child is (child.total_us / denom_us) * width.
    Returns list of (node, x, y, w) rectangles.
    """
    rects = []
    if denom_us <= 0 or width <= 0:
        return rects
    items = sorted(nodes.values(), key=lambda n: -n["total_us"])
    cx = x0
    for n in items:
        w = (n["total_us"] / denom_us) * width
        if w >= 0.4:   # don't bother drawing pixel-slivers
            rects.append((n, cx, y0, w))
            rects.extend(_layout_icicle(n["children"], cx, y0 + 1, w, n["total_us"]))
        cx += w
    return rects


def _truncate(label: str, max_chars: int) -> str:
    if max_chars <= 1:
        return ""
    if len(label) <= max_chars:
        return label
    if max_chars <= 3:
        return label[:max_chars]
    return label[: max_chars - 1] + "…"


def _render_icicle(forest, path, title):
    canvas_w = 2000.0
    total = sum(n["total_us"] for n in forest.values())
    if total <= 0:
        print(f"[icicle] no {TIME_LABEL.lower()} time recorded; skipping {path.name}")
        return
    rects = _layout_icicle(forest, 0.0, 0, canvas_w, total)
    if not rects:
        print(f"[icicle] no rectangles to draw; skipping {path.name}")
        return
    max_depth = max(r[2] for r in rects) + 1

    fig_h = max(5.0, 0.55 * max_depth + 1.5)
    fig, ax = plt.subplots(figsize=(22, fig_h))
    for n, x, y, w in rects:
        ax.add_patch(mpatches.Rectangle(
            (x, -(y + 1)), w, 1.0,
            facecolor=_node_color(n["name"]),
            edgecolor="white", linewidth=0.4,
        ))
        if w >= 25.0:
            # crude estimate of chars that fit
            max_chars = max(2, int(w / 8.0))
            ax.text(x + w / 2.0, -(y + 0.5),
                    _truncate(_shorten_kernel(n["name"]), max_chars),
                    ha="center", va="center", fontsize=6,
                    color="#222222")

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(-max_depth - 0.2, 0.2)
    ax.set_yticks([-(d + 0.5) for d in range(max_depth)])
    ax.set_yticklabels([f"depth {d}" for d in range(max_depth)], fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel(f"inclusive {TIME_LABEL} time  (full width = {_fmt_us(total).strip()})")
    ax.set_title(title, fontsize=12)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    legend = [
        mpatches.Patch(color="#d94545", label="kuiper:: (verified)"),
        mpatches.Patch(color="#e07b54", label="cuBLAS / CUTLASS GEMM"),
        mpatches.Patch(color="#f4a582", label="matmul / linear (dispatch)"),
        mpatches.Patch(color="#6baed6", label="softmax"),
        mpatches.Patch(color="#74c476", label="norm"),
        mpatches.Patch(color="#9e9ac8", label="activation"),
        mpatches.Patch(color="#fdbe85", label="embedding"),
        mpatches.Patch(color="#e9c46a", label="copy / cast"),
        mpatches.Patch(color="#bdbdbd", label="elementwise"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, ncol=3, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Icicle chart saved → {path}")


# ── Build & render the call tree ──────────────────────────────────────────────
print("\nAggregating profiler call tree …")
forest = _aggregate_call_tree(prof.events())
root_total = sum(n["total_us"] for n in forest.values())
print(f"  → tree built: {len(forest)} root nodes, "
      f"{_fmt_us(root_total).strip()} total {TIME_LABEL} time captured.\n")

# Stdout summary (top of the tree, deep enough to see the GEMM chain)
print(f"── Call tree (sorted by total {TIME_LABEL} time) ──────────────────────")
_print_call_tree(forest, root_total, sys.stdout, max_depth=8, min_pct=0.5)

# Full tree to a text file so nothing is lost
calltree_path = OUT_DIR / "call_tree.txt"
with calltree_path.open("w") as f:
    f.write(f"Profile of {MODEL_ID}  ({TIME_LABEL} time)\n")
    f.write(f"Kuiper: {'on' if USE_KUIPER else 'off'}\n")
    f.write(f"Total inclusive {TIME_LABEL} time at roots: {_fmt_us(root_total).strip()}\n\n")
    _print_call_tree(forest, root_total, f, max_depth=15, min_pct=0.05)
print(f"\nFull tree saved → {calltree_path}")

# Icicle / flame chart
_render_icicle(
    forest,
    OUT_DIR / "call_tree.png",
    title=f"Call tree (icicle, {TIME_LABEL} time) — {MODEL_ID}",
)

# Keep `kernel_times` for the FX graph renderer below (it expects a dict
# mapping op-name → ms). Extract per-name *self* time from the tree so that
# the colour-coded FX nodes are sized by actual self work, not duplicated
# inclusive time.
def _flatten_self(forest, out):
    for n in forest.values():
        out[n["name"]] = out.get(n["name"], 0.0) + n["self_us"] / 1000.0
        _flatten_self(n["children"], out)
    return out

kernel_times = _flatten_self(forest, {})

# ── Operator DAG via torch.fx ─────────────────────────────────────────────────
# ── Helper functions ──────────────────────────────────────────────────────────

def _op_color(op: str) -> str:
    """Map operation categories to colours."""
    op = op.lower()
    if "mm" in op or "matmul" in op or "linear" in op:
        return "#e07b54"   # orange  – matmul / linear
    if "softmax" in op or "attention" in op or "scaled_dot" in op:
        return "#6baed6"   # blue    – attention
    if "norm" in op or "layer_norm" in op or "rms" in op:
        return "#74c476"   # green   – norm
    if "silu" in op or "gelu" in op or "act" in op or "sigmoid" in op:
        return "#9e9ac8"   # purple  – activation
    if "embed" in op or "gather" in op:
        return "#fd8d3c"   # amber   – embedding
    if "add" in op or "mul" in op or "sub" in op:
        return "#bdbdbd"   # grey    – elementwise
    return "#f7f7f7"       # white   – other


def _render_graph(G, path, color_fn, ktimes, title):
    pos    = nx.planar_layout(G)
    colors = [color_fn(n) for n in G.nodes()]
    sizes  = []
    for n in G.nodes():
        t = max((v for k, v in ktimes.items() if k.lower() in n.lower()), default=0.1)
        sizes.append(300 + t * 80)

    fig, ax = plt.subplots(figsize=(20, 14))
    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=colors, node_size=sizes,
        font_size=6, arrows=True,
        edge_color="#cccccc", width=0.8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )
    legend = [
        mpatches.Patch(color="#e07b54", label="matmul / linear"),
        mpatches.Patch(color="#6baed6", label="attention / softmax"),
        mpatches.Patch(color="#74c476", label="norm"),
        mpatches.Patch(color="#9e9ac8", label="activation"),
        mpatches.Patch(color="#fd8d3c", label="embedding"),
        mpatches.Patch(color="#bdbdbd", label="elementwise"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Operator graph saved → {path}")


def _draw_fx_graph(G, path, color_fn, ktimes):
    """Draw the FX DAG, collapsing to unique op-type nodes for readability."""
    op_graph = nx.DiGraph()
    for n, data in G.nodes(data=True):
        op_graph.add_node(data["label"])
    for u, v in G.edges():
        ul = G.nodes[u]["label"]
        vl = G.nodes[v]["label"]
        if ul != vl:
            op_graph.add_edge(ul, vl)
    _render_graph(op_graph, path, color_fn, ktimes, title=f"Operator DAG — {MODEL_ID}")


def _draw_tree_fallback_graph(forest, path, color_fn, ktimes):
    """Fall-back when fx.symbolic_trace fails: draw the call tree as a chain
    of the top-N most-expensive nodes (by self time)."""
    flat = []
    def collect(nodes):
        for n in nodes.values():
            flat.append((n["name"], n["self_us"] / 1000.0))
            collect(n["children"])
    collect(forest)
    flat.sort(key=lambda x: -x[1])
    top = flat[:30]
    G = nx.DiGraph()
    for i, (name, _) in enumerate(top):
        G.add_node(name)
        if i > 0:
            G.add_edge(top[i - 1][0], name)
    _render_graph(G, path, color_fn, ktimes, title=f"Top operators (call-tree fallback) — {MODEL_ID}")


# ── Build operator graph ──────────────────────────────────────────────────────
print("\nBuilding operator graph via torch.fx …")

try:
    graph_module = torch.fx.symbolic_trace(model)
    g: torch.fx.Graph = graph_module.graph

    G = nx.DiGraph()
    for node in g.nodes:
        label = node.op if node.op != "call_function" else str(node.target).split(".")[-1]
        G.add_node(node.name, label=label, op=node.op, target=str(node.target))
    for node in g.nodes:
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                G.add_edge(arg.name, node.name)
        for arg in node.kwargs.values():
            if isinstance(arg, torch.fx.Node):
                G.add_edge(arg.name, node.name)

    print(f"FX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    _draw_fx_graph(G, OUT_DIR / "op_graph.png", _op_color, kernel_times)

except Exception as fx_err:
    print(f"torch.fx trace failed ({fx_err})\nFalling back to call-tree-based graph …")
    _draw_tree_fallback_graph(forest, OUT_DIR / "op_graph.png", _op_color, kernel_times)

print("\nDone.")
