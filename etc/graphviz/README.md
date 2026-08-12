# Operator-graph diagrams

Presentation figures rendered from the *actual* Inductor post-grad FX graph of
the compiled Qwen2.5-0.5B decode step (bf16, batch 256), not from memory.

* `dag_gemm_attention_crop.png` — attention block slice: RoPE / KV-cache writes
  into `kuiperjit::sdpa_cudnn`, the o-projection `kuiperjit::mm`, the residual
  add and the following `kuiperjit::hreduce_poly`.
* `dag_gemm_attention_stock_crop.png` — the same block under stock
  `torch.compile` (`--no-kuiper`), with the ops a Kuiper kernel claims styled
  exactly as in the Kuiper figure.
* `dag_rmsnorm_before_crop.png` — the same RMSNorm region as plain ATen, with
  the `pow → mean → add → rsqrt` chain highlighted.
* `dag_rmsnorm_after_crop.png` — that chain collapsed into a single
  `kuiperjit::hreduce_poly` by the map-fusion pass.
* `simple_graph.png` — a three-operator toy graph (`mm → add → relu`), for
  explaining what an operator graph *is* without any of the above context.

Pure metadata ops (`reshape`, `permute`, `view`, `expand`, `getitem`, ...) arecontracted away: they rename axes or pick a tuple element rather than moving
data, so consumers are wired straight to their source. When a contracted chain
bottoms out in a graph input, the source tensor is drawn as a blue leaf with its
real name (`layers[12].qkv_w`, `kc_12`, ...), recovered from the Dynamo-level
input names.

Each figure also ships as the uncropped layout (`*.png`), a vector version
(`*.svg`) and the Graphviz source (`*.dot`). The `_crop` variants are a viewport
cut out of the full layout, so the surrounding graph runs off the frame.

## Regenerating

```
python3 etc/graphviz/dump_graph.py              # needs CUDA; pre.json / post.json
python3 etc/graphviz/dump_graph.py --no-kuiper  # stock.json
python3 etc/graphviz/render.py       # needs graphviz + pillow
python3 etc/graphviz/simple_graph.py # standalone, CPU-only; --tb for a tall layout
```

`dump_graph.py` monkeypatches `KuiperPostGradPass.__call__` to snapshot the
graph before and after the pass, then aborts the compile. `render.py` slices a
topological window around the anchor ops, lays it out with `dot`, and crops.
The window sizes and viewports are the constants in `render.py::main`.
`pre.json` / `post.json` / `stock.json` are not committed — they are ~700 KB of regenerable
graph dumps.
