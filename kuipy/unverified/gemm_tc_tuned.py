"""Autotuned front-end for the templated tensor-core GEMM in gemm_tc.cu.

The kernel's tiling, split-K factor and L2 swizzle width are runtime arguments
into a compile-time table of instantiations, so every candidate is already
built: tuning is pure benchmarking with no JIT step. That is the whole reason
the table exists, and it lets an unverified kernel go through the same
``kuipy.autotune`` machinery as a Kuiper one.
"""
import torch

from .. import autotune

_configs = None
_raw_fn = None
_cache = {}


def _raw():
    """The untuned binding, bound once: the extension lookup is not free."""
    global _raw_fn
    if _raw_fn is None:
        from . import module
        _raw_fn = module().gemm_tc
    return _raw_fn


def configs():
    """The tiling table as {bm, bn, bk, wm, wn, stages, skew, warps, smem} dicts."""
    global _configs
    if _configs is None:
        from . import module
        fields = ("bm", "bn", "bk", "wm", "wn", "stages", "skew", "warps", "smem")
        _configs = [dict(zip(fields, row)) for row in module().gemm_tc_configs()]
    return _configs


def _candidates(M, N, K, device):
    props = torch.cuda.get_device_properties(device)
    smem_cap = getattr(props, "shared_memory_per_block_optin",
                       props.shared_memory_per_block)
    sms = props.multi_processor_count

    out = []
    for index, cfg in enumerate(configs()):
        if M % cfg["bm"] or N % cfg["bn"] or K % cfg["bk"]:
            continue
        if cfg["smem"] > smem_cap:
            continue
        tiles = (M // cfg["bm"]) * (N // cfg["bn"])
        ktiles = K // cfg["bk"]
        # Splitting K only buys anything while the M/N tiling leaves SMs idle,
        # and each split costs a pass over an fp32 (splits, M, N) workspace.
        splits = [1]
        if tiles < sms:
            splits += [s for s in (2, 4, 8, 16)
                       if tiles * s <= 2 * sms and ktiles // s >= 2]
        # The tiling goes in the spec as well as its index so that a recorded
        # winner stops matching if the table is ever reordered.
        tile = ("{bm}x{bn}x{bk}w{wm}x{wn}s{stages}".format(**cfg))
        for s in splits:
            out.append({"config": index, "tile": tile, "splits": s, "group": 8,
                        "tiles": tiles, "area": cfg["bm"] * cfg["bn"]})

    # Priority order, which is also the fallback when there is no tuning entry:
    # fill the machine first, then prefer the tile with the most reuse.
    out.sort(key=lambda c: (c["tiles"] * c["splits"] < sms, -c["area"], c["splits"]))
    for c in out:
        del c["tiles"], c["area"]
    return out


def _run(spec, args, kwargs):
    bias = args[0] if len(args) == 3 else None
    mat1, mat2 = args[-2], args[-1]
    return _raw()(bias, mat1, mat2, config=spec["config"], splits=spec["splits"],
                  group=spec["group"], **kwargs)


def _select(M, N, K, epi, args, kwargs, device):
    candidates = _candidates(M, N, K, device)
    if not candidates:
        raise RuntimeError(
            f"no gemm_tc tiling divides {M}x{N}x{K}; it requires M%bm==0, "
            "N%bn==0 and K%bk==0 for some table entry"
        )
    key = autotune.make_key("unverified.gemm_tc", args, {"epi": epi})
    return autotune.tune(key, kwargs, candidates, _run, device)


def gemm_tc(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    """D = alpha*(mat1@mat2) + beta*input, with the tiling chosen by autotuning.

    ``input`` may be None (pure matmul), an (M, N) matrix, or a length-N vector
    broadcast over the rows.
    """
    M, K = mat1.shape
    N = mat2.shape[1]
    epi = 0 if input is None else input.dim()
    # Hashing the autotune key and re-deriving the candidate list costs more
    # than the kernel does on the smaller shapes, so the selection is memoized
    # on everything it depends on.
    cache_key = (M, N, K, mat1.dtype, epi, mat1.device.index)
    spec = _cache.get(cache_key)
    kwargs = {"beta": beta, "alpha": alpha}
    if spec is None:
        args = [mat1, mat2] if input is None else [input, mat1, mat2]
        spec = _cache[cache_key] = _select(M, N, K, epi, args, kwargs,
                                           mat1.device)
    return _raw()(input, mat1, mat2, config=spec["config"],
                  splits=spec["splits"], group=spec["group"], **kwargs)
