"""Shape-specific autotuning for verified Kuiper kernel instantiations.

An operator's ``supported()`` hands ``tune`` every legal parameterization of a
call as a list of specs. Normal runs consume the committed tuning file (falling
back to the first candidate on a miss); ``KUIPY_AUTOTUNE=1`` benchmarks each
candidate and records the winner.

Benchmarking never sees the caller's tensors: it synthesizes its own from the
shapes/strides/dtypes in the key, so a selection made while tracing a graph
(where the caller's tensors are fake) is still measured on real memory.
"""
import hashlib
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filelock import FileLock
import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from . import compile as _compile
from . import config as C

# cache_key -> winning spec, so a hot call is a dict lookup.
_selected = {}
_locks = {}
_device_locks = {}
_registry_lock = threading.Lock()


def _normalize(value):
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    raise TypeError(f"unsupported autotune key value: {value!r}")


def make_key(operation, args, parameters=None):
    """Identify a call by its tensor inputs, the selection-relevant parameters
    and the GPU. The tensor descriptions double as the recipe for synthesizing
    inputs when benchmarking."""
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if not tensors:
        raise ValueError(f"{operation} has no tensor inputs")
    device = tensors[0].device
    if device.type != "cuda":
        raise ValueError(f"{operation} autotuning requires CUDA tensors")
    if any(t.device != device for t in tensors):
        raise ValueError(f"{operation} inputs must be on one CUDA device")
    return {
        "tuning_schema": C.TUNING_SCHEMA_VERSION,
        "operation": operation,
        "inputs": [
            {
                "shape": [int(v) for v in tensor.shape],
                "strides": [int(v) for v in tensor.stride()],
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            for tensor in tensors
        ],
        "parameters": _normalize(parameters or {}),
        "gpu": torch.cuda.get_device_name(device),
    }


def key_hash(key):
    encoded = json.dumps(
        key, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class TuneStore:
    def __init__(self, path=None):
        self.path = Path(path or C.TUNE_PARAMS_PATH).resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    @staticmethod
    def _empty():
        return {
            "tuning_schema": C.TUNING_SCHEMA_VERSION,
            "entries": {},
        }

    def _read(self):
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to read Kuiper tuning parameters from {self.path}"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            raise RuntimeError(f"invalid Kuiper tuning parameter file: {self.path}")
        if data.get("tuning_schema") != C.TUNING_SCHEMA_VERSION:
            return self._empty()
        return data

    def lookup(self, digest, key):
        entry = self._read()["entries"].get(digest)
        if entry is None:
            return None
        if entry.get("key") != key:
            raise RuntimeError(
                f"autotune key collision or corrupt entry {digest} in {self.path}"
            )
        if not isinstance(entry.get("spec"), dict):
            raise RuntimeError(f"invalid autotune entry {digest} in {self.path}")
        return entry

    def update(self, digest, key, spec):
        C.KUIPY_CACHE.mkdir(parents=True, exist_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            data = self._read()
            existing = data["entries"].get(digest)
            if existing is not None and existing.get("key") != key:
                raise RuntimeError(
                    f"autotune key collision for {digest} in {self.path}"
                )
            data["entries"][digest] = {"key": key, "spec": _normalize(spec)}
            ordered = {
                "tuning_schema": C.TUNING_SCHEMA_VERSION,
                "entries": {
                    k: data["entries"][k] for k in sorted(data["entries"])
                },
            }
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "w") as out:
                    json.dump(ordered, out, indent=2, sort_keys=True)
                    out.write("\n")
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(tmp, self.path)
            finally:
                if tmp.exists():
                    tmp.unlink()

    def referenced_modules(self):
        return {
            entry["spec"]["module"]
            for entry in self._read()["entries"].values()
            if isinstance(entry, dict) and isinstance(entry.get("spec"), dict)
            and entry["spec"].get("module") is not None
        }


def _key_lock(cache_key):
    with _registry_lock:
        return _locks.setdefault(cache_key, threading.Lock())


def _device_lock(device):
    with _registry_lock:
        return _device_locks.setdefault(str(device), threading.Lock())


def _gpu_lock_path(key, device):
    identity = f"{key['gpu']}|{device}"
    if device is not None:
        identity = str(torch.cuda.get_device_properties(device).uuid)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / (
        f"kuipy-autotune-{os.getuid()}-{digest}.lock"
    )


def _match(candidates, normalized):
    for spec in candidates:
        if _normalize(spec) == normalized:
            return spec
    return None


def _synthesize_args(key, device):
    return [
        torch.empty_strided(
            entry["shape"], entry["strides"],
            dtype=getattr(torch, entry["dtype"]), device=device,
        ).zero_()
        for entry in key["inputs"]
    ]


def _capture(spec, run_candidate, args, kwargs):
    """Record ``AUTOTUNE_BATCH`` back-to-back launches into a CUDA graph.

    Dispatching a candidate from Python costs tens of microseconds, which is
    more than a decode-shaped GEMM takes on the device: timed eagerly, every
    candidate measures the same and the winner is picked out of host noise.
    Replaying a graph launches the kernels with no host in the loop, which is
    also how they run in the model. Returns ``None`` if the candidate cannot be
    captured, and the caller falls back to eager timing."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run_candidate(spec, args, kwargs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(C.AUTOTUNE_GRAPH_BATCH):
                run_candidate(spec, args, kwargs)
    except Exception as exc:
        C.log(f"cuda-graph capture failed for {spec.get('module')}: {exc}")
        return None
    return graph


def _time(launch, iters):
    timings = []
    for _ in range(C.AUTOTUNE_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) / iters)
    # Interference from the rest of the machine can only ever make a sample
    # slower, so the fastest one is the best estimate of the kernel's cost.
    return min(timings)


def _benchmark(spec, run_candidate, args, kwargs, device):
    with torch.cuda.device(device):
        run_candidate(spec, args, kwargs)
        torch.cuda.synchronize(device)
        graph = _capture(spec, run_candidate, args, kwargs)
        if graph is not None:
            try:
                graph.replay()
                torch.cuda.synchronize(device)
                return _time(graph.replay, C.AUTOTUNE_GRAPH_BATCH)
            finally:
                # Every capture owns a private pool holding one output tensor
                # per recorded launch; on a wide GEMM that is hundreds of MB.
                graph.reset()
                del graph

        for _ in range(C.AUTOTUNE_WARMUP):
            run_candidate(spec, args, kwargs)
        torch.cuda.synchronize(device)

        def launch():
            for _ in range(C.AUTOTUNE_BATCH):
                run_candidate(spec, args, kwargs)
        return _time(launch, C.AUTOTUNE_BATCH)


def _prebuild(candidates, run_candidate, args, kwargs, device):
    """Build every candidate's kernel before any of them is timed.

    A candidate that is not in the on-disk cache costs an F* extraction and an
    nvcc compile, both external processes, so threads give real parallelism;
    the per-kernel file lock in ``compile`` keeps distinct kernels independent.
    Timing itself stays strictly serial. Failures are ignored here -- the
    timing loop hits them again and records them properly."""
    jobs = min(C.AUTOTUNE_BUILD_JOBS, len(candidates))
    if jobs <= 1:
        return

    def build(spec):
        try:
            with torch.cuda.device(device):
                run_candidate(spec, args, kwargs)
        except Exception:
            pass

    print(f"[kuipy-autotune] building {len(candidates)} candidates "
          f"({jobs} jobs)", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(build, candidates))
    torch.cuda.synchronize(device)


def _cleanup(candidates, winner, store):
    # Specs without a "module" name a kernel that was built ahead of time (the
    # unverified table-driven ones), so there is no losing artifact to remove.
    referenced = store.referenced_modules()
    with _registry_lock:
        referenced.update(spec["module"] for spec in _selected.values()
                          if spec.get("module") is not None)
    for spec in candidates:
        module = spec.get("module")
        if module is not None and module != winner.get("module") \
                and module not in referenced:
            try:
                _compile.delete_kernel(module)
            except OSError as exc:
                C.log(f"failed to remove losing kernel {module}: {exc}")


def _tune_now(key, kwargs, candidates, run_candidate, device, store, digest):
    if _compile.is_capturing():
        raise RuntimeError(
            "KUIPY_AUTOTUNE cannot run inside kuipy.batch_capture()"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "KUIPY_AUTOTUNE cannot run during CUDA graph capture; "
            "run the tuning pass without CUDA graphs"
        )

    print(
        f"[kuipy-autotune] tuning {key['operation']} "
        f"({len(candidates)} candidates, {digest[:12]})",
        flush=True,
    )
    args = _synthesize_args(key, device)
    _prebuild(candidates, run_candidate, args, kwargs, device)
    results = []
    failures = []
    for index, spec in enumerate(candidates, 1):
        print(f"[kuipy-autotune] [{index}/{len(candidates)}] {spec}", flush=True)
        try:
            timing = _benchmark(spec, run_candidate, args, kwargs, device)
        except RuntimeError as exc:
            failures.append((spec, str(exc)))
            print(f"[kuipy-autotune] candidate failed: {exc}", flush=True)
            continue
        results.append((timing, index, spec))
        print(f"[kuipy-autotune] {timing:.4f} ms", flush=True)

    if not results:
        detail = "\n".join(f"{spec}: {error}" for spec, error in failures)
        raise RuntimeError(
            f"all autotune candidates failed for {key['operation']}:\n{detail}"
        )

    _, _, winner = min(results, key=lambda result: (result[0], result[1]))
    store.update(digest, key, winner)
    _cleanup(candidates, winner, store)
    print(f"[kuipy-autotune] selected {winner}", flush=True)
    return winner


def tune(key, kwargs, candidates, run_candidate, device):
    """Pick one spec out of ``candidates`` (given in priority order).

    Returns ``None`` for an empty candidate list, so a caller can return the
    result of ``tune`` straight out of ``supported()``."""
    if not candidates:
        return None
    digest = key_hash(key)
    store = TuneStore()
    cache_key = (str(store.path), digest)

    with _key_lock(cache_key):
        with _registry_lock:
            chosen = _selected.get(cache_key)
        if chosen is not None:
            spec = _match(candidates, _normalize(chosen))
            if spec is not None:
                return spec
            with _registry_lock:
                _selected.pop(cache_key, None)

        if C.AUTOTUNE:
            C.KUIPY_CACHE.mkdir(parents=True, exist_ok=True)
            with (_device_lock(device),
                  FileLock(str(_gpu_lock_path(key, device))),
                  FileLock(str(C.KUIPY_CACHE / "autotune-artifacts.lock")),
                  unset_fake_temporarily()):
                spec = _tune_now(key, kwargs, candidates, run_candidate,
                                 device, store, digest)
        else:
            entry = store.lookup(digest, key)
            spec = _match(candidates, entry["spec"]) if entry is not None else None
            if spec is None:
                if entry is not None:
                    C.log(f"ignoring stale autotune entry {digest}")
                spec = candidates[0]

        with _registry_lock:
            _selected[cache_key] = spec
        return spec


def reset_state():
    """Clear process-local selections; intended for tests."""
    with _registry_lock:
        _selected.clear()
        _locks.clear()
        _device_locks.clear()
