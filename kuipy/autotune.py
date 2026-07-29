"""Shape-specific autotuning for verified Kuiper kernel instantiations."""
import hashlib
import json
import os
import statistics
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock
import torch

from . import compile as _compile
from . import config as C


@dataclass
class Candidate:
    params: dict
    spec: object
    module: str


@dataclass
class _Selection:
    params: dict
    module: str


_selected = {}
_tuned = set()
_locks = {}
_device_locks = {}
_locks_guard = threading.Lock()
_state_lock = threading.Lock()


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
        if not isinstance(entry.get("params"), dict):
            raise RuntimeError(f"invalid autotune entry {digest} in {self.path}")
        return entry

    def update(self, digest, key, candidate):
        C.KUIPY_CACHE.mkdir(parents=True, exist_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            data = self._read()
            existing = data["entries"].get(digest)
            if existing is not None and existing.get("key") != key:
                raise RuntimeError(
                    f"autotune key collision for {digest} in {self.path}"
                )
            data["entries"][digest] = {
                "key": key,
                "params": _normalize(candidate.params),
                "module": candidate.module,
            }
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
            entry["module"]
            for entry in self._read()["entries"].values()
            if isinstance(entry, dict) and isinstance(entry.get("module"), str)
        }


def _key_lock(cache_key):
    with _locks_guard:
        return _locks.setdefault(cache_key, threading.Lock())


def _device_lock(device):
    with _locks_guard:
        return _device_locks.setdefault(str(device), threading.Lock())


def _gpu_lock_path(key, device):
    identity = f"{key['gpu']}|{device}"
    if device is not None:
        properties = torch.cuda.get_device_properties(device)
        identity = str(properties.uuid)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / (
        f"kuipy-autotune-{os.getuid()}-{digest}.lock"
    )


def _find_candidate(candidates, params):
    normalized = _normalize(params)
    for candidate in candidates:
        if _normalize(candidate.params) == normalized:
            return candidate
    return None


def _benchmark_candidate(candidate, run_candidate, device):
    with torch.cuda.device(device):
        run_candidate(candidate.spec)
        torch.cuda.synchronize(device)
        for _ in range(C.AUTOTUNE_WARMUP):
            run_candidate(candidate.spec)
        torch.cuda.synchronize(device)

        timings = []
        for _ in range(C.AUTOTUNE_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_candidate(candidate.spec)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return statistics.median(timings)


def _cleanup(candidates, winner, store):
    referenced = store.referenced_modules()
    with _state_lock:
        referenced.update(selection.module for selection in _selected.values())
    for candidate in candidates:
        if candidate.module != winner.module and candidate.module not in referenced:
            try:
                _compile.delete_kernel(candidate.module)
            except OSError as exc:
                C.log(f"failed to remove losing kernel {candidate.module}: {exc}")


def select_candidate(key, candidates, run_candidate, device):
    if not candidates:
        raise ValueError("autotuning requires at least one candidate")
    digest = key_hash(key)
    store = TuneStore()
    cache_key = (str(store.path), digest)

    with _key_lock(cache_key):
        with _state_lock:
            selection = _selected.get(cache_key)
        if selection is not None:
            candidate = _find_candidate(candidates, selection.params)
            if candidate is not None:
                return candidate.spec
            with _state_lock:
                _selected.pop(cache_key, None)

        if not C.AUTOTUNE:
            entry = store.lookup(digest, key)
            candidate = (
                _find_candidate(candidates, entry["params"])
                if entry is not None else None
            )
            if candidate is None:
                candidate = candidates[0]
                if entry is not None:
                    C.log(f"ignoring stale autotune entry {digest}")
            with _state_lock:
                _selected[cache_key] = _Selection(
                    candidate.params, candidate.module
                )
            return candidate.spec

        C.KUIPY_CACHE.mkdir(parents=True, exist_ok=True)
        gpu_file_lock = FileLock(str(_gpu_lock_path(key, device)))
        cache_file_lock = FileLock(
            str(C.KUIPY_CACHE / "autotune-artifacts.lock")
        )
        with _device_lock(device), gpu_file_lock, cache_file_lock:
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
            results = []
            failures = []
            for index, candidate in enumerate(candidates, 1):
                print(
                    f"[kuipy-autotune] [{index}/{len(candidates)}] "
                    f"{candidate.params}",
                    flush=True,
                )
                try:
                    timing = _benchmark_candidate(
                        candidate, run_candidate, device
                    )
                except RuntimeError as exc:
                    failures.append((candidate.params, str(exc)))
                    print(
                        f"[kuipy-autotune] candidate failed: {exc}", flush=True
                    )
                    continue
                results.append((timing, index, candidate))
                print(f"[kuipy-autotune] {timing:.4f} ms", flush=True)

            if not results:
                detail = "\n".join(
                    f"{params}: {error}" for params, error in failures
                )
                raise RuntimeError(
                    f"all autotune candidates failed for "
                    f"{key['operation']}:\n{detail}"
                )

            _, _, winner = min(results, key=lambda result: (result[0], result[1]))
            store.update(digest, key, winner)
            with _state_lock:
                _selected[cache_key] = _Selection(
                    winner.params, winner.module
                )
                _tuned.add(cache_key)
            _cleanup(candidates, winner, store)
            print(f"[kuipy-autotune] selected {winner.params}", flush=True)
            return winner.spec


def reset_state():
    """Clear process-local selections; intended for tests."""
    with _state_lock:
        _selected.clear()
        _tuned.clear()
    with _locks_guard:
        _locks.clear()
        _device_locks.clear()
