import json

import torch

from kuipy import autotune
from kuipy import compile as jit_compile
from kuipy import config
from kuipy import kuiops


def _key(operation="aten.mm.default", shape=(64, 64)):
    return {
        "tuning_schema": config.TUNING_SCHEMA_VERSION,
        "operation": operation,
        "inputs": [
            {
                "shape": list(shape),
                "strides": [shape[1], 1],
                "dtype": "bfloat16",
            }
        ],
        "parameters": {},
        "gpu": "Test GPU",
    }


def _candidate(name, speed):
    return autotune.Candidate(
        params={"impl": "test", "tile": name},
        spec={"name": name, "speed": speed},
        module=f"Kuiops.Test.{name}",
    )


def _configure(monkeypatch, tmp_path, enabled):
    monkeypatch.setattr(config, "AUTOTUNE", enabled)
    monkeypatch.setattr(config, "TUNE_PARAMS_PATH", tmp_path / "tune_params.json")
    monkeypatch.setattr(config, "KUIPY_CACHE", tmp_path / ".kuipy_cache")
    autotune.reset_state()


def test_runtime_uses_committed_entry(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=False)
    key = _key()
    candidates = [_candidate("slow", 2.0), _candidate("fast", 1.0)]
    store = autotune.TuneStore()
    store.update(autotune.key_hash(key), key, candidates[1])

    def no_benchmark(*args):
        raise AssertionError("runtime mode must not benchmark")

    monkeypatch.setattr(autotune, "_benchmark_candidate", no_benchmark)
    selected = autotune.select_candidate(key, candidates, lambda spec: None, None)
    assert selected == candidates[1].spec


def test_runtime_cache_miss_uses_first_candidate(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=False)
    candidates = [_candidate("default", 2.0), _candidate("other", 1.0)]
    selected = autotune.select_candidate(
        _key(), candidates, lambda spec: None, None
    )
    assert selected == candidates[0].spec
    assert not config.TUNE_PARAMS_PATH.exists()


def test_autotune_ignores_json_once_per_process(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=True)
    key = _key()
    candidates = [_candidate("old", 3.0), _candidate("winner", 1.0)]
    store = autotune.TuneStore()
    store.update(autotune.key_hash(key), key, candidates[0])

    calls = []

    def benchmark(candidate, run_candidate, device):
        calls.append(candidate.module)
        return candidate.spec["speed"]

    removed = []
    monkeypatch.setattr(autotune, "_benchmark_candidate", benchmark)
    monkeypatch.setattr(jit_compile, "delete_kernel", removed.append)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    first = autotune.select_candidate(key, candidates, lambda spec: None, None)
    second = autotune.select_candidate(key, candidates, lambda spec: None, None)

    assert first == second == candidates[1].spec
    assert calls == [candidate.module for candidate in candidates]
    assert removed == [candidates[0].module]
    entry = json.loads(config.TUNE_PARAMS_PATH.read_text())["entries"][
        autotune.key_hash(key)
    ]
    assert entry["key"] == key
    assert entry["params"] == candidates[1].params
    assert entry["module"] == candidates[1].module


def test_schema_mismatch_invalidates_file(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=False)
    config.TUNE_PARAMS_PATH.write_text(
        json.dumps({"tuning_schema": 0, "entries": {"old": {}}})
    )
    assert autotune.TuneStore().lookup("old", _key()) is None


def test_hash_is_canonical():
    first = _key()
    second = {
        "gpu": first["gpu"],
        "parameters": {},
        "inputs": first["inputs"],
        "operation": first["operation"],
        "tuning_schema": first["tuning_schema"],
    }
    assert autotune.key_hash(first) == autotune.key_hash(second)


def test_tile_enumeration_preserves_defaults():
    bt2d = kuiops._bt2d_tiles(torch.float32, 128, 128, 64)
    tc2d = kuiops._tc2d_tiles(torch.bfloat16, 128, 128, 64)
    assert len(bt2d) > 1 and bt2d[0] == kuiops._bt2d_tile(
        torch.float32, 128, 128, 64
    )
    assert len(tc2d) > 1 and tc2d[0] == kuiops._tc2d_tile(
        torch.bfloat16, 128, 128, 64
    )


def test_delete_kernel_removes_only_instantiation(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KUIPY_CACHE", tmp_path / "cache")
    paths = {
        "KUIPY_JIT_SRC": tmp_path / "src",
        "KUIPY_CHECKED_DIR": tmp_path / "checked",
        "KUIPY_JIT_PRE": tmp_path / "pre",
        "KUIPY_JIT_CU": tmp_path / "cu",
        "KUIPY_JIT_BUILD": tmp_path / "build",
    }
    for name, path in paths.items():
        monkeypatch.setattr(config, name, path)
        path.mkdir()

    module = "Kuiops.Test.Tile"
    files, directories = jit_compile.kernel_artifacts(module)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("candidate")
    for path in directories:
        path.mkdir(parents=True)
        (path / "kernel.so").write_text("candidate")
    shared = config.KUIPY_CHECKED_DIR / "Kuiops.Shared.fst.checked"
    shared.write_text("shared")

    jit_compile.delete_kernel(module)

    assert all(not path.exists() for path in files + directories)
    assert shared.exists()
