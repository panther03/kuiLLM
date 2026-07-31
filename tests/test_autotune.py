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


def _spec(name, speed):
    return {"module": f"Kuiops.Test.{name}", "tile": name, "speed": speed}


def _configure(monkeypatch, tmp_path, enabled):
    monkeypatch.setattr(config, "AUTOTUNE", enabled)
    monkeypatch.setattr(config, "TUNE_PARAMS_PATH", tmp_path / "tune_params.json")
    monkeypatch.setattr(config, "KUIPY_CACHE", tmp_path / ".kuipy_cache")
    autotune.reset_state()


def test_no_candidates_is_unsupported(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=False)
    assert autotune.tune(_key(), {}, [], None, None) is None


def test_runtime_uses_committed_entry(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=False)
    key = _key()
    candidates = [_spec("slow", 2.0), _spec("fast", 1.0)]
    autotune.TuneStore().update(autotune.key_hash(key), key, candidates[1])

    def no_benchmark(*args):
        raise AssertionError("runtime mode must not benchmark")

    monkeypatch.setattr(autotune, "_benchmark", no_benchmark)
    assert autotune.tune(key, {}, candidates, None, None) is candidates[1]


def test_runtime_cache_miss_uses_first_candidate(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=False)
    candidates = [_spec("default", 2.0), _spec("other", 1.0)]
    assert autotune.tune(_key(), {}, candidates, None, None) is candidates[0]
    assert not config.TUNE_PARAMS_PATH.exists()


def test_autotune_ignores_json_once_per_process(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, enabled=True)
    key = _key()
    candidates = [_spec("old", 3.0), _spec("winner", 1.0)]
    autotune.TuneStore().update(autotune.key_hash(key), key, candidates[0])

    calls = []

    def benchmark(spec, run_candidate, args, kwargs, device):
        calls.append(spec["module"])
        return spec["speed"]

    removed = []
    monkeypatch.setattr(autotune, "_benchmark", benchmark)
    monkeypatch.setattr(autotune, "_synthesize_args", lambda key, device: [])
    monkeypatch.setattr(jit_compile, "delete_kernel", removed.append)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    first = autotune.tune(key, {}, candidates, None, None)
    second = autotune.tune(key, {}, candidates, None, None)

    assert first is second is candidates[1]
    assert calls == [spec["module"] for spec in candidates]
    assert removed == [candidates[0]["module"]]
    entry = json.loads(config.TUNE_PARAMS_PATH.read_text())["entries"][
        autotune.key_hash(key)
    ]
    assert entry["key"] == key
    assert entry["spec"] == candidates[1]


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


def test_tile_enumeration_is_grid_bounded():
    assert len(kuiops._tiles("bt2d", torch.float32, 1, 128, 128, 64)) > 1
    assert len(kuiops._tiles("tc2d", torch.bfloat16, 1, 128, 128, 64)) > 1
    # A grid that would exceed the block limit leaves no legal tiling.
    huge = kuiops._MAX_BLOCKS * 128
    assert kuiops._tiles("bt2d", torch.float32, huge, 128, 128, 64) == []


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
