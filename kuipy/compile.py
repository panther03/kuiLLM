"""Compile a JIT-extracted Kuiper kernel .cu into a loadable torch extension.

Generates a thin pybind wrapper (trivial glue) from a jinja template, then uses
``torch.utils.cpp_extension.load`` which caches the resulting ``.so`` on disk and
skips nvcc when sources are unchanged. A loaded module is also memoised in
process so a hot cache costs only a dict lookup.

Batch mode (``start_capture`` / ``finalize_capture``): while capturing, matched
kernels are only *extracted* (F* -> .cu, plus their pybind wrapper fragment) and
their actual execution is deferred (``CaptureDeferred``) so the caller falls back
to stock PyTorch. On ``finalize_capture`` every captured wrapper fragment is
concatenated into a single .cpp translation unit and compiled together with all
the device .cu files into one extension. Because the heavy ``torch/extension.h``
(pulled in via ``kuiops.h``, ``#pragma once``) is then parsed only once for the
whole batch instead of once per kernel, this amortizes almost all of the C++
host-compile cost across a full run. The ``_loaded`` map is intentionally
non-injective afterwards: every captured module key points at a proxy over the
same shared extension.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

from filelock import FileLock
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import config as C
from . import jitprofile as P
from . import toolchain

_TEMPLATES = C._REPO_ROOT / "kuiops"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATES)), undefined=StrictUndefined)

# In-process cache: ext_name -> object exposing ``.run(*args)`` (either a loaded
# torch extension module, or a ``_BatchKernel`` proxy over a shared batch module).
_loaded = {}
# Negative cache: ext_name -> exception, for kernels that failed to build, so we
# fall back to PyTorch immediately instead of re-running F*/nvcc on every call.
_failed = {}

# Active batch capture, or None. ext_name -> _CaptureRec (insertion-ordered).
_capture = None
_capture_artifact_lock = None


class CaptureDeferred(Exception):
    """Raised in place of running a kernel while a batch capture is active: the
    kernel has been extracted and queued for the batch build, and the caller
    should fall back to stock PyTorch for this call."""


class _DeferredKernel:
    """Stand-in returned during capture; running it defers to PyTorch."""
    def run(self, *args, **kwargs):
        raise CaptureDeferred


class _BatchKernel:
    """Proxy mapping ``.run`` onto one op of a shared batch extension module."""
    def __init__(self, mod, key):
        self._fn = getattr(mod, key)

    def run(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _CaptureRec:
    __slots__ = ("module", "ext_name", "cu_path", "fragment")

    def __init__(self, module, ext_name, cu_path, fragment):
        self.module = module
        self.ext_name = ext_name
        self.cu_path = cu_path
        self.fragment = fragment


def is_capturing() -> bool:
    return _capture is not None


def start_capture():
    """Begin a batch capture: matched kernels are extracted but not compiled,
    and their execution is deferred to stock PyTorch until ``finalize_capture``."""
    global _capture, _capture_artifact_lock
    if C.AUTOTUNE:
        raise RuntimeError("KUIPY_AUTOTUNE cannot be combined with batch capture")
    if _capture is None:
        C.KUIPY_CACHE.mkdir(parents=True, exist_ok=True)
        _capture_artifact_lock = FileLock(
            str(C.KUIPY_CACHE / "kernel-artifacts.lock")
        )
        _capture_artifact_lock.acquire()
        _capture = {}


def kernel_artifacts(module):
    """Return the exact per-instantiation files and directories for ``module``."""
    ext_name = module.replace(".", "_")
    files = [
        C.KUIPY_JIT_SRC / f"{module}.fst",
        C.KUIPY_CHECKED_DIR / f"{module}.fst.checked",
        C.KUIPY_JIT_PRE / f"{ext_name}.krml",
        C.KUIPY_JIT_PRE / f"{ext_name}.cu",
        C.KUIPY_JIT_PRE / f"{ext_name}.h",
        C.KUIPY_JIT_CU / f"{ext_name}.cu",
        C.KUIPY_JIT_CU / f"{ext_name}.h",
        C.KUIPY_JIT_CU / f"{ext_name}_decl.h",
        C.KUIPY_JIT_CU / f"{ext_name}_wrapper.cpp",
    ]
    return files, [C.KUIPY_JIT_BUILD / ext_name]


def delete_kernel(module):
    """Delete one compiled instantiation without touching shared support caches."""
    ext_name = module.replace(".", "_")
    C.ensure_dirs()
    artifact_lock = FileLock(str(C.KUIPY_CACHE / "kernel-artifacts.lock"))
    build_lock = FileLock(str(C.KUIPY_JIT_BUILD / f"{ext_name}.lock"))
    extract_lock = FileLock(str(C.KUIPY_CACHE / f"extract-{ext_name}.lock"))
    with artifact_lock, build_lock, extract_lock:
        _loaded.pop(ext_name, None)
        _failed.pop(ext_name, None)
        files, directories = kernel_artifacts(module)
        for path in files:
            path.unlink(missing_ok=True)
        for path in directories:
            if path.exists():
                shutil.rmtree(path)


def _nvcc_flags():
    flags = C.NVCC_BASE_FLAGS.copy()
    arch = C.nvcc_arch_flag()
    if arch:
        flags.append(arch)
    return flags


def _ensure_ninja_on_path():
    import shutil
    if shutil.which("ninja") is None:
        cand = Path(sys.executable).parent / "ninja"
        if cand.exists():
            os.environ["PATH"] = f"{cand.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        import ninja
        os.environ["PATH"] = str(Path(ninja.BIN_DIR)) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _render_wrapper(wrapper_template, wrapper_ctx, sym, header, batch):
    wctx = dict(wrapper_ctx)
    wctx.update(sym=sym, header=header, batch=batch)
    return _env.get_template(wrapper_template).render(**wctx)


def build_kernel(module: str,
                 fst_template: str, fst_ctx: dict,
                 wrapper_template: str, wrapper_ctx: dict):
    """Extract + compile a kernel, returning an object exposing ``.run(*args)``.

    ``module``          : F* module name (e.g. ``Klas.JitGemmBT2D_f32_...``).
    ``fst_template``     : jinja template filename producing the one-line .fst.
    ``fst_ctx``          : context for the .fst template (tile sizes, et, ...).
    ``wrapper_template`` : jinja template filename for the C++ pybind wrapper.
    ``wrapper_ctx``      : extra context for the wrapper (ctype, ...).

    While a batch capture is active, a not-yet-built kernel is only extracted
    and queued; the returned proxy raises ``CaptureDeferred`` when run.
    """
    ext_name = module.replace(".", "_")
    mod = _loaded.get(ext_name)
    if mod is not None:
        return mod
    if _capture is not None:
        return _capture_kernel(module, ext_name,
                               fst_template, fst_ctx, wrapper_template, wrapper_ctx)
    failed = _failed.get(ext_name)
    if failed is not None:
        raise failed
    try:
        # Cross-process lock keyed on the kernel's extension name: this lets
        # independent kernels build fully in parallel (e.g. parallel pytest
        # workers, or a batch pre-warming script) while serializing two
        # processes that race to build the *same* kernel, which would
        # otherwise corrupt the shared .kuipy_cache/cu and build directories.
        C.ensure_dirs()
        lock_path = C.KUIPY_JIT_BUILD / f"{ext_name}.lock"
        with P.stage("build-lock-wait", ext_name):
            lock = FileLock(str(lock_path))
            lock.acquire()
        try:
            mod = _loaded.get(ext_name)
            if mod is not None:
                return mod
            return _build_kernel(module, ext_name, fst_template, fst_ctx,
                                 wrapper_template, wrapper_ctx)
        finally:
            lock.release()
    except Exception as e:
        _failed[ext_name] = e
        raise


def _extract(module, ext_name, fst_template, fst_ctx, wrapper_template, wrapper_ctx,
             batch):
    """Run F* -> .cu/.h for ``module`` and render its pybind wrapper.

    Returns ``(cu_path, wrapper_text)``. Shared by the single-kernel build and
    the batch-capture path so the extraction logic lives in one place.
    """
    C.ensure_dirs()
    with P.stage("render", ext_name):
        fst_text = _env.get_template(fst_template).render(**fst_ctx)
    cu_path, h_path, decl_path = toolchain.extract_cu(module, fst_text)
    # The extracted host symbol is `<module-with-dots-as-underscores>_<letname>`.
    sym = f"{ext_name}_{fst_ctx['name']}"
    with P.stage("render", ext_name):
        wrapper_text = _render_wrapper(wrapper_template, wrapper_ctx,
                                       sym, decl_path.name, batch)
    return cu_path, wrapper_text


def _capture_kernel(module, ext_name,
                    fst_template, fst_ctx, wrapper_template, wrapper_ctx):
    if ext_name not in _capture:
        cu_path, fragment = _extract(module, ext_name, fst_template, fst_ctx,
                                     wrapper_template, wrapper_ctx, batch=True)
        _capture[ext_name] = _CaptureRec(module, ext_name, cu_path, fragment)
        C.log(f"captured {ext_name} (deferred)")
    return _DeferredKernel()


def _build_kernel(module, ext_name,
                  fst_template, fst_ctx, wrapper_template, wrapper_ctx):

    # LATER: would be nice if there was a way to load the built .so directly (not have to build it in ninja),
    # but it seems you have to go through this torch.utils.cpp_extension stuff to load it.

    cu_path, wrapper_text = _extract(module, ext_name, fst_template, fst_ctx,
                                     wrapper_template, wrapper_ctx, batch=False)

    # generate wrapper .cpp next to the kernel. It includes the declaration-only
    # header (just the launcher prototype, no kuiper.h), so it can be compiled by
    # the host compiler (g++) instead of nvcc: it's pure torch/pybind glue with
    # no device code, but nvcc would otherwise parse the heavy torch/extension.h
    # through its slower multi-pass pipeline purely to satisfy kuiper.h's CUDA
    # intrinsics pulled in by the full per-kernel header.
    wrapper_path = C.KUIPY_JIT_CU / f"{ext_name}_wrapper.cpp"
    wrapper_path.write_text(wrapper_text)

    # compile + load
    _ensure_ninja_on_path()
    from torch.utils.cpp_extension import load
    build_dir = C.KUIPY_JIT_BUILD / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)
    C.log(f"building {ext_name} -> {build_dir}")
    with P.stage("cpp-extension-load", ext_name):
        mod = load(
            name=ext_name,
            sources=[str(wrapper_path), str(cu_path)],
            extra_include_paths=[str(C.KUIPY_JIT_CU), str(C.KUIPER_INCLUDE), str(C._REPO_ROOT / "include")],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=_nvcc_flags(),
            build_directory=str(build_dir),
            verbose=(C.JIT_VERBOSITY > 0),
        )
    _loaded[ext_name] = mod
    return mod


def finalize_capture():
    global _capture_artifact_lock
    try:
        return _finalize_capture()
    finally:
        if _capture_artifact_lock is not None:
            _capture_artifact_lock.release()
            _capture_artifact_lock = None


def _finalize_capture():
    """Compile every captured kernel into one shared extension and wire each
    module key to it. Returns the loaded module (or None if nothing captured).

    All wrapper fragments are concatenated into a single .cpp translation unit so
    ``torch/extension.h`` is parsed once for the whole batch; the device .cu files
    stay separate nvcc units (ninja builds them in parallel) and link into the
    same .so."""
    global _capture
    recs = list(_capture.values()) if _capture is not None else []
    _capture = None
    if not recs:
        return None

    # Stable extension name keyed on the exact set of kernels, so re-capturing the
    # same set reuses the on-disk build cache instead of recompiling.
    keys = sorted(r.ext_name for r in recs)
    digest = hashlib.sha1("\n".join(keys).encode()).hexdigest()[:16]
    ext_name = f"kuipy_batch_{digest}"

    combined = _combined_source(recs)
    C.ensure_dirs()
    combined_path = C.KUIPY_JIT_CU / f"{ext_name}.cpp"
    combined_path.write_text(combined)

    _ensure_ninja_on_path()
    from torch.utils.cpp_extension import load
    build_dir = C.KUIPY_JIT_BUILD / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)
    C.log(f"batch-building {len(recs)} kernels -> {build_dir}")
    lock_path = C.KUIPY_JIT_BUILD / f"{ext_name}.lock"
    with FileLock(str(lock_path)), P.stage("cpp-extension-load-batch", ext_name):
        mod = load(
            name=ext_name,
            sources=[str(combined_path)] + [str(r.cu_path) for r in recs],
            extra_include_paths=[str(C.KUIPY_JIT_CU), str(C.KUIPER_INCLUDE),
                                 str(C._REPO_ROOT / "include")],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=_nvcc_flags(),
            build_directory=str(build_dir),
            verbose=(C.JIT_VERBOSITY > 0),
        )
    for r in recs:
        _loaded[r.ext_name] = _BatchKernel(mod, r.ext_name)
    return mod


def _combined_source(recs):
    """One translation unit: every wrapper fragment (each op renamed to a unique
    ``op_<ext_name>``, with no per-kernel PYBIND11_MODULE) followed by a single
    module that registers each op under its module key."""
    parts = [r.fragment for r in recs]
    binds = "\n".join(
        f'    m.def("{r.ext_name}", &op_{r.ext_name}, "run");' for r in recs)
    parts.append(
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n" + binds + "\n}\n")
    return "\n".join(parts)
