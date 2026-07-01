"""Compile a JIT-extracted Kuiper kernel .cu into a loadable torch extension.

Generates a thin pybind wrapper (trivial glue) from a jinja template, then uses
``torch.utils.cpp_extension.load`` which caches the resulting ``.so`` on disk and
skips nvcc when sources are unchanged. A loaded module is also memoised in
process so a hot cache costs only a dict lookup.
"""
import os
import sys
from pathlib import Path

from filelock import FileLock
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import config as C
from . import toolchain

_TEMPLATES = C._REPO_ROOT / "kuiops"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATES)), undefined=StrictUndefined)

# In-process cache: ext_name -> loaded module.
_loaded = {}
# Negative cache: ext_name -> exception, for kernels that failed to build, so we
# fall back to PyTorch immediately instead of re-running F*/nvcc on every call.
_failed = {}


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


def build_kernel(module: str, 
                 fst_template: str, fst_ctx: dict,
                 wrapper_template: str, wrapper_ctx: dict):
    """Extract + compile a kernel, returning the loaded torch extension module.

    ``module``          : F* module name (e.g. ``Klas.JitGemmBT2D_f32_...``).
    ``fst_template``     : jinja template filename producing the one-line .fst.
    ``fst_ctx``          : context for the .fst template (tile sizes, et, ...).
    ``wrapper_template`` : jinja template filename for the C++ pybind wrapper.
    ``wrapper_ctx``      : extra context for the wrapper (ctype, ...).
    """
    ext_name = module.replace(".", "_")
    mod = _loaded.get(ext_name)
    if mod is not None:
        return mod
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
        with FileLock(str(lock_path)):
            mod = _loaded.get(ext_name)
            if mod is not None:
                return mod
            return _build_kernel(module, ext_name, fst_template, fst_ctx,
                                 wrapper_template, wrapper_ctx)
    except Exception as e:
        _failed[ext_name] = e
        raise


def _build_kernel(module, ext_name,
                  fst_template, fst_ctx, wrapper_template, wrapper_ctx):
    
    # LATER: would be nice if there was a way to load the built .so directly (not have to build it in ninja),
    # but it seems you have to go through this torch.utils.cpp_extension stuff to load it.

    C.ensure_dirs()

    # 1) F* -> .cu/.h
    fst_text = _env.get_template(fst_template).render(**fst_ctx)
    cu_path, h_path, decl_path = toolchain.extract_cu(module, fst_text)

    # The extracted host symbol is `<module-with-dots-as-underscores>_<letname>`.
    sym = f"{ext_name}_{fst_ctx['name']}"

    # 2) generate wrapper .cpp next to the kernel. It includes the
    # declaration-only header (just the launcher prototype, no kuiper.h), so
    # it can be compiled by the host compiler (g++) instead of nvcc: it's
    # pure torch/pybind glue with no device code, but nvcc would otherwise
    # parse the heavy torch/extension.h through its slower multi-pass
    # pipeline purely to satisfy kuiper.h's CUDA intrinsics pulled in by the
    # full per-kernel header.
    wrapper_path = C.KUIPY_JIT_CU / f"{ext_name}_wrapper.cpp"
    if not wrapper_path.exists():
        wctx = dict(wrapper_ctx)
        wctx.update(sym=sym, header=decl_path.name)
        wrapper_path.write_text(_env.get_template(wrapper_template).render(**wctx))

    # 3) compile + load
    _ensure_ninja_on_path()
    from torch.utils.cpp_extension import load
    build_dir = C.KUIPY_JIT_BUILD / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)
    C.log(f"building {ext_name} -> {build_dir}")
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
