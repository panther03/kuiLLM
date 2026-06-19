"""Compile a JIT-extracted Kuiper kernel .cu into a loadable torch extension.

Generates a thin pybind wrapper (trivial glue) from a jinja template, then uses
``torch.utils.cpp_extension.load`` which caches the resulting ``.so`` on disk and
skips nvcc when sources are unchanged. A loaded module is also memoised in
process so a hot cache costs only a dict lookup.
"""
import os
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import config as C
from . import toolchain

_TEMPLATES = Path(__file__).resolve().parent / "templates"
_WRAPPERS = Path(__file__).resolve().parent / "csrc"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATES)), undefined=StrictUndefined)

# In-process cache: ext_name -> loaded module.
_loaded = {}


def _nvcc_flags():
    flags = [
        "-O3", "--use_fast_math",
        "-Xcompiler", "-fPIC",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
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

    C.ensure_dirs()

    # 1) F* -> .cu/.h
    fst_text = _env.get_template(fst_template).render(**fst_ctx)
    cu_path, h_path, header_name = toolchain.extract_cu(module, fst_text)

    # The extracted host symbol is `<module-with-dots-as-underscores>_<letname>`.
    sym = f"{ext_name}_{fst_ctx['name']}"

    # 2) generate wrapper .cu next to the kernel
    wrapper_path = C.JIT_CU / f"{ext_name}_wrapper.cu"
    if not wrapper_path.exists():
        wctx = dict(wrapper_ctx)
        wctx.update(sym=sym, header=header_name)
        wrapper_path.write_text(_env.get_template(wrapper_template).render(**wctx))

    # 3) compile + load
    _ensure_ninja_on_path()
    from torch.utils.cpp_extension import load
    build_dir = C.JIT_BUILD / ext_name
    build_dir.mkdir(parents=True, exist_ok=True)
    mod = load(
        name=ext_name,
        sources=[str(wrapper_path), str(cu_path)],
        extra_include_paths=[str(C.KUIPER_INCS), str(C.KUIPER_DIST), str(C.JIT_CU)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=_nvcc_flags(),
        build_directory=str(build_dir),
        verbose=C.JIT_VERBOSE,
    )
    _loaded[ext_name] = mod
    return mod
