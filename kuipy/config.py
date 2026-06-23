"""Paths and toolchain flags for JIT extraction/compilation of Kuiper kernels.

Everything here is derived from ``$KUIPER_HOME`` (the root of the kuiper repo)
so the JIT pipeline reproduces what ``verify.mk`` does, but out-of-tree: it
never regenerates the repo's ``.depend`` and never edits files under ``src/``.
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent          
_REPO_ROOT = _HERE.parent 

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------
KUIPER_ROOT = Path(os.path.join(_REPO_ROOT, "kuiper")).resolve()
KUIPER_SRC = KUIPER_ROOT / "src"
KUIPER_INCS = KUIPER_ROOT / "include"
KUIPER_DIST = KUIPER_ROOT / "dist"
KUIPER_SCRIPTS = KUIPER_ROOT / "scripts"

# Holds all the .checked dependencies. It would be nice to have separate cache dirs, 
# but FStar only supports one cache directory so it's really easiest to just have everything
# write to the cache of the original repo.
OBJ_CACHE_DIR = KUIPER_ROOT / "obj"          # holds the verified dependency .checked

FSTAR_EXE = KUIPER_ROOT / "inst" / "bin" / "fstar.exe"
KRML_EXE = KUIPER_ROOT / "inst" / "bin" / "krml"
# F* extraction plugin, without the .cmxs extension (matches verify.mk PLUGIN).
PLUGIN = KUIPER_ROOT / "extraction" / "dune" / "_build" / "default" / "kuiper_extr"

FIXUP_SED = KUIPER_SCRIPTS / "fixup.sed"

# --------------------------------------------------------------------------
# JIT working tree (kept entirely inside the python package, repo stays clean)
# --------------------------------------------------------------------------
JIT_CACHE = Path(os.environ.get("KUIPY_JIT_CACHE", os.path.join(_REPO_ROOT, ".jitcache"))).resolve()
JIT_SRC = JIT_CACHE / "src"        # generated Klas_<Mod>.fst
JIT_PRE = JIT_CACHE / "pre"        # our object files + raw karamel output
JIT_CU = JIT_CACHE / "cu"          # fixed-up .cu/.h
JIT_BUILD = JIT_CACHE / "build"    # torch cpp_extension build dirs (.so)

# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------

# Default: admit SMT queries (fast cold compile). Set KUIPY_JIT_VERIFY=1 to run
# full F* verification of each instantiation instead.
JIT_FULL_VERIFY = os.environ.get("KUIPY_JIT_VERIFY", "0") == "1"
JIT_VERBOSE = os.environ.get("KUIPY_JIT_VERBOSE", "0") == "1"

ENABLE_PRINT_PROFILING = os.environ.get("KUIPY_PRINT_PROFILING", "0") == "1"

# If set, JIT extraction/compilation errors propagate instead of falling back to
# stock PyTorch. Off by default so an unsupported shape never breaks a model.
JIT_STRICT = os.environ.get("KUIPY_JIT_STRICT", "1") == "1"

RE_TUNE = os.environ.get("KUIPY_RE_TUNE", "0") == "1"

# --------------------------------------------------------------------------
# F* flags (mirrors `make echo-fstar`)
# --------------------------------------------------------------------------
FSTAR_FLAGS = [
    "--silent",
    "--include", str(KUIPER_SRC),
    "--include", str(JIT_SRC),
    "--include", str(_REPO_ROOT / "kuiops"),
    "--cache_dir", str(OBJ_CACHE_DIR),
    "--odir", str(OBJ_CACHE_DIR),
    "--warn_error", "-291",
    "--warn_error", "-249-321",
    "--warn_error", "@242@250",
    "--z3version", "4.13.3",
    "--ext", "kuiper",
    "--ext", "__unrefine",
    "--ext", "no_krml_private",
    "--warn_error", "-288",
    "--ext", "context_pruning_no_ambients",
    "--ext", "freshen",
]

# --------------------------------------------------------------------------
# karamel flags (mirrors `make echo-krml`)
# --------------------------------------------------------------------------
KRML_FLAGS = [
    "-add-early-include", "<kuiper.h>",
    "-fc++-compat",
    "-fcast-allocations",
    "-skip-compilation",
    "-skip-makefiles",
    "-faggressive-inlining",
    "-fauto-for-loops",
    "-fnoshort-enums",
    "-cuda",
    "-dbacktrace",
    "-silent",
    "-drop", "Prims",
    "-minimal",
    "-header", "/dev/null",
    "-warn-error", "@6",
    "-warn-error", "-2@4-10@18",
]


def nvcc_arch_flag():
    """gencode flag for the current device, or None if torch/CUDA unavailable."""
    try:
        import torch
        cc = torch.cuda.get_device_capability(0)
        return f"-gencode=arch=compute_{cc[0]}{cc[1]},code=sm_{cc[0]}{cc[1]}"
    except Exception:
        return None


def ensure_dirs():
    for d in (JIT_SRC, JIT_PRE, JIT_CU, JIT_BUILD):
        d.mkdir(parents=True, exist_ok=True)


def log(*a):
    if JIT_VERBOSE:
        print("[kuipy-jit]", *a, flush=True)
