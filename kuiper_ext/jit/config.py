"""Paths and toolchain flags for JIT extraction/compilation of Kuiper kernels.

Everything here is derived from ``$KUIPER_HOME`` (the root of the kuiper repo)
so the JIT pipeline reproduces what ``verify.mk`` does, but out-of-tree: it
never regenerates the repo's ``.depend`` and never edits files under ``src/``.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------
KUIPER_ROOT = Path(os.environ["KUIPER_HOME"]).resolve()
KUIPER_SRC = KUIPER_ROOT / "src"
KUIPER_OBJ = KUIPER_ROOT / "obj"          # holds the verified dependency .checked
KUIPER_INCS = KUIPER_ROOT / "include"
KUIPER_DIST = KUIPER_ROOT / "dist"
KUIPER_SCRIPTS = KUIPER_ROOT / "scripts"

FSTAR_EXE = KUIPER_ROOT / "inst" / "bin" / "fstar.exe"
KRML_EXE = KUIPER_ROOT / "inst" / "bin" / "krml"
# F* extraction plugin, without the .cmxs extension (matches verify.mk PLUGIN).
PLUGIN = KUIPER_ROOT / "extraction" / "dune" / "_build" / "default" / "kuiper_extr"

FIXUP_SED = KUIPER_SCRIPTS / "fixup.sed"

# --------------------------------------------------------------------------
# JIT working tree (kept entirely inside the python package, repo stays clean)
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent          # kuiper_ext/
JIT_CACHE = Path(os.environ.get("KUIPER_JIT_CACHE", _HERE / ".jitcache")).resolve()
JIT_SRC = JIT_CACHE / "src"        # generated Klas_<Mod>.fst
JIT_OBJ = JIT_CACHE / "obj"        # symlinks to KUIPER_OBJ/*.checked + our .checked/.krml
JIT_PRE = JIT_OBJ / "pre"          # karamel raw output
JIT_CU = JIT_CACHE / "cu"          # fixed-up .cu/.h
JIT_BUILD = JIT_CACHE / "build"    # torch cpp_extension build dirs (.so)

# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------
# Default: admit SMT queries (fast cold compile). Set KUIPER_JIT_VERIFY=1 to run
# full F* verification of each instantiation instead.
JIT_FULL_VERIFY = os.environ.get("KUIPER_JIT_VERIFY", "0") == "1"
JIT_VERBOSE = os.environ.get("KUIPER_JIT_VERBOSE", "0") == "1"

# --------------------------------------------------------------------------
# F* flags (mirrors `make echo-fstar`)
# --------------------------------------------------------------------------
FSTAR_FLAGS = [
    "--silent",
    "--include", str(KUIPER_SRC),
    "--include", str(JIT_SRC),
    "--cache_dir", str(JIT_OBJ),
    "--odir", str(JIT_OBJ),
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
    for d in (JIT_SRC, JIT_OBJ, JIT_PRE, JIT_CU, JIT_BUILD):
        d.mkdir(parents=True, exist_ok=True)
