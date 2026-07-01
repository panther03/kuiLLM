"""Paths and toolchain flags for JIT extraction/compilation of Kuiper kernels."""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent          
_REPO_ROOT = _HERE.parent 

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
KUIPER_INST = Path(os.environ.get("KUIPER_INST", _REPO_ROOT / "inst")).resolve()
KUIOPS_SRC = Path(_REPO_ROOT / "kuiops").resolve()

FSTAR_EXE = KUIPER_INST / "bin" / "fstar.exe"
KRML_EXE = KUIPER_INST / "bin" / "krml"
# F* extraction plugin, without the .cmxs extension (matches verify.mk PLUGIN).
PLUGIN = KUIPER_INST / "kuiper_extr" / "kuiper_extr"
KUIPER_INCLUDE = KUIPER_INST / "include" / "kuiper"
FIXUP_SED = KUIPER_INST / "fixup.sed"

# --------------------------------------------------------------------------
# JIT cache dirs
# --------------------------------------------------------------------------
KUIPY_CACHE = Path(os.environ.get("KUIPY_CACHE", _REPO_ROOT / ".kuipy_cache")).resolve()
KUIPY_CHECKED_DIR = KUIPY_CACHE / "checked"  # holds the verified dependency .checked
KUIPY_JIT_PRE = KUIPY_CACHE / "pre"          # holds the raw karamel output
KUIPY_JIT_CU = KUIPY_CACHE / "cu"            # holds the fixed-up .cu/.h files
KUIPY_JIT_BUILD = KUIPY_CACHE / "build"      # holds the torch cpp_extension build dirs (.so)
KUIPY_JIT_SRC = KUIPY_CACHE / "src"          # instantiated fst files

# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------

# Default: admit SMT queries (fast cold compile). Set KUIPY_JIT_VERIFY=1 to run
# full F* verification of each instantiation instead.
JIT_FULL_VERIFY = os.environ.get("KUIPY_JIT_VERIFY", "0") == "1"
JIT_VERBOSITY = int(os.environ.get("KUIPY_JIT_VERBOSITY", "0")) # 2 = print mismatched ops
JIT_FLUSH_CACHE = os.environ.get("KUIPY_JIT_FLUSH_CACHE", "0") == "1"

ENABLE_PRINT_PROFILING = os.environ.get("KUIPY_PRINT_PROFILING", "0") == "1"

# 0 = no errors from JIT, even when compilation fails
# 1 = error when compilation fails only
# 2 = error when operator is not offloaded
JIT_STRICTNESS = int(os.environ.get("KUIPY_JIT_STRICTNESS", "1"))

JIT_NVCC_FAST = os.environ.get("KUIPY_JIT_NVCC_FAST", "0") == "1"

RE_TUNE = os.environ.get("KUIPY_RE_TUNE", "0") == "1"

# --------------------------------------------------------------------------
# F* flags (mirrors `make echo-fstar`)
# --------------------------------------------------------------------------
FSTAR_FLAGS = [
    "--silent",
    "--include", str(KUIOPS_SRC),
    "--cache_dir", str(KUIPY_CHECKED_DIR),
    "--odir", str(KUIPY_CHECKED_DIR),
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

# --------------------------------------------------------------------------
# nvcc base flags
# --------------------------------------------------------------------------

_NVCC_COMMON_FLAGS = [
    "--expt-relaxed-constexpr",
    "-std=c++17",
    # Needed so bf16/fp16 arithmetic operators/conversions used by the
    # generated wrapper code (e.g. casting alpha/beta scalars) are available;
    # must be present regardless of optimization level.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

# TODO: remove: probably does not offer meaningful speedup as apparently 
# most of the runtime is spent in fstar/karamel.
if JIT_NVCC_FAST:
    NVCC_BASE_FLAGS = _NVCC_COMMON_FLAGS + [
        "-Xcompiler", "-O0", "-Xptxas", "-O0", "-lineinfo",
    ]
else:
    NVCC_BASE_FLAGS = _NVCC_COMMON_FLAGS + [
        "-O3", "--use_fast_math", "-Xcompiler", "-fPIC",
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
    for d in (KUIPY_CHECKED_DIR, KUIPY_JIT_SRC, KUIPY_JIT_PRE, KUIPY_JIT_CU, KUIPY_JIT_BUILD):
        d.mkdir(parents=True, exist_ok=True)


def log(*a):
    if JIT_VERBOSITY > 0:
        print("[kuipy-jit]", *a, flush=True)
