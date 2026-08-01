"""Out-of-tree F* -> karamel -> .cu pipeline for a single JIT instantiation."""
import os
import subprocess
from pathlib import Path

from filelock import FileLock

from . import config as C
from . import jitprofile as P

_built = False

# Guards the one-time `make verify-kuiops` seeding pass: with parallel pytest
# workers every process runs it at startup, and concurrent makes would race on
# the shared .checked files they all write.
_SEED_LOCK = C.KUIPY_CACHE / "verify-kuiops.lock"

# `make verify-kuiops` (see verify.mk) dependency-orders and checks every
# kuiops/*.fst{i} support module into .kuipy_cache/checked, using the same
# flags as this module, so by the time any kernel is built the entire Kuiops
# namespace -- like Kuiper itself -- is already on disk as .checked. A JIT
# build therefore only ever *reads* the shared cache and writes its own
# per-instantiation outputs (.checked, .krml, .cu/.h), which is what makes it
# safe to extract many kernels concurrently.
#
# (The previous code passed `--already_cached *,-<module>,-Kuiops`, which made
# F* re-typecheck and rewrite the shared Kuiops .checked files on *every*
# build. That forced the whole F*/karamel stage under one global lock. It is
# measurably no faster and produces byte-identical .krml, so it is gone.)

def _run(cmd, what, kernel=None):
    C.log(f"({what})", " ".join(str(c) for c in cmd))
    with P.stage(what, kernel):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"kuipy-jit {what} failed (exit {proc.returncode}):\n"
            f"CMD: {' '.join(str(c) for c in cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc

def _ensure_built():
    """Verify the kuiops support modules (and Kuiper, if not installed yet) so
    every dependency of a JIT instantiation has a .checked file on disk."""
    global _built
    if _built:
        return
    C.KUIPY_CACHE.mkdir(parents=True, exist_ok=True)
    with P.stage("seed-lock-wait"):
        lock = FileLock(str(_SEED_LOCK))
        lock.acquire()
    try:
        # -j: verify.mk's per-module CHECK rules are dependency-ordered, so the
        # independent ones verify in parallel (~5min -> ~1m45 on a 128-core box).
        _run(["make", "-C", str(C._REPO_ROOT), "verify-kuiops",
              f"-j{C.SEED_JOBS}"], "repo-build")
    finally:
        lock.release()
    C.ensure_dirs()
    _built = True


def extract_cu(module: str, fst_text: str):
    """Verify+extract ``module`` (whose source is ``fst_text``) to a .cu/.h pair.

    Returns ``(cu_path, h_path, decl_path)``. Idempotent: if the .cu already
    exists it is returned without recompiling.

    ``decl_path`` is a declaration-only sibling of ``h_path`` (same launcher
    prototype, but without ``#include <kuiper.h>``). The kernel .cu needs the
    full ``kuiper.h`` machinery (device intrinsics, tensor-core types, ...) for
    its device code, but the pybind wrapper only ever calls the launcher by
    prototype -- it never touches kuiper.h's CUDA-only symbols. Including the
    full header in the wrapper forces it to be compiled with nvcc (multi-pass,
    slow) purely to satisfy kuiper.h's device intrinsics; including only the
    declaration lets the wrapper be compiled as plain .cpp via the host
    compiler instead, which is much faster for a file that is otherwise pure
    torch/pybind glue with no device code.
    """
    _ensure_built()

    underscored = module.replace(".", "_")        # e.g. Klas_JitGemm...
    cu_path = C.KUIPY_JIT_CU / f"{underscored}.cu"
    h_path = C.KUIPY_JIT_CU  / f"{underscored}.h"
    decl_path = C.KUIPY_JIT_CU / f"{underscored}_decl.h"

    if cu_path.exists() and h_path.exists():
        if C.JIT_FLUSH_CACHE:
            os.remove(cu_path)
            os.remove(h_path)
            if decl_path.exists():
                os.remove(decl_path)
        else:
            if not decl_path.exists():
                _make_decl_header(h_path, decl_path)
            return cu_path, h_path, decl_path

    fst_path = C.KUIPY_JIT_SRC / f"{module}.fst"
    checked = C.KUIPY_CHECKED_DIR / f"{module}.fst.checked"
    krml = C.KUIPY_JIT_PRE / f"{underscored}.krml"

    # Per-module lock: two processes racing to build the *same* kernel would
    # corrupt its shared .krml/.cu, but distinct kernels are independent and
    # extract fully in parallel.
    with P.stage("fstar-lock-wait", module):
        _lock = FileLock(str(C.KUIPY_CACHE / f"extract-{underscored}.lock"))
        _lock.acquire()
    try:
        # Re-check after acquiring the lock: another process may have built
        # this exact kernel while we were waiting.
        if cu_path.exists() and h_path.exists() and not C.JIT_FLUSH_CACHE:
            if not decl_path.exists():
                _make_decl_header(h_path, decl_path)
            return cu_path, h_path, decl_path

        fst_path.write_text(fst_text)
        # No .fsti: the single `let` must be exported (it is the host entry
        # point), and an interface would require its own .checked to exist first.

        # Only this instantiation is uncached: the Kuiops support modules are
        # checked by `make verify-kuiops` (via _ensure_built) and everything
        # else ships prebuilt with Kuiper.
        already = f"*,-{module}"
        admit = [] if C.JIT_FULL_VERIFY else ["--admit_smt_queries", "true"]

        # 1) check (produces <module>.fst.checked). Required even though the
        # extract pass typechecks too: cross-module inlining refuses to run
        # against a module that has no .checked file on disk.
        _run([str(C.FSTAR_EXE), *C.FSTAR_FLAGS, *admit,
              "--already_cached", already,
              "-c", str(fst_path), "-o", str(checked)],
             "fstar-check", module)

        # 2) extract to krml
        _run([str(C.FSTAR_EXE), *C.FSTAR_FLAGS, *admit,
              "--already_cached", already,
              "--codegen", "krml", "--load_cmxs", str(C.PLUGIN),
              "--extract", "-*,+Kuiops,+Kuiper",
              "-o", str(krml), str(fst_path)],
             "fstar-extract", module)

        # 3) karamel -> pre/<underscored>.cu + .h
        _run([str(C.KRML_EXE), *C.KRML_FLAGS,
              "-bundle", f"{module}=*",
              "-tmpdir", str(C.KUIPY_JIT_PRE), str(krml)],
             "karamel", module)

        pre_cu = C.KUIPY_JIT_PRE / f"{underscored}.cu"
        pre_h = C.KUIPY_JIT_PRE / f"{underscored}.h"

        # 4) fixup (sed + indent), matching verify.mk
        with P.stage("fixup", module):
            _fixup(pre_cu, cu_path)
            _fixup(pre_h, h_path)
            _make_decl_header(h_path, decl_path)
    finally:
        _lock.release()
    return cu_path, h_path, decl_path

def _make_decl_header(h_path: Path, decl_path: Path):
    """Strip the ``#include <kuiper.h>`` line from ``h_path``, writing the
    result to ``decl_path``. karamel always emits this include (via
    ``-add-early-include <kuiper.h>`` in KRML_FLAGS) so device code in the
    kernel .cu can use kuiper's CUDA intrinsics/types, but the launcher
    prototype itself only needs plain C types (uint32_t, float*, ...), so a
    caller that only wants to declare/call the launcher (i.e. the wrapper)
    doesn't need kuiper.h and can be compiled without nvcc.
    """
    lines = h_path.read_text().splitlines(keepends=True)
    decl_path.write_text("".join(
        l for l in lines if "#include <kuiper.h>" not in l))

def _fixup(src: Path, dst: Path):
    sed = subprocess.run(["sed", "-f", str(C.FIXUP_SED), str(src)],
                         capture_output=True, text=True, check=True)
    indent = subprocess.run(["indent", "-linux", "-i4", "-nut"],
                            input=sed.stdout, capture_output=True, text=True, check=True)
    dst.write_text(indent.stdout)
