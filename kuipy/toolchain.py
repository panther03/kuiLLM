"""Out-of-tree F* -> karamel -> .cu pipeline for a single JIT instantiation."""
import os
import subprocess
from pathlib import Path

from filelock import FileLock

from . import config as C

_built = False

# TODO: this whole thing is a giant mess and we should just figure out 
# why it is not working to run make verify-kuiops in the repo root 
# and have all these dependencies cached ahead of time (the original intent of
# _ensure_built()).

# F* checks/extracts every new instantiation against `--already_cached
# *,-<module>,-Kuiops`, which forces it to re-typecheck and re-extract the
# *whole* Kuiops namespace (the shared support .fst{i} files, not just the
# one-line instantiation), writing results into the shared
# .kuipy_cache/checked and .kuipy_cache/pre directories. Two different
# kernels built concurrently (e.g. by parallel pytest workers) therefore race
# on those shared support-module files. Serialize the F*/karamel stage
# globally with a cross-process lock; nvcc/ninja compilation of the
# resulting per-kernel .cu (the part that's actually safe to parallelize)
# happens outside this lock, in compile.py.
_FSTAR_LOCK = C.KUIPY_CACHE / "fstar.lock"

def _run(cmd, what):
    C.log(f"({what})", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"kuipy-jit {what} failed (exit {proc.returncode}):\n"
            f"CMD: {' '.join(str(c) for c in cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc

def _ensure_built():
    """run repository makefile and ensure deps are up to date"""
    global _built
    if _built:
        return
    _run(["make", "-C", str(C._REPO_ROOT), "verify-kuiops"], "repo build")
    C.ensure_dirs()
    with FileLock(str(_FSTAR_LOCK)):
        _ensure_kuiops_checked()
    _built = True


def _kuiops_support_files():
    """Real (non-template, non-instantiation) kuiops/*.fst{i} support
    modules, ordered so that Kuiops.Common (depended on by Gather/Scatter)
    is checked first, and each module's .fsti is checked before its .fst."""
    files = [p for p in C.KUIOPS_SRC.rglob("*.fst*") if p.suffix in (".fst", ".fsti")]
    common = C.KUIOPS_SRC / "Kuiops.Common.fsti"
    def key(p):
        return (p.resolve() != common, p.suffix == ".fst", str(p))
    return sorted(files, key=key)


def _ensure_kuiops_checked():
    """Pre-populate .checked files for the kuiops/*.fst{i} support modules.

    Each per-instantiation JIT build excludes the whole Kuiops namespace from
    --already_cached (see extract_cu below), so F* re-verifies these support
    modules every time anyway -- but F* only *persists* a module's .checked
    file if its own dependencies already have .checked files on disk (that's
    how the dependency-digest scheme works). On a completely empty cache
    there's nothing on disk yet, so even a successfully-reverified support
    module silently fails to be written, and the very next JIT build for
    *any* kernel then fails with "Module <kernel> was not checked". Run an
    explicit, dependency-ordered pass once per process to seed the cache;
    each kernel build's own reverification keeps these files fresh after that.
    """
    admit = [] if C.JIT_FULL_VERIFY else ["--admit_smt_queries", "true"]
    for src in _kuiops_support_files():
        checked = C.KUIPY_CHECKED_DIR / f"{src.name}.checked"
        if checked.exists() and checked.stat().st_mtime >= src.stat().st_mtime:
            continue
        _run([str(C.FSTAR_EXE), *C.FSTAR_FLAGS, *admit,
              "--already_cached", "*",
              "-c", str(src), "-o", str(checked)],
             f"kuiops-warm {src.name}")

def extract_cu(module: str, fst_text: str):
    """Verify+extract ``module`` (whose source is ``fst_text``) to a .cu/.h pair.

    Returns ``(cu_path, h_path, header_name)``. Idempotent: if the .cu already
    exists it is returned without recompiling.
    """
    _ensure_built()

    underscored = module.replace(".", "_")        # e.g. Klas_JitGemm...
    cu_path = C.KUIPY_JIT_CU / f"{underscored}.cu"
    h_path = C.KUIPY_JIT_CU  / f"{underscored}.h"

    if cu_path.exists() and h_path.exists(): 
        if C.JIT_FLUSH_CACHE:
            os.remove(cu_path)
            os.remove(h_path)
        else:
            return cu_path, h_path, h_path.name

    fst_path = C.KUIPY_JIT_SRC / f"{module}.fst"
    checked = C.KUIPY_CHECKED_DIR / f"{module}.fst.checked"
    krml = C.KUIPY_JIT_PRE / f"{underscored}.krml"

    with FileLock(str(_FSTAR_LOCK)):
        # Re-check after acquiring the lock: another process may have built
        # this exact kernel (or, more commonly, just refreshed the shared
        # Kuiops .checked files) while we were waiting.
        if cu_path.exists() and h_path.exists() and not C.JIT_FLUSH_CACHE:
            return cu_path, h_path, h_path.name

        fst_path.write_text(fst_text)
        # No .fsti: the single `let` must be exported (it is the host entry
        # point), and an interface would require its own .checked to exist first.

        # TODO: shouldnt have to reverify kuiops stuff, but the cache is not working right now
        already = f"*,-{module},-Kuiops"
        admit = [] if C.JIT_FULL_VERIFY else ["--admit_smt_queries", "true"]

        # 1) check (produces <module>.fst.checked)
        _run([str(C.FSTAR_EXE), *C.FSTAR_FLAGS, *admit,
              "--already_cached", already,
              "-c", str(fst_path), "-o", str(checked)],
             "check")

        # 2) extract to krml
        _run([str(C.FSTAR_EXE), *C.FSTAR_FLAGS, *admit,
              "--already_cached", already,
              "--codegen", "krml", "--load_cmxs", str(C.PLUGIN),
              "--extract", f"-*,+{module},+Kuiper,+Klas",
              "-o", str(krml), str(fst_path)],
             "extract")

        # 3) karamel -> pre/<underscored>.cu + .h
        _run([str(C.KRML_EXE), *C.KRML_FLAGS,
              "-bundle", f"{module}=*",
              "-tmpdir", str(C.KUIPY_JIT_PRE), str(krml)],
             "karamel")

        pre_cu = C.KUIPY_JIT_PRE / f"{underscored}.cu"
        pre_h = C.KUIPY_JIT_PRE / f"{underscored}.h"

        # 4) fixup (sed + indent), matching verify.mk
        _fixup(pre_cu, cu_path)
        _fixup(pre_h, h_path)
    return cu_path, h_path, h_path.name

def _fixup(src: Path, dst: Path):
    sed = subprocess.run(["sed", "-f", str(C.FIXUP_SED), str(src)],
                         capture_output=True, text=True, check=True)
    indent = subprocess.run(["indent", "-linux", "-i4", "-nut"],
                            input=sed.stdout, capture_output=True, text=True, check=True)
    dst.write_text(indent.stdout)
