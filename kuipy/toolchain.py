"""Out-of-tree F* -> karamel -> .cu pipeline for a single JIT instantiation."""
import os
import subprocess
from pathlib import Path

from . import config as C

_built = False

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
    _built = True

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
    fst_path.write_text(fst_text)
    # No .fsti: the single `let` must be exported (it is the host entry point),
    # and an interface would require its own .checked to exist first.

    checked = C.KUIPY_CHECKED_DIR / f"{module}.fst.checked"
    krml = C.KUIPY_JIT_PRE / f"{underscored}.krml"

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
