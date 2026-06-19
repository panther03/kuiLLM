"""GEMM JIT implementation: aten::mm and aten::addmm.

Routing:
  * float32  -> Klas.GEMM.BlockTiling2D (in-place alpha/beta GEMM)
  * bfloat16 -> Klas.GEMM.TensorCore2D  (bf16 inputs, f32 accumulator)

Tile selection is a deterministic heuristic: the largest valid block/warp tiling
that divides (M, N, K) and satisfies the same shared-memory / copy-fullness /
thread-count constraints encoded in the kuiper repo's ``*.fst.sh`` generators.
"""
from . import compile as _compile

_SHMEM_BYTES = 101376
_MAX_THREADS = 1024
_WARP = 32


# ---------------------------------------------------------------------------
# Tile heuristics (mirror the .fst.sh filters)
# ---------------------------------------------------------------------------

def _bt2d_tile(M, N, K):
    """BlockTiling2D (f32, chunk=4). Returns (bm,bn,bk,tm,tn) or None."""
    chunk = 4
    for bm in (128, 64, 32):
        if M % bm:
            continue
        for bn in (128, 64, 32):
            if N % bn or bn % chunk:
                continue
            for bk in (64, 32):
                if K % bk or bk % chunk:
                    continue
                if 4 * bm * bk + 4 * bk * bn > _SHMEM_BYTES:
                    continue
                for tm in (16, 8):
                    if bm % tm:
                        continue
                    for tn in (16, 8):
                        if bn % tn:
                            continue
                        threads = (bm // tm) * (bn // tn)
                        if threads > _MAX_THREADS:
                            continue
                        fill = chunk * threads
                        if (bm * bk) % fill or (bk * bn) % fill:
                            continue
                        return (bm, bn, bk, tm, tn)
    return None


def _tc2d_tile(M, N, K):
    """TensorCore2D (bf16->f32, chunk=8, tm=tn=tk=16). Returns tile dict or None."""
    chunk = 8
    tm = tn = tk = 16
    for bm in (128, 64):
        if M % bm or bm % tm:
            continue
        for bn in (128, 64):
            if N % bn or bn % chunk or bn % tn:
                continue
            for bk in (64, 32, 16):
                if K % bk or bk % chunk or bk % tk:
                    continue
                if 2 * bm * bk + 2 * bk * bn > _SHMEM_BYTES:
                    continue
                for wm in (16, 8, 4, 2):
                    if (bm % (tm * wm)) or (bm % (wm * tm)):
                        continue
                    for wn in (16, 8, 4, 2):
                        if bn % (tn * wn):
                            continue
                        warps = (bm // (wm * tm)) * (bn // (wn * tn))
                        if warps * _WARP > _MAX_THREADS:
                            continue
                        fill = chunk * warps * _WARP
                        if (bm * bk) % fill or (bk * bn) % fill:
                            continue
                        return dict(bm=bm, bn=bn, bk=bk, tm=tm, tn=tn, tk=tk, wm=wm, wn=wn)
    return None


# ---------------------------------------------------------------------------
# Impl
# ---------------------------------------------------------------------------

class GemmImpl:
    """Handles aten::mm and aten::addmm for f32 (BT2D) and bf16 (TC2D)."""

    def supported(self, func, args, kwargs):
        import torch
        aten = torch.ops.aten

        if func is aten.mm.default and len(args) == 2:
            A, B = args
            return self._plan(A, B, op="mm")
        if func is aten.addmm.default and len(args) == 3:
            bias, A, B = args
            beta = kwargs.get("beta", 1)
            alpha = kwargs.get("alpha", 1)
            if not (isinstance(beta, (int, float)) and isinstance(alpha, (int, float))):
                return None
            return self._plan(A, B, op="addmm", bias=bias,
                              alpha=float(alpha), beta=float(beta))
        return None

    def _plan(self, A, B, op, bias=None, alpha=1.0, beta=1.0):
        import torch
        if not (A.is_cuda and B.is_cuda):
            return None
        if A.dim() != 2 or B.dim() != 2:
            return None
        if A.dtype != B.dtype:
            return None
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            return None
        M, K, N = int(M), int(K), int(N)

        if bias is not None:
            if not bias.is_cuda or bias.dtype != A.dtype:
                return None
            if bias.dim() == 1:
                if bias.shape[0] != N:
                    return None
            elif bias.dim() == 2:
                if tuple(bias.shape) != (M, N):
                    return None
            else:
                return None

        if A.dtype == torch.float32:
            tile = _bt2d_tile(M, N, K)
            if tile is None:
                return None
            bm, bn, bk, tm, tn = tile
            return dict(kind="bt2d", op=op, dtype="f32", M=M, N=N, K=K,
                        bm=bm, bn=bn, bk=bk, tm=tm, tn=tn,
                        alpha=alpha, beta=beta)
        if A.dtype == torch.bfloat16:
            tile = _tc2d_tile(M, N, K)
            if tile is None:
                return None
            return dict(kind="tc2d", op=op, dtype="bf16", M=M, N=N, K=K,
                        alpha=alpha, beta=beta, **tile)
        return None

    # ------------------------------------------------------------------
    def run(self, spec, args, kwargs):
        if spec["kind"] == "bt2d":
            return self._run_bt2d(spec, args, kwargs)
        return self._run_tc2d(spec, args, kwargs)

    def _run_bt2d(self, spec, args, kwargs):
        import torch
        if spec["op"] == "mm":
            A, B = args
            bias = None
            alpha, beta = 1.0, 0.0
        else:
            bias, A, B = args
            alpha, beta = spec["alpha"], spec["beta"]

        module = ("Klas.JitGemmBT2D_f32_"
                  f"{spec['bm']}x{spec['bn']}x{spec['bk']}_{spec['tm']}x{spec['tn']}")
        mod = _compile.build_kernel(
            module=module,
            fst_template="gemm_blocktiling2d.fst.j2",
            fst_ctx=dict(module=module, name="kernel",
                         bm=spec["bm"], bn=spec["bn"], bk=spec["bk"],
                         tm=spec["tm"], tn=spec["tn"], et="f32"),
            wrapper_template="wrapper_blocktiling2d.cu.j2",
            wrapper_ctx=dict(ctype="float"),
        )
        if bias is None:
            Cin = torch.zeros((spec["M"], spec["N"]), dtype=A.dtype, device=A.device)
        else:
            Cin = bias
        return mod.run(A, B, Cin, float(alpha), float(beta))

    def _run_tc2d(self, spec, args, kwargs):
        import torch
        if spec["op"] == "mm":
            A, B = args
            bias = None
            alpha, beta = 1.0, 0.0
        else:
            bias, A, B = args
            alpha, beta = spec["alpha"], spec["beta"]

        module = ("Klas.JitGemmTC2D_bf16_"
                  f"{spec['bm']}x{spec['bn']}x{spec['bk']}_"
                  f"{spec['tm']}x{spec['tn']}x{spec['tk']}_{spec['wm']}x{spec['wn']}")
        mod = _compile.build_kernel(
            module=module,
            fst_template="gemm_tensorcore2d.fst.j2",
            fst_ctx=dict(module=module, name="kernel", et_ab="bf16", et_c="float",
                         bm=spec["bm"], bn=spec["bn"], bk=spec["bk"],
                         tm=spec["tm"], tn=spec["tn"], tk=spec["tk"],
                         wm=spec["wm"], wn=spec["wn"]),
            wrapper_template="wrapper_tensorcore2d.cu.j2",
            wrapper_ctx=dict(),
        )
        # f32 accumulator C = A @ B
        Cf = mod.run(A, B)
        if bias is None:
            return (Cf * alpha).to(torch.bfloat16) if alpha != 1.0 else Cf.to(torch.bfloat16)
        out = alpha * Cf + beta * bias.to(torch.float32)
        return out.to(torch.bfloat16)
