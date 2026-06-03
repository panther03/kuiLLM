"""High-level PyTorch integration for Kuiper kernels.

Three flavours of integration are offered, in increasing order of risk:

1. `KuiperLinear` + `replace_linears(model, ...)` — surgical, opt-in module
   replacement for `torch.nn.Linear`. Weight is pre-transposed at swap time
   so each forward call is just a matmul (+ bias add). Falls back to the
   original `nn.Linear.forward` whenever the inputs don't satisfy Kuiper's
   shape/dtype constraints.

2. `enable_softmax_patch()` / `disable_softmax_patch()` — monkey-patches
   `torch.nn.functional.softmax` to route last-dim f32/f16 calls through
   Kuiper. **OFF by default**: Kuiper's softmax has no max-shift and will
   overflow on attention logits.

3. `enable_matmul_patch()` / `disable_matmul_patch()` — monkey-patches
   `torch.matmul` to route rank-2 f32/f16 calls satisfying the tile
   divisibility through Kuiper.

For Qwen2.5-bf16 inference there is no zero-cost path: bf16 has no verified
Kuiper GEMM. `KuiperLinear(cast_bf16_to_f16=True)` will downcast bf16 inputs
to f16, run the verified GEMM, and upcast — *with measurable accuracy loss*.

See `KUIPER_INTEGRATION.md` for the per-kernel mapping and the trade-offs.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import kuiper_ext as _kx


# Module-level dispatch counters so callers (e.g., profile_ops.py) can
# verify whether KuiperLinear actually fired the verified kernel or fell
# back to F.linear. Reset with `reset_dispatch_stats()`.
_dispatch_stats = {"kuiper": 0, "fallback": 0}


def get_dispatch_stats() -> dict:
    """Return a copy of the (kuiper, fallback) call counters."""
    return dict(_dispatch_stats)


def reset_dispatch_stats() -> None:
    _dispatch_stats["kuiper"] = 0
    _dispatch_stats["fallback"] = 0


# ---------------------------------------------------------------------------
# Linear replacement
# ---------------------------------------------------------------------------

def _shape_ok_for_f32_matmul(M: int, N: int, K: int) -> bool:
    return M > 0 and N > 0 and K > 0 and (M % 32 == 0) and (N % 32 == 0) and (K % 32 == 0)


def _shape_ok_for_f16_matmul(M: int, N: int, K: int) -> bool:
    return M > 0 and N > 0 and K > 0 and (M % 64 == 0) and (N % 64 == 0) and (K % 64 == 0)


class KuiperLinear(nn.Module):
    """Drop-in replacement for `nn.Linear` that uses verified Kuiper GEMM.

    Weight is stored in row-major (K, out_features) form (i.e., already
    transposed relative to `nn.Linear.weight` which is (out, in)) so that a
    forward pass is exactly `X @ W` — Kuiper's native layout.

    Parameters
    ----------
    cast_bf16_to_f16 : bool
        When the input is bf16, cast to f16, run the f16 TensorCore GEMM,
        cast back. This trades the verified-correctness guarantee for an
        approximate result. Default False — bf16 inputs simply fall back to
        the original `F.linear`.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None,
                 *, cast_bf16_to_f16: bool = False,
                 bias_dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cast_bf16_to_f16 = cast_bf16_to_f16

        factory = {"device": device, "dtype": dtype}
        # Stored as (in_features, out_features) row-major.
        self.weight_t = nn.Parameter(torch.empty(in_features, out_features, **factory))
        if bias:
            b_factory = {"device": device, "dtype": bias_dtype or dtype}
            self.bias = nn.Parameter(torch.empty(out_features, **b_factory))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @classmethod
    def from_linear(cls, lin: nn.Linear, *, cast_bf16_to_f16: bool = False) -> "KuiperLinear":
        # When the source weight is bf16 and we have opted into the f16
        # cast, *pre-cast the weight once now* — that way the per-call cost
        # is just an input cast + output cast, and the (typically) much
        # larger weight conversion never re-runs.
        store_dtype = lin.weight.dtype
        if cast_bf16_to_f16 and lin.weight.dtype == torch.bfloat16:
            store_dtype = torch.float16
        m = cls(
            lin.in_features, lin.out_features,
            bias=lin.bias is not None,
            device=lin.weight.device, dtype=store_dtype,
            bias_dtype=lin.weight.dtype,   # keep bias in the activation dtype
            cast_bf16_to_f16=cast_bf16_to_f16,
        )
        with torch.no_grad():
            # nn.Linear.weight is (out, in); we want (in, out) contiguous row-major.
            w_t = lin.weight.t().contiguous()
            if store_dtype != lin.weight.dtype:
                w_t = w_t.to(store_dtype)
            m.weight_t.copy_(w_t)
            if lin.bias is not None:
                m.bias.copy_(lin.bias)
        return m

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_t, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, cast_bf16_to_f16={self.cast_bf16_to_f16}, "
                f"weight_dtype={self.weight_t.dtype}")

    # ------------------------------------------------------------------
    # The forward dispatcher mirrors all of Kuiper's KPR_GUARDs so a
    # mismatch falls back to native instead of aborting the process.
    # ------------------------------------------------------------------
    def _native_forward(self, x: torch.Tensor) -> torch.Tensor:
        _dispatch_stats["fallback"] += 1
        with torch.profiler.record_function("kuiper::fallback_linear"):
            # If we pre-cast the weight, we need to undo it for fallback.
            w_full = self.weight_t.t()
            if w_full.dtype != x.dtype:
                w_full = w_full.to(x.dtype)
            b = self.bias
            if b is not None and b.dtype != x.dtype:
                b = b.to(x.dtype)
            return F.linear(x, w_full, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self._native_forward(x)

        with torch.profiler.record_function("kuiper::KuiperLinear.forward"):
            return self._cuda_forward(x)

    def _cuda_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        in_f = self.in_features
        out_f = self.out_features
        x2 = x.reshape(-1, in_f).contiguous()
        M = x2.size(0)

        x_dtype = x2.dtype
        w = self.weight_t  # (in, out), row-major contiguous
        w_dtype = w.dtype

        out2: Optional[torch.Tensor] = None

        if x_dtype == torch.float32 and w_dtype == torch.float32:
            if _shape_ok_for_f32_matmul(M, out_f, in_f):
                try:
                    with torch.profiler.record_function("kuiper::matmul_f32"):
                        out2 = _kx.matmul_f32(x2, w)
                except RuntimeError as e:
                    warnings.warn(f"kuiper matmul_f32 failed, falling back: {e}")
        elif x_dtype == torch.float16 and w_dtype == torch.float16:
            if _shape_ok_for_f16_matmul(M, out_f, in_f):
                try:
                    with torch.profiler.record_function("kuiper::matmul_f16"):
                        out2 = _kx.matmul_f16(x2, w)
                except RuntimeError as e:
                    warnings.warn(f"kuiper matmul_f16 failed, falling back: {e}")
        elif x_dtype == torch.bfloat16 and w_dtype == torch.bfloat16:
            # Verified bf16 in / f32 out TensorCore2D kernel — no input
            # casts, only the unavoidable f32 -> bf16 cast on the output.
            if _shape_ok_for_f16_matmul(M, out_f, in_f):
                try:
                    with torch.profiler.record_function("kuiper::matmul_bf16"):
                        out_f32 = _kx.matmul_bf16_to_f32(x2, w)
                        out2 = out_f32.to(torch.bfloat16)
                except RuntimeError as e:
                    warnings.warn(f"kuiper matmul_bf16_to_f32 failed, falling back: {e}")
        elif (self.cast_bf16_to_f16 and x_dtype == torch.bfloat16
                and w_dtype == torch.float16):
            # Legacy path (cast_bf16_to_f16=True): weight was pre-cast at
            # module-swap time; we cast the input per-call and the f16
            # output back to bf16. Kept for comparison / when the bf16
            # TensorCore2D variant is unavailable.
            if _shape_ok_for_f16_matmul(M, out_f, in_f):
                try:
                    with torch.profiler.record_function("kuiper::matmul_f16_bf16cast"):
                        x_f16 = x2.to(torch.float16)
                        out2 = _kx.matmul_f16(x_f16, w)
                        out2 = out2.to(torch.bfloat16)
                except RuntimeError as e:
                    warnings.warn(f"kuiper matmul_f16 (bf16-cast) failed, falling back: {e}")

        if out2 is None:
            return self._native_forward(x)

        _dispatch_stats["kuiper"] += 1
        if self.bias is not None:
            out2 = out2 + self.bias

        return out2.reshape(*orig_shape[:-1], out_f)


def replace_linears(model: nn.Module, *,
                    cast_bf16_to_f16: bool = False,
                    name_filter=None) -> int:
    """Recursively replace every `nn.Linear` in `model` with `KuiperLinear`.

    `name_filter(qualified_name) -> bool` lets you keep some Linears un-
    replaced (e.g., the LM head). Returns the number of replacements made.
    """
    if name_filter is None:
        name_filter = lambda _name: True
    n_replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            qname = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, KuiperLinear):
                if name_filter(qname):
                    new = KuiperLinear.from_linear(child, cast_bf16_to_f16=cast_bf16_to_f16)
                    setattr(module, child_name, new)
                    n_replaced += 1
    return n_replaced


# ---------------------------------------------------------------------------
# Optional monkey patches (off by default)
# ---------------------------------------------------------------------------

_orig_softmax = None
_orig_matmul = None


def _patched_softmax(x, dim=None, _stacklevel=3, dtype=None):
    if (dim in (-1, x.dim() - 1) and x.is_cuda and x.is_contiguous() and dtype is None):
        if x.dtype == torch.float32:
            y = x.clone()
            try:
                with torch.profiler.record_function("kuiper::softmax_last_f32"):
                    _kx.softmax_last_f32_(y)
                return y
            except RuntimeError:
                pass
        elif x.dtype == torch.float16:
            y = x.clone()
            try:
                with torch.profiler.record_function("kuiper::softmax_last_f16"):
                    _kx.softmax_last_f16_(y)
                return y
            except RuntimeError:
                pass
    return _orig_softmax(x, dim=dim, _stacklevel=_stacklevel, dtype=dtype)


def enable_softmax_patch() -> None:
    """**Experimental.** Patch F.softmax to route last-dim f16/f32 to Kuiper.

    WARNING: Kuiper's softmax has no max-shift; large logits overflow to
    inf/NaN. Do NOT enable for attention softmax on un-pre-shifted logits.
    """
    global _orig_softmax
    if _orig_softmax is None:
        _orig_softmax = F.softmax
        F.softmax = _patched_softmax  # type: ignore[assignment]


def disable_softmax_patch() -> None:
    global _orig_softmax
    if _orig_softmax is not None:
        F.softmax = _orig_softmax  # type: ignore[assignment]
        _orig_softmax = None


def _patched_matmul(a, b, *, out=None):
    if out is None and a.is_cuda and b.is_cuda and a.dim() == 2 and b.dim() == 2:
        M, K = a.shape
        K2, N = b.shape
        if K == K2:
            if (a.dtype == torch.float32 and b.dtype == torch.float32
                    and _shape_ok_for_f32_matmul(M, N, K)):
                try:
                    with torch.profiler.record_function("kuiper::matmul_f32"):
                        return _kx.matmul_f32(a.contiguous(), b.contiguous())
                except RuntimeError:
                    pass
            elif (a.dtype == torch.float16 and b.dtype == torch.float16
                    and _shape_ok_for_f16_matmul(M, N, K)):
                try:
                    with torch.profiler.record_function("kuiper::matmul_f16"):
                        return _kx.matmul_f16(a.contiguous(), b.contiguous())
                except RuntimeError:
                    pass
    return _orig_matmul(a, b, out=out)


def enable_matmul_patch() -> None:
    """Patch `torch.matmul` to route rank-2 f16/f32 calls through Kuiper.

    Rank > 2 or non-divisible shapes fall through to PyTorch.
    """
    global _orig_matmul
    if _orig_matmul is None:
        _orig_matmul = torch.matmul
        torch.matmul = _patched_matmul  # type: ignore[assignment]


def disable_matmul_patch() -> None:
    global _orig_matmul
    if _orig_matmul is not None:
        torch.matmul = _orig_matmul  # type: ignore[assignment]
        _orig_matmul = None


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def enable_kuiper(model: Optional[nn.Module] = None,
                  *,
                  replace_linear: bool = True,
                  cast_bf16_to_f16: bool = False,
                  patch_softmax: bool = False,
                  patch_matmul: bool = False,
                  linear_filter=None) -> dict:
    """Apply all enabled Kuiper integrations to `model` and/or globals."""
    info = {"linears_replaced": 0, "softmax_patched": False, "matmul_patched": False}
    if replace_linear and model is not None:
        info["linears_replaced"] = replace_linears(
            model, cast_bf16_to_f16=cast_bf16_to_f16, name_filter=linear_filter)
    if patch_softmax:
        enable_softmax_patch()
        info["softmax_patched"] = True
    if patch_matmul:
        enable_matmul_patch()
        info["matmul_patched"] = True
    return info


def disable_kuiper() -> None:
    """Undo monkey-patches. Module replacements stay (use a fresh model)."""
    disable_softmax_patch()
    disable_matmul_patch()
