"""
Python bindings for KuiOps kernels. These classes:
 a) check if a given aten op is supported by an instantiation of a KuiOps template
 b) extract the relevant kernel from the template, compile it, and call it.
"""
from . import compile as _compile
from . import autotune as _autotune
from . import config as _config
from .config import log

import torch
aten = torch.ops.aten

_MAX_NUMEL = 2097152 * 1024

def _scalar(x):
    
    if isinstance(x, (float, int)):
        return x
    if isinstance(x, torch.Tensor) and x.dim() == 0 and x.dtype == torch.float64:
        return float(x.item())
    return None


def _norm_dim(d, rank):
    return d + rank if d < 0 else d


class _Family:
    """An operator family: ``supported(func, args, kwargs)`` returns the spec of
    the kernel instantiation serving this call (or ``None``), and ``run(spec,
    args, kwargs)`` extracts, compiles and invokes it."""

    fst_template = None
    wrapper_template = None
    # False for families whose kernel internally launches several GPU kernels and
    # synchronizes to observe the intermediate results: those cannot take a
    # caller-supplied stream, so they must not be recorded into a CUDA graph.
    # They stay callable directly (e.g. from the unit tests).
    graph_safe = True

    def _mod(self, module, fst_ctx, wrapper_ctx):
        return _compile.build_kernel(module,
            self.fst_template, fst_ctx,
            self.wrapper_template, wrapper_ctx)

def torch_dtype_to_fstar(dt):
    return {
        torch.float16: "f16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.bfloat16: "bf16",
        torch.int64: "i64",
        torch.int32: "i32",
        torch.int16: "i16",
        torch.int8: "i8",
        torch.uint64: "u64",
        torch.uint32: "u32",
        torch.uint16: "u16",
        torch.uint8: "u8",
        torch.bool: "u8"
    }[dt]

def torch_dtype_to_fstar_namespace(dt):
    return {
        torch.float16: "Kuiper.Float16",
        torch.float32: "Kuiper.Float32",
        torch.float64: "Kuiper.Float64",
        torch.bfloat16: "Kuiper.BFloat16",
        torch.int64:  "FStar.Int64",
        torch.int32:  "FStar.Int32",
        torch.int16:  "FStar.Int16",
        torch.int8:   "FStar.Int8",
        torch.uint64: "FStar.UInt64",
        torch.uint32: "FStar.UInt32",
        torch.uint16: "FStar.UInt16",
        torch.uint8:  "FStar.UInt8"
    }[dt]

def torch_dtype_to_ctype(dt):
    return {
        torch.float16: "__half",
        torch.float32: "float",
        torch.float64: "double",
        torch.bfloat16: "__nv_bfloat16",
        torch.int64: "int64_t",
        torch.int32: "int32_t",
        torch.int16: "int16_t",
        torch.int8: "int8_t",
        torch.uint64: "uint64_t",
        torch.uint32: "uint32_t",
        torch.uint16: "uint16_t",
        torch.uint8: "uint8_t",
        torch.bool: "uint8_t"
    }[dt]

def torch_dtype_to_aten_scalar(dt):
    """libtorch ScalarType enum name, for allocating C++ output tensors."""
    return {
        torch.float16: "torch::kFloat16",
        torch.float32: "torch::kFloat32",
        torch.float64: "torch::kFloat64",
        torch.bfloat16: "torch::kBFloat16",
        torch.bool: "torch::kBool",
        torch.int64: "torch::kInt64",
        torch.int32: "torch::kInt32",
        torch.int16: "torch::kInt16",
        torch.int8: "torch::kInt8",
        torch.uint64: "torch::kUInt64",
        torch.uint32: "torch::kUInt32",
        torch.uint16: "torch::kUInt16",
        torch.uint8: "torch::kUInt8",
    }[dt]

_FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64, torch.bfloat16]

# TODO: signed integers do not have the scalar typeclass in kuiper
# because we do not have total unconditional operations on them. 
# What to do about it?
_SCALAR_DTYPES = _FLOAT_DTYPES + [torch.uint16, torch.uint32, torch.uint64]

_INTEGER_DTYPES = [
    torch.bool,
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.uint8, torch.uint16, torch.uint32, torch.uint64,
]

_SIGNED_INTEGER_DTYPES = {
    torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64,
}

_UNSIGNED_INTEGER_DTYPES = {
    torch.bool: 8, torch.uint8: 8, torch.uint16: 16,
    torch.uint32: 32, torch.uint64: 64,
}
                
def cast_constarg(c,dt):
    if dt in _FLOAT_DTYPES:
        return f"(fcast (Kuiper.Float64.is_floating.of_literal \"{float(c):f}\"))"
    elif isinstance(c, int):    
        cast_typename = {
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.uint8: "uint8",
            torch.uint16: "uint16",
            torch.uint32: "uint32",
            torch.uint64: "uint64"
        }[dt]
        return f"(uint64_to_{cast_typename} (FStar.UInt64.uint_to_t {c:d}))"
    else:
        raise ValueError(c)

# LATER: be more dilligent about checking tensor layouts when a Kuiper operator expects row_major or column_major etc.
# Or if we flatten it on the Kuiper side then it doesn't matter.

# ---------------------------------------------------------------------------
# Elementwise (unary, binary, scalar-broadcast)
# ---------------------------------------------------------------------------

def _const_tag(constargs):
    """A filename-safe identifier fragment encoding baked-in constants, so two
    calls with different scalar operands don't collide in the kernel cache."""
    if not constargs:
        return ""
    parts = []
    for c in constargs:
        s = repr(c).replace("-", "n").replace(".", "p").replace("+", "")
        parts.append("".join(ch for ch in s if ch.isalnum()))
    return "C" + "_".join(parts)


class ElementwiseImpl(_Family):
    fst_template = "elementwise/Kuiops.Elementwise.Inst.fst.j2"
    wrapper_template = "elementwise/wrapper_elementwise.cu.j2"

    # aten op -> F* impl name (arithmetic ops via the scalar/floating typeclass).
    _IMPL = {
        aten.silu.default:      "silu",
        aten.relu.default:      "relu",
        aten.neg.default:       "neg",
        aten.rsqrt.default:     "rsqrt",
        aten.cos.default:       "cos",
        aten.sin.default:       "sin",
        aten.pow.Tensor_Scalar: "pow",
        aten.add.Tensor:        "add",
        aten.add.Scalar:        "add",
        aten.mul.Tensor:        "mul",
        aten.mul.Scalar:        "mul",
        aten.sub.Tensor:        "sub",
        aten.sub.Scalar:        "sub",
        aten.div.Tensor:        "div",
        aten.div.Scalar:        "div",
    }

    # Ops that require the `floating` typeclass (float dtypes only).
    _FLOATING_ONLY = {
        aten.silu.default, aten.neg.default, aten.rsqrt.default,
        aten.cos.default, aten.sin.default, aten.pow.Tensor_Scalar,
        aten.sub.Tensor, aten.sub.Scalar, aten.div.Tensor, aten.div.Scalar,
    }

    # Scalar comparisons: `T op scalar -> bool`, via map_gpu_notinplace.
    _COMPARE = {
        aten.eq.Scalar: "eq_u8",
        aten.le.Scalar: "le_u8",
        aten.lt.Scalar: "lt_u8",
    }

    # Bitwise ops, modelled on u8 (torch.bool is a byte); in-place same-type.
    _BITWISE1 = {aten.bitwise_not.default: "bnot"}
    _BITWISE2 = {aten.bitwise_and.Tensor: "band", aten.bitwise_or.Tensor: "bor"}

    _BOOL_DTYPES = (torch.bool, torch.uint8)

    def supported(self, func, args, kwargs):
        A = args[0]
        if not (isinstance(A, torch.Tensor) and A.is_cuda
                and 0 < A.numel() <= _MAX_NUMEL):
            return None

        def same(B):
            return (isinstance(B, torch.Tensor) and B.is_cuda
                    and B.dtype == A.dtype and tuple(B.shape) == tuple(A.shape))

        # Unary bitwise (bool), in-place.
        if func in ElementwiseImpl._BITWISE1:
            if A.dtype != torch.bool:
                return None
            return dict(kind="map", method=ElementwiseImpl._BITWISE1[func],
                        in_dtypes=[A.dtype], out_dtype=A.dtype,
                        constargs=[])

        # Binary bitwise (bool), in-place.
        if func in ElementwiseImpl._BITWISE2:
            if A.dtype != torch.bool or not same(args[1]):
                return None
            return dict(kind="map2", method=ElementwiseImpl._BITWISE2[func],
                        in_dtypes=[A.dtype, A.dtype], out_dtype=A.dtype,
                        constargs=[])

        # Scalar comparisons -> bool. Only the scalar-rhs form is supported:
        # there is no not-in-place *binary* map kernel, so the Tensor overloads
        # (le.Tensor / lt.Tensor) fall through to PyTorch.
        if func in ElementwiseImpl._COMPARE:
            if A.dtype not in _FLOAT_DTYPES:
                return None
            s = _scalar(args[1])
            if s is None:
                return None
            return dict(kind="map_nip", method=ElementwiseImpl._COMPARE[func],
                        in_dtypes=[A.dtype], out_dtype=torch.bool,
                        constargs=[s])

        # Ternary select: where(cond, x, y). No broadcasting: shapes must match.
        if func is aten.where.self:
            if len(args) < 3:
                return None
            Cnd, X, Y = args[0], args[1], args[2]
            if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in (Cnd, X, Y)):
                return None
            if (Cnd.dtype not in ElementwiseImpl._BOOL_DTYPES or X.dtype != Y.dtype
                    or not (tuple(Cnd.shape) == tuple(X.shape) == tuple(Y.shape))):
                return None
            if not (0 < X.numel() <= _MAX_NUMEL):
                return None
            return dict(kind="map3", method="bwhere",
                        in_dtypes=[Cnd.dtype, X.dtype, Y.dtype],
                        out_dtype=X.dtype, constargs=[])

        # Arithmetic path (unary / binary / scalar-broadcast).
        impl = ElementwiseImpl._IMPL.get(func)
        if impl is None:
            return None
        if A.dtype not in _SCALAR_DTYPES:
            return None
        if func in ElementwiseImpl._FLOATING_ONLY and A.dtype not in _FLOAT_DTYPES:
            return None

        if len(args) == 1:
            return dict(kind="map", method=impl, in_dtypes=[A.dtype],
                        out_dtype=A.dtype, constargs=[])

        constargs = []
        if func in (aten.add.Tensor, aten.add.Scalar) and kwargs.get("alpha", 1) != 1:
            impl += "_alpha"
            alpha = _scalar(kwargs["alpha"])
            assert alpha is not None
            constargs += [alpha]

        # Special-case x**2 -> square (better precision than generic pow).
        if (func is aten.pow.Tensor_Scalar and isinstance(args[1], int)
                and args[1] == 2):
            return dict(kind="map", method="square", in_dtypes=[A.dtype],
                        out_dtype=A.dtype, constargs=[])

        # Scalar second operand (by spec e.g. add.Scalar, or by overloading).
        if (s := _scalar(args[1])) is not None:
            constargs += [s]
            return dict(kind="map", method=impl, in_dtypes=[A.dtype],
                        out_dtype=A.dtype, constargs=constargs)
        if same(args[1]):
            return dict(kind="map2", method=impl, in_dtypes=[A.dtype, A.dtype],
                        out_dtype=A.dtype, constargs=constargs)
        return None

    def run(self, spec, args, kwargs):
        kind, method = spec["kind"], spec["method"]
        in_dtypes, out_dtype = spec["in_dtypes"], spec["out_dtype"]

        # TODO: proper PyTorch type promotion. For now constants are cast to the
        # input element type.
        fs = [torch_dtype_to_fstar(d) for d in in_dtypes]
        ins = " ".join(f"(i{i}: {fs[i]})" for i in range(len(in_dtypes)))
        consts = [cast_constarg(c, in_dtypes[0]) for c in spec["constargs"]]
        body = [f"i{i}" for i in range(len(in_dtypes))] + consts
        fun = f"fun {ins} -> {method} {' '.join(body)}"

        tag = "_".join(fs)
        ctag = _const_tag(spec["constargs"])
        module = (f"Kuiops.Elementwise.{kind.title().replace('_', '')}"
                  f".{method.title()}.{tag.title()}" + (f".{ctag}" if ctag else ""))
        name = f"elem_{kind}_{method}_{tag}_{ctag}_jit".lower()

        fst_ctx = dict(module=module, name=name, kind=kind, fun=fun)
        wrapper_ctx = dict(module=module.replace(".", "_"), name=name, kind=kind)

        if kind in ("map", "map2"):
            fst_ctx["et"] = fs[0]
            wrapper_ctx["cpp_et"] = torch_dtype_to_ctype(in_dtypes[0])
        elif kind == "map_nip":
            fst_ctx["et"], fst_ctx["ot"] = fs[0], torch_dtype_to_fstar(out_dtype)
            wrapper_ctx["cpp_et"] = torch_dtype_to_ctype(in_dtypes[0])
            wrapper_ctx["cpp_ot"] = torch_dtype_to_ctype(out_dtype)
            wrapper_ctx["out_scalar"] = torch_dtype_to_aten_scalar(out_dtype)
        elif kind == "map3":
            fst_ctx["eta"], fst_ctx["etb"], fst_ctx["etc"] = fs[0], fs[1], fs[2]
            fst_ctx["eto"] = torch_dtype_to_fstar(out_dtype)
            wrapper_ctx["cpp_et"] = torch_dtype_to_ctype(in_dtypes[1])

        return self._mod(module, fst_ctx, wrapper_ctx).run(*args[:len(in_dtypes)])


# ---------------------------------------------------------------------------
# Matmul family (mm / addmm / bmm)
# ---------------------------------------------------------------------------
#
# Three Kuiper GEMM backends, all row-major:
#   bt2d     BlockTiling2D, one element type throughout, batched (rank-2 runs at
#            batch = 1: an [m, n] buffer *is* a [1, m, n] batched one).
#   tc2d     TensorCore2D, accumulates in `acc` and writes `acc` in place.
#   tc2d_to  TensorCore2D.To, accumulates in `acc` and casts to `out`.

_SHMEM_BYTES = 101376
_MAX_THREADS = 1024
_MAX_BLOCKS = 2097152
_WARP = 32

# The GEMM operands are copied through `has_vec_cpy`, which exists for these
# element types only.
_VEC_CPY_DTYPES = (torch.float16, torch.float32, torch.bfloat16)

# Tensor-core input element types and the compute capability each needs.
_TC_INPUT_DTYPES = {torch.float16: (7, 0), torch.bfloat16: (8, 0)}

# Accumulator element types the tensor cores can pair with each input type
# (Kuiper's `valid_frag_et_dims` / `valid_frag_et_comb`).
_TC_ACC_DTYPES = {
    torch.float16: (torch.float16, torch.float32),
    torch.bfloat16: (torch.float32,),
}

_BACKEND_TAG = {"tc2d": "Tc2D", "tc2d_to": "Tc2DTo", "bt2d": "Bt2D"}


def _tc_device_supported(dtype, device):
    cc = _config.cuda_device_capability(device)
    return cc is not None and cc >= _TC_INPUT_DTYPES[dtype]


def _bt2d_tiles(dtype, M, N, K):
    """Every legal BlockTiling2D parameterization for this problem.

    chunk and the shared-memory byte budget scale with the element size
    (``chunk et = 16/sizeof(et)``)."""
    tiles = []
    itemsize = dtype.itemsize
    chunk = 16 // itemsize
    for bm in (128, 64, 32):
        if M % bm:
            continue
        for bn in (128, 64, 32):
            if N % bn or bn % chunk:
                continue
            for bk in (64, 32):
                if K % bk or bk % chunk:
                    continue
                if itemsize * bm * bk + itemsize * bk * bn > _SHMEM_BYTES:
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
                        tiles.append(dict(bm=bm, bn=bn, bk=bk, tm=tm, tn=tn))
    return tiles


def _tc2d_tiles(dtype, M, N, K):
    """Every legal TensorCore2D parameterization for this problem, ranges listed
    in preference order. The fragment dims are fixed at the 16x16x16 shape."""
    tiles = []
    chunk = 16 // dtype.itemsize
    tm = tn = tk = 16
    for bm in (128, 64):
        if M % bm or bm % tm:
            continue
        for bn in (128, 64):
            if N % bn or bn % chunk or bn % tn:
                continue
            for bk in (32, 64, 16):
                if K % bk or bk % chunk or bk % tk:
                    continue
                if 2 * (bm * bk + bk * bn) > _SHMEM_BYTES:
                    continue
                for wm in (8, 4, 2, 16):
                    if bm % (wm * tm):
                        continue
                    for wn in (4, 8, 2, 16):
                        if bn % (wn * tn):
                            continue
                        warps = (bm // (wm * tm)) * (bn // (wn * tn))
                        if warps * _WARP > _MAX_THREADS:
                            continue
                        fill = chunk * warps * _WARP
                        if (bm * bk) % fill or (bk * bn) % fill:
                            continue
                        tiles.append(dict(bm=bm, bn=bn, bk=bk, tm=tm, tn=tn,
                                          tk=tk, wm=wm, wn=wn))
    return tiles


def _tiles(backend, dtype, batch, M, N, K):
    """Legal tilings, in priority order, whose grid also fits in one launch."""
    enumerate_tiles = _bt2d_tiles if backend == "bt2d" else _tc2d_tiles
    return [
        tile for tile in enumerate_tiles(dtype, M, N, K)
        if batch * (M // tile["bm"]) * (N // tile["bn"]) <= _MAX_BLOCKS
    ]


def _backend_supports(backend, in_dtype, acc_dtype, out_dtype, device):
    if backend == "bt2d":
        return (in_dtype in _VEC_CPY_DTYPES
                and acc_dtype == in_dtype and out_dtype == in_dtype)
    if (in_dtype not in _TC_INPUT_DTYPES
            or not _tc_device_supported(in_dtype, device)
            or acc_dtype not in _TC_ACC_DTYPES[in_dtype]):
        return False
    if backend == "tc2d":
        # TensorCore2D writes the accumulator itself; no cast on the way out.
        return out_dtype == acc_dtype
    return out_dtype in _VEC_CPY_DTYPES


class _MatmulFamily(_Family):
    """Backend selection shared by mm / addmm / bmm.

    ``acc_dtype`` and ``out_dtype`` kwargs pin the accumulator and output
    element types; ``impl`` pins the backend (and rejects the call outright if
    that backend cannot serve it). An unset ``acc_dtype`` means f32 for the
    tensor-core backends, and the input type for BlockTiling2D."""

    # Backends this operator can use, in priority order.
    backends = ()
    operation = None

    def _plans(self, kwargs, in_dtype, device):
        """The (backend, acc_dtype, out_dtype) triples this call admits."""
        requested = kwargs.get("impl")
        if requested is not None and requested not in self.backends:
            return []
        acc_dtype = kwargs.get("acc_dtype")
        out_dtype = kwargs.get("out_dtype", in_dtype)
        plans = []
        for backend in self.backends:
            if requested is not None and backend != requested:
                continue
            acc = acc_dtype
            if acc is None:
                acc = in_dtype if backend == "bt2d" else torch.float32
            if _backend_supports(backend, in_dtype, acc, out_dtype, device):
                plans.append((backend, acc, out_dtype))
        return plans

    def _select(self, args, kwargs, specs):
        key = _autotune.make_key(self.operation, args, {
            "impl": kwargs.get("impl"),
            "acc_dtype": kwargs.get("acc_dtype"),
            "out_dtype": kwargs.get("out_dtype"),
        })
        return _autotune.tune(key, kwargs, specs, self.run, args[0].device)


def _tile_tag(tile):
    return "_".join(f"{k}{v}" for k, v in tile.items())


def _gemm_fst_ctx(spec, name):
    return dict(
        module=spec["module"], name=name, backend=spec["backend"],
        in_et=torch_dtype_to_fstar(spec["in_dtype"]),
        acc_et=torch_dtype_to_fstar(spec["acc_dtype"]),
        out_et=torch_dtype_to_fstar(spec["out_dtype"]),
        **spec["tile"])


def _gemm_wrapper_ctx(spec, name):
    return dict(
        module=spec["module"].replace(".", "_"), name=name,
        backend=spec["backend"],
        cpp_in_et=torch_dtype_to_ctype(spec["in_dtype"]),
        cpp_acc_et=torch_dtype_to_ctype(spec["acc_dtype"]),
        cpp_out_et=torch_dtype_to_ctype(spec["out_dtype"]),
        out_scalar=torch_dtype_to_aten_scalar(spec["out_dtype"]))


class MmImpl(_MatmulFamily):
    fst_template = "mm/Kuiops.Mm.Inst.fst.j2"
    wrapper_template = "mm/wrapper_mm.cu.j2"
    backends = ("tc2d", "tc2d_to", "bt2d")
    operation = "aten.mm.default"

    @staticmethod
    def _spec(backend, tile, in_dtype, acc_dtype, out_dtype):
        module = (
            f"Kuiops.Mm.{_BACKEND_TAG[backend]}"
            f".{torch_dtype_to_fstar(in_dtype).title()}"
            f".{torch_dtype_to_fstar(acc_dtype).title()}"
            f".{torch_dtype_to_fstar(out_dtype).title()}.P_{_tile_tag(tile)}")
        return dict(module=module, backend=backend, tile=tile,
                    in_dtype=in_dtype, acc_dtype=acc_dtype, out_dtype=out_dtype)

    def supported(self, func, args, kwargs):
        if len(args) != 2:
            return None
        A, B = args
        if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
                and A.is_cuda and B.is_cuda
                and A.dim() == 2 and B.dim() == 2 and A.dtype == B.dtype):
            return None
        M, K = (int(x) for x in A.shape)
        K2, N = (int(x) for x in B.shape)
        if K != K2:
            return None
        specs = [
            self._spec(backend, tile, A.dtype, acc_dtype, out_dtype)
            for backend, acc_dtype, out_dtype in self._plans(kwargs, A.dtype, A.device)
            for tile in _tiles(backend, A.dtype, 1, M, N, K)
        ]
        return self._select(args, kwargs, specs)

    def run(self, spec, args, kwargs):
        name = "mm_jit"
        mod = self._mod(spec["module"], _gemm_fst_ctx(spec, name),
                        _gemm_wrapper_ctx(spec, name))
        return mod.run(*args)


class BmmImpl(_MatmulFamily):
    """BlockTiling2D is batched, so one launch covers every page."""

    fst_template = "bmm/Kuiops.Bmm.Inst.fst.j2"
    wrapper_template = "bmm/wrapper_bmm.cu.j2"
    backends = ("bt2d",)
    operation = "aten.bmm.default"

    @staticmethod
    def _spec(backend, tile, in_dtype, acc_dtype, out_dtype):
        module = (f"Kuiops.Bmm.{torch_dtype_to_fstar(in_dtype).title()}"
                  f".P_{_tile_tag(tile)}")
        return dict(module=module, backend=backend, tile=tile,
                    in_dtype=in_dtype, acc_dtype=acc_dtype, out_dtype=out_dtype)

    def supported(self, func, args, kwargs):
        if len(args) != 2:
            return None
        A, B = args
        if not (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
                and A.is_cuda and B.is_cuda
                and A.dim() == 3 and B.dim() == 3 and A.dtype == B.dtype):
            return None
        batch, M, K = (int(x) for x in A.shape)
        batch2, K2, N = (int(x) for x in B.shape)
        if batch != batch2 or K != K2:
            return None
        # The three batched products must fit in a u32 index.
        if max(batch * M * K, batch * K * N, batch * M * N) >= 2 ** 32:
            return None
        specs = [
            self._spec(backend, tile, A.dtype, acc_dtype, out_dtype)
            for backend, acc_dtype, out_dtype in self._plans(kwargs, A.dtype, A.device)
            for tile in _tiles(backend, A.dtype, batch, M, N, K)
        ]
        return self._select(args, kwargs, specs)

    def run(self, spec, args, kwargs):
        name = "bmm_jit"
        mod = self._mod(spec["module"], _gemm_fst_ctx(spec, name),
                        _gemm_wrapper_ctx(spec, name))
        return mod.run(*args)


class AddmmImpl(_MatmulFamily):
    """addmm(C, A, B, alpha, beta) = beta*C + alpha*(A@B). The output is a copy
    of C, which the kernel updates through the `lincomb` combiner."""

    fst_template = "addmm/Kuiops.Addmm.Inst.fst.j2"
    wrapper_template = "addmm/wrapper_addmm.cu.j2"
    backends = ("tc2d", "tc2d_to", "bt2d")
    operation = "aten.addmm.default"

    @staticmethod
    def _spec(backend, tile, in_dtype, acc_dtype, out_dtype):
        module = (
            f"Kuiops.Addmm.{_BACKEND_TAG[backend]}"
            f".{torch_dtype_to_fstar(in_dtype).title()}"
            f".{torch_dtype_to_fstar(acc_dtype).title()}"
            f".{torch_dtype_to_fstar(out_dtype).title()}.P_{_tile_tag(tile)}")
        return dict(module=module, backend=backend, tile=tile,
                    in_dtype=in_dtype, acc_dtype=acc_dtype, out_dtype=out_dtype)

    def supported(self, func, args, kwargs):
        if len(args) != 3:
            return None
        Cin, A, B = args
        if not all(isinstance(t, torch.Tensor) and t.is_cuda and t.dim() == 2
                   for t in (Cin, A, B)):
            return None
        if A.dtype != B.dtype:
            return None
        M, K = (int(x) for x in A.shape)
        K2, N = (int(x) for x in B.shape)
        if K != K2 or tuple(int(x) for x in Cin.shape) != (M, N):
            return None
        alpha = _scalar(kwargs.get("alpha", 1))
        beta = _scalar(kwargs.get("beta", 1))
        if alpha is None or beta is None:
            return None
        specs = [
            self._spec(backend, tile, A.dtype, acc_dtype, out_dtype)
            for backend, acc_dtype, out_dtype in self._plans(kwargs, A.dtype, A.device)
            # C is the output buffer, so it already has to be the output type.
            if Cin.dtype == out_dtype
            for tile in _tiles(backend, A.dtype, 1, M, N, K)
        ]
        return self._select(args, kwargs, specs)

    def run(self, spec, args, kwargs):
        name = "addmm_jit"
        mod = self._mod(spec["module"], _gemm_fst_ctx(spec, name),
                        _gemm_wrapper_ctx(spec, name))
        alpha = float(_scalar(kwargs.get("alpha", 1)))
        beta = float(_scalar(kwargs.get("beta", 1)))
        return mod.run(*args, alpha, beta)


# ---------------------------------------------------------------------------
# softmax (row-wise, last dim)
# ---------------------------------------------------------------------------

# LATER: implement this as a proper n-dimensional batched operator on the kuiper side
class SoftmaxImpl(_Family):
    fst_template = "softmax/Kuiops.Softmax.Inst.fst.j2"
    wrapper_template = "softmax/wrapper_softmax.cu.j2"
    # RowSoftmax is a composite kernel: it launches several GPU kernels and syncs
    # in between to observe their results, so it cannot be graph-captured.
    graph_safe = False

    def supported(self, func, args, kwargs):
        # aten._softmax.default(self, dim, half_to_float)
        if len(args) != 3:
            return None
        X, dim, half_to_float = args
        if not (X.is_cuda and X.dtype in _FLOAT_DTYPES and not half_to_float):
            return None
        rank = X.dim()
        if rank < 1 or _norm_dim(dim, rank) != rank - 1:
            return None
        n = int(X.shape[-1])
        m = X.numel() // n if n else 0
        # RowSoftmax: m <= max_blocks, m*n <= max_blocks*max_threads.
        if m <= 0 or m > _MAX_BLOCKS or m * n > _MAX_NUMEL:
            return None
        return dict(dtype=X.dtype, rows=m, cols=n)

    def run(self, spec, args, kwargs):
        dtype, m, n = spec["dtype"], spec["rows"], spec["cols"]
        X = args[0]
        et = torch_dtype_to_fstar(dtype)
        module = f"Kuiops.Softmax.{et.title()}"
        name = "softmax_jit"
        fst_ctx = dict(module=module, name=name, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            cpp_et=torch_dtype_to_ctype(dtype),
            nth=_MAX_THREADS,
        )
        # TODO: move this stuff to C++ for consistency
        A = X.contiguous().reshape(m, n).clone()
        out = self._mod(module, fst_ctx, wrapper_ctx).run(A)
        return out.reshape(X.shape)


import math as _math

# ---------------------------------------------------------------------------
# Scaled dot-product attention (efficient_attention)
# ---------------------------------------------------------------------------

class SdpaImpl(_Family):
    fst_template = "sdpa/Kuiops.Sdpa.Inst.fst.j2"
    wrapper_template = "sdpa/wrapper_sdpa.cu.j2"
    # Composite (it runs a row softmax internally), so it syncs: not graph-safe.
    graph_safe = False

    def supported(self, func, args, kwargs):
        # _scaled_dot_product_efficient_attention(query, key, value, attn_bias,
        #   compute_log_sumexp, dropout_p=0.0, is_causal=False, *, scale=None)
        if len(args) < 5:
            return None
        Q, Kt, V, bias = args[0], args[1], args[2], args[3]
        compute_log_sumexp = args[4]
        dropout_p = args[5] if len(args) > 5 else kwargs.get("dropout_p", 0.0)
        is_causal = args[6] if len(args) > 6 else kwargs.get("is_causal", False)
        # The kernel does not produce the LSE needed by the backward pass.
        if bias is None or compute_log_sumexp or dropout_p != 0.0 or is_causal:
            return None
        if not all(isinstance(t, torch.Tensor) and t.is_cuda and t.dim() == 4
                   for t in (Q, Kt, V, bias)):
            return None
        if not all(t.device == Q.device for t in (Kt, V, bias)):
            return None
        N, H, L, E = (int(x) for x in Q.shape)
        N2, H2, S, E2 = (int(x) for x in Kt.shape)
        N3, H3, S2, Ev = (int(x) for x in V.shape)
        if min(N, H, L, S, E, Ev) <= 0:
            return None
        if (N, H, E) != (N2, H2, E2) or (N, H, S) != (N3, H3, S2):
            return None
        if tuple(int(x) for x in bias.shape) != (N, H, L, S):
            return None
        # Kernel index/grid constraints.
        if (L * S > _MAX_NUMEL or L * Ev > _MAX_NUMEL or
                N * H * L > _MAX_BLOCKS or N * H * L * S > _MAX_NUMEL):
            return None
        if max(N * H * L * E, N * H * S * E, N * H * S * Ev,
                N * H * L * Ev, N * H * L * S) >= 2 ** 32:
            return None
        scale = kwargs.get("scale", None)
        if scale is None:
            scale = 1.0 / _math.sqrt(E)
        return dict(out_dtype=bias.dtype, scale=float(scale))

    def run(self, spec, args, kwargs):
        out_dtype, scale = spec["out_dtype"], spec["scale"]
        dtype = torch.float32
        et = torch_dtype_to_fstar(dtype)
        module = "Kuiops.Sdpa.F32"
        name = "sdpa_jit"
        fst_ctx = dict(module=module, name=name, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"),
            name=name,
            cpp_et=torch_dtype_to_ctype(dtype),
        )
        Q, Kt, V, bias = args[:4]
        Q = Q if Q.dtype == dtype else Q.to(dtype)
        Kt = Kt if Kt.dtype == dtype else Kt.to(dtype)
        V = V if V.dtype == dtype else V.to(dtype)
        bias = bias if bias.dtype == dtype else bias.to(dtype)
        out, lse = self._mod(module, fst_ctx, wrapper_ctx).run(Q, Kt, V, bias, scale)
        # aten returns (output, log_sumexp[f32], philox_seed, philox_offset).
        if out_dtype != dtype:
            out = out.to(out_dtype)
        empty = torch.empty([], dtype=torch.int64)
        return (out, lse, empty, empty)


# ---------------------------------------------------------------------------
# Indexed data-movement (gather / scatter / cat)
# ---------------------------------------------------------------------------
#
# These kernels only move element payloads (no arithmetic), so they impose no
# `scalar` typeclass requirement on the element type -- any dtype with a layout
# is fine. The index tensor is int64 and reinterpreted bit-for-bit as the
# kernel's machine-word offset type. Shapes flow at runtime, so an instantiation
# is keyed by (element type, rank, axis) only, and the output is allocated on the
# C++ side per the wrapper conventions.

def _shape_le(small, large):
    """Pointwise ``small[d] <= large[d]`` over equal ranks (Kuiper ``shape_le``)."""
    return len(small) == len(large) and all(s <= l for s, l in zip(small, large))


def _numel(dims):
    n = 1
    for d in dims:
        n *= int(d)
    return n


class GatherImpl(_Family):
    fst_template = "gather/Kuiops.Gather.Inst.fst.j2"
    wrapper_template = "gather/wrapper_gather.cu.j2"

    def supported(self, func, args, kwargs):
        # aten.gather.default(self, dim, index, *, sparse_grad=False)
        if len(args) < 3 or kwargs.get("sparse_grad", False):
            return None
        Inp, dim, Idx = args[0], args[1], args[2]
        if not (isinstance(Inp, torch.Tensor) and isinstance(Idx, torch.Tensor)
                and Inp.is_cuda and Idx.is_cuda):
            return None
        if Idx.dtype != torch.int64:
            return None
        rank = Inp.dim()
        if Idx.dim() != rank or rank < 1:
            return None
        dim = _norm_dim(int(dim), rank)
        if not (0 <= dim < rank):
            return None
        inp_shape = [int(x) for x in Inp.shape]
        idx_shape = [int(x) for x in Idx.shape]
        # The kernel requires `shape_le idx inp` (pointwise over every axis).
        if not _shape_le(idx_shape, inp_shape):
            return None
        if not (0 < _numel(idx_shape) <= _MAX_NUMEL):
            return None
        return dict(dtype=Inp.dtype, dim=dim, rank=rank)

    def run(self, spec, args, kwargs):
        dtype, dim, rank = spec["dtype"], spec["dim"], spec["rank"]
        et = torch_dtype_to_fstar(dtype)
        # Dims flow at runtime, so one module per (element type, rank, axis).
        module = f"Kuiops.Gather.{et.title()}.R{rank}.Dim{dim}"
        name = "gather_jit"
        fst_ctx = dict(module=module, name=name, r=rank, dim=dim, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"), name=name, dim=dim, r=rank,
            cpp_et=torch_dtype_to_ctype(dtype))
        # wrapper: op(Input, Index)
        return self._mod(module, fst_ctx, wrapper_ctx).run(args[0], args[2])


class ScatterImpl(_Family):
    fst_template = "scatter/Kuiops.Scatter.Inst.fst.j2"
    wrapper_template = "scatter/wrapper_scatter.cu.j2"

    def supported(self, func, args, kwargs):
        # aten.scatter.src(self, dim, index, src)
        if len(args) < 4:
            return None
        Self, dim, Idx, Src = args[0], args[1], args[2], args[3]
        if not all(isinstance(t, torch.Tensor) for t in (Self, Idx, Src)):
            return None
        if not (Self.is_cuda and Idx.is_cuda and Src.is_cuda):
            return None
        if Src.dtype != Self.dtype:
            return None
        if Idx.dtype != torch.int64:
            return None
        rank = Self.dim()
        if Idx.dim() != rank or Src.dim() != rank or rank < 1:
            return None
        dim = _norm_dim(int(dim), rank)
        if not (0 <= dim < rank):
            return None
        self_shape = [int(x) for x in Self.shape]
        idx_shape = [int(x) for x in Idx.shape]
        src_shape = [int(x) for x in Src.shape]
        # The kernel models src and index with one shape `di`, so they must match.
        if src_shape != idx_shape:
            return None
        # Requires `shape_le src self` (pointwise over every axis).
        if not _shape_le(src_shape, self_shape):
            return None
        if not (0 < _numel(src_shape) <= _MAX_NUMEL):
            return None
        return dict(dtype=Self.dtype, dim=dim, rank=rank)

    def run(self, spec, args, kwargs):
        dtype, dim, rank = spec["dtype"], spec["dim"], spec["rank"]
        Self, Idx, Src = args[0], args[2], args[3]
        et = torch_dtype_to_fstar(dtype)
        module = f"Kuiops.Scatter.{et.title()}.R{rank}.Dim{dim}"
        name = "scatter_jit"
        fst_ctx = dict(module=module, name=name, r=rank, dim=dim, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"), name=name, dim=dim, r=rank,
            cpp_et=torch_dtype_to_ctype(dtype))
        # wrapper: op(Self, Index, Src) -> clone(Self) updated in place
        return self._mod(module, fst_ctx, wrapper_ctx).run(Self, Idx, Src)


class CatImpl(_Family):
    fst_template = "cat/Kuiops.Cat.Inst.fst.j2"
    wrapper_template = "cat/wrapper_cat.cu.j2"

    def supported(self, func, args, kwargs):
        # aten.cat.default(tensors, dim=0) -- the kernel is binary only.
        if len(args) < 1:
            return None
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        if not (isinstance(tensors, (list, tuple)) and len(tensors) == 2):
            return None
        A, B = tensors
        if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in (A, B)):
            return None
        if A.dtype != B.dtype:
            return None
        rank = A.dim()
        if B.dim() != rank or rank < 1:
            return None
        dim = _norm_dim(int(dim), rank)
        if not (0 <= dim < rank):
            return None
        a_shape = [int(x) for x in A.shape]
        b_shape = [int(x) for x in B.shape]
        # Every axis except `dim` must agree.
        if any(a_shape[d] != b_shape[d] for d in range(rank) if d != dim):
            return None
        out_shape = list(a_shape)
        out_shape[dim] = a_shape[dim] + b_shape[dim]
        if not (0 < _numel(out_shape) <= _MAX_NUMEL):
            return None
        return dict(dtype=A.dtype, dim=dim, rank=rank)

    def run(self, spec, args, kwargs):
        dtype, dim, rank = spec["dtype"], spec["dim"], spec["rank"]
        et = torch_dtype_to_fstar(dtype)
        module = f"Kuiops.Cat.{et.title()}.R{rank}.Dim{dim}"
        name = "cat_jit"
        fst_ctx = dict(module=module, name=name, r=rank, dim=dim, et=et)
        wrapper_ctx = dict(
            module=module.replace(".", "_"), name=name, dim=dim, r=rank,
            cpp_et=torch_dtype_to_ctype(dtype))
        # wrapper: op(A, B)
        return self._mod(module, fst_ctx, wrapper_ctx).run(*args[0])


class HReducePolyImpl(_Family):
    fst_template = "reduce/Kuiops.HReducePoly.Inst.fst.j2"
    wrapper_template = "reduce/wrapper_hreduce_poly.cu.j2"

    _OPS = {
        aten.sum.dim_IntList: "add",
        aten.prod.dim_int: "mul",
        aten.all.dim: "and",
        aten.any.dim: "or",
    }

    def supported(self, func, args, kwargs):
        if func not in self._OPS or len(args) < 1:
            return None
        if func is not aten.sum.dim_IntList and len(args) < 2:
            return None
        Inp = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        if not (isinstance(Inp, torch.Tensor) and Inp.is_cuda
                and Inp.is_contiguous() and Inp.dtype in _INTEGER_DTYPES):
            return None

        rank = Inp.dim()
        if func is aten.sum.dim_IntList:
            if dim is None or (
                    isinstance(dim, (list, tuple)) and len(dim) == 0):
                dims = list(range(rank))
            elif isinstance(dim, (list, tuple)):
                dims = [_norm_dim(int(d), rank) for d in dim]
            else:
                dims = [_norm_dim(int(dim), rank)]
        else:
            dims = [_norm_dim(int(dim), rank)]
        if rank == 0:
            if dims not in ([], [0], [-1]):
                return None
            dims = []
        if (len(set(dims)) != len(dims)
                or any(d < 0 or d >= rank for d in dims)):
            return None
        dims = sorted(dims)
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if not (1 <= rank <= 4) or dims != [rank - 1] or keepdim:
            return None

        shape = [int(d) for d in Inp.shape]
        if any(d <= 0 for d in shape):
            return None
        rows = _numel(shape[:-1])
        cols = shape[-1]
        if rows > _MAX_BLOCKS:
            return None
        nth = min(cols, 1024)
        if rows * cols >= 2**32 or cols + nth >= 2**32:
            return None

        op = self._OPS[func]
        if op in ("add", "mul"):
            out_dtype = kwargs.get("dtype")
            if out_dtype is None:
                out_dtype = torch.int64
            if out_dtype not in _INTEGER_DTYPES:
                return None
            if out_dtype is torch.bool:
                op = "or" if op == "add" else "and"
        else:
            out_dtype = torch.uint8 if Inp.dtype is torch.uint8 else torch.bool

        return dict(op=op, in_dtype=Inp.dtype, out_dtype=out_dtype, rank=rank)

    def run(self, spec, args, kwargs):
        op = spec["op"]
        in_dtype = spec["in_dtype"]
        out_dtype = spec["out_dtype"]
        if op in ("and", "or"):
            out_bits = 8
            reduce_op = f"{op}_u8"
            zero = {
                torch.bool: "0uy",
                torch.int8: "0y", torch.int16: "0s",
                torch.int32: "0l", torch.int64: "0L",
                torch.uint8: "0uy", torch.uint16: "0us",
                torch.uint32: "0ul", torch.uint64: "0UL",
            }[in_dtype]
            pre_map = f"fun x -> if x = {zero} then 0uy else 1uy"
        else:
            out_bits = (_SIGNED_INTEGER_DTYPES | _UNSIGNED_INTEGER_DTYPES)[out_dtype]
            reduce_op = f"{op}_u{out_bits}"
            if in_dtype in _SIGNED_INTEGER_DTYPES:
                in_bits = _SIGNED_INTEGER_DTYPES[in_dtype]
                cast = f"int{in_bits}_to_uint{out_bits}"
            else:
                in_bits = _UNSIGNED_INTEGER_DTYPES[in_dtype]
                cast = f"uint{in_bits}_to_uint{out_bits}"
            pre_map = (
                "fun x -> x" if in_bits == out_bits
                and in_dtype not in _SIGNED_INTEGER_DTYPES
                else f"FStar.Int.Cast.{cast}"
            )

        in_et = torch_dtype_to_fstar(in_dtype)
        out_et = f"u{out_bits}"
        rank = spec["rank"]
        module = (
            f"Kuiops.HReducePoly.{op.title()}."
            f"R{rank}.{in_et.title()}To{out_et.title()}As"
            f"{str(out_dtype).rsplit('.', 1)[-1].title()}"
        )
        name = "hreduce_poly_jit"
        fst_ctx = dict(
            module=module, name=name, r=rank, in_et=in_et, out_et=out_et,
            reduce_op=reduce_op, pre_map=pre_map)
        wrapper_ctx = dict(
            module=module.replace(".", "_"), name=name, r=rank,
            cpp_in=torch_dtype_to_ctype(in_dtype),
            cpp_kernel_out=torch_dtype_to_ctype(
                getattr(torch, f"uint{out_bits}")),
            out_scalar=torch_dtype_to_aten_scalar(out_dtype))
        return self._mod(module, fst_ctx, wrapper_ctx).run(args[0])


class MeanImpl(_Family):
    fst_template = "mean/Kuiops.Mean.Inst.fst.j2"
    wrapper_template = "mean/wrapper_mean.cu.j2"

    def supported(self, func, args, kwargs):
        # aten.mean.dim(self, dim, keepdim=False, *, dtype=None). The parallel
        # tree reduction (Kuiper.Kernel.HReduce.Block, one block per output row)
        # works on an [m, n] row-major matrix, so we only reduce the *last* axis
        # of a contiguous tensor (rank-N -> [prod(leading), last] reshape) and
        # require keepdim=True. Only the 1-dim case is supported, so an int[1]
        # tuple is unpacked as a singleton here.
        if len(args) < 2:
            return None
        Inp = args[0]
        dim = args[1]
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if kwargs.get("dtype", None) is not None:
            return None
        if not (isinstance(Inp, torch.Tensor) and Inp.is_cuda):
            return None
        if Inp.dtype not in _FLOAT_DTYPES or not keepdim:
            return None
        if isinstance(dim, (list, tuple)):
            if len(dim) != 1:
                return None
            dim = dim[0]
        if dim is None:
            return None
        rank = Inp.dim()
        if rank < 1:
            return None
        dim = _norm_dim(int(dim), rank)
        # Only the last axis: reducing a middle axis would need a strided view.
        if dim != rank - 1:
            return None
        length = int(Inp.shape[dim])
        if length < 1:
            return None
        # m = number of output rows (one GPU block each). Gate the kernel's
        # refinements: m <= max_blocks and m * n <= max_blocks * max_threads.
        m = Inp.numel() // length
        if not (0 < m <= _MAX_BLOCKS):
            return None
        if m * length > _MAX_NUMEL:
            return None
        return dict(dtype=Inp.dtype, length=length)

    def run(self, spec, args, kwargs):
        dtype, length = spec["dtype"], spec["length"]
        et = torch_dtype_to_fstar(dtype)
        # Keyed only by (element type, reduced length): m is a runtime arg, so
        # one module serves every rank/batch size with this last-dim length.
        module = f"Kuiops.Mean.{et.title()}.Len{length}"
        name = "mean_jit"
        fst_ctx = dict(module=module, name=name, et=et, length=length)
        wrapper_ctx = dict(
            module=module.replace(".", "_"), name=name,
            cpp_et=torch_dtype_to_ctype(dtype))
        # wrapper: op(Input)
        return self._mod(module, fst_ctx, wrapper_ctx).run(args[0])
