"""S1: pickle analysis without executing anything.

Two independent passes over each pickle stream:

1. A restricted unpickler whose find_class never returns a real object, only
   stubs defined here. It records every import the pickle asks for and rebuilds
   the structure (state dict names, tensor views) so the PyTorch parser can find
   the tensors. Nothing outside this module is ever called.
2. An opcode walk (pickletools.genops) that collects imports without building
   anything, so a stream that breaks the unpickler half way is still covered.

Imports are classified against an allowlist of what PyTorch checkpoints need
and a denylist of known code-execution gadgets.
"""
from __future__ import annotations

import _compat_pickle
import io
import pickle
import pickletools
from dataclasses import dataclass, field

from .model import Finding

# --- classification ------------------------------------------------------

SAFE = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor"), ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_parameter_with_state"),
    ("torch._tensor", "_rebuild_from_type_v2"), ("torch.nn.parameter", "Parameter"),
    ("torch", "Size"), ("torch", "device"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice"), ("builtins", "complex"),
    ("builtins", "range"), ("builtins", "bytearray"),
}
TORCH_DTYPES = {
    "float32", "float", "float64", "double", "float16", "half", "bfloat16", "int64", "long",
    "int32", "int", "int16", "short", "int8", "uint8", "bool", "complex64", "complex128",
    "float8_e4m3fn", "float8_e5m2",
}

# (module, name); name "*" means the whole module.
DANGEROUS = {
    ("os", "*"), ("posix", "*"), ("nt", "*"), ("subprocess", "*"), ("sys", "*"),
    ("builtins", "eval"), ("builtins", "exec"), ("builtins", "compile"), ("builtins", "__import__"),
    ("builtins", "open"), ("builtins", "getattr"), ("builtins", "setattr"), ("builtins", "delattr"),
    ("builtins", "globals"), ("builtins", "locals"), ("builtins", "breakpoint"), ("builtins", "input"),
    ("builtins", "vars"), ("builtins", "apply"),
    ("builtins", "execfile"),
    ("runpy", "*"), ("socket", "*"), ("shutil", "*"), ("importlib", "*"), ("pickle", "*"),
    ("_pickle", "*"), ("marshal", "*"), ("types", "*"), ("ctypes", "*"), ("code", "*"),
    ("pty", "*"), ("webbrowser", "*"), ("asyncio", "*"), ("multiprocessing", "*"),
    ("requests", "*"), ("urllib", "*"), ("http", "*"), ("ftplib", "*"), ("smtplib", "*"),
    ("telnetlib", "*"), ("pdb", "*"), ("bdb", "*"), ("timeit", "*"), ("cProfile", "*"),
    ("profile", "*"), ("platform", "*"), ("tempfile", "*"), ("pydoc", "*"),
    ("operator", "attrgetter"), ("operator", "methodcaller"), ("functools", "partial"),
    ("functools", "reduce"), ("torch", "load"), ("torch.serialization", "load"),
    ("torch.storage", "_load_from_bytes"), ("torch.hub", "*"), ("torch.jit", "*"),
    ("numpy", "load"), ("numpy.testing._private.utils", "runstring"), ("dill", "*"),
}


def normalize(module: str, name: str) -> tuple[str, str]:
    """Python 2 names (protocols 0-2), mapped the way the real unpickler maps them."""
    if (module, name) in _compat_pickle.NAME_MAPPING:
        return _compat_pickle.NAME_MAPPING[(module, name)]
    return _compat_pickle.IMPORT_MAPPING.get(module, module), name


def classify(module: str, name: str) -> str:
    """Return 'safe', 'dangerous', or 'unknown'."""
    if (module, name) in SAFE:
        return "safe"
    if module == "torch" and (name.endswith("Storage") or name in TORCH_DTYPES):
        return "safe"
    top = module.split(".")
    for i in range(len(top), 0, -1):
        prefix = ".".join(top[:i])
        if (prefix, "*") in DANGEROUS:
            return "dangerous"
    if (module, name) in DANGEROUS:
        return "dangerous"
    return "unknown"


# --- stubs ---------------------------------------------------------------

STORAGE_DTYPES = {
    "FloatStorage": "F32", "DoubleStorage": "F64", "HalfStorage": "F16", "BFloat16Storage": "BF16",
    "LongStorage": "I64", "IntStorage": "I32", "ShortStorage": "I16", "CharStorage": "I8",
    "ByteStorage": "U8", "BoolStorage": "BOOL", "ComplexFloatStorage": "C64",
    "ComplexDoubleStorage": "C128", "Float8_e4m3fnStorage": "F8_E4M3", "Float8_e5m2Storage": "F8_E5M2",
}


class Stub:
    """Stands in for any object the pickle asks to build. Records, never runs."""
    qualname = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args, obj.kwargs, obj.state, obj.items = args, kwargs, None, {}
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, k, v):
        self.items[k] = v

    def append(self, v):
        self.items[len(self.items)] = v

    def extend(self, vs):
        for v in vs:
            self.append(v)

    def __repr__(self):
        return f"<stub {self.qualname}>"


@dataclass(frozen=True)
class StorageType:
    name: str

    @property
    def dtype(self) -> str:
        return STORAGE_DTYPES.get(self.name, f"torch.{self.name}")


@dataclass(frozen=True)
class StorageRef:
    key: str
    dtype: str
    numel: int
    location: str


@dataclass
class TensorRef:
    storage: StorageRef | None
    offset: int          # in elements
    shape: tuple
    stride: tuple


class _ODict(dict):
    """OrderedDict stand-in; torch state dicts carry a `_metadata` attribute via BUILD."""

    def __setstate__(self, state):
        self.state = state


def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    st = storage if isinstance(storage, StorageRef) else None
    return TensorRef(st, int(storage_offset), tuple(size), tuple(stride))


def _rebuild_parameter(data, *rest):
    return data


def _rebuild_from_type_v2(func, new_type, args, state):
    return func(*args) if callable(func) else Stub()


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, f, imports: list):
        super().__init__(f)
        self.imports = imports

    def find_class(self, module, name):
        module, name = normalize(module, name)
        self.imports.append((module, name))
        if (module, name) in {("torch._utils", "_rebuild_tensor"), ("torch._utils", "_rebuild_tensor_v2")}:
            return _rebuild_tensor
        if (module, name) in {("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_parameter_with_state")}:
            return _rebuild_parameter
        if (module, name) == ("torch._tensor", "_rebuild_from_type_v2"):
            return _rebuild_from_type_v2
        if (module, name) == ("collections", "OrderedDict"):
            return _ODict
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        return type(name, (Stub,), {"qualname": f"{module}.{name}"})

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            st = pid[1]
            dtype = st.dtype if isinstance(st, StorageType) else "?"
            return StorageRef(str(pid[2]), dtype, int(pid[4]), str(pid[3]))
        return Stub()


# --- opcode walk ---------------------------------------------------------

_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
               "SHORT_BINSTRING", "BINSTRING", "STRING"}


def opcode_imports(data: bytes) -> tuple[list[tuple[str, str]], str | None]:
    """Imports named by GLOBAL/INST/STACK_GLOBAL, without building anything.

    STACK_GLOBAL takes its module and name from the stack; they are tracked as the
    last two strings pushed, including through the memo.
    """
    found: list[tuple[str, str]] = []
    recent: list = []
    memo: dict = {}
    err = None
    try:
        for op, arg, _ in pickletools.genops(io.BytesIO(data)):
            name = op.name
            if name in ("GLOBAL", "INST"):
                mod, _, nm = arg.partition(" ")
                found.append(normalize(mod, nm))
                recent.append(None)
            elif name in _STRING_OPS:
                recent.append(arg if isinstance(arg, str) else arg.decode("latin-1") if isinstance(arg, bytes) else arg)
            elif name in ("BINGET", "LONG_BINGET", "GET"):
                recent.append(memo.get(arg))
            elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
                memo[arg] = recent[-1] if recent else None
            elif name == "MEMOIZE":
                memo[len(memo)] = recent[-1] if recent else None
            elif name == "STACK_GLOBAL":
                mod, nm = (recent[-2], recent[-1]) if len(recent) >= 2 else (None, None)
                found.append(normalize(str(mod), str(nm)))
                recent.append(None)
            elif name == "STOP":
                break
            else:
                recent.append(None)
    except Exception as e:  # malformed stream; keep what was found
        err = f"{type(e).__name__}: {e}"
    return found, err


# --- entry point ---------------------------------------------------------

@dataclass
class PickleResult:
    obj: object = None
    end: int = 0                      # bytes consumed by the unpickler
    imports: list = field(default_factory=list)
    error: str | None = None


def safe_load(f) -> PickleResult:
    """Run the restricted unpickler on a file object positioned at a pickle."""
    start = f.tell()
    res = PickleResult()
    try:
        res.obj = _SafeUnpickler(f, res.imports).load()
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    res.end = f.tell() - start
    return res


def analyse(label: str, data: bytes) -> list[Finding]:
    """S1 findings for one pickle stream."""
    res = safe_load(io.BytesIO(data))
    ops, op_err = opcode_imports(data)
    seen = dict.fromkeys(res.imports + ops)  # ordered, deduplicated
    findings = []
    for mod, nm in seen:
        kind = classify(mod, nm)
        if kind == "safe":
            continue
        sev = "critical" if kind == "dangerous" else "medium"
        findings.append(Finding("S1", sev, f"{kind}-import",
                                f"{label}: imports {mod}.{nm}" + (" (code execution gadget)" if kind == "dangerous" else ""),
                                {"stream": label, "module": mod, "name": nm}))
    if res.error and op_err:
        # The specific exception text differs between interpreters, so the message
        # is kept generic to stay identical across the Python and TypeScript cores.
        findings.append(Finding("S1", "medium", "malformed-pickle", f"{label}: pickle data does not parse cleanly",
                                {"stream": label}))
    return findings
