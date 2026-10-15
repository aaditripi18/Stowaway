"""GGUF (llama.cpp): header, key/value metadata, tensor infos, aligned tensor data.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md. Tensor data is
aligned (default 32 bytes); the alignment padding is expected to be zero.
"""
from __future__ import annotations

import struct
from pathlib import Path

from ..model import Finding, ModelFile, Region, TensorInfo, prod

# ggml_type -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
}

# GGUF metadata value types
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
MAX_STRING = 64 * 1024 * 1024
MAX_ARRAY = 1 << 28
KEEP_ARRAY = 16  # arrays longer than this are summarized in the report


class _Reader:
    def __init__(self, f, size: int):
        self.f, self.size = f, size

    def take(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("unexpected end of file")
        return b

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > MAX_STRING:
            raise ValueError(f"string of {n} bytes")
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            itype, count = self.unpack("<I"), self.unpack("<Q")
            if count > MAX_ARRAY:
                raise ValueError(f"array of {count} items")
            items = [self.value(itype) for _ in range(count)]
            return items if count <= KEEP_ARRAY else {"array_of": count, "head": items[:KEEP_ARRAY]}
        raise ValueError(f"unknown metadata value type {vtype}")


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def parse(path: Path, f) -> ModelFile:
    size = path.stat().st_size
    mf = ModelFile(path=path, format="gguf", size=size)
    r = _Reader(f, size)
    try:
        if r.take(4) != b"GGUF":
            raise ValueError("bad magic")
        version = r.unpack("<I")
        if version not in (2, 3):
            mf.findings.append(Finding("parse", "low", "gguf-version", f"GGUF version {version} is not 2 or 3"))
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        meta = {"gguf.version": version}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("<I")
            if n_dims > 8:
                raise ValueError(f"tensor {name!r} has {n_dims} dims")
            ne = tuple(r.unpack("<Q") for _ in range(n_dims))
            infos.append((name, ne, r.unpack("<I"), r.unpack("<Q")))
    except (ValueError, struct.error):
        # The specific exception text differs between interpreters; keep it generic
        # so the Python and TypeScript cores report this identically.
        mf.findings.append(Finding("parse", "high", "bad-gguf", "could not parse the GGUF header"))
        return mf

    mf.metadata = meta
    header_end = f.tell()
    alignment = meta.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        mf.findings.append(Finding("parse", "medium", "bad-alignment", "general.alignment is not a positive power of two"))
        alignment = 32
    data_start = _align(header_end, alignment)
    mf.regions.append(Region(0, header_end, "header", "gguf header"))
    mf.regions.append(Region(header_end, min(data_start, size), "padding", "header alignment"))

    spans = []
    for name, ne, tcode, rel in infos:
        shape = tuple(reversed(ne))  # GGUF lists the innermost dimension first
        if tcode not in GGML_TYPES:
            mf.findings.append(Finding("parse", "low", "unknown-dtype", f"tensor {name!r} has ggml type {tcode}"))
            mf.tensors.append(TensorInfo(name, f"ggml_{tcode}", shape, data_start + rel, 0))
            continue
        tname, block, bsize = GGML_TYPES[tcode]
        n = prod(shape)
        if n % block:
            mf.findings.append(Finding("parse", "high", "bad-block", f"tensor {name!r}: {n} elements is not a multiple of {block}"))
        nbytes = n // block * bsize
        start = data_start + rel
        if rel % alignment:
            mf.findings.append(Finding("parse", "medium", "unaligned-tensor", f"tensor {name!r} is not aligned"))
        if start + nbytes > size:
            mf.findings.append(Finding("parse", "high", "tensor-out-of-bounds", f"tensor {name!r} points outside the file",
                                       {"tensor": name}))
            continue
        mf.tensors.append(TensorInfo(name, tname, shape, start, nbytes))
        mf.regions.append(Region(start, start + nbytes, "tensor", name))
        spans.append((start, start + nbytes, name))

    spans.sort()
    for i, (s, e, nm) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else size
        if i + 1 < len(spans) and nxt < e:
            mf.findings.append(Finding("parse", "high", "overlapping-tensors",
                                       f"tensors {nm!r} and {spans[i + 1][2]!r} share bytes",
                                       {"tensors": [nm, spans[i + 1][2]]}))
        pad_end = min(_align(e, alignment), nxt)
        if pad_end > e:
            mf.regions.append(Region(e, pad_end, "padding", f"alignment after {nm}"))
    return mf
