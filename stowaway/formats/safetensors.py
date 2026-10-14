"""safetensors: 8-byte little-endian header length, a JSON header, then raw tensor bytes.

Spec: https://github.com/huggingface/safetensors. Tensor offsets are relative to the
end of the header and must tile the data section exactly, so anything else in the
file is unaccounted for.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from ..model import ITEMSIZE, Finding, ModelFile, Region, TensorInfo, prod

MAX_HEADER = 100 * 1024 * 1024  # the reference implementation's limit


def _pairs_hook(pairs):
    seen: dict = {}
    dups = []
    for k, v in pairs:
        if k in seen:
            dups.append(k)
        seen[k] = v
    if dups:
        seen["\0duplicates"] = dups
    return seen


def parse(path: Path, f) -> ModelFile:
    size = path.stat().st_size
    mf = ModelFile(path=path, format="safetensors", size=size)
    head = f.read(8)
    if len(head) < 8:
        mf.findings.append(Finding("parse", "high", "truncated", "file is shorter than the 8-byte header length"))
        return mf
    (n,) = struct.unpack("<Q", head)
    if n > MAX_HEADER or 8 + n > size:
        mf.findings.append(Finding("parse", "high", "bad-header-length",
                                   f"header length {n} does not fit the file", {"header_length": n}))
        return mf
    raw = f.read(n)
    mf.regions.append(Region(0, 8 + n, "header", "safetensors header"))
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        mf.findings.append(Finding("parse", "high", "bad-header-json", f"header is not valid JSON: {e}"))
        return mf

    dups = header.pop("\0duplicates", None)
    if dups:
        # Different loaders keep the first or the last value: two readers can see two models.
        mf.findings.append(Finding("parse", "high", "duplicate-keys",
                                   f"header repeats {len(dups)} key(s); loaders may disagree on which tensor is meant",
                                   {"keys": sorted(set(dups))}))

    meta = header.pop("__metadata__", None)
    if meta is not None:
        mf.metadata = meta if isinstance(meta, dict) else {"__invalid__": repr(meta)[:200]}

    base = 8 + n
    spans = []
    for name, spec in header.items():
        try:
            dtype = spec["dtype"]
            shape = tuple(int(d) for d in spec["shape"])
            begin, end = (int(x) for x in spec["data_offsets"])
        except (KeyError, TypeError, ValueError):
            mf.findings.append(Finding("parse", "high", "bad-tensor-entry", f"malformed entry for tensor {name!r}"))
            continue
        nbytes = end - begin
        if begin < 0 or end < begin or base + end > size:
            mf.findings.append(Finding("parse", "high", "tensor-out-of-bounds",
                                       f"tensor {name!r} points outside the file", {"tensor": name}))
            continue
        itemsize = ITEMSIZE.get(dtype)
        if itemsize is None:
            mf.findings.append(Finding("parse", "info", "unknown-dtype", f"tensor {name!r} has dtype {dtype}"))
        elif prod(shape) * itemsize != nbytes:
            mf.findings.append(Finding("parse", "high", "size-mismatch",
                                       f"tensor {name!r}: shape says {prod(shape) * itemsize} bytes, offsets say {nbytes}",
                                       {"tensor": name}))
        mf.tensors.append(TensorInfo(name, dtype, shape, base + begin, nbytes))
        mf.regions.append(Region(base + begin, base + end, "tensor", name))
        spans.append((begin, end, name))

    spans.sort()
    for (b1, e1, n1), (b2, e2, n2) in zip(spans, spans[1:]):
        if b2 < e1:
            mf.findings.append(Finding("parse", "high", "overlapping-tensors",
                                       f"tensors {n1!r} and {n2!r} share bytes", {"tensors": [n1, n2]}))
    return mf
