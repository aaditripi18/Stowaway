"""Neutralise low-bit payloads instead of detecting them (PLAN.md, E9).

Where a payload cannot be seen (fair-coin float32 models, tensors a fine-tune
trained), it can still be removed: rounding every float32 weight to nearest at
mantissa bit `bits` clears bits 0..bits-1, overwriting whatever they held. Only
float32 tensors change; every other tensor, the header and the metadata are
copied byte for byte, so the output is a valid safetensors file with the same
tensor names, dtypes and shapes.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .bits import FLOAT_LAYOUT

F32_MANT = FLOAT_LAYOUT["F32"][1]  # 23
_EXP_MASK = np.uint32(0x7F800000)


def round_f32_bits(u: np.ndarray, bits: int) -> np.ndarray:
    """float32 values (as uint32 bit patterns) rounded to nearest, ties to even, at
    mantissa bit `bits`, so bits 0..bits-1 become zero. NaN and infinity are kept;
    a finite value that would round up to infinity is truncated instead."""
    if not 0 < bits < F32_MANT:
        raise ValueError(f"bits must be 1..{F32_MANT - 1}")
    u = u.astype(np.uint32, copy=False)
    w = u.astype(np.uint64)
    half_minus_one = np.uint64((1 << (bits - 1)) - 1)
    odd = (w >> np.uint64(bits)) & np.uint64(1)
    mask = np.uint64(~((1 << bits) - 1) & 0xFFFFFFFF)
    out = ((w + half_minus_one + odd) & mask).astype(np.uint32)
    special = (u & _EXP_MASK) == _EXP_MASK            # NaN / infinity: leave as is
    overflow = ((out & _EXP_MASK) == _EXP_MASK) & ~special
    out[overflow] = u[overflow] & np.uint32(int(mask))  # round toward zero instead
    out[special] = u[special]
    return out


def sanitize_safetensors(src: str | Path, dst: str | Path, bits: int) -> dict:
    """Write `dst`: `src` with every F32 tensor rounded at mantissa bit `bits`."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("write to a new file, not over the input")
    with open(src, "rb") as f:
        head = f.read(8)
        (n,) = struct.unpack("<Q", head)
        header_bytes = f.read(n)
        header = json.loads(header_bytes)
        start = 8 + n
        tensors = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                         key=lambda kv: kv[1]["data_offsets"][0])
        changed = rounded = 0
        with open(dst, "wb") as out:
            out.write(head + header_bytes)
            pos = 0
            for name, spec in tensors:
                b, e = spec["data_offsets"]
                if b > pos:  # bytes between tensors are copied unchanged
                    f.seek(start + pos)
                    out.write(f.read(b - pos))
                f.seek(start + b)
                raw = f.read(e - b)
                if spec["dtype"] == "F32" and raw:
                    u = np.frombuffer(raw, dtype="<u4")
                    v = round_f32_bits(u, bits)
                    changed += int((v != u).sum())
                    rounded += 1
                    raw = v.astype("<u4").tobytes()
                out.write(raw)
                pos = e
            f.seek(start + pos)
            out.write(f.read())  # anything after the last tensor, unchanged
    return {"f32_tensors": rounded, "elements_changed": changed, "bits": bits, "output": str(dst)}
