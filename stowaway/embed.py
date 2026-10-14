"""Test-fixture generator: write bytes into the low bits of weights, and read them back.

This exists so the detectors (S3/S4/S5) can be measured. A detector is only
meaningful if we know its true-positive rate against known positive samples and
its false-positive rate against clean models, so the experiments need a way to
produce positive samples on demand. The payloads used are random bytes or benign
test files; nothing here executes anything, and the reader exists only to confirm
in tests that the written bytes are recoverable.

A payload is framed with its 4-byte little-endian length, split into k-bit
chunks, and written into bits [0, k) of every `spread`-th float element of the
chosen tensors, in order. k = 24 on float32 mirrors the "replace the low three
bytes of each weight" construction from the EvilModel paper.
"""
from __future__ import annotations

import struct

import numpy as np

from .bits import FLOAT_LAYOUT, RawTensor

FRAME = 4  # bytes of length prefix


def _float_targets(tensors: dict[str, RawTensor], names) -> list[str]:
    chosen = list(tensors) if names is None else list(names)
    return [n for n in chosen if tensors[n].dtype in FLOAT_LAYOUT]


def _slots(tensors: dict[str, RawTensor], spread: int, names) -> int:
    return sum(len(range(0, tensors[n].data.size, spread)) for n in _float_targets(tensors, names))


def capacity(tensors: dict[str, RawTensor], k: int, spread: int = 1, names=None) -> int:
    """Payload bytes that fit after the length frame, for parameters (k, spread)."""
    total_bits = _slots(tensors, spread, names) * k
    return max(0, total_bits // 8 - FRAME)


def _bit_chunks(data: bytes, k: int) -> np.ndarray:
    """Little-endian bit stream of `data`, packed into k-bit integers."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    pad = (-len(bits)) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bits = bits.reshape(-1, k)
    weights = (1 << np.arange(k, dtype=np.uint64))
    return (bits.astype(np.uint64) * weights).sum(axis=1)


def embed(tensors: dict[str, RawTensor], payload: bytes, k: int, spread: int = 1,
          names=None) -> dict[str, RawTensor]:
    """Return copies of `tensors` with `payload` written into the low k bits.

    Raises ValueError if the payload does not fit.
    """
    if not 1 <= k <= 24:
        raise ValueError("k must be in 1..24")
    framed = struct.pack("<I", len(payload)) + payload
    chunks = _bit_chunks(framed, k)
    if chunks.size > _slots(tensors, spread, names):
        raise ValueError("payload does not fit in the chosen tensors")

    mask = np.uint64((1 << k) - 1)
    out = {n: RawTensor(t.dtype, t.shape, t.data.copy()) for n, t in tensors.items()}
    pos = 0
    for n in _float_targets(tensors, names):
        d = out[n].data
        idx = np.arange(0, d.size, spread)
        take = min(len(idx), chunks.size - pos)
        if take <= 0:
            break
        sel = idx[:take]
        d64 = d[sel].astype(np.uint64)
        d[sel] = ((d64 & ~mask) | chunks[pos:pos + take]).astype(d.dtype)
        pos += take
    return out


def extract(tensors: dict[str, RawTensor], k: int, spread: int = 1, names=None) -> bytes:
    """Recover a payload written by `embed` with the same parameters."""
    mask = (1 << k) - 1
    vals = []
    for n in _float_targets(tensors, names):
        d = tensors[n].data
        idx = np.arange(0, d.size, spread)
        vals.append(d[idx].astype(np.uint64) & np.uint64(mask))
    stream = np.concatenate(vals) if vals else np.zeros(0, dtype=np.uint64)
    bits = ((stream[:, None] >> np.arange(k, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8).reshape(-1)
    nbytes = (bits.size // 8)
    packed = np.packbits(bits[:nbytes * 8], bitorder="little")
    if packed.size < FRAME:
        return b""
    (length,) = struct.unpack("<I", packed[:FRAME].tobytes())
    return packed[FRAME:FRAME + length].tobytes()
