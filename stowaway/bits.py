"""Float tensors as raw bit patterns.

Weights are analysed as unsigned integers with the same bits as the floats, so
bf16 (which numpy cannot represent) needs no special casing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# dtype -> (unsigned view, mantissa bits, exponent bits)
FLOAT_LAYOUT = {
    "F32": (np.dtype("<u4"), 23, 8),
    "F16": (np.dtype("<u2"), 10, 5),
    "BF16": (np.dtype("<u2"), 7, 8),
}


@dataclass
class RawTensor:
    """A tensor's raw bits: floats as their unsigned view, anything else as bytes."""
    dtype: str
    shape: tuple[int, ...]
    data: np.ndarray  # 1-D

    @classmethod
    def from_bytes(cls, dtype: str, shape, raw: bytes) -> "RawTensor":
        view = FLOAT_LAYOUT[dtype][0] if dtype in FLOAT_LAYOUT else np.dtype("u1")
        return cls(dtype, tuple(shape), np.frombuffer(raw, dtype=view).copy())

    @classmethod
    def from_float(cls, arr: np.ndarray, dtype: str | None = None) -> "RawTensor":
        """From a float32/float16 numpy array; dtype="BF16" truncates float32 to bfloat16."""
        arr = np.ascontiguousarray(arr)
        if dtype == "BF16":
            u = arr.astype(np.float32).view(np.uint32)
            # round to nearest even, as torch does
            u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
            return cls("BF16", arr.shape, u.astype(np.uint16).ravel())
        dt = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}[arr.dtype]
        return cls(dt, arr.shape, arr.view(FLOAT_LAYOUT[dt][0]).ravel().copy())

    def to_bytes(self) -> bytes:
        return self.data.tobytes()

    def to_float32(self) -> np.ndarray:
        if self.dtype == "F32":
            return self.data.view(np.float32).reshape(self.shape)
        if self.dtype == "F16":
            return self.data.view(np.float16).astype(np.float32).reshape(self.shape)
        if self.dtype == "BF16":
            return (self.data.astype(np.uint32) << 16).view(np.float32).reshape(self.shape)
        raise TypeError(f"{self.dtype} is not a float type")


def fields(u: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """(exponent field, mantissa field) of a float tensor's unsigned view."""
    _, m, e = FLOAT_LAYOUT[dtype]
    return (u >> m) & ((1 << e) - 1), u & ((1 << m) - 1)


def informative(u: np.ndarray, dtype: str) -> np.ndarray:
    """Mask of elements whose mantissa says something: not zero, subnormal, inf/nan,
    or an exact power of two (pruned weights and constants like 1.0 would otherwise
    look like structure in every bit)."""
    _, m, e = FLOAT_LAYOUT[dtype]
    exp, mant = fields(u, dtype)
    return (exp != 0) & (exp != (1 << e) - 1) & (mant != 0)
