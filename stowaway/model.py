"""Format-independent description of a model file.

Every parser produces a ModelFile: the tensors it found, and a list of Regions
that says which byte ranges the format accounts for. The coverage check (S2)
then looks at what is left over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

# Region kinds that count as "accounted for". A "container" (a zip entry, a torch
# storage) does not: only the tensor views inside it do, so bytes of a storage that
# no tensor references show up as a gap.
COVERING_KINDS = frozenset({"header", "tensor", "padding", "structure"})
REGION_KINDS = COVERING_KINDS | {"container"}

SEVERITIES = ("info", "low", "medium", "high", "critical")

# Bytes per element. Quantized GGUF types are handled in formats/gguf.py.
ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "C64": 8, "C128": 16,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]  # numpy order (outermost first), for every format
    offset: int | None      # absolute offset of the first byte in the file; None if not addressable
    nbytes: int


@dataclass(frozen=True)
class Region:
    start: int
    end: int   # exclusive
    kind: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in REGION_KINDS:
            raise ValueError(f"unknown region kind {self.kind!r}")
        if not 0 <= self.start <= self.end:
            raise ValueError(f"bad region [{self.start}, {self.end})")


@dataclass
class Finding:
    check: str      # "S1", "S2", or "parse"
    severity: str
    code: str       # stable machine-readable id, e.g. "trailing-data"
    message: str
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")


@dataclass
class ModelFile:
    path: Path
    format: str
    size: int
    tensors: list[TensorInfo] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Pickle streams to analyse for S1: (label, absolute start, end).
    pickles: list[tuple[str, int, int]] = field(default_factory=list)
    # Pickles that are not stored verbatim in the file (a compressed zip entry).
    inline_pickles: dict[str, bytes] = field(default_factory=dict)

    def read(self, f: BinaryIO, start: int, length: int) -> bytes:
        f.seek(start)
        return f.read(length)

    def tensor_bytes(self, f: BinaryIO, t: TensorInfo) -> bytes:
        if t.offset is None:
            raise ValueError(f"tensor {t.name!r} is not addressable in the file")
        return self.read(f, t.offset, t.nbytes)


def prod(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n
