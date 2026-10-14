"""Format detection: by content, never by file extension."""
from __future__ import annotations

import pickletools
import struct
import zipfile
from pathlib import Path

from ..model import Finding, ModelFile
from . import gguf, pytorch, safetensors


def detect(f, size: int) -> str:
    f.seek(0)
    head = f.read(9)
    f.seek(0)
    if head[:4] == b"GGUF":
        return "gguf"
    if len(head) == 9 and head[8:9] == b"{":
        (n,) = struct.unpack("<Q", head[:8])
        if 8 + n <= size:
            return "safetensors"
    if head[:1] == b"\x80":
        return "pickle"
    if head[:4] == b"PK\x03\x04" or zipfile.is_zipfile(f):
        f.seek(0)
        return "zip"
    f.seek(0)
    if _is_old_pickle(f):  # protocols 0 and 1 have no \x80 marker
        return "pickle"
    f.seek(0)
    return "unknown"


def _is_old_pickle(f) -> bool:
    try:
        for i, (op, _, _) in enumerate(pickletools.genops(f)):
            if op.name == "STOP":
                return i > 0
            if i > 1_000_000:
                return False
    except Exception:
        return False
    return False


def open_model(path: Path, f) -> ModelFile:
    size = path.stat().st_size
    kind = detect(f, size)
    f.seek(0)
    if kind == "gguf":
        return gguf.parse(path, f)
    if kind == "safetensors":
        return safetensors.parse(path, f)
    if kind == "pickle":
        return pytorch.parse_pickles(path, f)
    if kind == "zip":
        return pytorch.parse_zip(path, f)
    mf = ModelFile(path=path, format="unknown", size=size)
    mf.findings.append(Finding("parse", "medium", "unknown-format", "not a recognised model format"))
    return mf
