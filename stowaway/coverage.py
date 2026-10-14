"""S2: every byte of the file must be accounted for.

A gap is a byte range that no header, tensor, padding or structure region
covers: data appended after the last tensor, junk before a zip archive, storage
bytes that no tensor references. Padding must be zero.
"""
from __future__ import annotations

import math
from collections import Counter

from .model import COVERING_KINDS, Finding, ModelFile

SAMPLE = 64 * 1024

MAGIC = [
    (b"MZ", "Windows executable (MZ)"), (b"\x7fELF", "ELF executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable"), (b"\xca\xfe\xba\xbe", "Mach-O universal / Java class"),
    (b"#!", "script with shebang"), (b"PK\x03\x04", "zip archive"), (b"\x1f\x8b", "gzip"),
    (b"7z\xbc\xaf\x27\x1c", "7-zip archive"), (b"Rar!", "rar archive"), (b"%PDF", "PDF"),
    (b"\x89PNG", "PNG image"), (b"\xff\xd8\xff", "JPEG image"), (b"GGUF", "GGUF model"),
    (b"\x80\x02", "pickle (protocol 2)"), (b"\x80\x03", "pickle (protocol 3)"),
    (b"\x80\x04", "pickle (protocol 4)"), (b"\x80\x05", "pickle (protocol 5)"),
]


def sniff(head: bytes) -> str | None:
    for magic, what in MAGIC:
        if head.startswith(magic):
            return what
    if head[4:8] == b"ftyp":
        return "MP4/QuickTime media"
    return None


def entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte (8.0 = indistinguishable from random)."""
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


def gaps(mf: ModelFile) -> list[tuple[int, int]]:
    spans = sorted((r.start, r.end) for r in mf.regions if r.kind in COVERING_KINDS and r.end > r.start)
    out, pos = [], 0
    for s, e in spans:
        if s > pos:
            out.append((pos, min(s, mf.size)))
        pos = max(pos, e)
    if pos < mf.size:
        out.append((pos, mf.size))
    return [(s, e) for s, e in out if e > s]


def _container(mf: ModelFile, start: int, end: int) -> str | None:
    for r in mf.regions:
        if r.kind == "container" and r.start <= start and end <= r.end:
            return r.label
    return None


def check(mf: ModelFile, f) -> list[Finding]:
    findings = []
    for start, end in gaps(mf):
        n = end - start
        f.seek(start)
        sample = f.read(min(n, SAMPLE))
        nonzero = any(sample)
        inside = _container(mf, start, end)
        what = sniff(sample)
        if inside and inside.startswith("storage"):
            code, where = "unreferenced-storage", f"{inside}: {n} bytes no tensor uses"
        elif end == mf.size:
            code, where = "trailing-data", f"{n} bytes after the end of the model"
        elif start == 0:
            code, where = "leading-data", f"{n} bytes before the model starts"
        else:
            code, where = "unaccounted-bytes", f"{n} unaccounted bytes at offset {start}"
        if not nonzero and n <= SAMPLE:
            sev = "low"
        elif what or n >= 1024:
            sev = "high"
        else:
            sev = "medium"
        msg = where + (f", looks like {what}" if what else "")
        findings.append(Finding("S2", sev, code, msg, {
            "start": start, "end": end, "bytes": n, "magic": what,
            "entropy": round(entropy(sample), 3), "all_zero": not nonzero and n <= SAMPLE,
            "container": inside,
        }))

    for r in mf.regions:
        if r.kind != "padding" or r.end <= r.start:
            continue
        f.seek(r.start)
        data = f.read(r.end - r.start)
        dirty = sum(1 for b in data if b)
        if dirty:
            findings.append(Finding("S2", "high", "dirty-padding",
                                    f"{dirty} non-zero byte(s) in padding ({r.label}) at offset {r.start}",
                                    {"start": r.start, "end": r.end, "nonzero": dirty}))
    return findings
