"""Run every check on one file and produce a JSON-serialisable report.

The report is deterministic (stable ordering, no timestamps) so the Python and
browser implementations can be compared byte for byte.
"""
from __future__ import annotations

import contextlib
import hashlib
from collections import Counter
from pathlib import Path

from . import coverage, pickle_scan, weights
from .formats import open_model
from .model import SEVERITIES

REPORT_VERSION = 1


def _sha256(f) -> str:
    h = hashlib.sha256()
    f.seek(0)
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def scan(path: str | Path, base: "str | Path | list | None" = None) -> dict:
    """Scan one model file. If `base` is given (a file, or the files of a sharded
    checkpoint), also run the S5 differential against that claimed base model."""
    path = Path(path)
    with contextlib.ExitStack() as stack:
        base_file = None
        if base is not None:  # read one tensor at a time while it is compared
            parts = []
            for b in (base if isinstance(base, (list, tuple)) else [base]):
                bf = stack.enter_context(open(b, "rb"))
                parts += [open_model(Path(b), bf), bf]
            base_file = weights.BaseFile(*parts)
        return _scan(path, base_file)


def _scan(path: Path, base_file) -> dict:
    with open(path, "rb") as f:
        mf = open_model(path, f)
        findings = list(mf.findings)
        for label, start, end in mf.pickles:
            findings += pickle_scan.analyse(label, mf.read(f, start, end - start))
        for label, data in mf.inline_pickles.items():
            findings += pickle_scan.analyse(label, data)
        if mf.format != "unknown":
            findings += coverage.check(mf, f)
            findings += weights.analyse(mf, f, base_file)
        digest = _sha256(f)

    rank = {s: i for i, s in enumerate(SEVERITIES)}
    findings.sort(key=lambda x: (-rank[x.severity], x.check, x.code, x.message))
    worst = max((rank[x.severity] for x in findings), default=-1)
    verdict = "clean" if worst < rank["medium"] else "suspicious" if worst < rank["high"] else "dangerous"
    return {
        "report_version": REPORT_VERSION,
        "file": path.name,
        "sha256": digest,
        "size": mf.size,
        "format": mf.format,
        "verdict": verdict,
        "tensors": {
            "count": len(mf.tensors),
            "bytes": sum(t.nbytes for t in mf.tensors),
            "dtypes": dict(sorted(Counter(t.dtype for t in mf.tensors).items())),
        },
        "findings": [
            {"check": x.check, "severity": x.severity, "code": x.code, "message": x.message, "detail": x.detail}
            for x in findings
        ],
    }
