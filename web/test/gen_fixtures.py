"""Generate safetensors fixtures and golden canonical-JSON reports for the
TypeScript parity tests. Run from the repo root with the project venv:

    web/.venv-less: use the repo venv
    PYTHONPATH=. .venv/Scripts/python web/test/gen_fixtures.py

Writes web/fixtures/<name>.safetensors and web/fixtures/<name>.json (the golden
report in the shared canonical form). The canonical form matches web/src/canon.ts.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "web" / "fixtures"

from stowaway.report import scan  # noqa: E402


def canon(o) -> str:
    """The exact string web/src/canon.ts produces."""
    if o is None:
        return "null"
    if isinstance(o, bool):
        return "true" if o else "false"
    if isinstance(o, int):
        return str(o)
    if isinstance(o, float):
        if o == int(o) and abs(o) < 1e15:
            return str(int(o))
        return repr(o)
    if isinstance(o, str):
        return json.dumps(o)
    if isinstance(o, (list, tuple)):
        return "[" + ",".join(canon(x) for x in o) + "]"
    if isinstance(o, dict):
        return "{" + ",".join(json.dumps(k) + ":" + canon(v) for k, v in sorted(o.items())) + "}"
    raise TypeError(f"cannot canonicalise {type(o)}")


def write(name: str, data: bytes):
    path = FIX / f"{name}.safetensors"
    path.write_bytes(data)
    rep = scan(path)
    (FIX / f"{name}.json").write_text(canon(rep) + "\n", encoding="utf-8")
    print(f"{name}: {rep['verdict']} ({len(rep['findings'])} findings)")


def st_bytes(tensors, metadata=None) -> bytes:
    tmp = FIX / "_tmp.safetensors"
    save_file(tensors, str(tmp), metadata=metadata)
    data = tmp.read_bytes()
    tmp.unlink()
    return data


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    clean = {
        "a": rng.standard_normal((64, 32)).astype(np.float32),
        "b": np.linspace(-1, 1, 50).astype(np.float16),
        "c": np.arange(16, dtype=np.int64),
    }
    write("clean", st_bytes(clean, {"producer": "stowaway-fixtures"}))

    write("trailing_mz", st_bytes(clean) + b"MZ" + bytes(range(256)) * 8)

    hdr = ('{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},'
           ' "w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}').encode()
    write("duplicate_keys", struct.pack("<Q", len(hdr)) + hdr + b"\0" * 8)

    hdr2 = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}).encode()
    write("size_mismatch", struct.pack("<Q", len(hdr2)) + hdr2 + b"\1" * 12)

    big = struct.pack("<Q", 10**9) + b"{}"
    write("bad_header_length", big)


if __name__ == "__main__":
    main()
