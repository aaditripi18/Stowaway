"""Write a large safetensors file for the opt-in large-file test, and its Python report.

    python scripts/make_large_fixture.py data/large/big.safetensors --gb 3.2

The file holds float32 tensors of 64 MiB each (random normal weights, so the
bit statistics have something to read), one tensor with a random payload in its
low 8 bits, and 4 KiB of executable-looking bytes after the end, so S2, S3/S4
and the hash all run at full size. The Python report is written next to it as
<name>.json, in the same canonical form the browser tests compare against.
Then:

    STOWAWAY_LARGE_FILE=data/large/big.safetensors npx vitest run test/large_file.test.ts   (in web/)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web" / "test"))
sys.path.insert(0, str(ROOT))

from gen_fixtures import canon  # noqa: E402
from stowaway.report import scan  # noqa: E402

TENSOR = 64 << 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--gb", type=float, default=3.2)
    a = ap.parse_args()
    n = max(2, int(a.gb * (1 << 30) // TENSOR))
    header, off = {}, 0
    for i in range(n):
        header[f"layers.{i}.weight"] = {"dtype": "F32", "shape": [TENSOR // 4], "data_offsets": [off, off + TENSOR]}
        off += TENSOR
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for i in range(n):
            w = (rng.standard_normal(TENSOR // 4).astype(np.float32) * 0.02)
            if i == n // 2:  # random bytes in the low 8 bits of one tensor
                u = w.view(np.uint32)
                u[:] = (u & np.uint32(0xFFFFFF00)) | rng.integers(0, 256, u.size, dtype=np.uint32)
            f.write(w.tobytes())
        f.write(b"MZ" + rng.bytes(4094))
    print(f"wrote {a.out} ({a.out.stat().st_size / (1 << 30):.2f} GiB, {n} tensors)")
    rep = scan(a.out)
    a.out.with_suffix(".json").write_text(canon(rep) + "\n", encoding="utf-8")
    print(f"python report: {rep['verdict']}, {len(rep['findings'])} findings -> {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
