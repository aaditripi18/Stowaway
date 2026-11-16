"""Why S5 flags 8 RMSNorm weights of Qwen/Qwen2-0.5B-Instruct against its true parent (E15).

For each tensor S5 flags in that pair: how many elements changed, the histogram of the highest
changed mantissa bit, how many units in the last place (ULP) each change is, and the share of
changed elements above S5c's bit. Writes results/e15_inspect.json; nothing goes in the ledger.

    python scripts/e15_inspect.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e15 import load  # noqa: E402
from stowaway import weights  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT  # noqa: E402

PAIRS = json.loads((ROOT / "results" / "e15_pairs.json").read_text(encoding="utf-8"))["pairs"]
OUT = ROOT / "results" / "e15_inspect.json"


def main():
    p = next(x for x in PAIRS if x["model"] == "Qwen/Qwen2-0.5B-Instruct")
    m = load(p["model"], p["model_commit"], p["model_file"])
    b = load(p["base"], p["base_commit"], p["base_file"])
    found = [f for f in weights.s5_differential(m, b) if f.code == "low-bit-only-change"]
    rows = []
    for f in found:
        n = f.detail["tensor"]
        a, c = m[n], b[n]
        mant = FLOAT_LAYOUT[a.dtype][1]
        diff = a.data != c.data
        xor = a.data[diff].astype(np.uint64) ^ c.data[diff].astype(np.uint64)
        hi = weights._highest_changed_bit(xor, mant)
        # signed difference in units in the last place (same sign and exponent: integer view difference)
        ulp = a.data[diff].astype(np.int64) - c.data[diff].astype(np.int64)
        vals = c.to_float32().ravel()
        rows.append({"tensor": n, "dtype": a.dtype, "elements": int(a.data.size), "changed": int(diff.sum()),
                     "hist_highest_bit": np.bincount(hi + 1, minlength=mant + 1).tolist(),
                     "ulp_abs_counts": {str(k): int(v) for k, v in zip(*np.unique(np.abs(ulp), return_counts=True))},
                     "base_value_range": [float(vals.min()), float(vals.max())], "finding": f.detail})
    OUT.write_text(json.dumps({"pair": p, "flagged": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} flagged tensors written to {OUT}")


if __name__ == "__main__":
    main()
