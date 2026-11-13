"""E12 dev (PLAN.md, phase 4): what S5c would say about honest tensors below its floor.

S5 skips a tensor whose fine-tune changed fewer than MIN_ELEMENTS = 256 elements, so a payload
spread thinly enough (E3: 1 KB at one weight in 64) puts fewer than 256 changed elements in
every tensor and is never judged. A lower floor catches it only if honest tensors with few
changes are not flagged. This reads the honest pairs E4/E4b already used (dev: seen before),
from the local download cache, and for every shared tensor with 1-255 changed elements records
S5c's statistics; then, for each candidate floor, how many honest tensors and pairs S5c flags.

Writes results/e12_dev.json. Nothing goes in the ledger: this chooses a rule, it does not judge it.

    STOWAWAY_CACHE=data/hub_cache python scripts/e12_dev.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e4b import _fetch  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT  # noqa: E402
from stowaway.weights import S5_MARGIN, _highest_changed_bit, align  # noqa: E402

import numpy as np  # noqa: E402

FLOORS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LISTS = ("results/ft_pairs_dev.json", "results/ft_pairs_fresh.json")


def s5c(a: np.ndarray, b: np.ndarray, dtype: str) -> dict | None:
    diff = a != b
    n = int(diff.sum())
    if n == 0:
        return None
    mant = FLOAT_LAYOUT[dtype][1]
    cutoff = mant - S5_MARGIN
    hi = _highest_changed_bit(a[diff].astype(np.uint64) ^ b[diff].astype(np.uint64), mant)
    low = np.bincount(hi + 1, minlength=mant + 1)[: cutoff + 2]
    j = int(np.argmax(low))
    share = float(low[j]) / n
    return {"changed": n, "bit": j - 1, "share": share, "flag": share >= 0.5 - 1.5 / n ** 0.5,
            "top": int(hi.max()), "mant": mant}


def main() -> None:
    pairs = [p for f in LISTS for p in json.loads((ROOT / f).read_text(encoding="utf-8"))["pairs"]]
    bases, rows = {}, []
    for i, p in enumerate(pairs, 1):
        try:
            if p["base"] not in bases:
                bases[p["base"]] = _fetch(p["base"])[1]
            base, ft = bases[p["base"]], _fetch(p["finetune"], p.get("file"))[1]
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] {p['finetune']}: skipped ({e})", flush=True)
            continue
        pairing, _, _ = align({n: (t.dtype, tuple(t.shape)) for n, t in ft.items()},
                              {n: (t.dtype, tuple(t.shape)) for n, t in base.items()})
        for n in sorted(pairing):
            a, b = ft[n], base[pairing[n]]
            if a.dtype != b.dtype or a.data.shape != b.data.shape or a.dtype not in FLOAT_LAYOUT:
                continue
            s = s5c(a.data, b.data, a.dtype)
            if s and s["changed"] < 256:
                rows.append({"finetune": p["finetune"], "base": p["base"], "tensor": n, "numel": int(a.data.size), **s})
        print(f"[{i}/{len(pairs)}] {p['finetune']}: {sum(r['finetune'] == p['finetune'] for r in rows)} tensors under 256",
              flush=True)
    by_floor = {}
    for F in FLOORS:
        flagged = [r for r in rows if r["changed"] >= F and r["flag"]]
        by_floor[F] = {"tensors": len(flagged), "pairs": len({r["finetune"] for r in flagged}),
                       "examples": [(r["finetune"], r["tensor"], r["changed"], r["bit"], round(r["share"], 3))
                                    for r in flagged[:8]]}
    out = {"pairs": len(pairs), "tensors_under_256": len(rows), "by_floor": by_floor, "rows": rows}
    (ROOT / "results" / "e12_dev.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({F: {k: v for k, v in x.items() if k != "examples"} for F, x in by_floor.items()}))


if __name__ == "__main__":
    main()
