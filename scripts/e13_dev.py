"""E13 dev (PLAN.md, phase 4): S5c's honest false alarms on LayerNorm gains, and a rule for them.

S5c flags a tensor when about half its changed elements share one highest changed bit j at or
below the cutoff, as a low-bit payload's do. A LayerNorm gain sits near 1.0 and a fine-tune moves
every element by a similar small amount, so its changes also pile up on one or two bits. The
difference: a payload in the low k bits changes nothing above bit k-1, so no changed element's
highest changed bit is above j; an additive change carries into higher bits.

This reads every honest pair already seen (E4/E4b dev and fresh, E12's fresh2, the E8 held-out
pairs S5 flagged), records S5c's statistics for each tensor S5c flags (share at j, share above j,
share at j-1), and the same statistics for E4b's planted payloads from the ledger. Then, for
candidate bounds A on the share above j, how many honest flags and payload catches remain.

Writes results/e13_dev.json. Nothing goes in the ledger: this chooses a rule, it does not judge it.

    python scripts/e13_dev.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import load  # noqa: E402
from scripts.run_e4b import fetch  # noqa: E402
from scripts.run_e12 import THIN, thin_copy  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT  # noqa: E402
from stowaway.weights import S5_MARGIN, _highest_changed_bit, align  # noqa: E402

LISTS = ("results/ft_pairs_dev.json", "results/ft_pairs_fresh.json", "results/ft_pairs_fresh2.json")
E8_FLAGGED = [{"base": "cross-encoder/ms-marco-MiniLM-L12-v2", "finetune": "cross-encoder/ms-marco-MiniLM-L4-v2"},
              {"base": "BAAI/bge-base-en-v1.5", "finetune": "datasocietyco/bge-base-en-v1.5-course-recommender-v5"}]
BOUNDS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35)
# where "above" starts: 0 = above S5c's bit j; 1 = above j + 1 (a thin payload's most common
# highest changed bit is one below its top bit by chance in some tensors)
OFFSETS = (0, 1)


def above(r: dict, offset: int) -> float:
    j = r["bit"] + 1  # histogram index of S5c's bit
    return float(sum(r["hist"][j + 1 + offset:])) / r["changed"]


def stats_from_hist(hist, n: int, cutoff: int) -> dict:
    """S5c's statistics from a histogram indexed by highest changed bit + 1."""
    hist = np.asarray(hist)
    low = hist[: cutoff + 2]
    j = int(np.argmax(low))
    share = float(low[j]) / n
    return {"changed": n, "bit": j - 1, "share": round(share, 4), "flag": bool(share >= 0.5 - 1.5 / n ** 0.5),
            "hist": [int(x) for x in hist],
            "above": round(float(hist[j + 1:].sum()) / n, 4), "below": round(float(hist[j - 1]) / n if j >= 1 else 0.0, 4)}


def tensor_stats(a: np.ndarray, b: np.ndarray, dtype: str) -> dict | None:
    diff = a != b
    n = int(diff.sum())
    if n == 0:
        return None
    mant = FLOAT_LAYOUT[dtype][1]
    hi = _highest_changed_bit(a[diff].astype(np.uint64) ^ b[diff].astype(np.uint64), mant)
    return stats_from_hist(np.bincount(hi + 1, minlength=mant + 1), n, mant - S5_MARGIN)


def main() -> None:
    pairs = [p for f in LISTS for p in json.loads((ROOT / f).read_text(encoding="utf-8"))["pairs"]] + E8_FLAGGED
    honest, bases, read = [], {}, 0
    thin, thinned, rng = [], set(), np.random.default_rng(1313)
    for i, p in enumerate(pairs, 1):
        try:
            if p["base"] not in bases:
                bases[p["base"]] = fetch(p["base"])[1]
            base, ft = bases[p["base"]], fetch(p["finetune"], p.get("file"))[1]
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] {p['finetune']}: skipped ({e})", flush=True)
            continue
        read += 1
        pairing, _, _ = align({n: (t.dtype, tuple(t.shape)) for n, t in ft.items()},
                              {n: (t.dtype, tuple(t.shape)) for n, t in base.items()})
        for n in sorted(pairing):
            a, b = ft[n], base[pairing[n]]
            if a.dtype != b.dtype or a.data.shape != b.data.shape or a.dtype not in FLOAT_LAYOUT:
                continue
            s = tensor_stats(a.data, b.data, a.dtype)
            if s and s["flag"]:
                honest.append({"finetune": p["finetune"], "base": p["base"], "tensor": n, **s})
        for repo, clean in ((p["base"], base), (p["finetune"], ft)):
            if repo in thinned:
                continue
            thinned.add(repo)
            for case, per in THIN.items():
                planted, tensors = thin_copy(clean, per, rng)
                rows = []
                for n, t in planted.items():
                    st = tensor_stats(t.data, clean[n].data, t.dtype) if t is not clean[n] else None
                    if st and st["flag"]:
                        rows.append(st)
                thin.append({"repo": repo, "case": case, "tensors": tensors, "flagged": rows})
        print(f"[{i}/{len(pairs)}] {p['finetune']}: {sum(r['finetune'] == p['finetune'] for r in honest)} flagged",
              flush=True)
    payloads = []
    for r in load():
        if r["experiment"] == "E4b" and r.get("kind") == "payload" and "hist" in r:
            s = stats_from_hist(r["hist"], r["changed"], r["cutoff"])
            if s["flag"]:
                payloads.append({"case": r["case"], "k": r["k"], "finetune": r["finetune"], "tensor": r["tensor"],
                                 "split": r["meta"].get("split"), **s})
    grid = {}
    for off in OFFSETS:
        for A in BOUNDS:
            grid[f"off{off}_A{A:g}"] = {
                "honest_flags": sum(above(r, off) <= A for r in honest),
                "honest_pairs": len({r["finetune"] for r in honest if above(r, off) <= A}),
                "payloads_kept_a": sum(above(r, off) <= A for r in payloads if r["case"].startswith("a")),
                "payloads_kept_b": sum(above(r, off) <= A for r in payloads if r["case"].startswith("b")),
                "thin_models": {c: sum(any(above(x, off) <= A for x in t["flagged"]) for t in thin if t["case"] == c)
                                for c in THIN},
                "thin_tensor_share": round(sum(sum(above(x, off) <= A for x in t["flagged"]) for t in thin)
                                           / max(1, sum(t["tensors"] for t in thin)), 4)}
    out = {"pairs": len(pairs), "pairs_read": read, "honest_flags": len(honest),
           "honest_pairs": len({r["finetune"] for r in honest}), "payload_catches": len(payloads),
           "honest_below_max": max((r["below"] for r in honest), default=None),
           "payload_below_min": min((r["below"] for r in payloads), default=None),
           "thin_models": {c: sum(t["case"] == c for t in thin) for c in THIN},
           "grid": grid, "honest": honest, "payloads": payloads, "thin": thin}
    (ROOT / "results" / "e13_dev.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("honest", "payloads", "thin")}, indent=1))


if __name__ == "__main__":
    main()
