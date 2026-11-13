"""E12: S5 without its 256-element floor, on fresh pairs (PLAN.md, phase 4; used once).

For the frozen list results/ft_pairs_fresh2.json, whole files (commit-pinned, SHA-256 checked):

  honest  every fine-tune against its base, S5 with the floor (256) and without it (1)
  thin    each of the list's clean models (bases and fine-tunes) against a copy of itself that
          carries random bytes at k = 8 in every float tensor of >= 1,024 elements, one weight in
          64: (a) 16 bytes per tensor, (b) 4 bytes per tensor; S5 with and without the floor

    python scripts/run_e12.py [--only N] [--finetunes REPO ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import Run  # noqa: E402
from scripts.run_e4b import fetch  # noqa: E402
from stowaway import weights  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402

FLOORS = {"floor256": 256, "nofloor": 1}
THIN = {"a": 16, "b": 4}
MIN_TENSOR = 1024


def s5(model: dict, base: dict, floor: int) -> list[str]:
    old = weights.S5_MIN_CHANGED
    weights.S5_MIN_CHANGED = floor
    try:
        return sorted(f.detail["tensor"] for f in weights.s5_differential(model, base) if f.severity == "high")
    finally:
        weights.S5_MIN_CHANGED = old


def thin_copy(clean: dict, per_tensor: int, rng) -> tuple[dict, int]:
    out, n = dict(clean), 0
    for name, t in sorted(clean.items()):
        # a tensor that cannot hold the payload and its 4-byte length frame is left clean
        # (roberta-large's 1,024-element biases; PLAN.md, E12 amendment 2026-11-14)
        if t.dtype in FLOAT_LAYOUT and t.data.size >= MIN_TENSOR and capacity({name: t}, 8, 64) >= per_tensor:
            out[name] = embed({name: t}, rng.bytes(per_tensor), k=8, spread=64, names=[name])[name]
            n += 1
    return out, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="first N pairs (smoke test; not recorded)")
    ap.add_argument("--pairs", default="results/ft_pairs_fresh2.json", help="smoke tests use a dev list")
    ap.add_argument("--finetunes", nargs="+", help="rerun only these pairs (e.g. ones a network outage skipped)")
    a = ap.parse_args(argv)
    lst = json.loads((ROOT / a.pairs).read_text(encoding="utf-8"))
    pairs = lst["pairs"][: a.only]
    if a.finetunes:
        pairs = [p for p in pairs if p["finetune"] in a.finetunes]
    run = Run("E12", pairs=a.pairs, floors=FLOORS, thin=THIN, k=8, spread=64, rerun=a.finetunes or None)
    if a.only:
        run.record = lambda **_: None
    rng = np.random.default_rng(12)
    bases, seen = {}, set()

    def thin(repo: str, clean: dict, commit: str) -> None:
        if repo in seen:
            return
        seen.add(repo)
        for case, per in THIN.items():
            planted, tensors = thin_copy(clean, per, rng)
            res = {k: s5(planted, clean, f) for k, f in FLOORS.items()}
            run.record(kind="thin", repo=repo, commit_hf=commit, case=case, bytes_per_tensor=per, tensors=tensors,
                       **{f"flagged_{k}": len(v) for k, v in res.items()},
                       **{f"detected_{k}": bool(v) for k, v in res.items()})
            print(f"  thin {case} {repo}: tensors {tensors}; flagged "
                  + ", ".join(f"{k} {len(v)}" for k, v in res.items()), flush=True)

    for i, p in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {p['finetune']} vs {p['base']}", flush=True)
        try:
            if p["base"] not in bases:
                bases[p["base"]] = fetch(p["base"])
            bres, base = bases[p["base"]]
            fres, ft = fetch(p["finetune"], p.get("file"))
        except Exception as e:  # noqa: BLE001
            run.record(kind="error", base=p["base"], finetune=p["finetune"], error=f"{type(e).__name__}: {e}"[:300])
            print(f"  skipped: {e}", flush=True)
            continue
        res = {k: s5(ft, base, f) for k, f in FLOORS.items()}
        run.record(kind="honest", base=p["base"], finetune=p["finetune"], base_commit=bres.commit, ft_commit=fres.commit,
                   **{f"flagged_{k}": v for k, v in res.items()})
        print("  honest: " + ", ".join(f"{k} {len(v)}" for k, v in res.items()), flush=True)
        thin(p["base"], base, bres.commit)
        thin(p["finetune"], ft, fres.commit)
    print("run", run.run_id)


if __name__ == "__main__":
    main()
