"""E13: S5d against S5c on fresh pairs (PLAN.md, phase 4; used once).

For the frozen list results/ft_pairs_fresh3.json, whole files (commit-pinned, SHA-256 checked),
S5 without a floor, each check run under S5c (S5_ABOVE_MAX = None) and S5d (A = 0.02):

  honest   every fine-tune against its base: the tensors each rule flags
  payload  E4b's procedure: a random payload in the low k bits (k = 8, 16, 20, 21) of (a) the
           largest tensor the fine-tune left identical to the base, (b) the changed tensor
           (>= 256 changed elements) with the lowest p99; is that tensor flagged
  thin     E12's thin payload in each clean model (k = 8, one weight in 64, every float tensor
           of >= 1,024 elements that holds it): (a) 16, (b) 4 bytes per tensor; any tensor flagged

    python scripts/run_e13.py [--only N] [--finetunes REPO ...]
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
from scripts.run_e12 import THIN, thin_copy  # noqa: E402
from scripts.run_e4b import KS, change_stats, fetch  # noqa: E402
from stowaway import weights  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402

RULES = {"s5c": None, "s5d": 0.02}


def flagged(model: dict, base: dict, rule: str) -> list[str]:
    old_floor, old_above = weights.S5_MIN_CHANGED, weights.S5_ABOVE_MAX
    weights.S5_MIN_CHANGED, weights.S5_ABOVE_MAX = 1, RULES[rule]
    try:
        return sorted(f.detail["tensor"] for f in weights.s5_differential(model, base) if f.severity == "high")
    finally:
        weights.S5_MIN_CHANGED, weights.S5_ABOVE_MAX = old_floor, old_above


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="first N pairs (smoke test; not recorded)")
    ap.add_argument("--pairs", default="results/ft_pairs_fresh3.json", help="smoke tests use a dev list")
    ap.add_argument("--finetunes", nargs="+", help="rerun only these pairs (e.g. ones a network outage skipped)")
    a = ap.parse_args(argv)
    pairs = json.loads((ROOT / a.pairs).read_text(encoding="utf-8"))["pairs"][: a.only]
    if a.finetunes:
        pairs = [p for p in pairs if p["finetune"] in a.finetunes]
    run = Run("E13", pairs=a.pairs, rules=RULES, ks=list(KS), thin=THIN, rerun=a.finetunes or None)
    if a.only:
        run.record = lambda **_: None
    rng = np.random.default_rng(13)
    bases, seen = {}, set()

    def thin(repo: str, clean: dict, commit: str) -> None:
        if repo in seen:
            return
        seen.add(repo)
        for case, per in THIN.items():
            planted, tensors = thin_copy(clean, per, rng)
            res = {r: flagged(planted, clean, r) for r in RULES}
            run.record(kind="thin", repo=repo, commit_hf=commit, case=case, bytes_per_tensor=per, tensors=tensors,
                       **{f"flagged_{r}": len(v) for r, v in res.items()},
                       **{f"detected_{r}": bool(v) for r, v in res.items()})
            print(f"  thin {case} {repo}: tensors {tensors}; " + ", ".join(f"{r} {len(v)}" for r, v in res.items()),
                  flush=True)

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
        common = dict(base=p["base"], finetune=p["finetune"], base_commit=bres.commit, ft_commit=fres.commit)
        pairing, _, _ = weights.align({n: (t.dtype, tuple(t.shape)) for n, t in ft.items()},
                                      {n: (t.dtype, tuple(t.shape)) for n, t in base.items()})
        ft_al = {pairing.get(n, n): t for n, t in ft.items()}  # the fine-tune under the base's names
        res = {r: flagged(ft_al, base, r) for r in RULES}
        run.record(kind="honest", **common, **{f"flagged_{r}": v for r, v in res.items()})
        print("  honest: " + ", ".join(f"{r} {len(v)}" for r, v in res.items()), flush=True)

        changed, unchanged = [], []
        for n, t in ft_al.items():
            b = base.get(n)
            if b is None or b.dtype != t.dtype or b.data.shape != t.data.shape:
                continue
            s = change_stats(t.data, b.data, t.dtype)
            if s["changed"] >= weights.MIN_ELEMENTS:
                changed.append((s["p99"], n))
            elif s["changed"] == 0:
                unchanged.append((t.data.size, n))
        targets = ([("a_base_matching", max(unchanged)[1])] if unchanged else []) + \
                  ([("b_low_gradient", min(changed)[1])] if changed else [])
        for case, n in targets:
            for k in KS:
                cap = capacity({n: ft_al[n]}, k, names=[n])
                if cap <= 16:
                    continue
                one = {n: embed({n: ft_al[n]}, rng.bytes(cap - 8), k, names=[n])[n]}
                hit = {r: n in flagged(one, {n: base[n]}, r) for r in RULES}
                run.record(kind="payload", case=case, tensor=n, k=k, **common, **{f"detected_{r}": v for r, v in hit.items()})
            print(f"  payload {case} {n}: done", flush=True)
        thin(p["base"], base, bres.commit)
        thin(p["finetune"], ft, fres.commit)
        del ft, ft_al
    print("run", run.run_id)


if __name__ == "__main__":
    main()
