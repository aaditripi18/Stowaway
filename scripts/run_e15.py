"""E15: is the base close enough for S5? (PLAN.md, phase 5; used once on results/e15_pairs.json)

For each frozen pair, both whole files (commit-pinned, SHA-256 checked), S5 with the closeness check
on: the median cosine of the aligned tensors, the identical ones, and whether `base-not-close`
fires; next to Pedigree's label (independent / related) and, for declared pairs, Pedigree's own
sampled median cosine.

    python scripts/run_e15.py [--only N]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import Run  # noqa: E402
from stowaway import hub, weights  # noqa: E402
from stowaway.formats import open_model  # noqa: E402

PEDIGREE = ROOT.parent / "Pedigree" / "results" / "ledger.jsonl"


def pedigree_cosines() -> dict:
    """Pedigree's own sampled cosines, when its repository is next to this one (not on Kaggle;
    the report joins them from Pedigree's ledger either way)."""
    out = {}
    if not PEDIGREE.exists():
        return out
    for line in PEDIGREE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("experiment") == "E2":
            for d in r.get("declared", []):
                out[(r["id"], d["base"])] = d.get("median_cosine")
        elif r.get("experiment") == "E1" and r.get("kind") == "independent":
            out[(r["model"], r["base"])] = r.get("median_cosine")
    return out


def load(repo: str, commit: str, file: str, tries: int = 3) -> dict:
    for t in range(tries):  # an interrupted transfer resumes; a failed one is tried again
        try:
            return _load(repo, commit, file)
        except hub.HubError:
            if t == tries - 1:
                raise


def _load(repo: str, commit: str, file: str) -> dict:
    res = hub.resolve(f"hf:{repo}@{commit}")
    res = hub.Resolved(res.repo, res.revision, res.commit, [file], {file: res.sizes[file]},
                       {file: res.sha256[file]}, res.base_model)
    (path,) = hub.download(res, log=lambda m: None)
    with open(path, "rb") as f:
        floats = weights._load_floats(open_model(path, f), f)
    Path(path).unlink(missing_ok=True)
    return floats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="first N pairs (smoke test; not recorded)")
    ap.add_argument("--pairs", default=str(ROOT / "results" / "e15_pairs.json"), help="smoke tests use a seen pair")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.pairs).read_text(encoding="utf-8"))
    pairs = spec["pairs"][: a.only]
    run = Run("E15", pairs="results/e15_pairs.json", close_min=weights.S5_CLOSE_MIN, n_pairs=len(pairs))
    if a.only:
        run.record = lambda **_: None
    ped = pedigree_cosines()
    weights.S5_BASE_CHECK = True
    for i, p in enumerate(pairs, 1):
        common = {k: p[k] for k in ("model", "base", "model_commit", "base_commit", "pedigree", "name_shape_match")}
        try:
            m = load(p["model"], p["model_commit"], p["model_file"])
            b = load(p["base"], p["base_commit"], p["base_file"])
            found = weights.s5_differential(m, b)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            run.record(kind="error", **common, error=f"{type(e).__name__}: {e}"[:300])
            continue
        close = next((f.detail for f in found if f.code == "base-closeness"), {})
        warned = any(f.code == "base-not-close" for f in found)
        s5_flags = sorted(f.detail["tensor"] for f in found if f.code == "low-bit-only-change")
        run.record(kind="pair", **common, warned=warned, median_cosine=close.get("median_cosine"),
                   compared=close.get("compared"), identical=close.get("identical"),
                   pedigree_cosine=ped.get((p["model"], p["base"])), s5_flagged=s5_flags[:20],
                   s5_flagged_n=len(s5_flags))
        print(f"[{i}/{len(pairs)}] {p['pedigree']:11s} {p['model']} <- {p['base']}: median cosine "
              f"{close.get('median_cosine')}, identical {close.get('identical')}/{close.get('compared')}, "
              f"warned {warned}", flush=True)
        del m, b
    print("run", run.run_id)


if __name__ == "__main__":
    main()
