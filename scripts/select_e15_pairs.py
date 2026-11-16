"""Freeze E15's fresh list (PLAN.md, phase 5): declared-base pairs from Pedigree's Hub audit.

Labels come from Pedigree (C:/Projects/Pedigree or --pedigree): its E2 ledger rows and its own
verdict rule (`scripts/analyse.py`: thresholds from its E1 development negatives). Here only
headers and file sizes are read, never weights: both files safetensors, one file each, <= 1.5 GB,
and S5's alignment (from the headers' names, dtypes and shapes) pairing >= 90% of the model's
tensors. Every pair Pedigree's weights contradict is kept; 30 of the rest are drawn with seed 15.

    python scripts/select_e15_pairs.py [--pedigree PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import hf  # noqa: E402
from stowaway.weights import align  # noqa: E402

MAX_BYTES = 1536 << 20
MAX_BYTES_NEG = 2560 << 20  # amendment: negatives may be up to 2.5 GB
MIN_MATCH = 0.9
DRAW = 30
OUT = ROOT / "results" / "e15_pairs.json"


def pedigree(path: Path):
    spec = importlib.util.spec_from_file_location("pedigree_analyse", path / "scripts" / "analyse.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path))
    spec.loader.exec_module(m)
    return m


def single_file(repo: str, commit: str) -> tuple[str, int] | None:
    import urllib.request
    url = f"{hf.API}/{repo}/revision/{commit}?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "stowaway-survey"}), timeout=60) as r:
        info = json.loads(r.read())
    st = [s for s in info.get("siblings", []) if s["rfilename"].endswith(".safetensors") and "/" not in s["rfilename"]]
    if len(st) != 1 or not st[0].get("size"):
        return None
    return st[0]["rfilename"], int(st[0]["size"])


def shapes(repo: str, commit: str, file: str) -> dict:
    _, header = hf.read_header(hf.resolve_url(repo, file, commit), hf.http_fetch)
    return {n: (v["dtype"], tuple(v["shape"])) for n, v in header.items() if n != "__metadata__"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pedigree", default=str(ROOT.parent / "Pedigree"))
    a = ap.parse_args(argv)
    P = pedigree(Path(a.pedigree))
    rows = P.load()
    e1 = [r for r in rows if r.get("experiment") == "E1"]
    T = P.thresholds(e1)
    rule = json.loads((Path(a.pedigree) / "results" / "verdicts.json").read_text(encoding="utf-8"))["rules"]
    rule = rule.get("verdict", "t") if isinstance(rule, dict) else "t"
    cands = []
    for m in (r for r in rows if r.get("experiment") == "E2"):
        if m.get("reason") or m.get("format") != "safetensors":
            continue
        for d in m.get("declared", []):
            if d.get("format") != "safetensors":
                continue
            v = P.related(d, T, rule)
            if v in ("related", "unrelated"):
                cands.append({"model": m["id"], "model_commit": m["commit"], "base": d["base"],
                              "base_commit": d["commit"], "pedigree": v})
    # amendment: Pedigree's E1 independent same-architecture pairs are negatives too
    seen = set()
    for r in e1:
        if r.get("kind") == "independent" and (r["model"], r["base"]) not in seen:
            seen.add((r["model"], r["base"]))
            cands.append({"model": r["model"], "model_commit": r["model_commit"], "base": r["base"],
                          "base_commit": r["base_commit"], "pedigree": "independent"})
    print(f"{len(cands)} declared safetensors pairs Pedigree could judge "
          f"({sum(c['pedigree'] == 'unrelated' for c in cands)} contradicted)", flush=True)
    random.Random(15).shuffle(cands)
    cands.sort(key=lambda c: c["pedigree"] == "related")  # negatives first, the rest in drawn order
    kept, drawn = [], 0
    for c in cands:
        if c["pedigree"] == "related" and drawn >= DRAW:
            break
        try:
            mf, bf = single_file(c["model"], c["model_commit"]), single_file(c["base"], c["base_commit"])
            cap = MAX_BYTES if c["pedigree"] == "related" else MAX_BYTES_NEG
            if not mf or not bf or mf[1] > cap or bf[1] > cap:
                continue
            ms, bs = shapes(c["model"], c["model_commit"], mf[0]), shapes(c["base"], c["base_commit"], bf[0])
        except Exception as e:  # noqa: BLE001
            print(f"  {c['model']}: skipped ({type(e).__name__})", flush=True)
            continue
        pairing, _, _ = align(ms, bs)
        match = len(pairing) / max(len(ms), 1)
        if match < MIN_MATCH:
            continue
        kept.append({**c, "model_file": mf[0], "base_file": bf[0], "model_bytes": mf[1], "base_bytes": bf[1],
                     "name_shape_match": round(match, 4)})
        drawn += c["pedigree"] == "related"
        print(f"  {c['pedigree']:9s} {c['model']} <- {c['base']} (match {match:.2f})", flush=True)
    OUT.write_text(json.dumps({"rule": "PLAN.md E15 (with its amendment)", "seed": 15, "max_bytes": MAX_BYTES,
                               "max_bytes_negatives": MAX_BYTES_NEG, "min_match": MIN_MATCH,
                               "pairs": kept}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(kept)} pairs ({sum(p['pedigree'] != 'related' for p in kept)} negatives)")


if __name__ == "__main__":
    main()
