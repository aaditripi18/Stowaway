"""E17 dev (PLAN.md, phase 5): S4's flags on whole files, false and true, by effect size.

Whole files flag trained tensors through S4 in four models (E14: Qwen2.5-VL, AST, gte-large,
Ornith). In a tensor of millions of elements a small, real difference in a bit plane's frequency
becomes significant. A payload pulls the plane towards a fair coin by roughly the share of
elements it writes times the reference's distance from 0.5.

For every model of the two lists whole files were already read for (E10 A: hf_models.json, E11:
hf_models_batch5.json), the whole file, commit-pinned: every S4 finding on the clean file (a false
alarm) with its effect (how much nearer 0.5 the plane is than its reference, in absolute terms)
and its z; then E11's planted cells (the survey sample's bytes of the target tensor, k = 1/4/8 or
1/4) with, for the target, S3's flag, S4's flag with its effect and z, and S4w's flag. Writes
results/e17_dev.json; nothing goes in the ledger: this chooses a rule, it does not judge it.

    python scripts/e17_dev.py [--lists ...] [--only N]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import hf  # noqa: E402
from scripts.run_e10a import MAX_BYTES, SAMPLING, classify  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402
from stowaway.formats import open_model  # noqa: E402
from stowaway.weights import _load_floats, s3_precision_history, s4_bit_planes, s4_windowed  # noqa: E402

OUT = ROOT / "results" / "e17_dev.json"


def s4_rows(floats, shapes) -> list[dict]:
    out = []
    for f in s4_bit_planes(floats, shapes=shapes):
        if f.code != "lowbit-anomaly":
            continue
        d = f.detail
        out.append({"tensor": d["tensor"], "bit": d["bit"], "z": d["z"], "mean": d["mean"],
                    "expected": d["expected"], "elements": d["elements"], "reference": d["reference"],
                    "effect": round(abs(d["expected"] - 0.5) - abs(d["mean"] - 0.5), 6)})
    return out


def model_rows(m: dict, rng) -> dict:
    repo = m["repo"]
    samples, info = hf.sample(repo, m["file"], **SAMPLING)
    shapes = {n: tuple(v) for n, v in info["shapes"].items()}
    _, _, flagged_s = classify(samples, shapes)
    pool = [n for n in samples if n not in flagged_s] or list(samples)
    target = max(pool, key=lambda n: (samples[n].data.size, n))
    res = hub.resolve("hf:" + repo)
    if res.sizes[m["file"]] > MAX_BYTES:
        return {"repo": repo, "skipped": "size"}
    res = hub.Resolved(res.repo, res.revision, res.commit, [m["file"]], {m["file"]: res.sizes[m["file"]]},
                       {m["file"]: res.sha256[m["file"]]}, res.base_model)
    (path,) = hub.download(res, log=lambda msg: None)
    with open(path, "rb") as f:
        whole = _load_floats(open_model(path, f), f)
    wshapes = {n: tuple(t.shape) for n, t in whole.items()}
    clean = s4_rows(whole, wshapes)
    cells = []
    for k in ([1, 4, 8] if samples[target].dtype == "F32" else [1, 4]):
        payload = rng.bytes(capacity(samples, k, names=[target]))
        planted = {**whole, target: embed({target: whole[target]}, payload, k=k, names=[target])[target]}
        s4 = [r for r in s4_rows(planted, wshapes) if r["tensor"] == target]
        s3 = any(f.detail.get("tensor") == target and f.severity in ("medium", "high")
                 for f in s3_precision_history(planted))
        win = any(f.detail["tensor"] == target for f in s4_windowed({target: planted[target]}))
        cells.append({"k": k, "payload_bytes": len(payload), "target": target,
                      "target_elements": int(whole[target].data.size), "s3": s3, "s4": s4[0] if s4 else None,
                      "s4w": win, "s4_clean_flag": any(r["tensor"] == target for r in clean)})
    del whole
    Path(path).unlink(missing_ok=True)
    return {"repo": repo, "commit": res.commit, "clean_s4": clean, "cells": cells}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", default=["hf_models.json", "hf_models_batch5.json"])
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args(argv)
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"models": []}
    have = {m["repo"] for m in done["models"]}
    rng = np.random.default_rng(17)
    for lst in a.lists:
        for m in json.loads((ROOT / "results" / lst).read_text(encoding="utf-8"))["models"][: a.only]:
            if m["repo"] in have:
                continue
            try:
                row = model_rows(m, rng)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = {"repo": m["repo"], "error": f"{type(e).__name__}: {e}"[:300]}
            row["list"] = lst
            done["models"].append(row)
            have.add(m["repo"])
            OUT.write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
            print(f"{m['repo']}: clean S4 flags {len(row.get('clean_s4', []))}; "
                  f"cells {[(c['k'], bool(c['s4']), c['s4w'], c['s3']) for c in row.get('cells', [])]}", flush=True)


if __name__ == "__main__":
    main()
