"""Freeze the Hugging Face survey list to results/hf_models.json.

Rule (fixed before any model is sampled): walk the most-downloaded public models
that ship safetensors, skip gated ones and ones without a top-level weights file,
classify each by the float dtype holding most of its parameters, and keep the
first N of each dtype. The frozen list keeps the survey reproducible after the
rankings change.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments import hf

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=1000, help="how many top models to walk")
    ap.add_argument("--per-dtype", nargs=3, type=int, default=[40, 20, 10], metavar=("F32", "BF16", "F16"))
    ap.add_argument("--out", default="hf_models.json", help="file in results/")
    ap.add_argument("--exclude", nargs="*", default=[], help="earlier lists in results/ whose repos to skip")
    args = ap.parse_args(argv)
    want = dict(zip(["F32", "BF16", "F16"], args.per_dtype))
    seen = {m["repo"] for name in args.exclude
            for m in json.loads((RESULTS / name).read_text(encoding="utf-8"))["models"]}

    models = hf.list_models(limit=args.scan)
    picked = {k: [] for k in want}
    skipped = {"gated": 0, "no_file": 0, "no_float": 0, "in_earlier_list": 0}
    for m in models:
        if m["id"] in seen:
            skipped["in_earlier_list"] += 1
            continue
        if m.get("gated"):
            skipped["gated"] += 1
            continue
        dt = hf.primary_dtype(m)
        if dt is None:
            skipped["no_float"] += 1
            continue
        f = hf.weights_file(m)
        if f is None:
            skipped["no_file"] += 1
            continue
        if dt in picked and len(picked[dt]) < want[dt]:
            picked[dt].append({"repo": m["id"], "file": f, "dtype": dt,
                               "params": m["safetensors"]["parameters"], "rank": models.index(m) + 1})
    out = {
        "selected": time.strftime("%Y-%m-%d", time.gmtime()),
        "rule": __doc__.strip().splitlines()[2:],
        "scanned": len(models), "skipped": skipped, "excluded_lists": args.exclude,
        "models": [x for k in want for x in picked[k]],
    }
    out_path = RESULTS / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in picked.items()}, "skipped", skipped)


if __name__ == "__main__":
    main()
