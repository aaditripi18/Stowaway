"""Freeze E10's random sample to results/hf_models_random.json (PLAN.md, E10 B).

Rule (fixed before any model is read): list the 10,000 most-downloaded public
models that ship safetensors, shuffle them with seed 0, and walk the shuffled
list keeping float32 models (by the dtype holding most parameters), skipping
gated ones, ones without a top-level safetensors file and every model of an
earlier list, until 40 are kept.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from experiments import hf

RESULTS = Path(__file__).resolve().parent.parent / "results"
POPULATION = 10_000
WANT = 40
EARLIER = ("hf_models.json", "hf_models_batch2.json", "hf_models_batch3.json", "hf_models_batch4.json")


def population() -> list[dict]:
    url = (f"{hf.API}?filter=safetensors&sort=downloads&direction=-1&limit=1000"
           "&expand[]=safetensors&expand[]=gated&expand[]=siblings")
    out = []
    while url and len(out) < POPULATION:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "stowaway-survey"}),
                                    timeout=120) as r:
            out += json.loads(r.read())
            nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url = nxt.group(1) if nxt else None
    return out[:POPULATION]


def main():
    pop = population()
    seen = {m["repo"] for name in EARLIER
            for m in json.loads((RESULTS / name).read_text(encoding="utf-8"))["models"]}
    order = np.random.default_rng(0).permutation(len(pop))
    picked, skipped = [], {"gated": 0, "no_file": 0, "no_float": 0, "not_f32": 0, "in_earlier_list": 0}
    for i in order:
        m = pop[i]
        if m["id"] in seen:
            skipped["in_earlier_list"] += 1
        elif m.get("gated"):
            skipped["gated"] += 1
        elif (dt := hf.primary_dtype(m)) is None:
            skipped["no_float"] += 1
        elif dt != "F32":
            skipped["not_f32"] += 1
        elif (f := hf.weights_file(m)) is None:
            skipped["no_file"] += 1
        else:
            picked.append({"repo": m["id"], "file": f, "dtype": dt,
                           "params": m["safetensors"]["parameters"], "rank": int(i) + 1})
            if len(picked) == WANT:
                break
    out = {"selected": time.strftime("%Y-%m-%d", time.gmtime()), "rule": __doc__.strip().splitlines()[2:],
           "population": len(pop), "seed": 0, "skipped": skipped, "excluded_lists": list(EARLIER),
           "models": picked}
    (RESULTS / "hf_models_random.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(picked)} models from a population of {len(pop)}; skipped {skipped}")


if __name__ == "__main__":
    main()
