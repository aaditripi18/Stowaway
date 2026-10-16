"""E0d: S4 against S4b (small role groups) on identical data (PLAN.md, phase 2).

  --split dev     the nine torchvision models, whole files
  --split fresh   results/hf_models_batch4.json, sampled as the survey samples
                  (6 roles x 4 layers, 32 KB each); used once

For each model: tensors with an S4 lowbit-anomaly under each rule (false
positives: the files are unmodified). Fresh only: random bytes in the low k bits
of the largest sampled tensor that neither rule flags on the clean sample, and
whether each rule then flags it (the E1b procedure), k by dtype as in E1b.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import corpus, hf  # noqa: E402
from experiments.ledger import Run  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402
from stowaway.weights import s4_bit_planes  # noqa: E402

KS = {"F32": (1, 4, 8), "F16": (1, 3), "BF16": (1, 2)}


RULES = {"S4": {}, "S4b": {"small_floor": True}, "S4c": {"min_signature": 0.02}}


def flags(samples, shapes, rule):
    return {f.detail["tensor"] for f in s4_bit_planes(samples, shapes=shapes, **RULES[rule])
            if f.code == "lowbit-anomaly"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "fresh"], required=True)
    a = ap.parse_args(argv)
    run = Run("E0d", split=a.split)
    rng = np.random.default_rng(0)
    if a.split == "dev":
        for name in corpus.VISION:
            model, _ = corpus.load_vision(name)
            raw = corpus.state_to_raw(model.state_dict())
            fp = {r: sorted(flags(raw, None, r)) for r in RULES}
            run.record(model=name, dtype="F32", fp=fp)
            print(f"{name:22s} " + "  ".join(f"{r} {len(v)}" for r, v in fp.items()), flush=True)
        print("run", run.run_id)
        return
    models = json.loads((ROOT / "results" / "hf_models_batch4.json").read_text(encoding="utf-8"))["models"]
    for m in models:
        try:
            samples, info = hf.sample(m["repo"], m["file"], chunk=32 * 1024, max_tensors=24, roles=6)
        except Exception as e:  # noqa: BLE001 -- recorded, as in the survey
            run.record(repo=m["repo"], dtype=m["dtype"], error=f"{type(e).__name__}: {e}"[:300])
            continue
        shapes = {n: tuple(v) for n, v in info["shapes"].items()}
        samples = {n: t for n, t in samples.items() if t.dtype in KS}
        if not samples:
            run.record(repo=m["repo"], dtype=m["dtype"], error="no float tensors large enough to sample")
            continue
        fp = {r: flags(samples, shapes, r) for r in RULES}
        rec = {"repo": m["repo"], "dtype": m["dtype"], "fp": {r: sorted(v) for r, v in fp.items()}, "payload": []}
        flagged_any = set().union(*fp.values())
        candidates = sorted((t.data.size, n) for n, t in samples.items() if n not in flagged_any)
        if candidates:
            target = candidates[-1][1]
            dtype = samples[target].dtype
            for k in KS[dtype]:
                cap = capacity({target: samples[target]}, k, names=[target])
                if cap < 64:
                    continue
                planted = dict(samples)
                planted[target] = embed({target: samples[target]}, rng.bytes(cap), k, names=[target])[target]
                rec["payload"].append({"tensor": target, "k": k,
                                       **{r: target in flags(planted, shapes, r) for r in RULES}})
        run.record(**rec)
        print(f"{m['dtype']:5s} {m['repo'][:50]:50s} " + "  ".join(f"{r} {len(v)}" for r, v in fp.items()) + "  "
              + " ".join(f"k{p['k']}:" + "".join(str(int(p[r])) for r in RULES) for p in rec["payload"]), flush=True)
    print("run", run.run_id)


if __name__ == "__main__":
    main()
