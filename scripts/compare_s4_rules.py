"""E0c: the week-4 S4 rule against the current one, on the same survey samples.

The week-4 rule read every tensor against the whole model and examined
min(8, mantissa) bit planes (all 7 of bf16's). The current rule reads a tensor
against its role peers and examines the low ~35% of the mantissa. Both run on
identical cached samples, so the difference is the rule alone. No network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments import hf
from experiments.ledger import Run
from stowaway.weights import s4_bit_planes

RESULTS = Path(__file__).resolve().parent.parent / "results"
OLD_BITS = {"F32": 8, "F16": 8, "BF16": 7}


def old_rule(samples):
    """Week-4 S4: no role peers (digit-free unique names) and the old bit counts."""
    alias = {n: "t" + "".join(chr(97 + int(c)) for c in str(i)) for i, n in enumerate(samples)}
    return [f for f in s4_bit_planes({alias[n]: t for n, t in samples.items()}, low_bits=OLD_BITS)
            if f.code == "lowbit-anomaly"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="hf_models_batch2.json")
    args = ap.parse_args(argv)
    models = json.loads((RESULTS / args.list).read_text(encoding="utf-8"))["models"]
    run = Run("E0c", list=args.list, n_models=len(models))
    for m in models:
        samples, info = hf.sample(m["repo"], m["file"], chunk=32 * 1024, max_tensors=24, roles=6)
        shapes = {n: tuple(v) for n, v in info["shapes"].items()}
        new = [f for f in s4_bit_planes(samples, shapes=shapes) if f.code == "lowbit-anomaly"]
        old = old_rule(samples)
        run.record(repo=m["repo"], dtype=m["dtype"], old_fp=len(old), new_fp=len(new),
                   new_flags=[{"tensor": f.detail["tensor"], "bit": f.detail["bit"], "mean": f.detail["mean"],
                               "expected": f.detail["expected"], "reference": f.detail["reference"]} for f in new])
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
