"""E0b + E1b: the single-file checks on public Hugging Face models, from samples.

For each model in results/hf_models.json, sample up to 12 float tensors (first
256 KB of each) with range requests, then:

  E0b  run S3 and S4 on the clean samples: false positives, the precision the
       weights actually carry (S3), and whether the low bits carry a signature (S4)
  E1b  fill the low k bits of the largest sampled tensor with random bytes and ask
       whether S3/S4 flag that tensor from the samples alone

Detection here is on samples, not whole files: it says whether the model's own
statistics would expose a random low-bit payload, which is what decides H2/H3.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from experiments import hf
from experiments.ledger import Run
from stowaway.embed import capacity, embed
from stowaway.weights import s3_precision_history, s4_bit_planes

LIST = Path(__file__).resolve().parent.parent / "results" / "hf_models.json"
FLAG = ("medium", "high", "critical")


def group(dtype: str, s3_origin: str | None, structured: bool) -> str:
    if dtype != "F32":
        return f"{dtype} file"
    if s3_origin:
        return f"F32 file, {s3_origin} precision"
    return "F32, low-bit signature" if structured else "F32, fair-coin low bits"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    models = json.loads(LIST.read_text(encoding="utf-8"))["models"][: args.limit]

    survey = Run("E0b", list=str(LIST.name), n_models=len(models))
    attack = Run("E1b", list=str(LIST.name), n_models=len(models))
    rng = np.random.default_rng(0)
    for i, m in enumerate(models, 1):
        repo = m["repo"]
        try:
            samples, info = hf.sample(repo, m["file"])
        except Exception as e:
            survey.record(repo=repo, dtype=m["dtype"], error=f"{type(e).__name__}: {e}"[:300])
            print(f"[{i}/{len(models)}] {repo}: ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        if not samples:
            survey.record(repo=repo, dtype=m["dtype"], error="no float tensors large enough to sample")
            continue
        try:
            dt = max({t.dtype for t in samples.values()}, key=lambda d: sum(t.data.size for t in samples.values() if t.dtype == d))
            s3 = s3_precision_history(samples)
            s4 = s4_bit_planes(samples)
            origin = next((f.detail["origin"] for f in s3 if f.code == "precision-origin"), None)
            sig = next((f.detail["signature"] for f in s4 if f.code == "lowbit-signature" and f.detail["dtype"] == dt), None)
            fps = [f for f in s3 + s4 if f.severity in FLAG]
            g = group(dt, origin, sig is not None)
            survey.record(repo=repo, dtype=dt, group=g, n_sampled=len(samples),
                          sampled_bytes=info["sampled_bytes"], s3_origin=origin,
                          signature=sig, false_positives=len(fps),
                          fp_tensors=sorted({f.detail.get("tensor") for f in fps if f.detail.get("tensor")}),
                          fp_codes=sorted({f.code for f in fps}))

            # The payload goes into the largest tensor the clean scan did not flag:
            # a detection on an already-flagged tensor would prove nothing.
            flagged = {f.detail.get("tensor") for f in fps}
            pool = [n for n in samples if n not in flagged] or list(samples)
            target = max(pool, key=lambda n: (samples[n].data.size, n))
            target_dirty = target in flagged
            ks = [1, 4, 8] if samples[target].dtype == "F32" else [1, 4]
            hits = []
            for k in ks:
                size = capacity(samples, k, names=[target])
                poisoned = embed(samples, rng.bytes(size), k=k, names=[target])
                found = [f for f in s3_precision_history(poisoned) + s4_bit_planes(poisoned)
                         if f.severity in FLAG and f.detail.get("tensor") == target]
                det3 = any(f.check == "S3" for f in found)
                det4 = any(f.check == "S4" for f in found)
                attack.record(repo=repo, dtype=dt, group=g, k=k, target=target,
                              payload_bytes=size, target_flagged_clean=target_dirty,
                              s3_detected=det3, s4_detected=det4, detected=det3 or det4)
                hits.append(f"k{k}:{'Y' if det3 or det4 else 'n'}")
            print(f"[{i}/{len(models)}] {repo}: {g}; fp={len(fps)}; payload {' '.join(hits)}", flush=True)
        except Exception as e:
            traceback.print_exc()
            survey.record(repo=repo, dtype=m["dtype"], error=f"analysis {type(e).__name__}: {e}"[:300])
    print(f"recorded runs {survey.run_id} and {attack.run_id}")


if __name__ == "__main__":
    main()
