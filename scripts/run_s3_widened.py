"""E1c: S3 on real bf16 weights saved as float32 (the `model.float()` upload).

Many models are trained in bf16 and then uploaded as float32. Widening is exact,
so the low 16 bits of every value are zero, and S3 should notice anything
written there. This takes the cached samples of every BF16 model from the
survey (run scripts/run_hf_survey.py first), widens them to float32 the way
torch does, and checks:

  clean     S3 reports bfloat16 precision and no anomaly (false positives)
  payload   random bytes in the low k bits of one tensor are flagged by S3

No network: it reads only the survey cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import hf
from experiments.ledger import Run
from stowaway.bits import RawTensor
from stowaway.embed import capacity, embed
from stowaway.weights import s3_precision_history

LIST = Path(__file__).resolve().parent.parent / "results" / "hf_models.json"
FLAG = ("medium", "high", "critical")


def widen(samples: dict[str, RawTensor]) -> dict[str, RawTensor]:
    """bf16 -> float32 exactly as torch's .float(): the bits move to the top half."""
    return {n: RawTensor("F32", t.shape, (t.data.astype(np.uint32) << 16))
            for n, t in samples.items() if t.dtype == "BF16"}


def main():
    models = [m for m in json.loads(LIST.read_text(encoding="utf-8"))["models"] if m["dtype"] == "BF16"]
    run = Run("E1c", n_models=len(models))
    rng = np.random.default_rng(0)
    for m in models:
        try:
            samples, _ = hf.sample(m["repo"], m["file"])  # served from the survey cache
        except Exception as e:
            print(f"{m['repo']}: not in cache ({type(e).__name__})")
            continue
        wide = widen(samples)
        if len(wide) < 3:
            print(f"{m['repo']}: fewer than 3 BF16 tensors sampled, skipped")
            continue
        clean = s3_precision_history(wide)
        origin = next((f.detail["origin"] for f in clean if f.code == "precision-origin"), None)
        fps = [f for f in clean if f.severity in FLAG]
        target = max(wide, key=lambda n: (wide[n].data.size, n))
        row = {"repo": m["repo"], "n_sampled": len(wide), "s3_origin": origin, "false_positives": len(fps)}
        for k in (1, 4, 8):
            # also a small payload: 1 KB, not the whole tensor
            for label, size in (("1KB", 1024), ("full", capacity(wide, k, names=[target]))):
                size = min(size, capacity(wide, k, names=[target]))
                found = [f for f in s3_precision_history(embed(wide, rng.bytes(size), k=k, names=[target]))
                         if f.severity in FLAG and f.detail.get("tensor") == target]
                run.record(**row, k=k, payload=label, payload_bytes=size, target=target, detected=bool(found))
                row_hit = "Y" if found else "n"
                print(f"{m['repo']}: origin={origin} fp={len(fps)} k={k} {label}:{row_hit}")
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
