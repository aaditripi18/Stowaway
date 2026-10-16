"""Re-score E6's single-file detection with the current checks (no inference).

E6 recorded single-file detection with the S4 of its run. The payloads are
deterministic (seeded generators, fixed loop order), so the same planted models
can be rebuilt and re-scanned after a detector change, e.g. S4c (E0d). Writes
one "E6s" ledger row per cell; the report prefers E6s over E6 for single-file
detection when both exist. Accuracy and agreement are E6's (they do not depend
on the detectors).

    python scripts/rescore_e6.py --models resnet18 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import corpus  # noqa: E402
from experiments.ledger import Run  # noqa: E402
from scripts.run_e1 import detected  # noqa: E402
from scripts.run_e6 import KINDS, KS, SEEDS, SIZES, payload  # noqa: E402
from stowaway import weights  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(corpus.VISION))
    a = ap.parse_args(argv)
    for name in a.models:
        model, _ = corpus.load_vision(name)
        clean = corpus.state_to_raw(model.state_dict())
        run = Run("E6s", model=name, s4_min_signature=weights.S4_MIN_SIGNATURE, s5_rule=weights.S5_RULE)
        clean_rep = corpus.scan_raw(clean)
        clean_flagged = {x["detail"].get("tensor") for x in clean_rep["findings"]
                         if x["severity"] in ("medium", "high", "critical")}
        for kind in KINDS:
            for seed in (SEEDS if kind == "random" else (0,)):
                rng = np.random.default_rng(seed)  # same draws, same order as run_e6
                for k in KS:
                    cap = capacity(clean, k)
                    for size in SIZES:
                        if size > cap:
                            continue
                        poisoned = embed(clean, payload(kind, size, rng), k=k)
                        written = {n for n in poisoned if not np.array_equal(poisoned[n].data, clean[n].data)}
                        rep = corpus.scan_raw(poisoned)
                        run.record(model=name, payload=kind, seed=seed, k=k, payload_bytes=size,
                                   single_file_detected=detected(rep, {"S3", "S4"}, written, clean_flagged))
        print(f"{name}: rescored, run {run.run_id}", flush=True)


if __name__ == "__main__":
    main()
