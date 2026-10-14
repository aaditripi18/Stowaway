"""E1: detection vs payload size.

For a model, embed a random (encrypted-style) payload of increasing size into the
low k bits of the float weights, then ask three detectors whether they see it:

  single : S3 + S4, from the file alone
  base   : S5, given the clean model as the claimed base

Random payloads are the hard case: their low bits are uniform, so S3/S4 cannot
separate them from natural fp32 noise. The curve this produces is the headline:
the largest payload single-file scanning misses, which E2 then prices in accuracy.
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments import corpus
from experiments.ledger import Run
from stowaway.embed import capacity, embed

SIZES = [1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20]  # 1 KB .. 1 MB


def detected(rep: dict, checks: set[str], written: set[str], clean_flagged: set[str]) -> bool:
    """A check detects the payload only if it flags a tensor the payload was written
    into and the clean scan did not already flag: a false positive elsewhere in the
    model is not a detection."""
    return any(x["check"] in checks and x["severity"] in ("medium", "high", "critical")
               and x["detail"].get("tensor") in written - clean_flagged
               for x in rep["findings"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--spread", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    args = ap.parse_args(argv)

    model, _ = corpus.load_vision(args.model)
    clean = corpus.state_to_raw(model.state_dict())
    run = Run("E1", model=args.model, spread=args.spread)
    rng = np.random.default_rng(0)
    clean_rep = corpus.scan_raw(clean)
    clean_flagged = {x["detail"].get("tensor") for x in clean_rep["findings"]
                     if x["severity"] in ("medium", "high", "critical")}

    for k in args.k:
        cap = capacity(clean, k, args.spread)
        for size in args.sizes:
            if size > cap:
                continue
            poisoned = embed(clean, rng.bytes(size), k=k, spread=args.spread)
            written = {n for n in poisoned if not np.array_equal(poisoned[n].data, clean[n].data)}
            rep = corpus.scan_raw(poisoned, base_raw=clean)
            single = detected(rep, {"S3", "S4"}, written, clean_flagged)
            base = detected(rep, {"S5"}, written, clean_flagged)
            run.record(model=args.model, k=k, spread=args.spread, payload_bytes=size,
                       capacity_bytes=cap, tensors_written=len(written),
                       clean_flagged=sorted(t for t in clean_flagged if t),
                       single_file_detected=single, base_detected=base, verdict=rep["verdict"])
            print(f"k={k} size={size:>8} tensors={len(written):>3} single={single!s:5} base={base}")
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
