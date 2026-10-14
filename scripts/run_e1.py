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


def detected(rep: dict, checks: set[str]) -> bool:
    return any(x["check"] in checks and x["severity"] in ("medium", "high", "critical")
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

    for k in args.k:
        cap = capacity(clean, k, args.spread)
        for size in args.sizes:
            if size > cap:
                continue
            poisoned = embed(clean, rng.bytes(size), k=k, spread=args.spread)
            rep = corpus.scan_raw(poisoned, base_raw=clean)
            run.record(model=args.model, k=k, spread=args.spread, payload_bytes=size,
                       capacity_bytes=cap,
                       single_file_detected=detected(rep, {"S3", "S4"}),
                       base_detected=detected(rep, {"S5"}),
                       verdict=rep["verdict"])
            print(f"k={k} size={size:>8} single={detected(rep, {'S3','S4'})!s:5} "
                  f"base={detected(rep, {'S5'})}")
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
