"""E0: false positives on clean public models.

Scans each clean model (single-file checks only, no base) and records every
finding at medium or above. A trustworthy detector produces none here; whatever
it does produce is its false-positive rate, which E1's detection numbers are
read against.
"""
from __future__ import annotations

import argparse

from experiments import corpus
from experiments.ledger import Run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(corpus.VISION))
    args = ap.parse_args(argv)

    run = Run("E0", models=args.models)
    for name in args.models:
        model, _ = corpus.load_vision(name)
        raw = corpus.state_to_raw(model.state_dict())
        rep = corpus.scan_raw(raw)
        flags = [x for x in rep["findings"] if x["severity"] in ("medium", "high", "critical")]
        by_check = {}
        for x in flags:
            by_check[x["check"]] = by_check.get(x["check"], 0) + 1
        run.record(model=name, split="dev" if name in corpus.DEV else "held_out", verdict=rep["verdict"], n_float_tensors=len(raw),
                   false_positives=len(flags), by_check=by_check,
                   codes=sorted({x["code"] for x in flags}))
        print(f"{name} [{'dev' if name in corpus.DEV else 'held out'}]: {rep['verdict']}  false positives={len(flags)}  {by_check}")
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
