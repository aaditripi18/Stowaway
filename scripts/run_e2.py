"""E2: what the payload costs the model.

For each payload size, embed a random payload into the low k bits, load the
poisoned weights back into the model, and measure top-1 accuracy on Imagenette.
The baseline is the clean model's accuracy on the same images. Overlaid on E1,
this answers the headline question: the largest payload detection can't see from
the file alone, and what hiding it costs.
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments import corpus, imagenette
from experiments.ledger import Run
from stowaway.embed import capacity, embed


def _accuracy(model, raw, state, weights, samples):
    poisoned_state = corpus.raw_into_state(state, raw)
    model.load_state_dict(poisoned_state)
    model.eval()
    return imagenette.accuracy(model, weights.transforms(), samples)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--spread", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[4 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20])
    ap.add_argument("--n-images", type=int, default=500)
    args = ap.parse_args(argv)

    model, weights = corpus.load_vision(args.model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    clean = corpus.state_to_raw(state)
    samples = imagenette.val_samples(limit=args.n_images)
    run = Run("E2", model=args.model, spread=args.spread, n_images=len(samples))

    base_acc = imagenette.accuracy(model, weights.transforms(), samples)[0]
    run.record(model=args.model, k=None, payload_bytes=0, accuracy=base_acc,
               accuracy_drop=0.0, n_images=len(samples))
    print(f"clean accuracy = {base_acc:.4f} on {len(samples)} images")

    rng = np.random.default_rng(0)
    for k in args.k:
        cap = capacity(clean, k, args.spread)
        for size in args.sizes:
            if size > cap:
                continue
            poisoned = embed(clean, rng.bytes(size), k=k, spread=args.spread)
            acc = _accuracy(model, poisoned, state, weights, samples)
            run.record(model=args.model, k=k, spread=args.spread, payload_bytes=size,
                       capacity_bytes=cap, accuracy=acc, accuracy_drop=base_acc - acc,
                       n_images=len(samples))
            print(f"k={k} size={size:>8} acc={acc:.4f} drop={base_acc - acc:+.4f}")
    print(f"recorded run {run.run_id}")


if __name__ == "__main__":
    main()
