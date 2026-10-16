"""E6: E1/E2 widened to nine models, five payload types and three seeds (PLAN.md).

For each torchvision model, payload type, k, size and seed: write the payload
into the low k bits of the float weights (stowaway.embed, spread 1), then

  detection   S3/S4 from the planted file alone (as E1: a flag on a written
              tensor the clean scan did not flag), and the per-tensor bit
              statistics S5 reads (as E4b), so any S5 rule can be scored later
  cost        top-1 accuracy and agreement with the clean model's top-1, on the
              same 1,000 Imagenette validation images

Accuracy is measured for every random-payload cell and for every other type at
seed 0. Forward passes only; nothing is trained.

    python scripts/run_e6.py --models resnet18 mobilenet_v3_small ...
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import corpus, imagenette  # noqa: E402
from experiments.ledger import Run  # noqa: E402
from scripts.run_e1 import detected  # noqa: E402
from scripts.run_e4b import change_stats  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402

KS = (1, 4, 8, 16)
SIZES = (1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20)
SEEDS = (0, 1, 2)
N_IMAGES = 1000

# Public-domain English (Darwin, On the Origin of Species, 1859, ch. IV opening),
# repeated to the payload size: a stand-in for any plain-text payload.
TEXT = ("Owing to this struggle for life, any variation, however slight and from whatever cause "
        "proceeding, if it be in any degree profitable to an individual of any species, in its "
        "infinitely complex relations to other organic beings and to external nature, will tend to "
        "the preservation of that individual, and will generally be inherited by its offspring. ").encode()


def payload(kind: str, size: int, rng: np.random.Generator) -> bytes:
    if kind == "random":
        return rng.bytes(size)
    if kind == "zeros":
        return bytes(size)
    if kind == "text":
        return (TEXT * (size // len(TEXT) + 1))[:size]
    if kind == "compressed":
        # text with a running counter in every sentence, compressed once: what a
        # packed payload looks like (close to random, but with a zlib header)
        n = 1 + size // 20
        text = b"".join(TEXT[: 40 + (i * 7919) % (len(TEXT) - 40)] + str(i * 2654435761 % 10**9).encode()
                        for i in range(n))
        out = zlib.compress(text, 9)
        while len(out) < size:
            n *= 2
            text = b"".join(TEXT[: 40 + (i * 7919) % (len(TEXT) - 40)] + str(i * 2654435761 % 10**9).encode()
                            for i in range(n))
            out = zlib.compress(text, 9)
        return out[:size]
    if kind == "executable":
        return b"MZ" + rng.bytes(size - 2)
    raise ValueError(kind)


KINDS = ("random", "zeros", "text", "compressed", "executable")


class Eval:
    """Preprocessed images for one model, and its clean top-1 predictions."""

    def __init__(self, model, weights, n: int):
        import torch
        from PIL import Image
        self.torch = torch
        samples = imagenette.val_samples(limit=n)
        tf = weights.transforms()
        self.x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in samples])
        self.y = np.array([t for _, t in samples])
        self.clean = self.predict(model)

    def predict(self, model) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(self.x), 100):
                out.append(model(self.x[i:i + 100]).argmax(1).numpy())
        return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(corpus.VISION))
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for name in a.models:
        model, weights = corpus.load_vision(name)
        sd = model.state_dict()
        meta = getattr(sd, "_metadata", None)  # per-module versions (MNASNet's loader requires them)
        state = {k: v.clone() for k, v in sd.items()}

        def with_meta(d):
            from collections import OrderedDict
            out = OrderedDict(d)
            if meta is not None:
                out._metadata = meta
            return out
        clean = corpus.state_to_raw(state)
        ev = Eval(model, weights, N_IMAGES)
        run = Run("E6", model=name, n_images=N_IMAGES, split="dev" if name in corpus.DEV else "held_out")
        run.record(kind="clean", model=name, accuracy=float((ev.clean == ev.y).mean()))
        clean_rep = corpus.scan_raw(clean)
        clean_flagged = {x["detail"].get("tensor") for x in clean_rep["findings"]
                         if x["severity"] in ("medium", "high", "critical")}
        print(f"== {name}: clean top-1 {(ev.clean == ev.y).mean():.4f}", flush=True)
        for kind in KINDS:
            for seed in (SEEDS if kind == "random" else (0,)):
                rng = np.random.default_rng(seed)
                for k in KS:
                    cap = capacity(clean, k)
                    for size in SIZES:
                        if size > cap:
                            continue
                        poisoned = embed(clean, payload(kind, size, rng), k=k)
                        written = {n for n in poisoned if not np.array_equal(poisoned[n].data, clean[n].data)}
                        rep = corpus.scan_raw(poisoned)
                        single = detected(rep, {"S3", "S4"}, written, clean_flagged)
                        s5 = {n: change_stats(poisoned[n].data, clean[n].data, clean[n].dtype) for n in written}
                        s5 = {n: {key: v for key, v in s.items() if key != "hist"} for n, s in s5.items()}
                        rec = dict(kind="cell", model=name, payload=kind, seed=seed, k=k, payload_bytes=size,
                                   capacity_bytes=cap, tensors_written=len(written), single_file_detected=single,
                                   s5_stats=s5)
                        model.load_state_dict(with_meta(corpus.raw_into_state(state, poisoned)))
                        pred = ev.predict(model)
                        rec.update(accuracy=float((pred == ev.y).mean()), agreement=float((pred == ev.clean).mean()))
                        model.load_state_dict(with_meta(state))
                        run.record(**rec)
                        print(f"  {kind:10s} seed {seed} k={k:2d} {size:>8} single={single!s:5} "
                              f"agree={rec['agreement']:.4f}", flush=True)
        print(f"recorded {run.run_id}", flush=True)


if __name__ == "__main__":
    main()
