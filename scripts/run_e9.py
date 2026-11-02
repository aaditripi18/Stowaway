"""E9: neutralise instead of detect: does clearing the low bits cost anything? (PLAN.md)

For each model, round every float32 weight to nearest at mantissa bit s
(stowaway.sanitize; s = 8, 12, 16) and measure prediction agreement with the
unmodified model:

  vision  the nine torchvision models of E6, top-1 on the same 1,000 Imagenette
          validation images (scripts/run_e6.Eval)
  text    distilbert/distilbert-base-uncased-finetuned-sst-2-english, labels on
          the 872 SST-2 validation sentences (GLUE, nyu-mll/glue parquet)

and check removal directly: a 64 KB random payload planted at k = 8 is not
recoverable after sanitising at s = 8 (every low bit is zero).

    python scripts/run_e9.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import corpus  # noqa: E402
from experiments.ledger import Run  # noqa: E402
from scripts.run_e6 import Eval  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.bits import RawTensor  # noqa: E402
from stowaway.embed import capacity, embed, extract  # noqa: E402
from stowaway.sanitize import round_f32_bits  # noqa: E402

BITS = (8, 12, 16)
N_IMAGES = 1000
PAYLOAD = 64 * 1024
TEXT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SST2_URL = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/sst2/validation-00000-of-00001.parquet"
DATA = ROOT / "data" / "sst2"


def sanitized(raw: dict[str, RawTensor], bits: int) -> dict[str, RawTensor]:
    return {n: RawTensor(t.dtype, t.shape, round_f32_bits(t.data, bits)) if t.dtype == "F32" else t
            for n, t in raw.items()}


def removal(raw: dict[str, RawTensor], rng) -> dict:
    """Plant a payload at k = 8, sanitise at s = 8, and look for it."""
    nbytes = min(PAYLOAD, capacity(raw, 8) - 8)
    secret = rng.bytes(nbytes)
    planted = embed(raw, secret, k=8)
    assert extract(planted, k=8)[:nbytes] == secret
    after = extract(sanitized(planted, 8), k=8)[:nbytes]
    return {"payload_bytes": nbytes, "recovered_bytes_equal": int(sum(a == b for a, b in zip(after, secret))),
            "low_bits_all_zero": not any(after)}


def vision(run, names, rng):
    for name in names:
        model, weights = corpus.load_vision(name)
        sd = model.state_dict()
        meta = getattr(sd, "_metadata", None)
        state = {k: v.clone() for k, v in sd.items()}

        def load(raw):
            from collections import OrderedDict
            d = OrderedDict(corpus.raw_into_state(state, raw))
            if meta is not None:
                d._metadata = meta
            model.load_state_dict(d)

        clean = corpus.state_to_raw(state)
        ev = Eval(model, weights, N_IMAGES)
        acc0 = float((ev.clean == ev.y).mean())
        rec = {"model": name, "modality": "vision", "n": N_IMAGES, "accuracy_clean": acc0}
        for bits in BITS:
            load(sanitized(clean, bits))
            pred = ev.predict(model)
            rec[f"agreement_{bits}"] = float((pred == ev.clean).mean())
            rec[f"accuracy_{bits}"] = float((pred == ev.y).mean())
        load(clean)
        rec.update(removal(clean, rng))
        run.record(kind="model", **rec)
        print(f"  {name}: " + "  ".join(f"s={b} agree {rec[f'agreement_{b}']:.4f}" for b in BITS), flush=True)


def sst2() -> tuple[list[str], np.ndarray, str]:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "validation.parquet"
    if not path.exists():
        with urllib.request.urlopen(urllib.request.Request(SST2_URL, headers={"User-Agent": hub.UA}),
                                    timeout=120) as r:
            path.write_bytes(r.read())
    import pandas as pd
    df = pd.read_parquet(path)
    return df["sentence"].tolist(), df["label"].to_numpy(), hashlib.sha256(path.read_bytes()).hexdigest()


def text(run, rng):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    commit = hub.resolve("hf:" + TEXT_MODEL).commit
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=commit)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL, revision=commit).eval()
    sentences, y, digest = sst2()
    enc = tok(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")

    def predict():
        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), 64):
                out.append(model(**{k: v[i:i + 64] for k, v in enc.items()}).logits.argmax(1).numpy())
        return np.concatenate(out)

    state = {k: v.clone() for k, v in model.state_dict().items()}
    clean = corpus.state_to_raw(state)
    base = predict()
    rec = {"model": TEXT_MODEL, "model_commit": commit, "modality": "text", "n": len(y),
           "data_sha256": digest, "accuracy_clean": float((base == y).mean())}
    for bits in BITS:
        model.load_state_dict(corpus.raw_into_state(state, sanitized(clean, bits)))
        pred = predict()
        rec[f"agreement_{bits}"] = float((pred == base).mean())
        rec[f"accuracy_{bits}"] = float((pred == y).mean())
    model.load_state_dict(state)
    rec.update(removal(clean, rng))
    run.record(kind="model", **rec)
    print(f"  {TEXT_MODEL}: " + "  ".join(f"s={b} agree {rec[f'agreement_{b}']:.4f}" for b in BITS), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(corpus.VISION))
    ap.add_argument("--no-text", action="store_true")
    a = ap.parse_args(argv)
    run = Run("E9", bits=list(BITS), n_images=N_IMAGES, payload=PAYLOAD, text_model=TEXT_MODEL)
    rng = np.random.default_rng(0)
    vision(run, a.models, rng)
    if not a.no_text:
        text(run, rng)
    print("run", run.run_id)


if __name__ == "__main__":
    main()
