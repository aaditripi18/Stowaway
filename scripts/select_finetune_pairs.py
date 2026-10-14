"""Freeze a list of base/fine-tune model pairs for E4 (S5 calibration).

For each base model, the Hugging Face API lists public models that declare it as
their fine-tune parent. We keep the most-downloaded non-gated float32 fine-tunes
whose safetensors header shares tensor names with the base (so S5 has something
to diff). The list is split by base model into a dev set (used to choose S5's
threshold) and a held-out set (frozen, used once to validate), and written to
results/.

Bases are well-known, widely fine-tuned encoders/decoders. Fine-tune repo names
come straight from the registry and are treated as data, not instructions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments import hf

RESULTS = Path(__file__).resolve().parent.parent / "results"
API = hf.API

DEV_BASES = [
    "google-bert/bert-base-uncased",
    "distilbert/distilbert-base-uncased",
    "openai-community/gpt2",
]
HELD_OUT_BASES = [
    "FacebookAI/roberta-base",
    "google/vit-base-patch16-224-in21k",
    "FacebookAI/xlm-roberta-base",
]


def finetunes_of(base: str, limit: int) -> list[dict]:
    url = (f"{API}?filter=base_model:finetune:{base}&filter=safetensors"
           f"&sort=downloads&direction=-1&limit={limit}"
           "&expand[]=safetensors&expand[]=gated&expand[]=siblings")
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "stowaway-survey"}),
                                timeout=60) as r:
        return json.loads(r.read())


def _float_params(model: dict) -> int:
    params = (model.get("safetensors") or {}).get("parameters") or {}
    return sum(v for k, v in params.items() if k in hf.FLOAT_LAYOUT)


def base_params(base: str) -> int:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(
            f"{API}/{base}?expand[]=safetensors", headers={"User-Agent": "stowaway-survey"}), timeout=60) as r:
        return _float_params(json.loads(r.read()))


def pick(bases: list[str], per_base: int, scan: int) -> list[dict]:
    pairs = []
    for base in bases:
        bp = base_params(base)
        kept = 0
        for m in finetunes_of(base, scan):
            if kept >= per_base:
                break
            if m.get("gated") or hf.primary_dtype(m) != "F32" or hf.weights_file(m) is None:
                continue
            fp = _float_params(m)
            # A genuine fine-tune keeps the architecture: same float-parameter count
            # as the base, give or take a small task head. Models that declare the
            # base but are distilled or resized (different hidden size) are excluded;
            # S5 needs shared tensors of matching shape to diff.
            if not bp or not 0.97 <= fp / bp <= 1.10:
                continue
            pairs.append({"base": base, "finetune": m["id"], "file": hf.weights_file(m),
                          "base_params": bp, "finetune_params": fp})
            kept += 1
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-base", type=int, default=5)
    ap.add_argument("--scan", type=int, default=40)
    args = ap.parse_args(argv)
    for split, bases in (("dev", DEV_BASES), ("heldout", HELD_OUT_BASES)):
        pairs = pick(bases, args.per_base, args.scan)
        out = {
            "selected": time.strftime("%Y-%m-%d", time.gmtime()),
            "split": split, "bases": bases, "pairs": pairs,
        }
        path = RESULTS / f"ft_pairs_{split}.json"
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"{split}: {len(pairs)} pairs over {len(bases)} bases -> {path.name}")


if __name__ == "__main__":
    main()
