"""Imagenette: a 10-class subset of ImageNet with genuine ImageNet labels.

Used for E2 (accuracy cost). The ImageNet-trained models predict over all 1000
classes; we score top-1 against the true ImageNet index of each image. The ten
WNID -> ImageNet-index mappings are the standard torchvision ordering and are
fixed here so the eval needs no extra download.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARCHIVE = DATA / "imagenette2-160.tgz"
EXTRACT = DATA / "imagenette2-160"

# WNID -> ImageNet class index (torchvision IMAGENET1K ordering)
WNID_TO_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def ensure_extracted() -> Path:
    if EXTRACT.is_dir():
        return EXTRACT
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"{ARCHIVE} not found; run the download step first")
    with tarfile.open(ARCHIVE) as tf:
        tf.extractall(DATA, filter="data")  # the archive is from a trusted source (fast.ai)
    return EXTRACT


def val_samples(limit: int | None = None, seed: int = 0) -> list[tuple[Path, int]]:
    """(image path, true ImageNet index) for the validation split, shuffled
    deterministically and capped at `limit`."""
    import random
    root = ensure_extracted() / "val"
    samples = []
    for wnid, idx in sorted(WNID_TO_INDEX.items()):
        for img in sorted((root / wnid).glob("*.JPEG")):
            samples.append((img, idx))
    random.Random(seed).shuffle(samples)
    return samples[:limit] if limit else samples


def accuracy(model, transform, samples, log_every: int = 0) -> tuple[float, int]:
    """Top-1 accuracy of `model` over `samples`. Forward passes only."""
    import torch
    from PIL import Image
    correct = 0
    with torch.no_grad():
        for i, (path, target) in enumerate(samples):
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0)
            pred = int(model(x).argmax(1).item())
            correct += (pred == target)
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1}/{len(samples)} acc={correct / (i + 1):.3f}", flush=True)
    n = len(samples)
    return (correct / n if n else 0.0), n
