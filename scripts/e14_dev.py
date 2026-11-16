"""E14 dev (PLAN.md, phase 5): what separates a clean tensor whole files flag from a payload host.

Whole files flag clean tensors in 8 models already read (E10 A: 5 of 67, E11: 3 of 65). In the
ones inspected, the flagged tensors really differ from the rest of their file: tensors that were
never trained (still at their random initialisation) or a whole component kept at full float32.
A payload goes into the low bits of a tensor the model uses, which training shaped.

For every float tensor of these 8 whole files this records whether S3/S4/S4w flags it and
statistics of its values read at the file's own precision (low bits the payload would use are
cut first, so a payload cannot change them): how close they are to an i.i.d. draw from the
usual initialisations (normal, uniform), how uneven its rows are, and how much of its component
(name prefix) is flagged. A rule is chosen here; it is judged on a fresh list.

Writes results/e14_dev.json; nothing goes in the ledger.

    python scripts/e14_dev.py [--repos ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stowaway import hub  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT  # noqa: E402
from stowaway.formats import open_model  # noqa: E402
from stowaway.weights import (MIN_ELEMENTS, _load_floats, s3_precision_history, s4_bit_planes,  # noqa: E402
                              s4_windowed)

DEV = [("google-t5/t5-small", "model.safetensors"), ("FacebookAI/xlm-roberta-base", "model.safetensors"),
       ("openai/clip-vit-large-patch14", "model.safetensors"), ("facebook/esm2_t33_650M_UR50D", "model.safetensors"),
       ("Qwen/Qwen2.5-VL-7B-Instruct", "model-00001-of-00005.safetensors"),
       ("MIT/ast-finetuned-audioset-10-10-0.4593", "model.safetensors"),
       ("Alibaba-NLP/gte-large-en-v1.5", "model.safetensors"),
       ("cyankiwi/Ornith-1.0-9B-AWQ-INT4", "model-00001-of-00002.safetensors")]
SAMPLE = 200_000  # elements per tensor for the distribution statistics (seeded)
OUT = ROOT / "results" / "e14_dev.json"


def component(name: str) -> str:
    """The tensor's component: its name up to the first numbered layer (text_model.encoder, ...)."""
    parts = []
    for p in name.split("."):
        if p.isdigit():
            break
        parts.append(p)
    return ".".join(parts[:3])


def ks(x: np.ndarray, cdf) -> float:
    x = np.sort(x)
    n = len(x)
    f = cdf(x)
    return float(max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n)))


def norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF (Abramowitz and Stegun 7.1.26, |error| < 1.5e-7), vectorised."""
    x = np.abs(z) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return 0.5 * (1 + np.sign(z) * y)


def stats(name: str, t, keep_low: int, rng) -> dict:
    """Value statistics after cutting the low `keep_low` mantissa bits."""
    v = t.data
    if keep_low:
        v = v & ~np.array((1 << keep_low) - 1, dtype=v.dtype)
    full = type(t)(t.dtype, t.shape, v).to_float32().astype(np.float64).ravel()
    vals = full[np.isfinite(full)]
    if vals.size < 2:
        return {}
    s = vals if vals.size <= SAMPLE else rng.choice(vals, SAMPLE, replace=False)
    mu, sd = float(s.mean()), float(s.std())
    out = {"mean_over_std": mu / sd if sd else None, "std": sd}
    if sd > 0:
        zs = (s - mu) / sd
        out["kurtosis"] = float(np.mean(zs ** 4) - 3)
        out["ks_normal"] = ks(s, lambda q: norm_cdf((q - mu) / sd))
        lo, hi = float(s.min()), float(s.max())
        out["ks_uniform"] = ks(s, lambda q: np.clip((q - lo) / (hi - lo), 0, 1)) if hi > lo else None
        out["distinct_share"] = float(len(np.unique(s)) / len(s))
    shape = tuple(t.shape)
    if len(shape) == 2 and shape[0] >= 8 and shape[1] >= 8:
        m = full.reshape(shape) if vals.size == full.size else None
        if m is not None:
            rs = m.std(axis=1)
            expected = 1 / np.sqrt(2 * (shape[1] - 1))  # CV of row stds for i.i.d. normal rows
            out["row_cv_ratio"] = float(rs.std() / rs.mean() / expected) if rs.mean() > 0 else None
            rm = m.mean(axis=1)
            out["row_mean_ratio"] = float(rm.std() / (m.std() / np.sqrt(shape[1])))  # 1 for i.i.d.
    return out


def analyse(repo: str, file: str) -> dict:
    res = hub.resolve("hf:" + repo)
    res = hub.Resolved(res.repo, res.revision, res.commit, [file], {file: res.sizes[file]},
                       {file: res.sha256[file]}, res.base_model)
    (path,) = hub.download(res, log=lambda m: None)
    with open(path, "rb") as f:
        floats = _load_floats(open_model(path, f), f)
    shapes = {n: tuple(t.shape) for n, t in floats.items()}
    s3 = s3_precision_history(floats)
    s4 = s4_bit_planes(floats, shapes=shapes)
    s4w = s4_windowed(floats)
    origin = next((f.detail for f in s3 if f.code == "precision-origin"), None)
    flags: dict[str, list[str]] = {}
    for f in s3 + s4 + s4w:
        if f.severity in ("high", "medium") and f.detail.get("tensor"):
            flags.setdefault(f.detail["tensor"], []).append(f"{f.check}:{f.code}")
    comp_n, comp_f = {}, {}
    for n in floats:
        c = component(n)
        comp_n[c] = comp_n.get(c, 0) + 1
        comp_f[c] = comp_f.get(c, 0) + (n in flags)
    rng = np.random.default_rng(14)
    rows = []
    for n, t in sorted(floats.items()):
        if t.data.size < MIN_ELEMENTS:
            continue
        mant = FLOAT_LAYOUT[t.dtype][1]
        keep_low = origin["exact_low_bits"] if origin and t.dtype == "F32" else round(mant * 8 / 23)
        rows.append({"tensor": n, "dtype": t.dtype, "shape": list(t.shape), "n": int(t.data.size),
                     "flags": flags.get(n, []), "component": component(n),
                     "component_flagged": comp_f[component(n)] / comp_n[component(n)],
                     "component_tensors": comp_n[component(n)], **stats(n, t, keep_low, rng)})
    del floats
    Path(path).unlink(missing_ok=True)
    return {"repo": repo, "file": file, "commit": res.commit, "origin": origin["origin"] if origin else None,
            "flagged": len(flags), "tensors": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+")
    a = ap.parse_args(argv)
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"models": []}
    have = {m["repo"] for m in done["models"]}
    for repo, file in DEV:
        if (a.repos and repo not in a.repos) or repo in have:
            continue
        try:
            m = analyse(repo, file)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            continue
        done["models"].append(m)
        OUT.write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
        print(f"{repo}: {m['flagged']} flagged of {len(m['tensors'])} tensors", flush=True)


if __name__ == "__main__":
    main()
