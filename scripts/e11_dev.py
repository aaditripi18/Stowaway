"""E11 dev (PLAN.md, phase 4): choose the windowed S4 rule (S4w) on the E10 A whole files.

Reads the window counts scripts/run_e11_record.py wrote (data/e11/e11/*.npz). For a window
size W = m x 32,768 (consecutive recorded windows merged) and a threshold Z, a tensor of at
least three windows is flagged when one window's low bit plane is nearer a fair coin than the
tensor's own windows are, by more than Z (S4's statistic: the reference is the median of the
tensor's windows, its spread tau the MAD beyond sampling noise; S4c's 0.02 minimum signature
applies). Quantisation scales and embedding tables are not windowed (see windowed()). Reported per (m, Z): clean files with a flagged window, planted cells caught, and the
E10 A cells the whole file missed but the sample caught (H11's ten differences).

Writes results/e11_dev.json.

    python scripts/e11_dev.py [data/e11/e11]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MS = (1, 2, 4, 8)
ZS = (6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0)
MIN_SIGNATURE = 0.02


def windowed(name: str) -> bool:
    """S4w skips quantisation metadata (as S4 does) and embedding tables, whose windows are rows
    with different training histories (positions past the training length, unseen tokens)."""
    n = name.lower()
    return "scale" not in n and "embed" not in n


def merge(n: np.ndarray, ones: np.ndarray, m: int):
    k = (len(n) // m) * m
    return n[:k].reshape(-1, m).sum(1), ones[:k].reshape(-1, m, ones.shape[1]).sum(1)


def zmax(n: np.ndarray, ones: np.ndarray, m: int) -> float:
    """The largest 'nearer a fair coin than its own windows' z of any window and bit."""
    n, ones = merge(n, ones, m)
    keep = n > 0
    n, ones = n[keep], ones[keep]
    if len(n) < 3:
        return -np.inf
    p = ones / n[:, None]
    sig = np.median(p, axis=0)
    mad = 1.4826 * np.median(np.abs(p - sig), axis=0)
    sampling = np.median(sig * (1 - sig) / n[:, None], axis=0)
    tau = np.sqrt(np.maximum(0.0, mad ** 2 - sampling))
    se = np.sqrt(np.maximum(sig * (1 - sig), 1e-6) / n[:, None])
    z = (np.abs(sig - 0.5) - np.abs(p - 0.5)) / np.sqrt(se ** 2 + tau ** 2)
    z = np.where(np.abs(sig - 0.5) >= MIN_SIGNATURE, z, -np.inf)
    return float(z.max())


def main(src: str) -> None:
    ledger = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    e10 = {(r["repo"], r["k"]): r for r in ledger if r.get("experiment") == "E10a" and r.get("kind") == "cell"}
    clean, cells = {}, []
    for f in sorted(Path(src).glob("*.npz")):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        repo = meta["repo"]
        tested = [k for k, t in meta["tensors"].items() if windowed(t["name"])]
        clean[repo] = {m: max((zmax(z[k + "_n"], z[k + "_ones"], m) for k in tested), default=-np.inf) for m in MS}
        target_key = next((k for k, t in meta["tensors"].items() if t["name"] == meta["target"]), None)
        for kk in meta["planted"]:
            if not windowed(meta["target"]):
                cells.append({"repo": repo, "k": int(kk[1:]), "sample": e10.get((repo, int(kk[1:])), {}).get("detected_sample"),
                              "whole_s4": e10.get((repo, int(kk[1:])), {}).get("detected_whole"),
                              "z": {m: -np.inf for m in MS}, "clean_target_z": None, "not_windowed": True})
                continue
            k = int(kk[1:])
            e = e10.get((repo, k), {})
            cells.append({"repo": repo, "k": k, "sample": e.get("detected_sample"), "whole_s4": e.get("detected_whole"),
                          "z": {m: zmax(z[kk + "_n"], z[kk + "_ones"], m) for m in MS},
                          "clean_target_z": {m: zmax(z[target_key + "_n"], z[target_key + "_ones"], m) for m in MS}
                          if target_key else None})
    grid = {}
    for m in MS:
        for Z in ZS:
            fp = sorted(r for r, v in clean.items() if v[m] > Z)
            caught = [c for c in cells if c["z"][m] > Z]
            h11 = [c for c in cells if c["sample"] and not c["whole_s4"]]
            grid[f"m{m}_z{Z:g}"] = {"clean_files_flagged": len(fp), "flagged": fp[:10],
                                    "planted_caught": len(caught), "planted": len(cells),
                                    "h11_recovered": sum(c["z"][m] > Z for c in h11), "h11": len(h11),
                                    "whole_or_window_vs_sample_agree": sum(
                                        bool(c["sample"]) == (bool(c["whole_s4"]) or c["z"][m] > Z) for c in cells)}
    out = {"files": len(clean), "cells": len(cells), "grid": grid,
           "clean_zmax": {r: {m: round(v[m], 2) for m in MS} for r, v in clean.items()},
           "cell_rows": [{**c, "z": {m: round(v, 2) for m, v in c["z"].items()}} for c in cells]}
    (ROOT / "results" / "e11_dev.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    for key, g in grid.items():
        print(key, {k: v for k, v in g.items() if k != "flagged"})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "e11" / "e11"))
