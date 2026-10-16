"""E4b: S5 on whole files, and the S5b/S5c refinements (PLAN.md, phase 2).

For each base/fine-tune pair in a frozen list, download both (commit-pinned,
SHA-256 checked, via stowaway.hub), and for every tensor the two share (same
dtype and shape) record how the fine-tune changed it:

  changed     elements that differ
  p99         99th percentile of the highest changed mantissa bit (what S5 reads)
  top         the highest changed bit of any element
  top_share   share of changed elements whose highest changed bit is `top` (S5b)
  low_bin, low_share
              the bit j <= cutoff that is the highest changed bit of the most
              changed elements, and that share (S5c)

Then plant a random payload in the low k bits (k = 8, 16, 20, 21) of
  (a) the largest float tensor the fine-tune left identical to the base, and
  (b) the changed tensor with the smallest p99 (the low-gradient case),
and record the same statistics for each. Rules are applied in the report, so
every threshold can be read off the same records.

    python scripts/run_e4b.py --pairs results/ft_pairs_dev.json --split dev
    python scripts/run_e4b.py --pairs results/ft_pairs_fresh.json --split fresh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import Run  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402
from stowaway.formats import open_model  # noqa: E402
from stowaway.weights import MIN_ELEMENTS, S5_MARGIN, S5_PERCENTILE, _load_floats, align  # noqa: E402

KS = (8, 16, 20, 21)


def change_stats(a: np.ndarray, b: np.ndarray, dtype: str) -> dict | None:
    """How tensor a differs from b, bit by bit (unsigned views of the same dtype)."""
    mant = FLOAT_LAYOUT[dtype][1]
    cutoff = mant - S5_MARGIN
    diff = a != b
    changed = int(diff.sum())
    if changed < MIN_ELEMENTS:
        return {"changed": changed}
    x = (a[diff].astype(np.uint64) ^ b[diff].astype(np.uint64)) & np.uint64((1 << mant) - 1)
    hi = np.full(x.size, -1, dtype=np.int64)
    nz = x != 0
    hi[nz] = np.floor(np.log2(x[nz].astype(np.float64))).astype(np.int64)
    # exact for x < 2**53; float log2 of an exact power of two is exact, and the
    # floor of others never crosses an integer for integers this small
    hist = np.bincount(hi + 1, minlength=mant + 1)  # index = bit + 1
    top = int(hi.max())
    low = hist[: cutoff + 2]  # bins for bits -1 .. cutoff
    low_bin = int(np.argmax(low)) - 1
    return {"changed": changed, "p99": float(np.percentile(hi, S5_PERCENTILE)), "top": top,
            "top_share": float((hi == top).mean()), "low_bin": low_bin,
            "low_share": float(low[low_bin + 1] / changed), "cutoff": cutoff, "mant": mant,
            "hist": hist.tolist()}


def fetch(repo: str, file: str | None = None, tries: int = 4):
    """Resolve and download, retrying transient network failures."""
    import time
    import urllib.error
    for attempt in range(tries):
        try:
            return _fetch(repo, file)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == tries - 1:
                raise hub.HubError(f"network: {e}") from None
            print(f"  network error ({e}); retrying", flush=True)
            time.sleep(10 * (attempt + 1))


def _fetch(repo: str, file: str | None = None):
    res = hub.resolve("hf:" + repo)
    if file and file in res.files:
        res = hub.Resolved(res.repo, res.revision, res.commit, [file], {file: res.sizes[file]},
                           {file: res.sha256[file]}, res.base_model)
    paths = hub.download(res, log=lambda m: print("  " + m, flush=True))
    floats = {}
    for p in paths:
        with open(p, "rb") as f:
            floats.update(_load_floats(open_model(p, f), f))
    return res, floats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--only", type=int, default=None, help="first N pairs (smoke test; not recorded)")
    ap.add_argument("--missing", action="store_true",
                    help="only the pairs no earlier run of this split read (e.g. after a network failure)")
    a = ap.parse_args(argv)
    pairs = json.loads(Path(a.pairs).read_text(encoding="utf-8"))["pairs"]
    if a.missing:
        from experiments.ledger import load
        done = {r["finetune"] for r in load() if r["experiment"] == "E4b" and r.get("kind") == "pair"
                and r.get("meta", {}).get("split") == a.split}
        pairs = [p for p in pairs if p["finetune"] not in done]
        print(f"{len(pairs)} pair(s) not yet read", flush=True)
    if a.only:
        pairs = pairs[: a.only]
    run = Run("E4b", split=a.split, pairs=a.pairs, ks=list(KS))
    if a.only:
        run.record = lambda **_: None
    rng = np.random.default_rng(0)
    bases = {}
    for i, pair in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] {pair['finetune']}  vs  {pair['base']}", flush=True)
        try:
            if pair["base"] not in bases:
                bases[pair["base"]] = fetch(pair["base"])
            bres, base = bases[pair["base"]]
            fres, ft = fetch(pair["finetune"], pair.get("file"))
        except hub.HubError as e:
            print(f"  skipped: {e}", flush=True)
            run.record(kind="error", base=pair["base"], finetune=pair["finetune"], error=str(e))
            continue
        common = dict(base=pair["base"], finetune=pair["finetune"], base_commit=bres.commit, ft_commit=fres.commit)
        # pair tensors exactly as the scanner's S5 does (weights.align)
        pairing, dm, db = align({n: (t.dtype, tuple(t.shape)) for n, t in ft.items()},
                                {n: (t.dtype, tuple(t.shape)) for n, t in base.items()})
        common.update(dropped_model=dm, dropped_base=db)
        shared = sorted(pairing)
        changed, unchanged = [], []
        for n in shared:
            s = change_stats(ft[n].data, base[pairing[n]].data, ft[n].dtype)
            if s["changed"] >= MIN_ELEMENTS:
                run.record(kind="honest", tensor=n, dtype=ft[n].dtype, numel=int(ft[n].data.size), **common, **s)
                changed.append((s["p99"], n))
            elif s["changed"] == 0:
                unchanged.append((ft[n].data.size, n))
        run.record(kind="pair", shared=len(shared), changed=len(changed), unchanged=len(unchanged), **common)
        targets = []
        if unchanged:
            targets.append(("a_base_matching", max(unchanged)[1]))
        if changed:
            targets.append(("b_low_gradient", min(changed)[1]))
        for case, n in targets:
            for k in KS:
                t = {n: ft[n]}
                cap = capacity(t, k, names=[n])
                if cap <= 16:
                    continue
                poisoned = embed(t, rng.bytes(cap - 8), k, names=[n])[n]
                s = change_stats(poisoned.data, base[pairing[n]].data, ft[n].dtype)
                run.record(kind="payload", case=case, tensor=n, k=k, dtype=ft[n].dtype,
                           numel=int(ft[n].data.size), **common, **s)
        print(f"  {len(shared)} shared tensors, {len(changed)} changed, {len(unchanged)} unchanged", flush=True)
        del ft
    print("run", run.run_id)


if __name__ == "__main__":
    main()
