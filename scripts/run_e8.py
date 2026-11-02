"""E8: can the base diff close the single-file blind spot on its own? (PLAN.md, phase 3)

For each float32 model of the frozen survey list that E1b classified as having
fair-coin low bits (the models the file alone says nothing about):

1. Does its card declare a `base_model`, and can that repository be fetched
   (non-gated, safetensors)?
2. Lineage check: align target and base as S5 does (weights.align) and record
   the share of the target's float tensors matched by name and shape, and how
   the matched tensors differ (identical / changed only in the low half of the
   mantissa / changed above it). A declared base is *verified* if >= 90% of the
   target's float tensors are matched.
3. On verified pairs, whole-file S5 (the scanner's own s5_differential) on the
   unmodified target, and on the target with a 64 KB random payload at k = 8 in
   (a) the largest float tensor identical to the base and (b) the largest
   float tensor the fine-tune changed.

    python scripts/run_e8.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import Run, load  # noqa: E402
from scripts.run_e4b import change_stats, fetch  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402
from stowaway.weights import MIN_ELEMENTS, align, s5_differential  # noqa: E402

LIST = "hf_models.json"
BLIND = "F32, fair-coin low bits"
VERIFIED_MATCH = 0.90
PAYLOAD = 64 * 1024
K = 8


def blind_models() -> list[dict]:
    """The fair-coin float32 models of the latest complete E1b run on the survey list."""
    runs = collections.OrderedDict()
    for r in load():
        if r["experiment"] == "E1b" and r.get("meta", {}).get("list") == LIST:
            runs.setdefault(r["run_id"], []).append(r)
    full = [rs for rs in runs.values() if len({r["repo"] for r in rs if r["dtype"] == "F32"}) == 39]
    if not full:
        sys.exit("no complete E1b run on the survey list")
    repos = sorted({r["repo"] for r in full[-1] if r["group"] == BLIND})
    listed = {m["repo"]: m for m in json.loads((ROOT / "results" / LIST).read_text(encoding="utf-8"))["models"]}
    return [listed[r] for r in repos]


def flagged(findings) -> set[str]:
    return {f.detail["tensor"] for f in findings if f.check == "S5" and "tensor" in f.detail}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="first N models (smoke test; not recorded)")
    a = ap.parse_args(argv)
    models = blind_models()
    print(f"{len(models)} single-file-blind models", flush=True)
    if a.only:
        models = models[: a.only]
    run = Run("E8", list=LIST, group=BLIND, verified_match=VERIFIED_MATCH, payload=PAYLOAD, k=K)
    if a.only:
        run.record = lambda **_: None
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        repo = m["repo"]
        print(f"[{i + 1}/{len(models)}] {repo}", flush=True)
        try:
            tres = hub.resolve("hf:" + repo)
        except hub.HubError as e:
            run.record(kind="model", repo=repo, declared=[], status="target_unavailable", error=str(e))
            continue
        declared = list(tres.base_model)
        if not declared:
            print("  no base_model declared", flush=True)
            run.record(kind="model", repo=repo, declared=[], status="no_declared_base")
            continue
        base_repo = declared[0]
        try:
            bres, base = fetch(base_repo)
            _, target = fetch(repo, m["file"])
        except hub.HubError as e:
            print(f"  base not usable: {e}", flush=True)
            run.record(kind="model", repo=repo, declared=declared, base=base_repo, status="base_unavailable",
                       error=str(e))
            continue
        pairing, dm, db = align({n: (t.dtype, tuple(t.shape)) for n, t in target.items()},
                                {n: (t.dtype, tuple(t.shape)) for n, t in base.items()})
        identical, low_only, above = [], [], []
        for n, bn in pairing.items():
            s = change_stats(target[n].data, base[bn].data, target[n].dtype)
            if s["changed"] == 0:
                identical.append(n)
            elif s["changed"] >= MIN_ELEMENTS and s["top"] < FLOAT_LAYOUT[target[n].dtype][1] // 2:
                low_only.append(n)
            else:
                above.append(n)
        match = len(pairing) / max(len(target), 1)
        verified = match >= VERIFIED_MATCH
        common = dict(repo=repo, declared=declared, base=base_repo, base_commit=bres.commit,
                      target_commit=tres.commit)
        run.record(kind="model", status="verified" if verified else "unverified", n_target=len(target),
                   n_base=len(base), matched=len(pairing), match_share=match, identical=len(identical),
                   low_only=len(low_only), changed_above=len(above), **common)
        print(f"  base {base_repo}: {len(pairing)}/{len(target)} matched ({match:.0%}), "
              f"{len(identical)} identical, {len(above)} changed", flush=True)
        if not verified:
            continue
        honest = flagged(s5_differential(target, base))
        run.record(kind="honest", flags=sorted(honest), n_flags=len(honest), **common)
        size = lambda n: target[n].data.size  # noqa: E731
        cases = []
        if identical:
            cases.append(("a_base_identical", max(identical, key=size)))
        trained = low_only + above
        if trained:
            cases.append(("b_trained", max(trained, key=size)))
        for case, n in cases:
            cap = capacity({n: target[n]}, K, names=[n])
            nbytes = min(PAYLOAD, cap - 8)
            if nbytes <= 16:
                run.record(kind="payload", case=case, tensor=n, status="too_small", **common)
                continue
            poisoned = dict(target)
            poisoned[n] = embed({n: target[n]}, rng.bytes(nbytes), K, names=[n])[n]
            hits = flagged(s5_differential(poisoned, base))
            run.record(kind="payload", case=case, tensor=n, payload_bytes=nbytes, caught=n in hits,
                       other_flags=sorted(hits - {n} - honest), **common)
            print(f"  {case}: {n} ({nbytes} B) -> {'caught' if n in hits else 'MISSED'}", flush=True)
        del target
    print("run", run.run_id)


if __name__ == "__main__":
    main()
