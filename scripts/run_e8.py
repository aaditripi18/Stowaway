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

from experiments import hf  # noqa: E402
from experiments.ledger import Run, load  # noqa: E402
from scripts.run_e4b import change_stats, fetch  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.bits import FLOAT_LAYOUT, RawTensor  # noqa: E402
from stowaway.model import ITEMSIZE  # noqa: E402
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


def lineage(target: dict, base: dict) -> dict:
    """How the target's float tensors relate to a candidate base (S5's alignment)."""
    pairing, _, _ = align({n: (t.dtype, tuple(t.shape)) for n, t in target.items()},
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
    cos = [cosine(target[n], base[bn]) for n, bn in pairing.items()]
    return {"pairing": pairing, "identical": identical, "low_only": low_only, "above": above,
            "match": len(pairing) / max(len(target), 1),
            "median_cosine": float(np.nanmedian(cos)) if cos else None}


# --- E8b: base discovery (PLAN.md amendment) ---------------------------------

CANDIDATES = 20
DIMS = ("hidden_size", "num_hidden_layers", "vocab_size")
SCREEN_TENSORS = 12
MIN_COSINE = 0.9  # amendment 2: median cosine of the screened tensors
DEV = {"BAAI/bge-reranker-base", "ProsusAI/finbert", "SamLowe/roberta-base-go_emotions"}


def config_of(repo: str) -> dict | None:
    try:
        with hub._open(hf.resolve_url(repo, "config.json")) as r:
            return json.loads(r.read())
    except Exception:  # no config, gated, network: not a candidate
        return None


def candidates(repo: str, cfg: dict) -> list[str]:
    """The most-downloaded hub models with the same model_type and dimensions."""
    mt = cfg.get("model_type")
    if not mt:
        return []
    with hub._open(f"{hf.API}?filter={mt}&sort=downloads&direction=-1&limit=100") as r:
        listed = [m["id"] for m in json.loads(r.read())]
    out = []
    for c in listed:
        if c == repo:
            continue
        cc = config_of(c)
        if cc and cc.get("model_type") == mt and all(cc.get(d) == cfg.get(d) for d in DIMS):
            out.append(c)
        if len(out) == CANDIDATES:
            break
    return out


def _grab(url: str, start: int, header: dict, name: str) -> RawTensor:
    """The first 32 KB of one tensor, by HTTP range request."""
    spec = header[name]
    isz = ITEMSIZE[spec["dtype"]]
    b, e = spec["data_offsets"]
    length = min(e - b, 32 * 1024) // isz * isz
    data = hf.http_fetch(url, start + b, start + b + length - 1)
    return RawTensor.from_bytes(spec["dtype"], (length // isz,), data)


def values(t: RawTensor) -> np.ndarray:
    """A float tensor's values as float64, from its raw bits."""
    if t.dtype == "BF16":
        return (t.data.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return t.data.view({"F32": np.float32, "F16": np.float16, "F64": np.float64}[t.dtype]).astype(np.float64)


def cosine(a: RawTensor, b: RawTensor) -> float:
    m = min(a.data.size, b.data.size)
    x, y = values(a)[:m], values(b)[:m]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else float("nan")


def screen(target_url: str, target_header: tuple, cand: str) -> dict | None:
    """Lineage check on headers plus the first 32 KB of up to 12 matched tensors."""
    try:
        cand_url = hf.resolve_url(cand, hf.weights_file_for(cand))
        cstart, cheader = hf.read_header(cand_url, hf.http_fetch)
    except Exception:  # no top-level safetensors, gated, network: not screenable
        return None
    tstart, theader = target_header

    def shapes(h):
        return {n: (v["dtype"], tuple(v["shape"])) for n, v in h.items()
                if n != "__metadata__" and v["dtype"] in FLOAT_LAYOUT}

    tsh = shapes(theader)
    pairing, _, _ = align(tsh, shapes(cheader))
    match = len(pairing) / max(len(tsh), 1)
    if match < VERIFIED_MATCH:
        return {"candidate": cand, "match": match, "close": None}
    big = [n for n in sorted(pairing) if int(np.prod(tsh[n][1])) >= 4096]
    names = hf._spaced(big, SCREEN_TENSORS)
    cos = [cosine(_grab(target_url, tstart, theader, n), _grab(cand_url, cstart, cheader, pairing[n]))
           for n in names]
    med = float(np.nanmedian(cos)) if cos else None
    return {"candidate": cand, "match": match, "close": med, "screened": len(names)}


def discover(repo: str, file: str) -> tuple[str | None, list[dict]]:
    cfg = config_of(repo)
    if cfg is None:
        return None, []
    url = hf.resolve_url(repo, file)
    header = hf.read_header(url, hf.http_fetch)
    results = [r for r in (screen(url, header, c) for c in candidates(repo, cfg)) if r]
    ok = [r for r in results if r["close"] is not None and r["close"] >= MIN_COSINE]
    best = max(ok, key=lambda r: (r["close"], r["match"]))["candidate"] if ok else None
    return best, results


def test_pair(run, rng, repo, base_repo, route, declared, file) -> bool:
    """Full lineage check and, if verified, honest + payload S5 on whole files."""
    try:
        bres, base = fetch(base_repo)
        tres, target = fetch(repo, file)
    except hub.HubError as e:
        print(f"  {route} base not usable: {e}", flush=True)
        run.record(kind="model", repo=repo, route=route, declared=declared, base=base_repo,
                   status="base_unavailable", error=str(e))
        return False
    lin = lineage(target, base)
    verified = lin["match"] >= VERIFIED_MATCH
    common = dict(repo=repo, split="dev" if repo in DEV else "held_out", route=route, declared=declared,
                  base=base_repo, base_commit=bres.commit,
                  target_commit=tres.commit)
    run.record(kind="model", status="verified" if verified else "unverified", n_target=len(target),
               n_base=len(base), matched=len(lin["pairing"]), match_share=lin["match"],
               identical=len(lin["identical"]), low_only=len(lin["low_only"]),
               changed_above=len(lin["above"]), median_cosine=lin["median_cosine"], **common)
    print(f"  {route} base {base_repo}: {len(lin['pairing'])}/{len(target)} matched ({lin['match']:.0%}), "
          f"{len(lin['identical'])} identical, {len(lin['above'])} changed", flush=True)
    if not verified:
        return False
    honest = flagged(s5_differential(target, base))
    run.record(kind="honest", flags=sorted(honest), n_flags=len(honest), **common)
    cases = []
    if lin["identical"]:
        cases.append(("a_base_identical", max(lin["identical"], key=lambda n: target[n].data.size)))
    trained = lin["low_only"] + lin["above"]
    if trained:
        cases.append(("b_trained", max(trained, key=lambda n: target[n].data.size)))
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
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="first N models (smoke test; not recorded)")
    a = ap.parse_args(argv)
    models = blind_models()
    print(f"{len(models)} single-file-blind models", flush=True)
    if a.only:
        models = models[: a.only]
    run = Run("E8", list=LIST, group=BLIND, verified_match=VERIFIED_MATCH, payload=PAYLOAD, k=K,
              candidates=CANDIDATES, dims=list(DIMS), min_cosine=MIN_COSINE, dev=sorted(DEV))
    if a.only:
        run.record = lambda **_: None
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        repo = m["repo"]
        print(f"[{i + 1}/{len(models)}] {repo}", flush=True)
        try:
            declared = list(hub.resolve("hf:" + repo).base_model)
        except hub.HubError as e:
            run.record(kind="model", repo=repo, route="declared", declared=[], status="target_unavailable",
                       error=str(e))
            continue
        if declared and test_pair(run, rng, repo, declared[0], "declared", declared, m["file"]):
            continue
        if not declared:
            print("  no base_model declared", flush=True)
            run.record(kind="model", repo=repo, split="dev" if repo in DEV else "held_out", route="declared",
                       declared=[], status="no_declared_base")
        best, screened = discover(repo, m["file"])
        run.record(kind="discovery", repo=repo, split="dev" if repo in DEV else "held_out", found=best,
                   screened=screened)
        print(f"  discovery: {len(screened)} candidates screened -> {best or 'none'}", flush=True)
        if best:
            test_pair(run, rng, repo, best, "discovered", declared, m["file"])
    print("run", run.run_id)


if __name__ == "__main__":
    main()
