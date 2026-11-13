"""E11 fresh: the windowed S4 (S4w) on whole files of a list no rule was chosen on (PLAN.md, phase 4).

The E10 A procedure on results/hf_models_batch5.json (frozen before any of its weights was read):
for each model, the survey's 32 KB samples and the whole file (commit-pinned, SHA-256 checked),
S3/S4 on both; then a random payload in the low k bits of the sample's target tensor, as many
bytes as its sample holds, written from its first element. Added: S4w on the whole file, clean
(its findings are false positives) and planted (does it flag the target).

    python scripts/run_e11.py [--list hf_models_batch5.json] [--only N] [--dry]
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

from experiments import hf  # noqa: E402
from experiments.ledger import Run  # noqa: E402
from scripts.run_e10a import MAX_BYTES, SAMPLING, classify, detected  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402
from stowaway.formats import open_model  # noqa: E402
from stowaway.weights import S4W_WINDOW, S4W_Z, _load_floats, s4_windowed  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="hf_models_batch5.json")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="smoke test: record nothing")
    ap.add_argument("--repos", nargs="+")
    a = ap.parse_args(argv)
    models = json.loads((ROOT / "results" / a.list).read_text(encoding="utf-8"))["models"][: a.only]
    if a.repos:
        models = [m for m in models if m["repo"] in a.repos]
    run = Run("E11", list=a.list, n_models=len(models), window=S4W_WINDOW, z=S4W_Z, **SAMPLING)
    if a.dry:
        run.record = lambda **_: None
    rng = np.random.default_rng(0)
    for i, m in enumerate(models, 1):
        repo = m["repo"]
        try:
            samples, info = hf.sample(repo, m["file"], **SAMPLING)
            shapes = {n: tuple(v) for n, v in info["shapes"].items()}
            g_sample, fps_s, flagged_s = classify(samples, shapes)
            pool = [n for n in samples if n not in flagged_s] or list(samples)
            target = max(pool, key=lambda n: (samples[n].data.size, n))
            res = hub.resolve("hf:" + repo)
            res = hub.Resolved(res.repo, res.revision, res.commit, [m["file"]], {m["file"]: res.sizes[m["file"]]},
                               {m["file"]: res.sha256[m["file"]]}, res.base_model)
            if res.sizes[m["file"]] > MAX_BYTES:
                run.record(kind="model", repo=repo, status="skipped_size", bytes=res.sizes[m["file"]])
                continue
            (path,) = hub.download(res, log=lambda msg: None)
            with open(path, "rb") as f:
                whole = _load_floats(open_model(path, f), f)
            wshapes = {n: tuple(t.shape) for n, t in whole.items()}
            g_whole, fps_w, flagged_w = classify(whole, wshapes)
            w_clean = s4_windowed(whole)
            run.record(kind="model", repo=repo, commit_hf=res.commit, dtype=m["dtype"], group_sample=g_sample,
                       group_whole=g_whole, fp_sample=len(fps_s), fp_whole=len(fps_w), fp_window=len(w_clean),
                       window_flagged=[f.detail["tensor"] for f in w_clean][:20],
                       tensors_sample=len(samples), tensors_whole=len(whole), target=target)
            ks = [1, 4, 8] if samples[target].dtype == "F32" else [1, 4]
            cells = []
            for k in ks:
                size = capacity(samples, k, names=[target])
                payload = rng.bytes(size)
                det_s = detected(embed(samples, payload, k=k, names=[target]), shapes, target)
                planted = embed({target: whole[target]}, payload, k=k, names=[target])[target]
                det_w = detected({**whole, target: planted}, wshapes, target)
                det_win = any(f.detail["tensor"] == target for f in s4_windowed({target: planted}))
                run.record(kind="cell", repo=repo, k=k, target=target, payload_bytes=size, detected_sample=det_s,
                           detected_whole=det_w, detected_window=det_win,
                           detected_whole_or_window=det_w or det_win,
                           target_flagged_clean_whole=target in flagged_w,
                           target_flagged_clean_window=any(f.detail["tensor"] == target for f in w_clean))
                cells.append(f"k{k}:{'Y' if det_s else 'n'}/{'Y' if det_w else 'n'}/{'Y' if det_win else 'n'}")
            print(f"[{i}/{len(models)}] {repo}: {g_sample} | {g_whole}; window FP {len(w_clean)}; {' '.join(cells)}",
                  flush=True)
            del whole
            Path(path).unlink(missing_ok=True)
        except Exception as e:
            traceback.print_exc()
            run.record(kind="model", repo=repo, status="error", error=f"{type(e).__name__}: {e}"[:300])
    print("run", run.run_id)


if __name__ == "__main__":
    main()
