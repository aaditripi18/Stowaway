"""E10 A: do whole files agree with the 32 KB samples the survey used? (PLAN.md)

For each model of a frozen list, with the scanner's current code:

  sample  the survey's sampling (4 layers of the 6 largest roles, first 32 KB
          each, up to 24 tensors) and its S3/S4 verdicts, exactly as E0b/E1b
  whole   the whole file downloaded (commit-pinned, SHA-256 checked) and the same
          S3/S4 run on every float tensor at full size

Both sides classify the model (half-precision origin / low-bit signature /
fair-coin) and judge the same planted payload: random bytes in the low k bits of
the sample's target tensor, as many bytes as fit in its 32 KB sample, written
from the tensor's first element, so the whole file carries the payload in exactly
the bytes the sample read. Each downloaded file is deleted after use.

    python scripts/run_e10a.py [--list hf_models.json] [--only N]
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
from scripts.run_hf_survey import FLAG, group  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402
from stowaway.formats import open_model  # noqa: E402
from stowaway.weights import _load_floats, s3_precision_history, s4_bit_planes  # noqa: E402

SAMPLING = {"chunk": 32 * 1024, "max_tensors": 24, "roles": 6}
MAX_BYTES = 6 << 30  # files larger than this are recorded as skipped (memory)


def classify(floats, shapes) -> tuple[str, list, set]:
    dt = max({t.dtype for t in floats.values()},
             key=lambda d: sum(t.data.size for t in floats.values() if t.dtype == d))
    s3 = s3_precision_history(floats)
    s4 = s4_bit_planes(floats, shapes=shapes)
    origin = next((f.detail["origin"] for f in s3 if f.code == "precision-origin"), None)
    sig = next((f.detail["signature"] for f in s4 if f.code == "lowbit-signature" and f.detail["dtype"] == dt), None)
    fps = [f for f in s3 + s4 if f.severity in FLAG]
    return group(dt, origin, sig is not None), fps, {f.detail.get("tensor") for f in fps}


def detected(floats, shapes, target) -> bool:
    found = [f for f in s3_precision_history(floats) + s4_bit_planes(floats, shapes=shapes)
             if f.severity in FLAG and f.detail.get("tensor") == target]
    return bool(found)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="hf_models.json")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="smoke test: record nothing")
    ap.add_argument("--repos", nargs="+", help="only these models of the list (e.g. ones a run lost)")
    a = ap.parse_args(argv)
    models = json.loads((ROOT / "results" / a.list).read_text(encoding="utf-8"))["models"][: a.only]
    if a.repos:
        models = [m for m in models if m["repo"] in a.repos]
    run = Run("E10a", list=a.list, n_models=len(models), **SAMPLING)
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
            rec = dict(kind="model", repo=repo, commit_hf=res.commit, dtype=m["dtype"], group_sample=g_sample,
                       group_whole=g_whole, fp_sample=len(fps_s), fp_whole=len(fps_w),
                       tensors_sample=len(samples), tensors_whole=len(whole), target=target)
            run.record(**rec)
            ks = [1, 4, 8] if samples[target].dtype == "F32" else [1, 4]
            cells = []
            for k in ks:
                size = capacity(samples, k, names=[target])
                payload = rng.bytes(size)
                det_s = detected(embed(samples, payload, k=k, names=[target]), shapes, target)
                det_w = detected({**whole, target: embed({target: whole[target]}, payload, k=k,
                                                         names=[target])[target]}, wshapes, target)
                run.record(kind="cell", repo=repo, k=k, target=target, payload_bytes=size,
                           detected_sample=det_s, detected_whole=det_w,
                           target_flagged_clean_whole=target in flagged_w)
                cells.append(f"k{k}:{'Y' if det_s else 'n'}/{'Y' if det_w else 'n'}")
            print(f"[{i}/{len(models)}] {repo}: {g_sample} | {g_whole}; {' '.join(cells)}", flush=True)
            del whole
            Path(path).unlink(missing_ok=True)  # the file only: other runs may share the cache
        except Exception as e:
            traceback.print_exc()
            run.record(kind="model", repo=repo, status="error", error=f"{type(e).__name__}: {e}"[:300])
    print("run", run.run_id)


if __name__ == "__main__":
    main()
