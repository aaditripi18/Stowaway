"""E11 recording (PLAN.md, phase 4): per-window low-bit counts of whole files, so that a
windowed S4 rule can be chosen and judged offline without downloading the models again.

For each model of a frozen list (files up to 6 GB): the whole file, commit-pinned and SHA-256
checked; for every float tensor of at least two windows (W = 32,768 elements, the survey's
sample size), each complete window's count of informative elements (S4's definition) and of
ones in each of S4's low bit planes. Then the E10 A experiment again on the same file: a random
payload in the low k bits of the target tensor, written from its first element, as many bytes
as the survey's 32 KB sample holds; the planted target's window counts are recorded too.

Writes one compressed .npz per model to OUT (default /kaggle/working/e11) and a summary line
per model to the ledger (E11rec). Nothing is judged here.

    python scripts/run_e11_record.py [--list hf_models.json] [--out DIR] [--only N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import hf  # noqa: E402
from experiments.ledger import Run  # noqa: E402
from scripts.run_e10a import MAX_BYTES, SAMPLING, classify  # noqa: E402
from stowaway import hub  # noqa: E402
from stowaway.bits import informative  # noqa: E402
from stowaway.embed import capacity, embed  # noqa: E402
from stowaway.formats import open_model  # noqa: E402
from stowaway.weights import S4_BITS, _load_floats  # noqa: E402

W = 32 * 1024


def windows(t) -> tuple[np.ndarray, np.ndarray] | None:
    """(informative elements per window, ones per window and low bit plane) or None."""
    nw = t.data.size // W
    if nw < 2:
        return None
    v = t.data[: nw * W].reshape(nw, W)
    inf = informative(v.reshape(-1), t.dtype).reshape(nw, W)
    nbits = S4_BITS[t.dtype]
    ones = np.stack([(((v >> np.uint64(b)) & np.uint64(1)).astype(bool) & inf).sum(1) for b in range(nbits)], axis=1)
    return inf.sum(1).astype(np.uint32), ones.astype(np.uint32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="hf_models.json")
    ap.add_argument("--out", default=os.environ.get("STOWAWAY_E11_OUT", "/kaggle/working/e11"))
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--repos", nargs="+")
    ap.add_argument("--dry", action="store_true", help="smoke test: record nothing in the ledger")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = json.loads((ROOT / "results" / a.list).read_text(encoding="utf-8"))["models"][: a.only]
    if a.repos:
        models = [m for m in models if m["repo"] in a.repos]
    run = Run("E11rec", list=a.list, n_models=len(models), window=W, **SAMPLING)
    if a.dry:
        run.record = lambda **_: None
    rng = np.random.default_rng(11)
    for i, m in enumerate(models, 1):
        repo = m["repo"]
        try:
            samples, info = hf.sample(repo, m["file"], **SAMPLING)
            shapes = {n: tuple(v) for n, v in info["shapes"].items()}
            _, _, flagged_s = classify(samples, shapes)
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
            arrays, meta = {}, {"repo": repo, "commit": res.commit, "target": target, "tensors": {}, "planted": {}}
            for name, t in whole.items():
                w = windows(t)
                if w is None:
                    continue
                key = f"t{len(meta['tensors'])}"
                arrays[key + "_n"], arrays[key + "_ones"] = w
                meta["tensors"][key] = {"name": name, "dtype": t.dtype, "shape": list(t.shape), "size": int(t.data.size)}
            ks = [1, 4, 8] if samples[target].dtype == "F32" else [1, 4]
            for k in ks:
                size = capacity(samples, k, names=[target])
                planted = embed({target: whole[target]}, rng.bytes(size), k=k, names=[target])[target]
                w = windows(planted)
                if w is not None:
                    arrays[f"k{k}_n"], arrays[f"k{k}_ones"] = w
                    meta["planted"][f"k{k}"] = {"bytes": size}
            safe = repo.replace("/", "__")
            np.savez_compressed(out / f"{safe}.npz", meta=np.array(json.dumps(meta)), **arrays)
            run.record(kind="model", repo=repo, commit_hf=res.commit, target=target, tensors=len(meta["tensors"]),
                       windows=int(sum(arrays[k].size for k in arrays if k.endswith("_n") and k.startswith("t"))),
                       planted=sorted(meta["planted"]))
            print(f"[{i}/{len(models)}] {repo}: {len(meta['tensors'])} tensors recorded", flush=True)
            del whole
            Path(path).unlink(missing_ok=True)
        except Exception as e:
            traceback.print_exc()
            run.record(kind="model", repo=repo, status="error", error=f"{type(e).__name__}: {e}"[:300])
    print("run", run.run_id)


if __name__ == "__main__":
    main()
