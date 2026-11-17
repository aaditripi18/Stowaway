"""E17: whole-tensor S4 where S4w already looks (PLAN.md, phase 5; used once on hf_models_batch6.json).

The dev procedure of scripts/e17_dev.py on the frozen fresh list, recorded in the ledger: per model
the S4 flags on the clean whole file (with each tensor's size), and E11's planted cells with S3, S4
and S4w's verdict on the target. Both rules (the current one and S4x) are scored from these reads
by scripts/report.py.

    python scripts/run_e17.py [--only N]
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

from experiments.ledger import Run  # noqa: E402
from scripts.e17_dev import model_rows  # noqa: E402
from stowaway.weights import S4W_WINDOW  # noqa: E402

LIST = "hf_models_batch6.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="first N models (smoke test; not recorded)")
    a = ap.parse_args(argv)
    models = json.loads((ROOT / "results" / LIST).read_text(encoding="utf-8"))["models"][: a.only]
    run = Run("E17", list=LIST, n_models=len(models), window=S4W_WINDOW)
    if a.only:
        run.record = lambda **_: None
    rng = np.random.default_rng(17)
    for i, m in enumerate(models, 1):
        try:
            row = model_rows(m, rng)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            run.record(kind="model", repo=m["repo"], status="error", error=f"{type(e).__name__}: {e}"[:300])
            continue
        if "skipped" in row:
            run.record(kind="model", repo=m["repo"], status="skipped_size")
            continue
        run.record(kind="model", repo=m["repo"], commit_hf=row["commit"], dtype=m["dtype"], clean_s4=row["clean_s4"])
        for c in row["cells"]:
            run.record(kind="cell", repo=m["repo"], **c)
        print(f"[{i}/{len(models)}] {m['repo']}: clean S4 {len(row['clean_s4'])}; cells "
              f"{[(c['k'], bool(c['s4']), c['s4w'], c['s3']) for c in row['cells']]}", flush=True)
    print("run", run.run_id)


if __name__ == "__main__":
    main()
