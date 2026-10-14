"""Append-only results ledger.

Every experiment run appends one JSON line per measurement, stamped with the git
commit and whether the tree was dirty, so any number in RESULTS.md can be traced
back to the exact code that produced it.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "results" / "ledger.jsonl"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=LEDGER.parent.parent).stdout.strip()
    except Exception:
        return ""


def git_state() -> dict:
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": commit or "unknown", "dirty": dirty}


class Run:
    """A single invocation of an experiment. Records share its run id and metadata."""

    def __init__(self, experiment: str, **meta):
        self.experiment = experiment
        self.meta = meta
        self.git = git_state()
        self.started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.run_id = f"{experiment}-{self.started}-{self.git['commit']}"

    def record(self, **fields) -> None:
        row = {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "ts": self.started,
            "commit": self.git["commit"],
            "dirty": self.git["dirty"],
            "meta": self.meta,
            **fields,
        }
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    with open(LEDGER, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
