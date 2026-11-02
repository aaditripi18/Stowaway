"""Run a download-heavy experiment (E8, E10 A) on Kaggle: a CPU session with fast downloads.

  python scripts/kaggle_e10.py push                 # upload the committed code (git archive + COMMIT stamp)
  python scripts/kaggle_e10.py run [e10a|e8]        # push, then start that experiment's notebook
  python scripts/kaggle_e10.py status [e10a|e8]
  python scripts/kaggle_e10.py pull [e10a|e8]       # append the session's new ledger lines to results/ledger.jsonl

Only committed code runs: the upload is `git archive HEAD`, stamped with the
commit, so every ledger line the session writes names the exact code. The Kaggle
API key stays in ~/.kaggle/kaggle.json and is read only by the Kaggle CLI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = os.environ.get("STOWAWAY_KAGGLE_USER", "aadityae5545454")
DATASET = "stowaway-code"
JOBS = {"e10a": ("stowaway-e10-whole-files", "scripts/run_e10a.py"),
        "e8": ("stowaway-e8-base-discovery", "scripts/run_e8.py")}
LEDGER = ROOT / "results" / "ledger.jsonl"

NOTEBOOK = r'''
import os, pathlib, shutil, subprocess, sys
src = next(pathlib.Path("/kaggle/input").rglob("Stowaway/COMMIT")).parent
work = pathlib.Path("/kaggle/working/Stowaway")
shutil.copytree(src, work)
print("code", (work / "COMMIT").read_text().strip(), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(work)], check=True)
os.environ["STOWAWAY_CACHE"] = "/tmp/stowaway-hub"
ledger = work / "results" / "ledger.jsonl"
n0 = len(ledger.read_text(encoding="utf-8").splitlines())
subprocess.run([sys.executable, SCRIPT], cwd=work, check=False)
new = ledger.read_text(encoding="utf-8").splitlines()[n0:]
pathlib.Path("/kaggle/working/new_ledger_lines.jsonl").write_text("\n".join(new) + "\n", encoding="utf-8")
print(len(new), "new ledger lines", flush=True)
shutil.rmtree(work)
'''


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    exe = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run([str(exe), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}{r.stderr}")
    return r


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def push() -> str:
    if git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results"):
        sys.exit("commit your changes first: Kaggle runs only committed code")
    sha = git("rev-parse", "--short", "HEAD")
    ds = f"{USER}/{DATASET}"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        archive = d / f"stowaway-{sha}.zip"
        subprocess.run(["git", "archive", "--format=zip", "--prefix=Stowaway/", "-o", str(archive), "HEAD"],
                       cwd=ROOT, check=True)
        with zipfile.ZipFile(archive, "a") as z:
            z.writestr("Stowaway/COMMIT", sha + "\n")
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASET, "id": ds, "licenses": [{"name": "other"}]}), encoding="utf-8")
        if kaggle("datasets", "status", ds, check=False).returncode == 0:
            kaggle("datasets", "version", "-p", str(d), "-m", f"code {sha}")
        else:
            kaggle("datasets", "create", "-p", str(d))  # private unless --public
    for _ in range(90):  # Kaggle can report "ready" before it serves the new version
        with tempfile.TemporaryDirectory() as d:
            r = kaggle("datasets", "download", ds, "-f", "Stowaway/COMMIT", "-p", d, check=False)
            got = [p for p in Path(d).rglob("*") if p.is_file()]
            if r.returncode == 0 and got:
                data = got[0].read_bytes()
                if data[:2] == b"PK":
                    with zipfile.ZipFile(got[0]) as z:
                        data = z.read(z.namelist()[0])
                if data.decode().strip() == sha:
                    print(f"Kaggle serves code {sha}")
                    return sha
        time.sleep(10)
    sys.exit("Kaggle did not serve the new code within 15 minutes")


def notebook_source(job: str, sha: str) -> str:
    """The script Kaggle runs for one experiment (tests/test_kaggle_e10.py: it must define SCRIPT)."""
    return f"# Stowaway {job}, code {sha}\nSCRIPT = {JOBS[job][1]!r}\n" + NOTEBOOK


def run(job: str) -> None:
    kernel, _ = JOBS[job]
    sha = push()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run.py").write_text(notebook_source(job, sha), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{kernel}", "title": kernel, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USER}/{DATASET}"],
            "competition_sources": [], "kernel_sources": []}), encoding="utf-8")
        print(kaggle("kernels", "push", "-p", str(d)).stdout.strip())


def status(job: str) -> None:
    print(kaggle("kernels", "status", f"{USER}/{JOBS[job][0]}").stdout.strip())


def pull(job: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        kaggle("kernels", "output", f"{USER}/{JOBS[job][0]}", "-p", d)
        f = next(Path(d).rglob("new_ledger_lines.jsonl"), None)
        if f is None:
            sys.exit(f"no new_ledger_lines.jsonl in the output ({[p.name for p in Path(d).rglob('*')]})")
        new = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    have = set(LEDGER.read_text(encoding="utf-8").splitlines())
    add = [ln for ln in new if ln not in have]
    with open(LEDGER, "a", encoding="utf-8") as out:
        out.writelines(ln + "\n" for ln in add)
    print(f"appended {len(add)} of {len(new)} lines to {LEDGER}")


if __name__ == "__main__":
    cmd, job = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "e10a")
    push() if cmd == "push" else {"run": run, "status": status, "pull": pull}[cmd](job)
