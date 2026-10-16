"""E5: the same model files through Stowaway and the scanners people already use.

H1 (pre-registered in PLAN.md): Stowaway's S1 matches picklescan on code
payloads, and every weight-payload model passes picklescan and a safetensors
conversion untouched.

The corpus is built here, deterministically, from one clean model (torchvision
ResNet-18) and a tiny state dict, in four groups:

  clean       unmodified files in each format -- the false-positive column
  code        a pickle global that would run on load (exec, os.system, ...);
              the arguments are inert ("pass", "true") and nothing is ever
              unpickled with real classes, by any tool here
  weight      64 KB of random bytes in the low 8 bits of the float weights
  structure   bytes no parser reads: data after the end of the file, an extra
              entry in a .pt zip

Each file goes through picklescan, ModelScan and fickling (their Python APIs,
default settings) and through Stowaway alone and with the clean model as base.
Then every .pt file is converted to .safetensors the way users are told to --
torch.load(weights_only=True) and safetensors.torch.save_file -- to record what
the conversion removes and what it carries across.

Writes one ledger row per (file, tool) and per conversion, and the raw tool
outputs to results/e5/.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import traceback
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from experiments import corpus
from experiments.ledger import Run
from stowaway.embed import embed, extract
from stowaway.report import scan

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "e5"
RAW = ROOT / "results" / "e5"
PAYLOAD_BYTES = 64 * 1024
K = 8
FLAG = ("suspicious", "dangerous")


# ---------------------------------------------------------------- corpus

class _Global:
    """Pickles as a call to `fn(*args)`. Only ever pickled, never loaded."""

    def __init__(self, fn, args):
        self.fn, self.args = fn, args

    def __reduce__(self):
        return self.fn, self.args




def _tiny_state():
    import torch
    g = torch.Generator().manual_seed(0)
    return {"fc.weight": torch.randn(16, 32, generator=g), "fc.bias": torch.randn(16, generator=g)}


def _bf16_origin(state):
    """The model as if it had been trained in bf16 and saved as float32:
    every float32 weight rounded to bf16 precision (low 16 bits zero)."""
    import torch
    return {k: v.to(torch.bfloat16).to(torch.float32) if v.dtype == torch.float32 else v.clone()
            for k, v in state.items()}


def _with_payload(state, payload):
    raw = corpus.state_to_raw(state)
    return corpus.raw_into_state(state, embed(raw, payload, K)), raw


def _save_st(state, path):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))


def _gguf(path, extra=b""):
    import gguf as gguf_lib
    rng = np.random.default_rng(1)
    w = gguf_lib.GGUFWriter(str(path), "llama")
    w.add_string("name", "e5")
    w.add_tensor("blk.0.weight", rng.standard_normal((64, 64)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    if extra:
        with open(path, "ab") as f:
            f.write(extra)


def build():
    """Write the corpus. Returns [(name, group, path, base_path_or_None)]."""
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    payload = rng.bytes(PAYLOAD_BYTES)
    exe = b"MZ" + rng.bytes(4094)  # an executable-looking blob; random, not a program

    model, _ = corpus.load_vision("resnet18")
    clean = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bf16 = _bf16_origin(clean)
    files = []

    def add(name, group, base=None):
        files.append((name, group, OUT / name, base))

    # clean
    torch.save(clean, OUT / "clean.pt"); add("clean.pt", "clean")
    torch.save(clean, OUT / "clean_legacy.pt", _use_new_zipfile_serialization=False); add("clean_legacy.pt", "clean")
    _save_st(clean, OUT / "clean.safetensors"); add("clean.safetensors", "clean")
    _save_st(bf16, OUT / "clean_bf16origin.safetensors"); add("clean_bf16origin.safetensors", "clean")
    (OUT / "clean_numpy.pkl").write_bytes(pickle.dumps({"w": np.zeros((4, 4), np.float32)}, protocol=4))
    add("clean_numpy.pkl", "clean")
    _gguf(OUT / "clean.gguf"); add("clean.gguf", "clean")

    # code: a dangerous global next to real tensors
    tiny = _tiny_state()
    calls = {
        "exec": (exec, ("pass",)),
        "eval": (eval, ("0",)),
        "os_system": (os.system, ("true",)),
        "builtins_getattr": (getattr, (0, "real")),
    }
    for label, (fn, args) in calls.items():
        obj = {"state_dict": tiny, "extra": _Global(fn, args)}
        torch.save(obj, OUT / f"code_{label}.pt"); add(f"code_{label}.pt", "code")
    for label in ("exec", "os_system"):
        fn, args = calls[label]
        obj = {"state_dict": tiny, "extra": _Global(fn, args)}
        torch.save(obj, OUT / f"code_{label}_legacy.pt", _use_new_zipfile_serialization=False)
        add(f"code_{label}_legacy.pt", "code")
    for proto in (0, 2):
        (OUT / f"code_exec_p{proto}.pkl").write_bytes(
            pickle.dumps({"w": [1.0], "x": _Global(exec, ("pass",))}, protocol=proto))
        add(f"code_exec_p{proto}.pkl", "code")

    # weight: random bytes in the low bits, in a fair-coin model and a bf16-origin one
    clean_st = OUT / "clean.safetensors"
    bf16_st = OUT / "clean_bf16origin.safetensors"
    poisoned, _ = _with_payload(clean, payload)
    torch.save(poisoned, OUT / "weight.pt"); add("weight.pt", "weight", clean_st)
    _save_st(poisoned, OUT / "weight.safetensors"); add("weight.safetensors", "weight", clean_st)
    poisoned16, _ = _with_payload(bf16, payload)
    torch.save(poisoned16, OUT / "weight_bf16origin.pt"); add("weight_bf16origin.pt", "weight", bf16_st)
    _save_st(poisoned16, OUT / "weight_bf16origin.safetensors"); add("weight_bf16origin.safetensors", "weight", bf16_st)

    # structure: bytes outside anything the format describes
    (OUT / "trailing.safetensors").write_bytes(clean_st.read_bytes() + exe); add("trailing.safetensors", "structure")
    zbuf = io.BytesIO((OUT / "clean.pt").read_bytes())
    with zipfile.ZipFile(zbuf, "a") as z:
        prefix = z.namelist()[0].split("/")[0]
        z.writestr(f"{prefix}/extra/payload.bin", exe)
    (OUT / "extra_entry.pt").write_bytes(zbuf.getvalue()); add("extra_entry.pt", "structure")
    _gguf(OUT / "trailing.gguf", extra=exe); add("trailing.gguf", "structure")
    return files, payload


# ---------------------------------------------------------------- scanners

def run_picklescan(path):
    from picklescan.scanner import scan_file_path
    r = scan_file_path(str(path))
    flagged = r.issues_count > 0 or r.infected_files > 0
    if r.scan_err:
        verdict = "error"
    elif r.scanned_files == 0:
        verdict = "not scanned"
    else:
        verdict = "flagged" if flagged else ("suspicious only" if r.suspicious_count else "clean")
    raw = {"scanned_files": r.scanned_files, "issues_count": r.issues_count,
           "infected_files": r.infected_files, "suspicious_count": r.suspicious_count,
           "scan_err": r.scan_err,
           "globals": sorted({f"{g.module}.{g.name}:{g.safety.name}" for g in r.globals})}
    return verdict, raw


def run_modelscan(path):
    from modelscan.modelscan import ModelScan
    r = ModelScan().scan(path)
    s = r.get("summary", {})
    issues = r.get("issues", [])
    scanned = s.get("scanned", {}).get("total_scanned", 0)
    if r.get("errors"):
        verdict = "error" if not issues else "flagged"
    elif issues:
        verdict = "flagged"
    elif scanned == 0:
        verdict = "not scanned"
    else:
        verdict = "clean"
    keep = {"total_issues": s.get("total_issues"), "total_issues_by_severity": s.get("total_issues_by_severity"),
            "scanned": scanned, "skipped": s.get("skipped", {}).get("total_skipped"),
            "issues": [{k: i.get(k) for k in ("description", "operator", "module", "severity")} for i in issues],
            "errors": [str(e)[:300] for e in r.get("errors", [])]}
    return verdict, keep


def run_fickling(path):
    """Routed the way fickling's own Hugging Face scanner routes files: zip
    archives (.pt) member by member, raw pickles directly, and formats it
    treats as known safe (.safetensors) or does not handle (.gguf) skipped."""
    import fickling
    if path.suffix in (".safetensors", ".gguf"):
        return "not scanned", {}
    try:
        if zipfile.is_zipfile(path):
            results = fickling.scan_zip_archive(str(path), graceful=True)
        else:
            results = {path.name: fickling.scan_file(str(path), graceful=True)}
    except Exception as e:  # noqa: BLE001 -- record any failure as the tool's answer
        return "error", {"exception": f"{type(e).__name__}: {str(e)[:300]}"}
    raw = {m: {"severity": getattr(r.severity, "name", str(r.severity)), "is_safe": r.is_safe,
               "errors": [str(e)[:300] for e in r.errors],
               "results": [str(x)[:300] for x in (r.results or [])][:10]} for m, r in results.items()}
    if not results:
        verdict = "not scanned"
    elif all(r.is_safe for r in results.values()):
        verdict = "clean"
    elif any(not r.is_safe and r.results for r in results.values()):
        verdict = "flagged"
    else:
        verdict = "error"
    return verdict, {"members": raw}


def run_fickling_ml(path):
    """fickling's ML allowlist analysis (fickling.ml.MLAllowlist), which knows
    the imports PyTorch checkpoints need. Not the CLI default, so reported as
    its own column."""
    from fickling.analysis import Analyzer, Severity, check_safety
    from fickling.fickle import Pickled, StackedPickle
    from fickling.ml import MLAllowlist
    if path.suffix in (".safetensors", ".gguf"):
        return "not scanned", {}
    raw, verdicts = {}, []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                streams = {n: Pickled.load(z.read(n)) for n in z.namelist() if n.endswith(".pkl")}
        else:  # a raw pickle, or legacy torch.save: several pickles back to back
            stacked = StackedPickle.load(io.BytesIO(path.read_bytes()), fail_on_decode_error=False)
            streams = {f"{path.name}#{i}": p for i, p in enumerate(stacked)}
    except Exception as e:  # noqa: BLE001
        return "error", {"exception": f"{type(e).__name__}: {str(e)[:200]}"}
    for name, pickled in streams.items():
        try:
            r = check_safety(pickled, analyzer=Analyzer([MLAllowlist()]))
            safe = r.severity == Severity.LIKELY_SAFE
            raw[name] = {"severity": r.severity.name, "text": str(r)[:300]}
            verdicts.append("clean" if safe else "flagged")
        except Exception as e:  # noqa: BLE001
            raw[name] = {"exception": f"{type(e).__name__}: {str(e)[:200]}"}
            verdicts.append("error")
    verdict = ("flagged" if "flagged" in verdicts else "error" if "error" in verdicts
               else "clean" if verdicts else "not scanned")
    return verdict, {"members": raw}


def run_stowaway(path, base=None):
    rep = scan(path, base=base)
    verdict = "flagged" if rep["verdict"] in FLAG else "clean"
    return verdict, {"verdict": rep["verdict"],
                     "findings": sorted({f"{x['check']}:{x['code']}:{x['severity']}" for x in rep["findings"]
                                         if x["severity"] in ("medium", "high", "critical")})}


TOOLS = {"picklescan": run_picklescan, "modelscan": run_modelscan, "fickling": run_fickling,
         "fickling-ml": run_fickling_ml, "stowaway": run_stowaway}


# ---------------------------------------------------------------- conversion

def convert(path, payload):
    """torch.load(weights_only=True) then safetensors.torch.save_file, as a user
    following the usual advice would. Never loads with weights_only=False."""
    import torch
    from safetensors.torch import load_file, save_file
    dst = path.with_suffix(".converted.safetensors")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        return {"outcome": "load refused", "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    save_file({k: v.contiguous() for k, v in state.items() if hasattr(v, "contiguous")}, str(dst))
    loaded = load_file(str(dst))
    # safetensors stores tensors sorted by name; read them back in the order the
    # payload was written (the original state dict's order).
    raw = corpus.state_to_raw({k: loaded[k] for k in state if k in loaded})
    bit_identical = all(torch.equal(state[k].view(torch.int32), loaded[k].view(torch.int32))
                        for k in loaded if loaded[k].dtype == torch.float32)
    try:
        survived = extract(raw, K) == payload
    except Exception:  # noqa: BLE001 -- no framed payload present
        survived = False
    after = run_stowaway(dst)
    return {"outcome": "converted", "weights_bit_identical": bool(bit_identical),
            "weight_payload_survives": bool(survived),
            "bytes_in": path.stat().st_size, "bytes_out": dst.stat().st_size,
            "stowaway_after": after[0], "stowaway_after_findings": after[1]["findings"]}


# ---------------------------------------------------------------- main

def main():
    corpus._torch_home()
    files, payload = build()
    versions = {p: version(p) for p in ("picklescan", "modelscan", "fickling", "safetensors", "torch")}
    run = Run("E5", payload_bytes=PAYLOAD_BYTES, k=K, versions=versions)
    if os.environ.get("E5_DRY"):  # check the harness without touching the ledger
        run.record = lambda **_: None
    RAW.mkdir(parents=True, exist_ok=True)
    raw_all = {}
    for name, group, path, base in files:
        raw_all[name] = {}
        for tool, fn in TOOLS.items():
            try:
                verdict, raw = fn(path)
            except Exception as e:  # noqa: BLE001
                verdict, raw = "error", {"exception": f"{type(e).__name__}: {str(e)[:300]}",
                                         "trace": traceback.format_exc()[-800:]}
            raw_all[name][tool] = {"verdict": verdict, **raw}
            run.record(file=name, group=group, tool=tool, verdict=verdict)
            print(f"{name:34s} {tool:11s} {verdict}")
        if base is not None:
            verdict, raw = run_stowaway(path, base)
            raw_all[name]["stowaway+base"] = {"verdict": verdict, **raw}
            run.record(file=name, group=group, tool="stowaway+base", verdict=verdict)
            print(f"{name:34s} {'stowaway+base':11s} {verdict}")
        if path.suffix == ".pt":
            c = convert(path, payload)
            raw_all[name]["conversion"] = c
            run.record(file=name, group=group, tool="conversion", **c)
            print(f"{name:34s} conversion  {c}")
    (RAW / "raw_outputs.json").write_text(json.dumps(raw_all, indent=1, sort_keys=True, default=str) + "\n",
                                          encoding="utf-8")
    print("run", run.run_id)


if __name__ == "__main__":
    main()
