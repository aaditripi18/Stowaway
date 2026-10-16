"""Generate safetensors fixtures and golden canonical-JSON reports for the
TypeScript parity tests. Run from the repo root with the project venv:

    web/.venv-less: use the repo venv
    PYTHONPATH=. .venv/Scripts/python web/test/gen_fixtures.py

Writes web/fixtures/<name>.safetensors and web/fixtures/<name>.json (the golden
report in the shared canonical form). The canonical form matches web/src/canon.ts.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "web" / "fixtures"

from stowaway.report import scan  # noqa: E402
from stowaway import pickle_scan as ps  # noqa: E402

PICKLE_FIX = FIX / "pickle"


def canon(o) -> str:
    """The exact string web/src/canon.ts produces."""
    if o is None:
        return "null"
    if isinstance(o, bool):
        return "true" if o else "false"
    if isinstance(o, int):
        return str(o)
    if isinstance(o, float):
        if o == int(o) and abs(o) < 1e15:
            return str(int(o))
        return repr(o)
    if isinstance(o, str):
        return json.dumps(o)
    if isinstance(o, (list, tuple)):
        return "[" + ",".join(canon(x) for x in o) + "]"
    if isinstance(o, dict):
        return "{" + ",".join(json.dumps(k) + ":" + canon(v) for k, v in sorted(o.items())) + "}"
    raise TypeError(f"cannot canonicalise {type(o)}")


def write(name: str, data: bytes):
    path = FIX / f"{name}.safetensors"
    path.write_bytes(data)
    rep = scan(path)
    (FIX / f"{name}.json").write_text(canon(rep) + "\n", encoding="utf-8")
    print(f"{name}: {rep['verdict']} ({len(rep['findings'])} findings)")


def st_bytes(tensors, metadata=None) -> bytes:
    tmp = FIX / "_tmp.safetensors"
    save_file(tensors, str(tmp), metadata=metadata)
    data = tmp.read_bytes()
    tmp.unlink()
    return data


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    clean = {
        "a": rng.standard_normal((64, 32)).astype(np.float32),
        "b": np.linspace(-1, 1, 50).astype(np.float16),
        "c": np.arange(16, dtype=np.int64),
    }
    write("clean", st_bytes(clean, {"producer": "stowaway-fixtures"}))

    write("trailing_mz", st_bytes(clean) + b"MZ" + bytes(range(256)) * 8)

    hdr = ('{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},'
           ' "w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}').encode()
    write("duplicate_keys", struct.pack("<Q", len(hdr)) + hdr + b"\0" * 8)

    hdr2 = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}).encode()
    write("size_mismatch", struct.pack("<Q", len(hdr2)) + hdr2 + b"\1" * 12)

    big = struct.pack("<Q", 10**9) + b"{}"
    write("bad_header_length", big)

    gen_pickles()
    gen_weights()
    gen_s5()
    gen_gguf()
    gen_pytorch()


def gen_pytorch():
    """PyTorch zip fixtures: a clean checkpoint, one with an appended blob, one
    with a code payload in the pickle, and one with an extra zip entry."""
    import torch
    import zipfile
    d = FIX / "pytorch"
    d.mkdir(parents=True, exist_ok=True)

    def golden(name, path):
        rep = scan(path)
        (d / f"{name}.json").write_text(canon(rep) + "\n", encoding="utf-8")
        print(f"pytorch/{name}: {rep['format']} {rep['verdict']} ({len(rep['findings'])} findings)")

    torch.manual_seed(0)
    sd = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.BatchNorm1d(6)).state_dict()
    torch.save(sd, d / "clean.pt")
    golden("clean", d / "clean.pt")

    raw = (d / "clean.pt").read_bytes()
    (d / "trailing.pt").write_bytes(raw + b"MZ" + bytes(range(256)) * 4)
    golden("trailing", d / "trailing.pt")

    import copy
    torch.save(sd, d / "extra_entry.pt")
    with zipfile.ZipFile(d / "extra_entry.pt", "a") as zf:
        zf.writestr("archive/data/loader.py", "print('hi')")
    golden("extra_entry", d / "extra_entry.pt")

    # A code payload: an object whose __reduce__ names a dangerous global.
    class Evil:
        def __reduce__(self):
            return exec, ("pass",)
    torch.save({"w": sd["0.weight"], "x": Evil()}, d / "evil.pt")
    golden("evil", d / "evil.pt")
    copy.copy  # keep import tidy


def gen_gguf():
    """GGUF fixtures: clean, trailing data, and a dirty alignment-padding byte."""
    import gguf as gguf_lib
    d = FIX / "gguf"
    d.mkdir(parents=True, exist_ok=True)

    def build(path):
        rng = np.random.default_rng(1)
        w = gguf_lib.GGUFWriter(str(path), "llama")
        w.add_uint32("answer", 42)
        w.add_string("name", "fixture")
        w.add_tensor("blk.0.weight", rng.standard_normal((32, 64)).astype(np.float32))
        w.add_tensor("blk.1.weight", rng.standard_normal((40,)).astype(np.float16))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    def golden(name, path):
        rep = scan(path)
        (d / f"{name}.json").write_text(canon(rep) + "\n", encoding="utf-8")
        print(f"gguf/{name}: {rep['verdict']} ({len(rep['findings'])} findings)")

    build(d / "clean.gguf")
    golden("clean", d / "clean.gguf")

    raw = (d / "clean.gguf").read_bytes()
    (d / "trailing.gguf").write_bytes(raw + np.random.default_rng(0).bytes(4096))
    golden("trailing", d / "trailing.gguf")

    # Flip a byte inside the header alignment padding (between the header and the
    # first aligned tensor) to trigger S2 dirty-padding.
    with open(d / "clean.gguf", "rb") as f:
        mf = __import__("stowaway.formats", fromlist=["open_model"]).open_model(Path(d / "clean.gguf"), f)
    pad = [r for r in mf.regions if r.kind == "padding" and r.end > r.start]
    buf = bytearray(raw)
    if pad:
        buf[pad[0].start] = 0x41
    (d / "dirty_padding.gguf").write_bytes(bytes(buf))
    golden("dirty_padding", d / "dirty_padding.gguf")

    # A header larger than the browser's first 1 MiB read, as in real GGUFs that
    # carry a tokenizer vocabulary: exercises the growing-prefix parse.
    rng = np.random.default_rng(2)
    w = gguf_lib.GGUFWriter(str(d / "big_header.gguf"), "llama")
    w.add_array("tokenizer.ggml.tokens", [f"tok{i:06d}" for i in range(300_000)])
    w.add_tensor("blk.0.weight", rng.standard_normal((64, 64)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    golden("big_header", d / "big_header.gguf")


def _biased_f32(shape, frac_bit0, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(shape).astype(np.float32).view(np.uint32).copy()
    ones = rng.random(shape) < frac_bit0
    u = np.where(ones, u | np.uint32(1), u & ~np.uint32(1)).astype(np.uint32)
    return u.view(np.float32)


def _payload_low_byte(arr, seed):
    rng = np.random.default_rng(seed)
    u = arr.view(np.uint32).copy()
    u = ((u & ~np.uint32(0xFF)) | rng.integers(0, 256, size=u.shape, dtype=np.uint32)).astype(np.uint32)
    return u.view(np.float32)


def gen_weights():
    """safetensors that exercise S3 and S4, so the TypeScript weight checks are
    covered by the full-scan parity test."""
    # Structured fp32 model: bit 0 set in ~67% of weights across five role peers.
    sig = {f"blk.{i}.w": _biased_f32((64, 64), 0.67, seed=10 + i) for i in range(5)}
    write("weights_signature", st_bytes(sig))
    # Same, but a random payload in one tensor's low byte pulls it towards 0.5.
    payload = dict(sig)
    payload["blk.0.w"] = _payload_low_byte(sig["blk.0.w"].copy(), seed=99)
    write("weights_payload", st_bytes(payload))
    # bf16 weights saved as fp32: the low 16 bits are zero (S3 precision-origin).
    rng = np.random.default_rng(3)
    bf = {}
    for i in range(4):
        f = (0.02 * rng.standard_normal((64, 64))).astype(np.float32)
        u = f.view(np.uint32)
        u = ((u + np.uint32(0x8000)) & ~np.uint32(0xFFFF)).astype(np.uint32)  # round to bf16, widen
        bf[f"blk.{i}.w"] = u.view(np.float32)
    write("weights_bf16_origin", st_bytes(bf))


def gen_s5():
    """A base model and the same model with a random low-bit payload in one
    tensor, for the S5 differential parity test."""
    d = FIX / "s5"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base = {f"layer.{i}.weight": rng.standard_normal((128, 64)).astype(np.float32) for i in range(3)}
    (d / "base.safetensors").write_bytes(st_bytes(base))
    poisoned = dict(base)
    poisoned["layer.1.weight"] = _payload_low_byte(base["layer.1.weight"].copy(), seed=42)
    (d / "poisoned.safetensors").write_bytes(st_bytes(poisoned))
    rep = scan(d / "poisoned.safetensors", base=d / "base.safetensors")
    (d / "poisoned.json").write_text(canon(rep) + "\n", encoding="utf-8")
    print(f"s5/poisoned: {rep['verdict']} ({len(rep['findings'])} findings)")


def _findings_json(label, data) -> str:
    findings = ps.analyse(label, data)
    rows = [{"check": f.check, "severity": f.severity, "code": f.code, "message": f.message, "detail": f.detail}
            for f in findings]
    return canon(rows)


def gen_pickles():
    """Raw pickle streams and the golden S1 findings for the TypeScript
    analysePickles parity test, plus full-file reports (write_pickle_reports)
    that check the pickle format dispatch end to end."""
    import pickle
    PICKLE_FIX.mkdir(parents=True, exist_ok=True)

    class Evil:
        def __reduce__(self):
            return exec, ("pass",)

    class OsSystem:
        def __reduce__(self):
            import os
            return os.system, ("true",)

    class Unknown:
        def __reduce__(self):
            import itertools
            return itertools.chain, ([],)

    cases = {}
    for proto in (0, 2, 4):
        cases[f"evil_exec_p{proto}"] = pickle.dumps({"x": Evil()}, protocol=proto)
    cases["os_system"] = pickle.dumps(OsSystem(), protocol=4)
    cases["unknown_import"] = pickle.dumps(Unknown(), protocol=4)
    cases["clean_data"] = pickle.dumps({"a": [1, 2, 3], "b": "hello", "c": (4, 5)}, protocol=4)
    evil = pickle.dumps({"x": Evil()}, protocol=4)
    cases["truncated"] = evil[:-3]  # breaks STOP; exec already emitted

    for name, data in cases.items():
        (PICKLE_FIX / f"{name}.pkl").write_bytes(data)
        (PICKLE_FIX / f"{name}.findings.json").write_text(_findings_json(name, data) + "\n", encoding="utf-8")
        print(f"pickle/{name}: {len(ps.analyse(name, data))} findings")
    write_pickle_reports()


def write_pickle_reports():
    """Golden full-file reports for the pickle fixtures, so the browser's format
    detection (including protocols 0 and 1, which have no 0x80 marker) is
    checked end to end, not only analysePickles()."""
    for path in sorted(PICKLE_FIX.glob("*.pkl")):
        rep = scan(path)
        (PICKLE_FIX / f"{path.stem}.json").write_text(canon(rep) + "\n", encoding="utf-8")
        print(f"pickle/{path.stem}: {rep['format']} {rep['verdict']}")


if __name__ == "__main__":
    main()
