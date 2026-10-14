import json
import struct

import numpy as np
from safetensors.numpy import save_file

from stowaway import scan
from conftest import codes

TENSORS = {
    "a": np.arange(12, dtype=np.float32).reshape(3, 4),
    "b": np.linspace(-1, 1, 5).astype(np.float16),
    "c": np.array([[1, 2], [3, 4]], dtype=np.int64),
}


def _write(path):
    save_file(TENSORS, str(path), metadata={"format": "np"})
    return path


def test_tensors_match_reference(tmp_path, parse):
    mf, data = parse(_write(tmp_path / "m.safetensors"))
    assert mf.format == "safetensors"
    assert mf.metadata == {"format": "np"}
    assert {t.name: (t.dtype, t.shape) for t in mf.tensors} == {
        "a": ("F32", (3, 4)), "b": ("F16", (5,)), "c": ("I64", (2, 2))}
    for name, arr in TENSORS.items():
        assert data[name] == arr.tobytes()


def test_clean_file_has_no_findings(tmp_path):
    rep = scan(_write(tmp_path / "m.safetensors"))
    assert rep["verdict"] == "clean" and rep["findings"] == []


def test_appended_executable_is_found(tmp_path):
    p = _write(tmp_path / "m.safetensors")
    with open(p, "ab") as f:
        f.write(b"MZ" + bytes(range(256)) * 8)
    rep = scan(p)
    assert rep["verdict"] == "dangerous"
    (x,) = [x for x in rep["findings"] if x["code"] == "trailing-data"]
    assert x["detail"]["bytes"] == 2 + 256 * 8
    assert x["detail"]["magic"] == "Windows executable (MZ)"


def _raw(path, header: str, body: bytes):
    h = header.encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + body)
    return path


def test_duplicate_keys(tmp_path):
    hdr = ('{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},'
           ' "w": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}')
    rep = scan(_raw(tmp_path / "d.safetensors", hdr, b"\0" * 8))
    assert "duplicate-keys" in codes(rep)


def test_size_mismatch_and_gap(tmp_path):
    hdr = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}})
    rep = scan(_raw(tmp_path / "s.safetensors", hdr, b"\1" * 12))
    assert "size-mismatch" in codes(rep)
    assert "trailing-data" in codes(rep, "S2")


def test_bad_header_length(tmp_path):
    p = tmp_path / "b.safetensors"
    p.write_bytes(struct.pack("<Q", 10**9) + b"{}")
    rep = scan(p)
    assert rep["format"] in ("safetensors", "unknown")
    assert rep["verdict"] != "clean"
