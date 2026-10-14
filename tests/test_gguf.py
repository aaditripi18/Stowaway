import gguf
import numpy as np

from stowaway import scan
from conftest import codes

A = np.arange(3, dtype=np.float32)           # 12 bytes: leaves 20 bytes of padding
B = np.arange(40, dtype=np.float16).reshape(5, 8)


def _write(path):
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_uint32("test.answer", 42)
    w.add_string("test.name", "fixture")
    w.add_tensor("a", A)
    w.add_tensor("b", B)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_tensors_match_reference(tmp_path, parse):
    mf, data = parse(_write(tmp_path / "m.gguf"))
    assert mf.format == "gguf"
    assert mf.metadata["test.answer"] == 42 and mf.metadata["test.name"] == "fixture"
    shapes = {t.name: (t.dtype, t.shape) for t in mf.tensors}
    assert shapes == {"a": ("F32", (3,)), "b": ("F16", (5, 8))}
    assert data["a"] == A.tobytes() and data["b"] == B.tobytes()


def test_clean(tmp_path):
    rep = scan(_write(tmp_path / "m.gguf"))
    assert rep["verdict"] == "clean", rep["findings"]


def test_trailing(tmp_path):
    p = _write(tmp_path / "m.gguf")
    with open(p, "ab") as f:
        f.write(np.random.default_rng(0).bytes(4096))
    rep = scan(p)
    (x,) = [x for x in rep["findings"] if x["code"] == "trailing-data"]
    assert x["severity"] == "high" and x["detail"]["entropy"] > 7.5


def test_dirty_padding(tmp_path, parse):
    p = _write(tmp_path / "m.gguf")
    mf, _ = parse(p)
    (a,) = [t for t in mf.tensors if t.name == "a"]
    raw = bytearray(p.read_bytes())
    raw[a.offset + a.nbytes + 3] = 0x41  # inside the alignment padding after "a"
    p.write_bytes(bytes(raw))
    assert "dirty-padding" in codes(scan(p), "S2")
