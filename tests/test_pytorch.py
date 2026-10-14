import pickle
import zipfile

import pytest
import torch

from stowaway import scan
from conftest import Evil, codes


def _state():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    return m.state_dict()


def _same(mf, data, state):
    assert {t.name for t in mf.tensors} == set(state)
    for name, t in state.items():
        assert data[name] == t.contiguous().numpy().tobytes(), name


def test_zip_state_dict(tmp_path, parse):
    p = tmp_path / "m.pt"
    torch.save(_state(), p)
    mf, data = parse(p)
    assert mf.format == "pytorch-zip"
    _same(mf, data, _state())
    rep = scan(p)
    assert rep["verdict"] == "clean", rep["findings"]


def test_legacy_state_dict(tmp_path, parse):
    p = tmp_path / "m.pt"
    torch.save(_state(), p, _use_new_zipfile_serialization=False)
    mf, data = parse(p)
    assert mf.format == "pytorch-legacy"
    _same(mf, data, _state())
    assert scan(p)["verdict"] == "clean"


def test_view_leaves_unreferenced_storage(tmp_path, parse):
    p = tmp_path / "v.pt"
    base = torch.arange(100, dtype=torch.float32)
    torch.save({"v": base[10:20]}, p)
    mf, data = parse(p)
    assert data["v"] == base[10:20].numpy().tobytes()
    rep = scan(p)
    gaps = [x for x in rep["findings"] if x["code"] == "unreferenced-storage"]
    assert sum(x["detail"]["bytes"] for x in gaps) == 90 * 4


def test_non_contiguous_view_is_covered(tmp_path, parse):
    p = tmp_path / "t.pt"
    torch.save({"t": torch.arange(12.0).reshape(3, 4).t()}, p)
    mf, _ = parse(p)
    (t,) = mf.tensors
    assert t.shape == (4, 3) and t.offset is None
    assert "unreferenced-storage" not in codes(scan(p))


def test_extra_zip_entry(tmp_path):
    p = tmp_path / "m.pt"
    torch.save(_state(), p)
    with zipfile.ZipFile(p, "a") as zf:
        zf.writestr("archive/data/loader.py", "print('hi')")
    assert "unexpected-zip-entry" in codes(scan(p))


@pytest.mark.parametrize("legacy", [False, True])
def test_code_payload_is_found_and_never_run(tmp_path, legacy):
    marker = tmp_path / "pwned"
    p = tmp_path / "evil.pt"
    torch.save({"w": torch.ones(3), "x": Evil(marker)}, p, _use_new_zipfile_serialization=not legacy)
    rep = scan(p)
    assert not marker.exists()
    assert rep["verdict"] == "dangerous"
    crit = [x for x in rep["findings"] if x["severity"] == "critical"]
    assert any(x["detail"]["name"] == "exec" for x in crit)


@pytest.mark.parametrize("protocol", range(0, pickle.HIGHEST_PROTOCOL + 1))
def test_plain_pickle_payload(tmp_path, protocol):
    marker = tmp_path / "pwned"
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps({"x": Evil(marker)}, protocol=protocol))
    rep = scan(p)
    assert not marker.exists()
    assert rep["format"] == "pickle"
    assert rep["verdict"] == "dangerous"
    assert any(x["severity"] == "critical" for x in rep["findings"])


def test_prepended_bytes_before_zip(tmp_path):
    p = tmp_path / "m.pt"
    torch.save(_state(), p)
    p.write_bytes(b"#!/bin/sh\necho hi\n" + p.read_bytes())
    rep = scan(p)
    assert rep["format"] == "pytorch-zip"
    assert "leading-data" in codes(rep, "S2")
