import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import hf  # noqa: E402


@pytest.fixture
def local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "CACHE", tmp_path / "cache")
    rng = np.random.default_rng(0)
    tensors = {f"layer{i}.weight": rng.standard_normal((64, 128)).astype(np.float32) for i in range(20)}
    tensors["small.bias"] = rng.standard_normal(16).astype(np.float32)
    tensors["ids"] = np.arange(5000, dtype=np.int64)
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    data = path.read_bytes()
    calls = []

    def fetch(url, start, end):
        calls.append((start, end))
        return data[start:end + 1]

    return tensors, fetch, calls


def test_sample_reads_exact_bytes_by_range(local_file):
    tensors, fetch, calls = local_file
    got, info = hf.sample("me/model", "model.safetensors", fetch=fetch, chunk=1024, max_tensors=5, url="x")
    assert len(got) == 5 and info["file_params_by_dtype"]["F32"] == 20 * 64 * 128 + 16
    for name, rt in got.items():
        assert rt.dtype == "F32" and rt.data.size == 256
        assert rt.to_bytes() == tensors[name].tobytes()[:1024]
    # never fetched whole tensors: header length, header, then one chunk per tensor
    assert len(calls) == 2 + 5 and all(e - s + 1 <= 1024 for s, e in calls[2:])


def test_choose_skips_small_and_non_float(local_file):
    _, fetch, _ = local_file
    _, header = hf.read_header("x", fetch)
    names = hf.choose_tensors(header, max_tensors=50)
    assert "small.bias" not in names and "ids" not in names and len(names) == 20


def test_cache_is_used_on_second_call(local_file):
    _, fetch, calls = local_file
    a, _ = hf.sample("me/model", "model.safetensors", fetch=fetch, chunk=1024, max_tensors=3, url="x")
    n = len(calls)
    b, _ = hf.sample("me/model", "model.safetensors", fetch=fetch, chunk=1024, max_tensors=3, url="x")
    assert len(calls) == n
    assert {k: v.to_bytes() for k, v in a.items()} == {k: v.to_bytes() for k, v in b.items()}
