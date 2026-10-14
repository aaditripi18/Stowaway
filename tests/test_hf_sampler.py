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


def test_role_sampling_takes_peers_from_each_role():
    header = {}
    off = 0
    for i in range(10):
        for r in ("q_proj", "up_proj", "norm"):
            header[f"layers.{i}.{r}.weight"] = {"dtype": "BF16", "shape": [4096], "data_offsets": [off, off + 8192]}
            off += 8192
    header["lm_head.weight"] = {"dtype": "BF16", "shape": [8192], "data_offsets": [off, off + 16384]}
    names = hf.choose_tensors(header, max_tensors=14, roles=3, per_role=4)
    from stowaway.weights import role
    from collections import Counter
    c = Counter(role(n) for n in names)
    assert all(c[f"layers.#.{r}.weight"] >= 4 for r in ("q_proj", "up_proj", "norm"))
    assert len(names) == 14 and len(set(names)) == 14  # topped up with two others


def test_sample_pair_aligns_shared_tensors(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    rng = np.random.default_rng(0)
    base_t = {f"layers.{i}.w": rng.standard_normal((64, 128)).astype(np.float32) for i in range(6)}
    base_t["only_in_base"] = rng.standard_normal((64, 64)).astype(np.float32)
    ft_t = dict(base_t)
    ft_t["classifier"] = rng.standard_normal((64, 128)).astype(np.float32)  # new head
    del ft_t["only_in_base"]
    ft_t["layers.0.w"] = ft_t["layers.0.w"] + 0.01  # a trained layer
    # different tensor order in the two files
    base_path = tmp_path / "base.safetensors"; save_file(base_t, str(base_path))
    ft_path = tmp_path / "ft.safetensors"; save_file({k: ft_t[k] for k in reversed(list(ft_t))}, str(ft_path))
    bd, fd = base_path.read_bytes(), ft_path.read_bytes()

    def fetch(url, start, end):
        return (fd if url.endswith("ft.safetensors") else bd)[start:end + 1]

    import experiments.hf as hf
    orig = hf.resolve_url
    hf.resolve_url = lambda repo, filename, revision="main": f"http://x/{filename}"
    try:
        ftr, br, info = hf.sample_pair("me/ft", "ft.safetensors", "me/base", "base.safetensors",
                                       fetch=fetch, chunk=4096, max_tensors=10, roles=0)
    finally:
        hf.resolve_url = orig
    assert "classifier" not in info["shared_tensors"] and "only_in_base" not in info["shared_tensors"]
    assert set(info["shared_tensors"]) == set(f"layers.{i}.w" for i in range(6))
    for n in info["shared_tensors"]:
        assert ftr[n].to_bytes() == ft_t[n].tobytes()[:ftr[n].data.size * 4]
        assert br[n].to_bytes() == base_t[n].tobytes()[:br[n].data.size * 4]
