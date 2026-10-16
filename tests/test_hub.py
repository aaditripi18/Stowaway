"""stowaway.hub against a fake Hub: file choice, commit pinning, SHA-256 checks,
resumed downloads, and `scan hf:... --base declared` end to end."""
import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from stowaway import cli, hub

FIX = Path(__file__).resolve().parent.parent / "web" / "fixtures" / "s5"


class FakeHub:
    """Serves /api/models/<repo>/revision/<rev> and /<repo>/resolve/<commit>/<file>."""

    def __init__(self):
        self.repos = {}
        self.requests = []
        self.corrupt = set()

    def add(self, repo, files: dict[str, bytes], base_model=None, commit="c0ffee" * 6 + "abcd"):
        self.repos[repo] = (commit, files, base_model)

    def __call__(self, url, headers=None, timeout=60):
        self.requests.append((url, dict(headers or {})))
        path = url.split("huggingface.co/", 1)[1]
        if path.startswith("api/models/"):
            rest = path[len("api/models/"):].split("?")[0]
            repo, _, _rev = rest.partition("/revision/")
            if repo not in self.repos:
                raise urllib.error.HTTPError(url, 404, "not found", {}, None)
            commit, files, base = self.repos[repo]
            body = {"sha": commit, "siblings": [
                {"rfilename": n, "size": len(b), "lfs": {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}}
                for n, b in files.items()]}
            if base:
                body["cardData"] = {"base_model": base}
            return _Resp(json.dumps(body).encode())
        repo_a, repo_b, _resolve, commit, name = path.split("/", 4)
        _, files, _ = self.repos[f"{repo_a}/{repo_b}"]
        data = files[name]
        if name in self.corrupt:
            data = data[:-1] + bytes([data[-1] ^ 1])
        rng = (headers or {}).get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return _Resp(data[start:], status=206)
        return _Resp(data)


class _Resp(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def fake(monkeypatch, tmp_path):
    h = FakeHub()
    monkeypatch.setattr(hub, "_open", h)
    monkeypatch.setenv("STOWAWAY_CACHE", str(tmp_path / "cache"))
    return h


def test_spec_parsing():
    assert hub.parse_spec("hf:org/name") == ("org/name", "main")
    assert hub.parse_spec("hf:org/name.v2@refs/pr/3") == ("org/name.v2", "refs/pr/3")
    for bad in ("org/name", "hf:name", "hf:org/name@", "hf:org/na me"):
        with pytest.raises(hub.HubError):
            hub.parse_spec(bad)


def test_choose_files():
    idx = {"model.safetensors.index.json": {"weight_map": {"a": "model-00002.safetensors", "b": "model-00001.safetensors"}}}
    pick = lambda names: hub.choose_files(names, idx.__getitem__)  # noqa: E731
    assert pick(["model.safetensors", "pytorch_model.bin"]) == ["model.safetensors"]
    assert pick(["model.safetensors.index.json", "model-00001.safetensors", "model-00002.safetensors"]) == \
        ["model-00001.safetensors", "model-00002.safetensors"]
    assert pick(["pytorch_model.bin", "config.json"]) == ["pytorch_model.bin"]
    assert pick(["q4.gguf", "sub/x.safetensors"]) == ["q4.gguf"]
    with pytest.raises(hub.HubError):
        pick(["model.safetensors.index.json", "model-00001.safetensors"])  # a shard is missing
    with pytest.raises(hub.HubError):
        pick(["config.json"])


def test_download_is_pinned_verified_and_cached(fake):
    data = (FIX / "base.safetensors").read_bytes()
    fake.add("org/base", {"model.safetensors": data, "config.json": b"{}"})
    res = hub.resolve("hf:org/base")
    assert res.files == ["model.safetensors"] and res.commit.startswith("c0ffee")
    (p,) = hub.download(res, log=lambda m: None)
    assert p.read_bytes() == data and res.commit in str(p)
    assert all(res.commit in u for u, _ in fake.requests if "/resolve/" in u)  # pinned, not "main"
    n = len(fake.requests)
    hub.download(res, log=lambda m: None)
    assert len(fake.requests) == n  # reused from the cache


def test_download_rejects_wrong_bytes(fake):
    fake.add("org/base", {"model.safetensors": (FIX / "base.safetensors").read_bytes()})
    fake.corrupt.add("model.safetensors")
    res = hub.resolve("hf:org/base")
    with pytest.raises(hub.HubError, match="SHA-256"):
        hub.download(res, log=lambda m: None)
    assert not list(hub.cache_root().rglob("model.safetensors"))  # nothing unverified is kept


def test_download_resumes(fake):
    data = (FIX / "base.safetensors").read_bytes()
    fake.add("org/base", {"model.safetensors": data})
    res = hub.resolve("hf:org/base")
    part = hub.cache_root() / "org--base" / res.commit / "model.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:1000])
    (p,) = hub.download(res, log=lambda m: None)
    assert p.read_bytes() == data
    assert any(h.get("Range") == "bytes=1000-" for _, h in fake.requests)


def test_scan_hf_target_against_its_declared_base(fake, capsys):
    fake.add("org/base", {"model.safetensors": (FIX / "base.safetensors").read_bytes()}, commit="b" * 40)
    fake.add("org/finetune", {"model.safetensors": (FIX / "poisoned.safetensors").read_bytes()},
             base_model="org/base", commit="f" * 40)
    status = cli.main(["scan", "hf:org/finetune", "--base", "declared", "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert status == 2
    assert rep["source"]["commit"] == "f" * 40 and rep["source"]["declared_base_model"] == ["org/base"]
    assert rep["base"] == {"hub": "org/base", "revision": "main", "commit": "b" * 40, "files": ["model.safetensors"]}
    assert any(x["check"] == "S5" and x["code"] == "low-bit-only-change" for x in rep["findings"])


def test_declared_without_a_card_base_is_an_error(fake, capsys):
    fake.add("org/lonely", {"model.safetensors": (FIX / "base.safetensors").read_bytes()})
    assert cli.main(["scan", "hf:org/lonely", "--base", "declared"]) == 3
    assert "declares no base_model" in capsys.readouterr().err


def test_sharded_base(tmp_path):
    """S5 against a base split over two files equals S5 against the whole base."""
    from safetensors.numpy import load_file, save_file

    from stowaway.report import scan
    base = load_file(str(FIX / "base.safetensors"))
    names = sorted(base)
    save_file({n: base[n] for n in names[: len(names) // 2]}, str(tmp_path / "b1.safetensors"))
    save_file({n: base[n] for n in names[len(names) // 2:]}, str(tmp_path / "b2.safetensors"))
    whole = scan(FIX / "poisoned.safetensors", base=FIX / "base.safetensors")
    split = scan(FIX / "poisoned.safetensors", base=[tmp_path / "b1.safetensors", tmp_path / "b2.safetensors"])
    assert whole["findings"] == split["findings"]


def test_truncated_transfers_are_resumed(fake):
    """The CDN can end a transfer early without an error: keep resuming."""
    data = (FIX / "base.safetensors").read_bytes()
    fake.add("org/base", {"model.safetensors": data})
    real = fake.__call__
    calls = {"n": 0}

    def short(url, headers=None, timeout=60):
        resp = real(url, headers, timeout)
        if "/resolve/" in url:
            calls["n"] += 1
            if calls["n"] <= 2:  # first two transfers stop after 700 bytes
                return _Resp(resp.read()[:700], status=resp.status)
        return resp
    hub._open = short
    res = hub.resolve("hf:org/base")
    (p,) = hub.download(res, log=lambda m: None)
    assert p.read_bytes() == data and calls["n"] >= 3
