"""Fetch a model's weight files from the Hugging Face Hub, for scanning.

    hf:org/name            the default branch
    hf:org/name@revision   a branch, tag or commit

Every download is pinned to the commit the revision resolves to, and checked
against the SHA-256 the Hub publishes for the file before it is used, so a
report names exactly which bytes it describes. Files go to a local cache
(STOWAWAY_CACHE, default ~/.cache/stowaway/hub) and are reused. Nothing
downloaded is ever loaded or unpickled: the scanner only reads bytes.

Which files: model.safetensors, else the shards listed in
model.safetensors.index.json, else the PyTorch equivalents (pytorch_model.bin
and its index), else every *.safetensors / *.gguf at the top level.

A token in HF_TOKEN is sent for gated or private repositories.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HOST = "https://huggingface.co"
UA = "stowaway (+https://github.com/aaditripi18/Stowaway)"
SPEC = re.compile(r"^hf:(?P<repo>[\w.-]+/[\w.-]+)(?:@(?P<rev>[\w./-]+))?$")


class HubError(Exception):
    pass


@dataclass(frozen=True)
class Resolved:
    repo: str
    revision: str          # what was asked for
    commit: str            # the commit it resolved to
    files: list[str]       # weight files, in order
    sizes: dict[str, int]
    sha256: dict[str, str]  # from the Hub (LFS files)
    base_model: list[str]  # the model card's declared base_model, if any


def is_spec(s: str) -> bool:
    return isinstance(s, str) and s.startswith("hf:")


def parse_spec(s: str) -> tuple[str, str]:
    m = SPEC.match(s)
    if not m:
        raise HubError(f"not a Hugging Face model spec: {s!r} (expected hf:org/name or hf:org/name@revision)")
    return m["repo"], m["rev"] or "main"


def _open(url: str, headers: dict | None = None, timeout: int = 60):
    h = {"User-Agent": UA, **(headers or {})}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def _json(url: str) -> dict:
    try:
        with _open(url) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        hint = " (gated or private: set HF_TOKEN)" if e.code in (401, 403) else ""
        raise HubError(f"{url}: HTTP {e.code}{hint}") from None


def choose_files(names: list[str], read_index) -> list[str]:
    """The weight files of a repository, by the rule in the module docstring.
    `read_index(name)` returns the parsed JSON of an index file."""
    top = [n for n in names if "/" not in n]
    for single, index in (("model.safetensors", "model.safetensors.index.json"),
                          ("pytorch_model.bin", "pytorch_model.bin.index.json")):
        if single in top:
            return [single]
        if index in top:
            shards = sorted(set(read_index(index).get("weight_map", {}).values()))
            missing = [s for s in shards if s not in names]
            if missing:
                raise HubError(f"{index} lists shards that are not in the repository: {missing[:3]}")
            if shards:
                return shards
    rest = sorted(n for n in top if n.endswith((".safetensors", ".gguf")))
    if not rest:
        raise HubError("no weight files found (looked for safetensors, pytorch_model.bin and gguf)")
    return rest


def resolve(spec: str) -> Resolved:
    repo, rev = parse_spec(spec)
    q = urllib.parse.quote
    info = _json(f"{HOST}/api/models/{repo}/revision/{q(rev, safe='')}?blobs=true")
    commit = info.get("sha")
    if not commit:
        raise HubError(f"could not resolve {repo}@{rev} to a commit")
    sib = {s["rfilename"]: s for s in info.get("siblings", [])}

    def read_index(name):
        with _open(f"{HOST}/{repo}/resolve/{commit}/{q(name)}") as r:
            return json.loads(r.read().decode("utf-8"))

    files = choose_files(list(sib), read_index)
    base = (info.get("cardData") or {}).get("base_model") or []
    if isinstance(base, str):
        base = [base]
    return Resolved(repo, rev, commit, files,
                    {f: int(sib[f].get("size") or (sib[f].get("lfs") or {}).get("size") or 0) for f in files},
                    {f: (sib[f].get("lfs") or {}).get("sha256", "") for f in files},
                    [str(b) for b in base])


def cache_root() -> Path:
    return Path(os.environ.get("STOWAWAY_CACHE", Path.home() / ".cache" / "stowaway" / "hub"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(res: Resolved, log=print) -> list[Path]:
    """Download (or reuse) each weight file, verified against the Hub's SHA-256.
    Interrupted downloads resume."""
    out = []
    for name in res.files:
        dst = cache_root() / res.repo.replace("/", "--") / res.commit / name
        want = res.sha256.get(name, "")
        if dst.exists() and (not want or _sha256(dst) == want):
            out.append(dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        url = f"{HOST}/{res.repo}/resolve/{res.commit}/{urllib.parse.quote(name)}"
        log(f"downloading {res.repo}@{res.commit[:10]} {name} ({res.sizes.get(name, 0) / 2**20:,.1f} MiB)")
        try:
            with _open(url, {"Range": f"bytes={have}-"} if have else None, timeout=120) as r:
                if have and r.status != 206:  # server ignored the range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
        except urllib.error.HTTPError as e:
            if e.code == 416:  # nothing left to fetch
                pass
            else:
                raise HubError(f"{url}: HTTP {e.code}") from None
        if want and _sha256(part) != want:
            part.unlink()
            raise HubError(f"{name}: downloaded bytes do not match the Hub's SHA-256; not used")
        part.replace(dst)
        out.append(dst)
    return out
