"""Sample tensors from public safetensors files on Hugging Face without downloading them.

A safetensors header gives every tensor's exact byte range, so a model can be
surveyed with a few HTTP range requests: the 8-byte header length, the header,
then the first `chunk` bytes of a handful of tensors spread through the network.
Samples are cached under data/hf_cache so reruns do not refetch.

The fetch function is a parameter so tests can serve ranges from a local file.
"""
from __future__ import annotations

import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from stowaway.bits import FLOAT_LAYOUT, RawTensor
from stowaway.model import ITEMSIZE

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "hf_cache"
API = "https://huggingface.co/api/models"

Fetch = Callable[[str, int, int], bytes]  # (url, start, end inclusive) -> bytes


def http_fetch(url: str, start: int, end: int, retries: int = 4) -> bytes:
    """Bytes [start, end] of `url`, following redirects, with retries."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                               "User-Agent": "stowaway-survey"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if r.status != 206 and start > 0:
                raise IOError(f"server ignored the range request (status {r.status})")
            return data[: end - start + 1]
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def list_models(limit: int = 500) -> list[dict]:
    """Most-downloaded models that ship safetensors, with dtype parameter counts."""
    url = (f"{API}?filter=safetensors&sort=downloads&direction=-1&limit={limit}"
           "&expand[]=safetensors&expand[]=gated&expand[]=siblings")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "stowaway-survey"}),
                                timeout=60) as r:
        return json.loads(r.read())


def primary_dtype(model: dict) -> str | None:
    params = (model.get("safetensors") or {}).get("parameters") or {}
    floats = {k: v for k, v in params.items() if k in FLOAT_LAYOUT}
    return max(floats, key=floats.get) if floats else None


def weights_file(model: dict) -> str | None:
    """model.safetensors, else the first shard."""
    files = sorted(s["rfilename"] for s in model.get("siblings", [])
                   if s["rfilename"].endswith(".safetensors") and "/" not in s["rfilename"])
    if "model.safetensors" in files:
        return "model.safetensors"
    return files[0] if files else None


def resolve_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def read_header(url: str, fetch: Fetch) -> tuple[int, dict]:
    (n,) = struct.unpack("<Q", fetch(url, 0, 7))
    if n > 100 * 1024 * 1024:
        raise ValueError(f"header length {n} is implausible")
    return 8 + n, json.loads(fetch(url, 8, 8 + n - 1))


def choose_tensors(header: dict, max_tensors: int = 12, min_elements: int = 4096) -> list[str]:
    """Float tensors big enough for statistics, evenly spaced through the file."""
    cands = []
    for name, spec in header.items():
        if name == "__metadata__" or spec.get("dtype") not in FLOAT_LAYOUT:
            continue
        n = int(np.prod(spec["shape"])) if spec["shape"] else 1
        if n >= min_elements:
            cands.append((spec["data_offsets"][0], name))
    cands.sort()
    if len(cands) <= max_tensors:
        return [n for _, n in cands]
    idx = np.linspace(0, len(cands) - 1, max_tensors).round().astype(int)
    return [cands[i][1] for i in sorted(set(idx))]


def _cache_dir(repo: str, filename: str, chunk: int, max_tensors: int) -> Path:
    return CACHE / repo.replace("/", "__") / f"{filename.replace('/', '__')}.{chunk}x{max_tensors}"


def sample(repo: str, filename: str, fetch: Fetch = http_fetch, chunk: int = 64 * 1024,
           max_tensors: int = 12, url: str | None = None,
           workers: int = 6) -> tuple[dict[str, RawTensor], dict]:
    """Sample tensors of one remote safetensors file.

    Returns ({name: RawTensor (1-D, first `chunk` bytes)}, info), cached on disk
    per (chunk, max_tensors) so samples of different sizes never mix."""
    from concurrent.futures import ThreadPoolExecutor
    cdir = _cache_dir(repo, filename, chunk, max_tensors)
    meta_path, data_path = cdir / "meta.json", cdir / "samples.npz"
    if meta_path.exists() and data_path.exists():
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(data_path) as z:
            return {n: RawTensor.from_bytes(info["dtypes"][n], (int(z[n].size) // ITEMSIZE[info["dtypes"][n]],),
                                            z[n].tobytes())
                    for n in info["tensors"]}, info

    url = url or resolve_url(repo, filename)
    base, header = read_header(url, fetch)
    names = choose_tensors(header, max_tensors)
    def one(name):
        spec = header[name]
        isz = ITEMSIZE[spec["dtype"]]
        b, e = spec["data_offsets"]
        length = min(e - b, chunk) // isz * isz
        return name, spec["dtype"], isz, fetch(url, base + b, base + b + length - 1)

    tensors, raw = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, dtype, isz, data in pool.map(one, names):  # map keeps the order
            tensors[name] = RawTensor.from_bytes(dtype, (len(data) // isz,), data)
            raw[name] = np.frombuffer(data, dtype=np.uint8)
    dtypes = {}
    for spec in header.values():
        if isinstance(spec, dict) and "dtype" in spec:
            dtypes.setdefault(spec["dtype"], 0)
            dtypes[spec["dtype"]] += int(np.prod(spec["shape"])) if spec["shape"] else 1
    info = {
        "repo": repo, "file": filename,
        "n_tensors_in_file": sum(1 for k in header if k != "__metadata__"),
        "file_params_by_dtype": dtypes,
        "tensors": names, "dtypes": {n: header[n]["dtype"] for n in names},
        "chunk": chunk, "sampled_bytes": int(sum(r.size for r in raw.values())),
    }
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez(data_path, **raw)
    meta_path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return tensors, info
