"""PyTorch checkpoints: the zip format (torch >= 1.6), the legacy format, and plain pickles.

Zip format: `<prefix>/data.pkl` holds the structure; each storage is a stored
(uncompressed) entry `<prefix>/data/<key>`. Tensors are views into storages, so
a tensor region covers only the bytes its view touches; storage bytes no tensor
references are left uncovered and reported by S2.

Legacy format: five pickles in a row (magic number, protocol, sys info, the
object, the sorted storage keys), then each storage as an int64 element count
followed by its raw bytes.
"""
from __future__ import annotations

import io
import pickletools
import struct
import zipfile
from pathlib import Path

from ..model import ITEMSIZE, Finding, ModelFile, Region, TensorInfo, prod
from ..pickle_scan import Stub, StorageRef, TensorRef, safe_load

LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
KNOWN_ENTRIES = {"version", ".format_version", "byteorder", ".storage_alignment",
                 ".data/serialization_id", ".data/version"}
_SKIP_SEGMENTS = {"_parameters", "_buffers", "_modules"}


# --- tensors from the unpickled structure ----------------------------------

def walk_tensors(obj, path=(), seen=None, depth=0):
    """Yield (name, TensorRef) for every tensor reachable from obj."""
    if seen is None:
        seen = set()
    if depth > 64 or id(obj) in seen:
        return
    if isinstance(obj, TensorRef):
        seen.add(id(obj))
        yield ".".join(path) or "<root>", obj
        return
    if isinstance(obj, (dict, list, tuple, Stub)):
        seen.add(id(obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_tensors(v, path + (str(k),), seen, depth + 1)
        if isinstance(getattr(obj, "state", None), dict):
            yield from walk_tensors(obj.state, path, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk_tensors(v, path + (str(i),), seen, depth + 1)
    elif isinstance(obj, Stub):
        for part in (obj.state, obj.items, obj.args, obj.kwargs):
            if isinstance(part, dict):
                for k, v in part.items():
                    seg = () if k in _SKIP_SEGMENTS else (str(k),)
                    yield from walk_tensors(v, path + seg, seen, depth + 1)
            elif isinstance(part, (list, tuple)):
                yield from walk_tensors(part, path, seen, depth + 1)


def _view_span(t: TensorRef) -> tuple[int, int, bool]:
    """(first element, end element, contiguous) touched by a strided view."""
    n = prod(t.shape)
    if n == 0:
        return t.offset, t.offset, True
    contiguous, expect = True, 1
    for size, stride in zip(reversed(t.shape), reversed(t.stride)):
        if size != 1 and stride != expect:
            contiguous = False
        expect *= size
    last = t.offset + sum((s - 1) * st for s, st in zip(t.shape, t.stride))
    return t.offset, last + 1, contiguous


def _add_tensors(mf: ModelFile, obj, storage_base: dict[str, int | None], storage_len: dict[str, int]):
    for name, t in walk_tensors(obj):
        st = t.storage
        if st is None or st.key not in storage_base:
            mf.findings.append(Finding("parse", "medium", "missing-storage",
                                       f"tensor {name!r} refers to a storage that is not in the file", {"tensor": name}))
            continue
        isz = ITEMSIZE.get(st.dtype)
        if isz is None:
            mf.findings.append(Finding("parse", "info", "unknown-dtype", f"tensor {name!r} has dtype {st.dtype}"))
            continue
        first, end, contiguous = _view_span(t)
        if first < 0 or end * isz > storage_len[st.key]:
            mf.findings.append(Finding("parse", "high", "tensor-out-of-bounds",
                                       f"tensor {name!r} reaches past its storage", {"tensor": name}))
            continue
        base = storage_base[st.key]
        nbytes = prod(t.shape) * isz
        offset = base + first * isz if (base is not None and contiguous) else None
        mf.tensors.append(TensorInfo(name, st.dtype, tuple(t.shape), offset, nbytes))
        if base is not None:
            mf.regions.append(Region(base + first * isz, base + end * isz, "tensor", name))


# --- zip format --------------------------------------------------------------

def _eocd_regions(f, size: int) -> list[Region]:
    """Central directory, zip64 records, end-of-central-directory record and comment."""
    tail = min(size, 22 + 65535 + 20 + 56)
    f.seek(size - tail)
    buf = f.read(tail)
    i = buf.rfind(b"PK\x05\x06")
    if i < 0:
        return []
    eocd = size - tail + i
    (_, _, _, _, _, cd_size, cd_off, clen) = struct.unpack("<IHHHHIIH", buf[i:i + 22])
    regions = [Region(eocd, min(eocd + 22 + clen, size), "structure", "end of central directory")]
    if i >= 20 and buf[i - 20:i - 16] == b"PK\x06\x07":
        loc = eocd - 20
        regions.append(Region(loc, eocd, "structure", "zip64 locator"))
        (z_off,) = struct.unpack("<Q", buf[i - 12:i - 4])
        f.seek(z_off)
        rec = f.read(56)
        if rec[:4] == b"PK\x06\x06":
            (rec_size,) = struct.unpack("<Q", rec[4:12])
            (cd_size, cd_off) = struct.unpack("<QQ", rec[40:56])
            regions.append(Region(z_off, z_off + 12 + rec_size, "structure", "zip64 end of central directory"))
    # cd_off is relative to the start of the archive; zipfile resolves prepended data for us later.
    regions.append(Region(cd_off, cd_off + cd_size, "structure", "central directory"))
    return regions


def parse_zip(path: Path, f) -> ModelFile:
    size = path.stat().st_size
    mf = ModelFile(path=path, format="pytorch-zip", size=size)
    try:
        zf = zipfile.ZipFile(f)
        infos = zf.infolist()
    except (zipfile.BadZipFile, OSError):
        mf.findings.append(Finding("parse", "high", "bad-zip", "not a readable zip archive"))
        return mf

    regions = _eocd_regions(f, size)
    cd = [r for r in regions if r.label == "central directory"]
    # Offsets stored in the archive are relative to its start; zipfile has worked out
    # how many bytes were prepended (start_dir is the true central directory offset).
    shift = zf.start_dir - cd[0].start if cd and hasattr(zf, "start_dir") else 0
    for r in regions:
        if r.label in ("central directory", "zip64 end of central directory"):
            r = Region(r.start + shift, r.end + shift, r.kind, r.label)
        mf.regions.append(r)
    if zf.comment:
        mf.findings.append(Finding("S2", "medium", "zip-comment",
                                   f"archive carries a {len(zf.comment)}-byte zip comment", {"bytes": len(zf.comment)}))

    pkl = [i for i in infos if i.filename.endswith("/data.pkl") or i.filename == "data.pkl"]
    if not pkl:
        mf.findings.append(Finding("parse", "medium", "no-data-pkl", "zip has no data.pkl; not a torch checkpoint"))
    prefix = pkl[0].filename[: -len("data.pkl")] if pkl else ""
    if any(i.filename.startswith(prefix + "code/") for i in infos):
        mf.format = "torchscript"

    storage_base: dict[str, int | None] = {}
    storage_len: dict[str, int] = {}
    for info in infos:
        f.seek(info.header_offset)
        lh = f.read(30)
        if lh[:4] != b"PK\x03\x04":
            mf.findings.append(Finding("parse", "high", "bad-local-header", f"entry {info.filename!r} has no local header"))
            continue
        flags, nlen, xlen = struct.unpack("<H", lh[6:8])[0], *struct.unpack("<HH", lh[26:30])
        data_start = info.header_offset + 30 + nlen + xlen
        data_end = data_start + info.compress_size
        mf.regions.append(Region(info.header_offset, data_start, "structure", f"local header {info.filename}"))
        if flags & 0x08:  # data descriptor follows the data
            f.seek(data_end)
            sig = f.read(4) == b"PK\x07\x08"
            big = info.compress_size >= 0xFFFFFFFF or info.file_size >= 0xFFFFFFFF
            dlen = (4 if sig else 0) + 4 + (16 if big else 8)
            mf.regions.append(Region(data_end, data_end + dlen, "structure", f"data descriptor {info.filename}"))

        rel = info.filename[len(prefix):] if info.filename.startswith(prefix) else None
        stored = info.compress_type == zipfile.ZIP_STORED
        if rel == "data.pkl" or (rel is not None and rel.endswith(".pkl")):
            mf.regions.append(Region(data_start, data_end, "header", info.filename))
            if stored:
                mf.pickles.append((info.filename, data_start, data_end))
        elif rel is not None and rel.startswith("data/") and "/" not in rel[5:]:
            key = rel[5:]
            storage_base[key] = data_start if stored else None
            storage_len[key] = info.file_size
            mf.regions.append(Region(data_start, data_end, "container", f"storage {key}"))
        elif rel in KNOWN_ENTRIES or (mf.format == "torchscript" and rel is not None):
            mf.regions.append(Region(data_start, data_end, "structure", info.filename))
        else:
            mf.regions.append(Region(data_start, data_end, "container", info.filename))
            mf.findings.append(Finding("S2", "medium", "unexpected-zip-entry",
                                       f"archive contains {info.filename!r}, which torch does not use",
                                       {"entry": info.filename, "bytes": info.file_size}))
        if not stored:
            mf.findings.append(Finding("parse", "low", "compressed-entry",
                                       f"entry {info.filename!r} is compressed; torch writes entries uncompressed",
                                       {"entry": info.filename}))

    if pkl:
        data = zf.read(pkl[0])
        if pkl[0].compress_type != zipfile.ZIP_STORED:
            mf.inline_pickles[pkl[0].filename] = data
        res = safe_load(io.BytesIO(data))
        if res.error is None:
            _add_tensors(mf, res.obj, storage_base, storage_len)
        else:
            mf.findings.append(Finding("parse", "medium", "structure-unreadable",
                                       "could not rebuild the checkpoint structure"))
    return mf


# --- legacy format and plain pickles ------------------------------------------

def _pickle_end(f) -> int:
    """Offset just past the STOP opcode of the pickle at the current position."""
    for op, _, _ in pickletools.genops(f):
        if op.name == "STOP":
            return f.tell()
    raise ValueError("pickle has no STOP opcode")


def parse_pickles(path: Path, f) -> ModelFile:
    size = path.stat().st_size
    mf = ModelFile(path=path, format="pickle", size=size)
    objs = []
    while f.tell() < size and len(objs) < 5:
        start = f.tell()
        if objs and f.read(1) != b"\x80":  # only the first pickle may use protocol 0/1
            break
        f.seek(start)
        try:
            end = _pickle_end(f)
        except Exception as e:
            mf.findings.append(Finding("parse", "medium", "malformed-pickle", f"pickle at {start} does not parse cleanly"))
            mf.pickles.append((f"pickle@{start}", start, size))
            mf.regions.append(Region(start, size, "header", f"pickle@{start}"))
            return mf
        f.seek(start)
        mf.pickles.append((f"pickle@{start}", start, end))
        mf.regions.append(Region(start, end, "header", f"pickle@{start}"))
        objs.append(safe_load(io.BytesIO(f.read(end - start))).obj)
        f.seek(end)
        if len(objs) == 1 and objs[0] != LEGACY_MAGIC:
            return mf  # a plain pickle; anything after it is for S2

    if len(objs) < 5 or objs[0] != LEGACY_MAGIC or not isinstance(objs[4], list):
        mf.findings.append(Finding("parse", "medium", "bad-legacy", "legacy torch file is incomplete"))
        return mf
    mf.format = "pytorch-legacy"
    main, keys = objs[3], [str(k) for k in objs[4]]
    dtypes = {t.storage.key: t.storage.dtype for _, t in walk_tensors(main) if t.storage}
    storage_base, storage_len = {}, {}
    pos = f.tell()
    for key in keys:
        if pos + 8 > size:
            mf.findings.append(Finding("parse", "high", "truncated", f"storage {key} is missing"))
            break
        f.seek(pos)
        (numel,) = struct.unpack("<q", f.read(8))
        isz = ITEMSIZE.get(dtypes.get(key, "?"))
        if isz is None or numel < 0 or pos + 8 + numel * isz > size:
            mf.findings.append(Finding("parse", "high", "bad-storage", f"storage {key} cannot be located"))
            break
        mf.regions.append(Region(pos, pos + 8, "structure", f"storage {key} size"))
        mf.regions.append(Region(pos + 8, pos + 8 + numel * isz, "container", f"storage {key}"))
        storage_base[key], storage_len[key] = pos + 8, numel * isz
        pos += 8 + numel * isz
    _add_tensors(mf, main, storage_base, storage_len)
    return mf
