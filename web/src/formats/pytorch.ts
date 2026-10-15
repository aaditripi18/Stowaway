// PyTorch checkpoints: the zip format (torch >= 1.6), the legacy format, and
// plain pickles. Mirrors stowaway/formats/pytorch.py. Storages are mapped to the
// tensors that view them (via the restricted pickle VM), so bytes no tensor
// references show up as S2 gaps; the embedded data.pkl is scanned by S1.
import { ITEMSIZE, Finding, finding, ModelFile, newModelFile, prod, region, TensorInfo } from "../model.js";
import { pyRepr } from "../pyrepr.js";
import { safeLoad, walkTensors, TensorRef, StorageRef, SafeUnpickler } from "../pickle_vm.js";

const KNOWN_ENTRIES = new Set(["version", ".format_version", "byteorder", ".storage_alignment", ".data/serialization_id", ".data/version"]);
const LEGACY_MAGIC = 0x1950a86a20f9469cn;  // low 64 bits are enough to recognise the magic
const SKIP = new Set(["_parameters", "_buffers", "_modules"]);

interface ZipEntry { filename: string; headerOffset: number; compressSize: number; fileSize: number; method: number; flags: number }

function u16(dv: DataView, o: number) { return dv.getUint16(o, true); }
function u32(dv: DataView, o: number) { return dv.getUint32(o, true); }

function eq(b: Uint8Array, o: number, sig: number[]): boolean {
  for (let i = 0; i < sig.length; i++) if (b[o + i] !== sig[i]) return false;
  return true;
}

// --- view span & tensor mapping (pytorch._view_span / _add_tensors) ---------

function viewSpan(t: TensorRef): [number, number, boolean] {
  const n = prod(t.shape);
  if (n === 0) return [t.offset, t.offset, true];
  let contiguous = true, expect = 1;
  for (let i = t.shape.length - 1; i >= 0; i--) {
    const size = t.shape[i]!, stride = t.stride[i] ?? 0;
    if (size !== 1 && stride !== expect) contiguous = false;
    expect *= size;
  }
  let last = t.offset;
  for (let i = 0; i < t.shape.length; i++) last += (t.shape[i]! - 1) * (t.stride[i] ?? 0);
  return [t.offset, last + 1, contiguous];
}

function addTensors(mf: ModelFile, obj: unknown, storageBase: Map<string, number | null>, storageLen: Map<string, number>): void {
  for (const [name, t] of walkTensors(obj)) {
    const st = t.storage as StorageRef | null;
    if (st === null || !storageBase.has(st.key)) {
      mf.findings.push(finding("parse", "medium", "missing-storage",
        `tensor ${pyRepr(name)} refers to a storage that is not in the file`, { tensor: name }));
      continue;
    }
    const isz = ITEMSIZE[st.dtype];
    if (isz === undefined) {
      mf.findings.push(finding("parse", "info", "unknown-dtype", `tensor ${pyRepr(name)} has dtype ${st.dtype}`));
      continue;
    }
    const [first, end, contiguous] = viewSpan(t);
    if (first < 0 || end * isz > storageLen.get(st.key)!) {
      mf.findings.push(finding("parse", "high", "tensor-out-of-bounds", `tensor ${pyRepr(name)} reaches past its storage`, { tensor: name }));
      continue;
    }
    const base = storageBase.get(st.key)!;
    const nbytes = prod(t.shape) * isz;
    const offset = base !== null && contiguous ? base + first * isz : null;
    mf.tensors.push({ name, dtype: st.dtype, shape: t.shape, offset, nbytes } as TensorInfo);
    if (base !== null) mf.regions.push(region(base + first * isz, base + end * isz, "tensor", name));
  }
  void SKIP;
}

// --- zip parsing ------------------------------------------------------------

function u64le(dv: DataView, o: number): number {
  const v = dv.getBigUint64(o, true);
  return v > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(v);
}

function eocdRegions(b: Uint8Array, dv: DataView, size: number): { regions: ReturnType<typeof region>[]; cdOff: number; cdSize: number; total: number; commentLen: number } {
  const tail = Math.min(size, 22 + 65535 + 20 + 56);
  const start = size - tail;
  let i = -1;
  for (let p = tail - 22; p >= 0; p--) if (eq(b, start + p, [0x50, 0x4b, 0x05, 0x06])) { i = start + p; break; }
  if (i < 0) return { regions: [], cdOff: 0, cdSize: 0, total: 0, commentLen: 0 };
  let cdSize = u32(dv, i + 12), cdOff = u32(dv, i + 16);
  const clen = u16(dv, i + 20), total = u16(dv, i + 10);
  const regions = [region(i, Math.min(i + 22 + clen, size), "structure", "end of central directory")];
  // zip64: a locator (PK0607) sits just before the EOCD and points at a zip64
  // end-of-central-directory record (PK0606). torch.save writes these.
  if (i >= 20 && eq(b, i - 20, [0x50, 0x4b, 0x06, 0x07])) {
    regions.push(region(i - 20, i, "structure", "zip64 locator"));
    const zOff = u64le(dv, i - 12);
    if (zOff + 56 <= size && eq(b, zOff, [0x50, 0x4b, 0x06, 0x06])) {
      const recSize = u64le(dv, zOff + 4);
      cdSize = u64le(dv, zOff + 40);
      cdOff = u64le(dv, zOff + 48);
      regions.push(region(zOff, zOff + 12 + recSize, "structure", "zip64 end of central directory"));
    }
  }
  regions.push(region(cdOff, cdOff + cdSize, "structure", "central directory"));
  return { regions, cdOff, cdSize, total, commentLen: clen };
}

function readCentralDirectory(b: Uint8Array, dv: DataView, cdOff: number, total: number): ZipEntry[] {
  const entries: ZipEntry[] = [];
  let p = cdOff;
  for (let n = 0; n < total; n++) {
    if (!eq(b, p, [0x50, 0x4b, 0x01, 0x02])) break;
    const flags = u16(dv, p + 8), method = u16(dv, p + 10);
    const compressSize = u32(dv, p + 20), fileSize = u32(dv, p + 24);
    const nlen = u16(dv, p + 28), xlen = u16(dv, p + 30), clen = u16(dv, p + 32);
    const headerOffset = u32(dv, p + 42);
    const filename = new TextDecoder("utf-8").decode(b.subarray(p + 46, p + 46 + nlen));
    entries.push({ filename, headerOffset, compressSize, fileSize, method, flags });
    p += 46 + nlen + xlen + clen;
  }
  return entries;
}

export function parseZip(bytes: Uint8Array): ModelFile {
  const size = bytes.byteLength;
  const mf = newModelFile("pytorch-zip", size);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocd = eocdRegions(bytes, dv, size);
  if (eocd.total === 0 && eocd.cdSize === 0) {
    mf.findings.push(finding("parse", "high", "bad-zip", "not a readable zip archive"));
    return mf;
  }
  const infos = readCentralDirectory(bytes, dv, eocd.cdOff, eocd.total);
  for (const r of eocd.regions) mf.regions.push(r);

  const pkl = infos.filter((i) => i.filename.endsWith("/data.pkl") || i.filename === "data.pkl");
  if (pkl.length === 0) mf.findings.push(finding("parse", "medium", "no-data-pkl", "zip has no data.pkl; not a torch checkpoint"));
  const prefix = pkl.length ? pkl[0]!.filename.slice(0, -"data.pkl".length) : "";
  if (infos.some((i) => i.filename.startsWith(prefix + "code/"))) mf.format = "torchscript";

  const storageBase = new Map<string, number | null>();
  const storageLen = new Map<string, number>();
  for (const info of infos) {
    if (!eq(bytes, info.headerOffset, [0x50, 0x4b, 0x03, 0x04])) {
      mf.findings.push(finding("parse", "high", "bad-local-header", `entry ${pyRepr(info.filename)} has no local header`));
      continue;
    }
    const flags = u16(dv, info.headerOffset + 6), nlen = u16(dv, info.headerOffset + 26), xlen = u16(dv, info.headerOffset + 28);
    const dataStart = info.headerOffset + 30 + nlen + xlen;
    const dataEnd = dataStart + info.compressSize;
    mf.regions.push(region(info.headerOffset, dataStart, "structure", `local header ${info.filename}`));
    if (flags & 0x08) {
      const sig = eq(bytes, dataEnd, [0x50, 0x4b, 0x07, 0x08]);
      const big = info.compressSize >= 0xffffffff || info.fileSize >= 0xffffffff;
      const dlen = (sig ? 4 : 0) + 4 + (big ? 16 : 8);
      mf.regions.push(region(dataEnd, dataEnd + dlen, "structure", `data descriptor ${info.filename}`));
    }
    const rel = info.filename.startsWith(prefix) ? info.filename.slice(prefix.length) : null;
    const stored = info.method === 0;
    if (rel === "data.pkl" || (rel !== null && rel.endsWith(".pkl"))) {
      mf.regions.push(region(dataStart, dataEnd, "header", info.filename));
      if (stored) mf.pickles.push([info.filename, dataStart, dataEnd]);
    } else if (rel !== null && rel.startsWith("data/") && !rel.slice(5).includes("/")) {
      const key = rel.slice(5);
      storageBase.set(key, stored ? dataStart : null);
      storageLen.set(key, info.fileSize);
      mf.regions.push(region(dataStart, dataEnd, "container", `storage ${key}`));
    } else if ((rel !== null && KNOWN_ENTRIES.has(rel)) || (mf.format === "torchscript" && rel !== null)) {
      mf.regions.push(region(dataStart, dataEnd, "structure", info.filename));
    } else {
      mf.regions.push(region(dataStart, dataEnd, "container", info.filename));
      mf.findings.push(finding("S2", "medium", "unexpected-zip-entry",
        `archive contains ${pyRepr(info.filename)}, which torch does not use`, { entry: info.filename, bytes: info.fileSize }));
    }
    if (!stored) mf.findings.push(finding("parse", "low", "compressed-entry",
      `entry ${pyRepr(info.filename)} is compressed; torch writes entries uncompressed`, { entry: info.filename }));
  }

  if (pkl.length) {
    const p = pkl[0]!;
    if (p.method === 0) {
      const dataStart = (() => { const ho = p.headerOffset; return ho + 30 + u16(dv, ho + 26) + u16(dv, ho + 28); })();
      const data = bytes.subarray(dataStart, dataStart + p.compressSize);
      const res = safeLoad(data);
      if (res.error === null) addTensors(mf, res.obj, storageBase, storageLen);
      else mf.findings.push(finding("parse", "medium", "structure-unreadable", "could not rebuild the checkpoint structure"));
    }
  }
  return mf;
}

// --- plain pickle and legacy ------------------------------------------------

void LEGACY_MAGIC;

function pickleEnd(bytes: Uint8Array): number | null {
  // Walk opcodes to the STOP; return the offset just past it, or null if none.
  try {
    const u = new SafeUnpickler(bytes, new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    u.load();
    return u.pos;
  } catch { return null; }
}

export function parsePickles(bytes: Uint8Array): ModelFile {
  const size = bytes.byteLength;
  const mf = newModelFile("pickle", size);
  const end = pickleEnd(bytes);
  const label = "pickle@0";
  if (end === null) {
    mf.findings.push(finding("parse", "medium", "malformed-pickle", "pickle at 0 does not parse cleanly"));
    mf.pickles.push([label, 0, size]);
    mf.regions.push(region(0, size, "header", label));
    return mf;
  }
  mf.pickles.push([label, 0, end]);
  mf.regions.push(region(0, end, "header", label));
  // A legacy torch file (first pickle is the magic long) is read by the Python
  // tool; a plain standalone pickle is fully handled here (S1 + S2).
  return mf;
}
