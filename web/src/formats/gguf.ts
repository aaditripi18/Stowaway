// GGUF (llama.cpp) parser. Mirrors stowaway/formats/gguf.py. The report does not
// include GGUF metadata, so values are parsed only to advance the cursor and to
// read general.alignment; tensors, regions and findings match the Python core.
import { finding, ModelFile, newModelFile, prod, region, TensorInfo } from "../model.js";
import { pyRepr } from "../pyrepr.js";

// ggml_type -> [name, elements per block, bytes per block]
const GGML_TYPES: Record<number, [string, number, number]> = {
  0: ["F32", 1, 4], 1: ["F16", 1, 2], 2: ["Q4_0", 32, 18], 3: ["Q4_1", 32, 20],
  6: ["Q5_0", 32, 22], 7: ["Q5_1", 32, 24], 8: ["Q8_0", 32, 34], 9: ["Q8_1", 32, 36],
  10: ["Q2_K", 256, 84], 11: ["Q3_K", 256, 110], 12: ["Q4_K", 256, 144],
  13: ["Q5_K", 256, 176], 14: ["Q6_K", 256, 210], 15: ["Q8_K", 256, 292],
  16: ["IQ2_XXS", 256, 66], 17: ["IQ2_XS", 256, 74], 18: ["IQ3_XXS", 256, 98],
  19: ["IQ1_S", 256, 50], 20: ["IQ4_NL", 32, 18], 21: ["IQ3_S", 256, 110],
  22: ["IQ2_S", 256, 82], 23: ["IQ4_XS", 256, 136], 24: ["I8", 1, 1],
  25: ["I16", 1, 2], 26: ["I32", 1, 4], 27: ["I64", 1, 8], 28: ["F64", 1, 8],
  29: ["IQ1_M", 256, 56], 30: ["BF16", 1, 2], 34: ["TQ1_0", 256, 54], 35: ["TQ2_0", 256, 66],
};

const SCALAR_SIZE: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
const STRING = 8, ARRAY = 9;
const MAX_STRING = 64 * 1024 * 1024;
const MAX_ARRAY = 1 << 28;

class GgufReader {
  pos = 0;
  constructor(readonly b: Uint8Array, readonly dv: DataView) {}
  take(n: number): Uint8Array { if (this.pos + n > this.b.length) throw new Error("eof"); const s = this.b.subarray(this.pos, this.pos + n); this.pos += n; return s; }
  u32(): number { const v = this.dv.getUint32(this.pos, true); this.pos += 4; return v; }
  u64(): number { const v = this.dv.getBigUint64(this.pos, true); this.pos += 8; if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("huge"); return Number(v); }
  string(): string { const n = this.u64(); if (n > MAX_STRING) throw new Error("string too long"); return new TextDecoder("utf-8").decode(this.take(n)); }
  scalar(vtype: number): number { // returns the numeric value (enough for general.alignment)
    const sz = SCALAR_SIZE[vtype]!; const p = this.pos; this.pos += sz;
    switch (vtype) {
      case 0: return this.dv.getUint8(p); case 1: return this.dv.getInt8(p);
      case 2: return this.dv.getUint16(p, true); case 3: return this.dv.getInt16(p, true);
      case 4: return this.dv.getUint32(p, true); case 5: return this.dv.getInt32(p, true);
      case 6: return this.dv.getFloat32(p, true); case 7: return this.dv.getUint8(p);
      case 10: return Number(this.dv.getBigUint64(p, true)); case 11: return Number(this.dv.getBigInt64(p, true));
      case 12: return this.dv.getFloat64(p, true); default: throw new Error("bad scalar");
    }
  }
  value(vtype: number): unknown {
    if (vtype in SCALAR_SIZE) return this.scalar(vtype);
    if (vtype === STRING) return this.string();
    if (vtype === ARRAY) {
      const itype = this.u32(); const count = this.u64();
      if (count > MAX_ARRAY) throw new Error("array too long");
      for (let i = 0; i < count; i++) this.value(itype); // consume; metadata not reported
      return null;
    }
    throw new Error("unknown metadata value type " + vtype);
  }
}

const align = (x: number, a: number) => Math.floor((x + a - 1) / a) * a;

export function parse(bytes: Uint8Array): ModelFile {
  const mf = newModelFile("gguf", bytes.byteLength);
  const r = new GgufReader(bytes, new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  let version = 0, nTensors = 0;
  const meta: Record<string, unknown> = {};
  const infos: Array<[string, number[], number, number]> = [];
  try {
    const magic = r.take(4);
    if (!(magic[0] === 0x47 && magic[1] === 0x47 && magic[2] === 0x55 && magic[3] === 0x46)) throw new Error("bad magic");
    version = r.u32();
    if (version !== 2 && version !== 3) mf.findings.push(finding("parse", "low", "gguf-version", `GGUF version ${version} is not 2 or 3`));
    nTensors = r.u64();
    const nKv = r.u64();
    meta["gguf.version"] = version;
    for (let i = 0; i < nKv; i++) { const key = r.string(); meta[key] = r.value(r.u32()); }
    for (let i = 0; i < nTensors; i++) {
      const name = r.string();
      const nDims = r.u32();
      if (nDims > 8) throw new Error(`tensor ${name} has ${nDims} dims`);
      const ne: number[] = [];
      for (let d = 0; d < nDims; d++) ne.push(r.u64());
      infos.push([name, ne, r.u32(), r.u64()]);
    }
  } catch {
    mf.findings.push(finding("parse", "high", "bad-gguf", "could not parse the GGUF header"));
    return mf;
  }

  mf.metadata = meta;
  const headerEnd = r.pos;
  let alignment = meta["general.alignment"];
  if (typeof alignment !== "number" || !Number.isInteger(alignment) || alignment <= 0 || (alignment & (alignment - 1)) !== 0) {
    if (alignment !== undefined) mf.findings.push(finding("parse", "medium", "bad-alignment", "general.alignment is not a positive power of two"));
    alignment = 32;
  }
  const a = alignment as number;
  const dataStart = align(headerEnd, a);
  const size = bytes.byteLength;
  mf.regions.push(region(0, headerEnd, "header", "gguf header"));
  mf.regions.push(region(headerEnd, Math.min(dataStart, size), "padding", "header alignment"));

  const spans: Array<[number, number, string]> = [];
  for (const [name, ne, tcode, rel] of infos) {
    const shape = [...ne].reverse();
    const t = GGML_TYPES[tcode];
    if (t === undefined) {
      mf.findings.push(finding("parse", "low", "unknown-dtype", `tensor ${pyRepr(name)} has ggml type ${tcode}`));
      mf.tensors.push({ name, dtype: `ggml_${tcode}`, shape, offset: dataStart + rel, nbytes: 0 } as TensorInfo);
      continue;
    }
    const [tname, block, bsize] = t;
    const n = prod(shape);
    if (n % block) mf.findings.push(finding("parse", "high", "bad-block", `tensor ${pyRepr(name)}: ${n} elements is not a multiple of ${block}`));
    const nbytes = Math.floor(n / block) * bsize;
    const start = dataStart + rel;
    if (rel % a) mf.findings.push(finding("parse", "medium", "unaligned-tensor", `tensor ${pyRepr(name)} is not aligned`));
    if (start + nbytes > size) {
      mf.findings.push(finding("parse", "high", "tensor-out-of-bounds", `tensor ${pyRepr(name)} points outside the file`, { tensor: name }));
      continue;
    }
    mf.tensors.push({ name, dtype: tname, shape, offset: start, nbytes });
    mf.regions.push(region(start, start + nbytes, "tensor", name));
    spans.push([start, start + nbytes, name]);
  }

  spans.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  for (let i = 0; i < spans.length; i++) {
    const [s, e, nm] = spans[i]!;
    const nxt = i + 1 < spans.length ? spans[i + 1]![0] : size;
    if (i + 1 < spans.length && nxt < e) {
      mf.findings.push(finding("parse", "high", "overlapping-tensors",
        `tensors ${pyRepr(nm)} and ${pyRepr(spans[i + 1]![2])} share bytes`, { tensors: [nm, spans[i + 1]![2]] }));
    }
    const padEnd = Math.min(align(e, a), nxt);
    if (padEnd > e) mf.regions.push(region(e, padEnd, "padding", `alignment after ${nm}`));
  }
  return mf;
}
