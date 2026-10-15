// safetensors parser. Mirrors stowaway/formats/safetensors.py.
import { Reader, decodeUtf8 } from "../bytes.js";
import { finding, ITEMSIZE, ModelFile, newModelFile, prod, region, TensorInfo } from "../model.js";
import { pyRepr } from "../pyrepr.js";

const MAX_HEADER = 100 * 1024 * 1024;

// Parse the header JSON while detecting duplicate keys (which JSON.parse silently
// collapses). Returns the object plus the list of repeated keys.
function parseHeader(text: string): { header: Record<string, any>; duplicates: string[] } {
  const duplicates: string[] = [];
  const seen = new Set<string>();
  // Track duplicates via a reviver is not enough (reviver sees post-merge), so we
  // do a light scan: JSON.parse for the value, and a separate key tally.
  const header = JSON.parse(text);
  // Re-tokenise object keys at depth 1 to find repeats.
  let depth = 0, inStr = false, esc = false, expectKey = false;
  let key = "", capturing = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (inStr) {
      if (esc) { if (capturing) key += c; esc = false; }
      else if (c === "\\") { if (capturing) key += c; esc = true; }
      else if (c === '"') { inStr = false; }
      else if (capturing) key += c;
      continue;
    }
    if (c === '"') {
      inStr = true;
      if (depth === 1 && expectKey) { capturing = true; key = ""; }
      continue;
    }
    if (c === "{") { depth++; if (depth === 1) expectKey = true; }
    else if (c === "}") depth--;
    else if (c === "[") depth++;
    else if (c === "]") depth--;
    else if (c === ":" && depth === 1 && capturing) {
      capturing = false; expectKey = false;
      const decoded = JSON.parse('"' + key + '"');
      if (seen.has(decoded)) duplicates.push(decoded);
      seen.add(decoded);
    } else if (c === "," && depth === 1) expectKey = true;
  }
  return { header, duplicates };
}

export function parse(bytes: Uint8Array): ModelFile {
  const r = new Reader(bytes);
  const mf = newModelFile("safetensors", r.size);
  if (r.size < 8) {
    mf.findings.push(finding("parse", "high", "truncated", "file is shorter than the 8-byte header length"));
    return mf;
  }
  const n = r.u64(0);
  if (n > MAX_HEADER || 8 + n > r.size) {
    mf.findings.push(finding("parse", "high", "bad-header-length",
      `header length ${n} does not fit the file`, { header_length: n }));
    return mf;
  }
  mf.regions.push(region(0, 8 + n, "header", "safetensors header"));
  let header: Record<string, any>, duplicates: string[];
  try {
    const text = decodeUtf8(r.slice(8, 8 + n));
    ({ header, duplicates } = parseHeader(text));
  } catch (e) {
    mf.findings.push(finding("parse", "high", "bad-header-json", `header is not valid JSON: ${e}`));
    return mf;
  }

  if (duplicates.length) {
    mf.findings.push(finding("parse", "high", "duplicate-keys",
      `header repeats ${duplicates.length} key(s); loaders may disagree on which tensor is meant`,
      { keys: [...new Set(duplicates)].sort() }));
  }

  if ("__metadata__" in header) {
    const meta = header["__metadata__"];
    delete header["__metadata__"];
    mf.metadata = meta && typeof meta === "object" && !Array.isArray(meta)
      ? meta : { __invalid__: JSON.stringify(meta).slice(0, 200) };
  }

  const base = 8 + n;
  const spans: Array<[number, number, string]> = [];
  for (const name of Object.keys(header)) {
    const spec = header[name];
    let dtype: string, shape: number[], begin: number, end: number;
    try {
      dtype = spec.dtype;
      shape = (spec.shape as number[]).map((d) => Number(d));
      const offs = spec.data_offsets as number[];
      if (!Array.isArray(offs) || offs.length !== 2) throw new Error("bad");
      begin = Number(offs[0]);
      end = Number(offs[1]);
      if (typeof dtype !== "string" || !Array.isArray(spec.shape) || shape.some((d) => !Number.isInteger(d)))
        throw new Error("bad");
    } catch {
      mf.findings.push(finding("parse", "high", "bad-tensor-entry", `malformed entry for tensor ${pyRepr(name)}`));
      continue;
    }
    const nbytes = end - begin;
    if (begin < 0 || end < begin || base + end > r.size) {
      mf.findings.push(finding("parse", "high", "tensor-out-of-bounds",
        `tensor ${pyRepr(name)} points outside the file`, { tensor: name }));
      continue;
    }
    const itemsize = ITEMSIZE[dtype];
    if (itemsize === undefined) {
      mf.findings.push(finding("parse", "info", "unknown-dtype", `tensor ${pyRepr(name)} has dtype ${dtype}`));
    } else if (prod(shape) * itemsize !== nbytes) {
      mf.findings.push(finding("parse", "high", "size-mismatch",
        `tensor ${pyRepr(name)}: shape says ${prod(shape) * itemsize} bytes, offsets say ${nbytes}`,
        { tensor: name }));
    }
    const t: TensorInfo = { name, dtype, shape, offset: base + begin, nbytes };
    mf.tensors.push(t);
    mf.regions.push(region(base + begin, base + end, "tensor", name));
    spans.push([begin, end, name]);
  }

  spans.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  for (let i = 0; i + 1 < spans.length; i++) {
    const [, e1, n1] = spans[i]!;
    const [b2, , n2] = spans[i + 1]!;
    if (b2 < e1) {
      mf.findings.push(finding("parse", "high", "overlapping-tensors",
        `tensors ${pyRepr(n1)} and ${pyRepr(n2)} share bytes`, { tensors: [n1, n2] }));
    }
  }
  return mf;
}
