// Run every ported check on one file and produce the same JSON-serialisable
// report object as stowaway/report.py. The file is read through a Reader, so a
// browser File is scanned in slices, never loaded whole.
import { Reader, withPrefix } from "./bytes.js";
import { Sha256 } from "./sha256.js";
import * as coverage from "./coverage.js";
import { Finding, ModelFile, newModelFile, SEVERITIES } from "./model.js";
import { parse as parseSafetensors } from "./formats/safetensors.js";
import { parse as parseGguf } from "./formats/gguf.js";
import { parseZip, parsePickles } from "./formats/pytorch.js";
import { analysePickles, isOldPickle } from "./pickle_scan.js";
import { analyseWeights, BaseModel } from "./weights.js";

export const REPORT_VERSION = 1;

function detect(r: Reader): string {
  const h = r.head(9);
  if (h.length >= 4 && h[0] === 0x47 && h[1] === 0x47 && h[2] === 0x55 && h[3] === 0x46) return "gguf";
  if (h.length >= 9 && h[8] === 0x7b /* { */) {
    const n = r.u64(0);
    if (8 + n <= r.size) return "safetensors";
  }
  if (h.length >= 1 && h[0] === 0x80) return "pickle";
  if (h.length >= 4 && h[0] === 0x50 && h[1] === 0x4b && h[2] === 0x03 && h[3] === 0x04) return "zip";
  // protocols 0 and 1 have no 0x80 marker
  if (withPrefix(r, (prefix, whole) => isOldPickle(prefix, whole))) return "pickle";
  return "unknown";
}

export function openModel(r: Reader): ModelFile {
  const kind = detect(r);
  if (kind === "safetensors") return parseSafetensors(r);
  if (kind === "gguf") return parseGguf(r);
  if (kind === "zip") return parseZip(r);
  if (kind === "pickle") return parsePickles(r);
  // Exactly as stowaway/formats/__init__.py for an unrecognised file.
  const mf = newModelFile("unknown", r.size);
  mf.findings.push({
    check: "parse", severity: "medium", code: "unknown-format",
    message: "not a recognised model format", detail: {},
  });
  return mf;
}

const asReader = (x: Uint8Array | Reader) => (x instanceof Reader ? x : Reader.of(x));

/** Parse a base model for S5. Its tensors are read only while each is compared. */
export function openBase(x: Uint8Array | Reader): BaseModel {
  const r = asReader(x);
  return { mf: openModel(r), r };
}

const HASH_CHUNK = 16 << 20;

function sha256Hex(r: Reader): string {
  const h = new Sha256();
  for (let o = 0; o < r.size; o += HASH_CHUNK) h.update(r.slice(o, Math.min(r.size, o + HASH_CHUNK)));
  return h.hex();
}

export async function scan(x: Uint8Array | Reader, base?: BaseModel): Promise<Record<string, unknown>> {
  const r = asReader(x);
  const mf = openModel(r);
  const findings: Finding[] = [...mf.findings];
  for (const [label, start, end] of mf.pickles) findings.push(...analysePickles(label, r.slice(start, end)));
  for (const [label, data] of Object.entries(mf.inlinePickles)) findings.push(...analysePickles(label, data));
  if (mf.format !== "unknown") {
    findings.push(...coverage.check(mf, r));
    findings.push(...analyseWeights(mf, r, base));
  }
  const digest = sha256Hex(r);

  const rank = Object.fromEntries(SEVERITIES.map((s, i) => [s, i]));
  findings.sort((a, b) =>
    (rank[b.severity]! - rank[a.severity]!) ||
    cmp(a.check, b.check) || cmp(a.code, b.code) || cmp(a.message, b.message));
  const worst = findings.reduce((w, x) => Math.max(w, rank[x.severity]!), -1);
  const verdict = worst < rank["medium"]! ? "clean" : worst < rank["high"]! ? "suspicious" : "dangerous";

  const dtypes: Record<string, number> = {};
  for (const t of mf.tensors) dtypes[t.dtype] = (dtypes[t.dtype] ?? 0) + 1;
  const sortedDtypes: Record<string, number> = {};
  for (const k of Object.keys(dtypes).sort()) sortedDtypes[k] = dtypes[k]!;

  return {
    report_version: REPORT_VERSION,
    file: "", // filled by the caller; Python uses the file name
    sha256: digest,
    size: mf.size,
    format: mf.format,
    verdict,
    tensors: {
      count: mf.tensors.length,
      bytes: mf.tensors.reduce((s, t) => s + t.nbytes, 0),
      dtypes: sortedDtypes,
    },
    findings: findings.map((x) => ({
      check: x.check, severity: x.severity, code: x.code, message: x.message, detail: x.detail,
    })),
  };
}

function cmp(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}
