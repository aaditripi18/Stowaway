// Run every ported check on one file and produce the same JSON-serialisable
// report object as stowaway/report.py. scan() is async because the file hash
// uses Web Crypto (available in the browser and in Node).
import { Reader } from "./bytes.js";
import * as coverage from "./coverage.js";
import { Finding, ModelFile, newModelFile, SEVERITIES } from "./model.js";
import { parse as parseSafetensors } from "./formats/safetensors.js";
import { parse as parseGguf } from "./formats/gguf.js";
import { analysePickles } from "./pickle_scan.js";
import { analyseWeights, RawFloats } from "./weights.js";

export const REPORT_VERSION = 1;

function detect(r: Reader): string {
  if (r.size >= 4 && r.bytes[0] === 0x47 && r.bytes[1] === 0x47 && r.bytes[2] === 0x55 && r.bytes[3] === 0x46)
    return "gguf";
  if (r.size >= 9 && r.bytes[8] === 0x7b /* { */) {
    const n = r.u64(0);
    if (8 + n <= r.size) return "safetensors";
  }
  if (r.size >= 1 && r.bytes[0] === 0x80) return "pickle";
  if (r.size >= 4 && r.bytes[0] === 0x50 && r.bytes[1] === 0x4b && r.bytes[2] === 0x03 && r.bytes[3] === 0x04)
    return "zip";
  return "unknown";
}

export function openModel(r: Reader): ModelFile {
  const kind = detect(r);
  if (kind === "safetensors") return parseSafetensors(r.bytes);
  if (kind === "gguf") return parseGguf(r.bytes);
  if (kind === "unknown") {
    // Exactly as stowaway/formats/__init__.py for an unrecognised file.
    const mf = newModelFile("unknown", r.size);
    mf.findings.push({
      check: "parse", severity: "medium", code: "unknown-format",
      message: "not a recognised model format", detail: {},
    });
    return mf;
  }
  // Recognised (gguf/pickle/zip) but not yet ported to TypeScript. The Python
  // core parses these; the browser build says so rather than guessing. These
  // formats are excluded from the parity fixtures until they are ported.
  const mf = newModelFile("unsupported", r.size);
  mf.findings.push({
    check: "parse", severity: "medium", code: "unsupported-format",
    message: "this build can only scan safetensors; use the Python tool for other formats",
    detail: { recognised_as: kind },
  });
  return mf;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function scan(bytes: Uint8Array, base?: RawFloats): Promise<Record<string, unknown>> {
  const r = new Reader(bytes);
  const mf = openModel(r);
  const findings: Finding[] = [...mf.findings];
  for (const [label, start, end] of mf.pickles) findings.push(...analysePickles(label, r.slice(start, end)));
  for (const [label, data] of Object.entries(mf.inlinePickles)) findings.push(...analysePickles(label, data));
  if (mf.format !== "unknown" && mf.format !== "unsupported") {
    findings.push(...coverage.check(mf, r));
    findings.push(...analyseWeights(mf, r, base));
  }
  const digest = await sha256Hex(bytes);

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
