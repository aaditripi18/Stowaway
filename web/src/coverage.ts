// S2: every byte must belong to a header, a tensor or zero padding.
// Mirrors stowaway/coverage.py.
import { Reader } from "./bytes.js";
import { COVERING_KINDS, Finding, finding, ModelFile } from "./model.js";

const SAMPLE = 64 * 1024;

const MAGIC: Array<[number[], string]> = [
  [[0x4d, 0x5a], "Windows executable (MZ)"],
  [[0x7f, 0x45, 0x4c, 0x46], "ELF executable"],
  [[0xcf, 0xfa, 0xed, 0xfe], "Mach-O executable"],
  [[0xca, 0xfe, 0xba, 0xbe], "Mach-O universal / Java class"],
  [[0x23, 0x21], "script with shebang"],
  [[0x50, 0x4b, 0x03, 0x04], "zip archive"],
  [[0x1f, 0x8b], "gzip"],
  [[0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c], "7-zip archive"],
  [[0x52, 0x61, 0x72, 0x21], "rar archive"],
  [[0x25, 0x50, 0x44, 0x46], "PDF"],
  [[0x89, 0x50, 0x4e, 0x47], "PNG image"],
  [[0xff, 0xd8, 0xff], "JPEG image"],
  [[0x47, 0x47, 0x55, 0x46], "GGUF model"],
  [[0x80, 0x02], "pickle (protocol 2)"],
  [[0x80, 0x03], "pickle (protocol 3)"],
  [[0x80, 0x04], "pickle (protocol 4)"],
  [[0x80, 0x05], "pickle (protocol 5)"],
];

function startsWith(data: Uint8Array, magic: number[]): boolean {
  if (data.length < magic.length) return false;
  for (let i = 0; i < magic.length; i++) if (data[i] !== magic[i]) return false;
  return true;
}

export function sniff(head: Uint8Array): string | null {
  for (const [magic, what] of MAGIC) if (startsWith(head, magic)) return what;
  if (head.length >= 8 && head[4] === 0x66 && head[5] === 0x74 && head[6] === 0x79 && head[7] === 0x70)
    return "MP4/QuickTime media";
  return null;
}

export function entropy(data: Uint8Array): number {
  if (data.length === 0) return 0;
  const counts = new Array(256).fill(0);
  for (const b of data) counts[b]++;
  const n = data.length;
  let h = 0;
  for (const c of counts) if (c) { const p = c / n; h -= p * Math.log2(p); }
  return h;
}

// Round half away from zero to `digits`, matching Python's round() for the
// values here (entropy >= 0, so half-even and half-up agree except at exact .5
// ties, which these never hit in practice). Python round is banker's rounding;
// we match it below.
function roundPy(x: number, digits: number): number {
  const f = Math.pow(10, digits);
  const scaled = x * f;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let r: number;
  if (diff > 0.5) r = floor + 1;
  else if (diff < 0.5) r = floor;
  else r = floor % 2 === 0 ? floor : floor + 1; // banker's rounding
  return r / f;
}

export function gaps(mf: ModelFile): Array<[number, number]> {
  const spans = mf.regions
    .filter((r) => COVERING_KINDS.has(r.kind) && r.end > r.start)
    .map((r) => [r.start, r.end] as [number, number])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out: Array<[number, number]> = [];
  let pos = 0;
  for (const [s, e] of spans) {
    if (s > pos) out.push([pos, Math.min(s, mf.size)]);
    pos = Math.max(pos, e);
  }
  if (pos < mf.size) out.push([pos, mf.size]);
  return out.filter(([s, e]) => e > s);
}

function container(mf: ModelFile, start: number, end: number): string | null {
  for (const r of mf.regions) if (r.kind === "container" && r.start <= start && end <= r.end) return r.label;
  return null;
}

export function check(mf: ModelFile, r: Reader): Finding[] {
  const findings: Finding[] = [];
  for (const [start, end] of gaps(mf)) {
    const nbytes = end - start;
    const sample = r.slice(start, start + Math.min(nbytes, SAMPLE));
    const nonzero = sample.some((b) => b !== 0);
    const inside = container(mf, start, end);
    const what = sniff(sample);
    let code: string, where: string;
    if (inside && inside.startsWith("storage")) {
      code = "unreferenced-storage"; where = `${inside}: ${nbytes} bytes no tensor uses`;
    } else if (end === mf.size) {
      code = "trailing-data"; where = `${nbytes} bytes after the end of the model`;
    } else if (start === 0) {
      code = "leading-data"; where = `${nbytes} bytes before the model starts`;
    } else {
      code = "unaccounted-bytes"; where = `${nbytes} unaccounted bytes at offset ${start}`;
    }
    const allZero = !nonzero && nbytes <= SAMPLE;
    const severity = allZero ? "low" : (what || nbytes >= 1024) ? "high" : "medium";
    const msg = where + (what ? `, looks like ${what}` : "");
    findings.push(finding("S2", severity, code, msg, {
      start, end, bytes: nbytes, magic: what,
      entropy: roundPy(entropy(sample), 3), all_zero: allZero, container: inside,
    }));
  }

  for (const rg of mf.regions) {
    if (rg.kind !== "padding" || rg.end <= rg.start) continue;
    const data = r.slice(rg.start, rg.end);
    let dirty = 0;
    for (const b of data) if (b) dirty++;
    if (dirty) {
      findings.push(finding("S2", "high", "dirty-padding",
        `${dirty} non-zero byte(s) in padding (${rg.label}) at offset ${rg.start}`,
        { start: rg.start, end: rg.end, nonzero: dirty }));
    }
  }
  return findings;
}
