// The limits page's worker: holds one model in memory, plants random bytes in a
// copy (plant.ts, identical to stowaway.embed), and scans the copy with the real
// checks, from the file alone and against the unmodified model as base.
//
// Started from a blob: URL like the scan worker, so it runs under the page's
// Content-Security-Policy (connect-src 'none'). Payloads are random bytes from
// crypto.getRandomValues; nothing is read back or executed.
import { Reader } from "./bytes.js";
import { FLOAT_LAYOUT, RawFloat, summarize } from "./floatbits.js";
import type { ModelFile } from "./model.js";
import { capacity, plant, PlantOptions } from "./plant.js";
import { openBase, openModel, scan } from "./report.js";
import { role } from "./weights.js";

type Finding = { check: string; severity: string; code: string; message: string; detail: Record<string, unknown> };
type Report = { verdict: string; findings: Finding[] };

export type Target = "all" | "fair" | "largest" | `role:${string}`;
export interface TryOptions { size: number; k: number; spread: number; target: Target }
export interface TensorRow { name: string; dtype: string; elements: number; role: string; bit0: number | null }
export interface TryResult {
  written: string[]; capacity: number; size: number; k: number; spread: number;
  single: { caught: boolean; findings: Finding[] }; base: { caught: boolean; findings: Finding[] };
}

const FLAG = new Set(["medium", "high", "critical"]);
let original: Uint8Array | null = null;   // the model as loaded (or rounded to bf16 precision)
let mf: ModelFile | null = null;
let cleanFlagged = new Set<string>();     // tensors the clean scan already flags (not counted as catches)
let rows: TensorRow[] = [];

function bf16Rounded(bytes: Uint8Array, m: ModelFile): Uint8Array {
  // As if the model had been trained in bf16 and saved as float32: every float32
  // weight rounded to bf16 precision (round to nearest even; NaN/inf untouched).
  const out = bytes.slice();
  const dv = new DataView(out.buffer);
  for (const t of m.tensors) {
    if (t.dtype !== "F32" || t.offset === null) continue;
    for (let at = t.offset; at < t.offset + t.nbytes; at += 4) {
      const u = dv.getUint32(at, true);
      if (((u >>> 23) & 0xff) === 0xff) continue;
      const r = (u + 0x7fff + ((u >>> 16) & 1)) >>> 0;
      dv.setUint32(at, (r & 0xffff0000) >>> 0, true);
    }
  }
  return out;
}

function tensorRows(bytes: Uint8Array, m: ModelFile): TensorRow[] {
  const out: TensorRow[] = [];
  for (const t of m.tensors) {
    if (!(t.dtype in FLOAT_LAYOUT) || t.offset === null || !t.nbytes) continue;
    const rf = RawFloat.fromBytes(t.dtype, t.shape, bytes.subarray(t.offset, t.offset + t.nbytes));
    const s = summarize(rf, 1);
    out.push({ name: t.name, dtype: t.dtype, elements: rf.size, role: role(t.name), bit0: s.n >= 256 ? s.counts[0]! / s.n : null });
  }
  return out;
}

function names(target: Target): string[] {
  if (target === "all") return rows.map((r) => r.name);
  if (target === "largest") return [rows.reduce((a, b) => (b.elements > a.elements ? b : a)).name];
  if (target === "fair") return rows.filter((r) => r.bit0 !== null && Math.abs(r.bit0 - 0.5) < 0.01).map((r) => r.name);
  const want = target.slice(5);
  return rows.filter((r) => r.role === want).map((r) => r.name);
}

function randomBytes(n: number): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i += 65536) crypto.getRandomValues(out.subarray(i, Math.min(n, i + 65536)));
  return out;
}

function caughtBy(rep: Report, checks: string[], written: Set<string>): Finding[] {
  return rep.findings.filter((f) => checks.includes(f.check) && FLAG.has(f.severity) &&
    typeof f.detail.tensor === "string" && written.has(f.detail.tensor) && !cleanFlagged.has(f.detail.tensor));
}

async function attempt(o: TryOptions, keepBytes: boolean): Promise<{ result: TryResult; bytes?: Uint8Array }> {
  const opts: PlantOptions = { k: o.k, spread: o.spread, names: names(o.target) };
  const cap = capacity(mf!, opts);
  if (!opts.names.length) throw new Error("no tensor matches that target");
  const size = Math.min(o.size, cap);
  const planted = plant(original!, mf!, randomBytes(size), opts);
  const written = new Set(opts.names);
  const single = (await scan(planted)) as unknown as Report;
  const withBase = (await scan(planted, openBase(original!))) as unknown as Report;
  const s = caughtBy(single, ["S3", "S4"], written), b = caughtBy(withBase, ["S5"], written);
  return {
    result: { written: opts.names, capacity: cap, size, k: o.k, spread: o.spread,
              single: { caught: s.length > 0, findings: s }, base: { caught: b.length > 0, findings: b } },
    bytes: keepBytes ? planted : undefined,
  };
}

const post = (m: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

self.onmessage = async (ev: MessageEvent) => {
  const m = ev.data as { id: number; kind: string; [k: string]: unknown };
  try {
    if (m.kind === "load") {
      let bytes = m.bytes as Uint8Array;
      const m0 = openModel(Reader.of(bytes));
      if (m0.format !== "safetensors") throw new Error("the limits page takes a .safetensors file");
      if (m.bf16) bytes = bf16Rounded(bytes, m0);
      original = bytes; mf = openModel(Reader.of(bytes));
      rows = tensorRows(bytes, mf);
      const clean = (await scan(bytes)) as unknown as Report;
      cleanFlagged = new Set(clean.findings.filter((f) => FLAG.has(f.severity) && typeof f.detail.tensor === "string")
        .map((f) => f.detail.tensor as string));
      post({ id: m.id, kind: "loaded", rows, cleanFlagged: [...cleanFlagged], cleanVerdict: clean.verdict,
             dtypes: [...new Set(rows.map((r) => r.dtype))] });
    } else if (m.kind === "try") {
      const { result, bytes } = await attempt(m.opts as TryOptions, Boolean(m.keep));
      post({ id: m.id, kind: "tried", result, bytes }, bytes ? [bytes.buffer] : []);
    } else if (m.kind === "sweep") {
      const { sizes, ks, spread, target } = m as unknown as { sizes: number[]; ks: number[]; spread: number; target: Target };
      const cells: Array<{ size: number; k: number; fits: boolean; single?: boolean; base?: boolean }> = [];
      let done = 0;
      for (const k of ks) {
        for (const size of sizes) {
          const cap = capacity(mf!, { k, spread, names: names(target) });
          if (size > cap) cells.push({ size, k, fits: false });
          else {
            const { result } = await attempt({ size, k, spread, target }, false);
            cells.push({ size, k, fits: true, single: result.single.caught, base: result.base.caught });
          }
          post({ id: m.id, kind: "progress", done: ++done, total: sizes.length * ks.length });
        }
      }
      post({ id: m.id, kind: "swept", cells });
    }
  } catch (e) {
    post({ id: m.id, kind: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
