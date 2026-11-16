// S3/S4/S5 weight statistics. Mirrors stowaway/weights.py. All arithmetic is on
// IEEE-754 doubles with the same operations as numpy here, and bit-plane means
// are exact integer counts, so the findings match the Python core byte for byte.
import { Reader } from "./bytes.js";
import { Finding, finding, ModelFile } from "./model.js";
import { FLOAT_LAYOUT, RawFloat, BitSummary, summarize } from "./floatbits.js";
import { pyRepr, pyStr } from "./pyrepr.js";

const MIN_ELEMENTS = 256;
const FP16_EXACT_LOW = 13;
const BF16_EXACT_LOW = 16;
const SIGNATURE_MIN_ELEMENTS = 4096;
const Z_FLAG = 6.0;
const S5_MARGIN = 3;
const S5_PERCENTILE = 99.0;
// S5 judges a tensor when at least this many elements changed; must equal weights.S5_MIN_CHANGED
// in Python (the old floor of 256 was removed by E12, PLAN.md: H13 and H14 supported).
export const S5_MIN_CHANGED = 1;
// S5d (PLAN.md, E13): S5c, and at most this share of changed elements has its highest changed
// bit above S5c's bit; null = S5c alone. Must equal weights.S5_ABOVE_MAX in Python (adopted by
// E13: H17 and H18 supported).
export const S5_ABOVE_MAX: number | null = 0.02;
// E15 (PLAN.md): report how close the base is; below S5_CLOSE_MIN a base-not-close finding.
// Must equal weights.S5_BASE_CHECK and weights.S5_CLOSE_MIN in Python (adopted by E15).
export const S5_BASE_CHECK = true;
export const S5_CLOSE_MIN = 0.9;
// Which S5 rule runs; must equal weights.S5_RULE in Python (see there).
export const S5_RULE: "p99" | "s5c" = "s5c"; // adopted by E4b (PLAN.md)
const S4_BITS: Record<string, number> = { F32: 8, F16: 3, BF16: 2 }; // round(mant*8/23)
// S4c (PLAN.md, E0d): test a bit plane only if its reference is at least this far
// from a fair coin; 0 = off. Must equal weights.S4_MIN_SIGNATURE in Python.
export const S4_MIN_SIGNATURE = 0.02; // adopted by E0d (PLAN.md)

export type RawFloats = Record<string, RawFloat>;

/** One float tensor, reduced to what S3 and S4 read. */
export interface TensorSummary extends BitSummary { name: string; dtype: string; ndim: number }

/** A base model for S5: its parsed structure and a way to read any tensor,
 * so each base tensor is loaded only while it is being compared. */
export interface BaseModel { mf: ModelFile; r: Reader }

function round(x: number, digits: number): number {
  // round-half-to-even, matching Python's round() for the rationals here.
  const f = Math.pow(10, digits);
  const s = x * f;
  const fl = Math.floor(s);
  const d = s - fl;
  let r: number;
  if (d > 0.5) r = fl + 1; else if (d < 0.5) r = fl; else r = fl % 2 === 0 ? fl : fl + 1;
  return r / f;
}

function median(xs: number[]): number {
  const a = [...xs].sort((p, q) => p - q);
  const L = a.length;
  if (L === 0) return NaN;
  return L % 2 ? a[(L - 1) / 2]! : (a[L / 2 - 1]! + a[L / 2]!) / 2;
}

export function role(name: string): string {
  return name.replace(/\d+/g, "#");
}

// --- S3 --------------------------------------------------------------------

export function s3PrecisionHistory(sums: Map<string, TensorSummary>): Finding[] {
  const findings: Finding[] = [];
  const f32 = [...sums.values()].filter((t) => t.dtype === "F32");
  if (f32.length === 0) return findings;

  const profiles: Record<string, number> = {};
  for (const { name, n, orBits } of f32) {
    if (n < MIN_ELEMENTS) continue;
    let trailing = 23;
    if (orBits) { for (let b = 0; b < 32; b++) if ((orBits >>> b) & 1) { trailing = b; break; } }
    profiles[name] = Math.min(trailing, 23);
  }
  const names = Object.keys(profiles);
  if (names.length === 0) return findings;

  const counts = new Array(24).fill(0);
  for (const n of names) counts[profiles[n]!]++;
  let modelFloor = 0, best = -1;
  for (let i = 0; i < 24; i++) if (counts[i] > best) { best = counts[i]; modelFloor = i; }

  let origin: string, exact: number;
  if (modelFloor >= BF16_EXACT_LOW) { origin = "bfloat16"; exact = BF16_EXACT_LOW; }
  else if (modelFloor >= FP16_EXACT_LOW) { origin = "float16"; exact = FP16_EXACT_LOW; }
  else return findings;

  findings.push(finding("S3", "info", "precision-origin",
    `stored as float32 but the weights carry only ${origin} precision (low ${exact} bits are zero in most tensors)`,
    { origin, exact_low_bits: exact, tensors: names.length }));
  for (const name of names.sort()) {
    const floor = profiles[name]!;
    if (floor < exact - 1) {
      const dirty = exact - floor;
      findings.push(finding("S3", "high", "precision-anomaly",
        `tensor ${pyRepr(name)} uses ${dirty} low bit(s) that a ${origin} model leaves zero`,
        { tensor: name, unexpected_low_bits: dirty, origin }));
    }
  }
  return findings;
}

// --- S4 --------------------------------------------------------------------

type Stat = { name: string; n: number; m: number[] };

function signature(stats: Stat[], nbits: number): { sig: number[]; tau: number[]; hasPop: boolean } {
  const big = stats.filter((s) => s.n >= SIGNATURE_MIN_ELEMENTS);
  const pool = big.length >= 3 ? big : stats.length >= 3 ? stats : [];
  if (pool.length === 0) return { sig: new Array(nbits).fill(0.5), tau: new Array(nbits).fill(0), hasPop: false };
  const sig: number[] = [], tau: number[] = [];
  for (let b = 0; b < nbits; b++) {
    const col = pool.map((s) => s.m[b]!);
    const sb = median(col);
    const mad = 1.4826 * median(col.map((x) => Math.abs(x - sb)));
    const sampling = median(pool.map((s) => (sb * (1 - sb)) / s.n));
    sig.push(sb);
    tau.push(Math.sqrt(Math.max(0, mad * mad - sampling)));
  }
  return { sig, tau, hasPop: true };
}

export function s4BitPlanes(sums: Map<string, TensorSummary>): Finding[] {
  const findings: Finding[] = [];
  const byDtype = new Map<string, Stat[]>();
  for (const name of [...sums.keys()].sort()) {
    if (name.toLowerCase().includes("scale")) continue;
    const t = sums.get(name)!;
    if (t.n < MIN_ELEMENTS) continue;
    (byDtype.get(t.dtype) ?? byDtype.set(t.dtype, []).get(t.dtype)!).push({ name, n: t.n, m: t.counts.map((c) => c / t.n) });
  }

  for (const [dtype, stats] of byDtype) {
    const kb = stats[0]!.m.length;
    const modelRef = signature(stats, kb);
    if (modelRef.hasPop && Math.max(...modelRef.sig.map((x) => Math.abs(x - 0.5))) > 0.02) {
      const means = modelRef.sig.slice(0, 4).map((x) => pyStr(round(x, 2))).join(", ");
      findings.push(finding("S4", "info", "lowbit-signature",
        `${dtype} weights carry a low-bit signature (bit means ${means}); tensors are tested against it`,
        { dtype, signature: modelRef.sig.map((x) => round(x, 4)) }));
    }
    const key = (name: string) => `${role(name)}|${sums.get(name)!.ndim}`;
    const roles = new Map<string, Stat[]>();
    for (const s of stats) (roles.get(key(s.name)) ?? roles.set(key(s.name), []).get(key(s.name))!).push(s);

    for (const { name, n, m } of stats) {
      const peers = roles.get(key(name))!.filter((s) => s.name !== name);
      let sig: number[], tau: number[], against: string;
      if (peers.length >= 3) { ({ sig, tau } = signature(peers, kb)); against = `its ${peers.length} role peers`; }
      else { sig = modelRef.sig; tau = modelRef.tau; against = "the model"; }
      let bestB = 0, bestScore = -Infinity;
      for (let b = 0; b < kb; b++) {
        const se = Math.sqrt(Math.max(sig[b]! * (1 - sig[b]!), 1e-6) / n);
        if (S4_MIN_SIGNATURE && Math.abs(sig[b]! - 0.5) < S4_MIN_SIGNATURE) continue;
        const closer = (Math.abs(sig[b]! - 0.5) - Math.abs(m[b]! - 0.5)) / Math.sqrt(se * se + tau[b]! * tau[b]!);
        if (closer > bestScore) { bestScore = closer; bestB = b; }
      }
      if (bestScore > Z_FLAG) {
        const mean = round(m[bestB]!, 4), expected = round(sig[bestB]!, 4), z = round(bestScore, 2);
        findings.push(finding("S4", "medium", "lowbit-anomaly",
          `tensor ${pyRepr(name)}: low bit ${bestB} is nearer a fair coin (${pyStr(mean)}) than ${against} (${pyStr(expected)}); z ${pyStr(z)}`,
          { tensor: name, bit: bestB, z, mean, expected, reference: peers.length >= 3 ? "role" : "model", elements: n }));
      }
    }
  }
  return findings;
}

// --- S4w ---------------------------------------------------------------------
// Mirrors weights.s4_windowed (PLAN.md, phase 4, E11): each window of S4W_WINDOW
// elements of one tensor against the tensor's own other windows. Must equal
// weights.S4W / S4W_WINDOW / S4W_Z in Python.
export const S4W = true; // adopted by E11 (PLAN.md: H15 and H16 supported)
export const S4W_WINDOW = 32 * 1024;
const S4W_Z = Z_FLAG;

/** S4w for one tensor: a finding, or null. */
export function s4Windowed(name: string, rf: RawFloat): Finding | null {
  const low = name.toLowerCase();
  const nbits = S4_BITS[rf.dtype];
  if (nbits === undefined || low.includes("scale") || low.includes("embed")) return null;
  const nw = Math.floor(rf.u.length / S4W_WINDOW);
  if (nw < 3) return null;
  const { mant, exp } = FLOAT_LAYOUT[rf.dtype]!;
  const expMax = (1 << exp) - 1;
  const mantMask = mant < 31 ? (1 << mant) - 1 : 0x7fffffff;
  const u = rf.u;
  const ns: number[] = [], ps: number[][] = [], idx: number[] = [];
  for (let w = 0; w < nw; w++) {
    const counts = new Array(nbits).fill(0);
    let n = 0;
    for (let i = w * S4W_WINDOW, end = i + S4W_WINDOW; i < end; i++) {
      const v = u[i]!;
      const e = (v >>> mant) & expMax;
      if (e === 0 || e === expMax || (v & mantMask) === 0) continue;
      n++;
      for (let b = 0; b < nbits; b++) counts[b] += (v >>> b) & 1;
    }
    if (n > 0) { ns.push(n); ps.push(counts.map((c) => c / n)); idx.push(w); }
  }
  if (ns.length < 3) return null;
  const sig: number[] = [], tau: number[] = [];
  for (let b = 0; b < nbits; b++) {
    const col = ps.map((p) => p[b]!);
    const sb = median(col);
    const mad = 1.4826 * median(col.map((x) => Math.abs(x - sb)));
    const sampling = median(ns.map((n) => (sb * (1 - sb)) / n));
    sig.push(sb);
    tau.push(Math.sqrt(Math.max(0, mad * mad - sampling)));
  }
  // numpy.argmax over (window, bit), row-major: the first maximum.
  let bestW = 0, bestB = 0, best = -Infinity;
  for (let w = 0; w < ns.length; w++) {
    for (let b = 0; b < nbits; b++) {
      let z = -Infinity;
      if (!S4_MIN_SIGNATURE || Math.abs(sig[b]! - 0.5) >= S4_MIN_SIGNATURE) {
        const se = Math.sqrt(Math.max(sig[b]! * (1 - sig[b]!), 1e-6) / ns[w]!);
        z = (Math.abs(sig[b]! - 0.5) - Math.abs(ps[w]![b]! - 0.5)) / Math.sqrt(se * se + tau[b]! * tau[b]!);
      }
      if (z > best) { best = z; bestW = w; bestB = b; }
    }
  }
  if (!(best > S4W_Z)) return null;
  const window = idx[bestW]!;
  const z = round(best, 2), mean = round(ps[bestW]![bestB]!, 4), expected = round(sig[bestB]!, 4);
  return finding("S4", "medium", "lowbit-window-anomaly",
    `tensor ${pyRepr(name)}: in elements ${window * S4W_WINDOW}-${(window + 1) * S4W_WINDOW - 1}, ` +
    `low bit ${bestB} is nearer a fair coin (${pyStr(mean)}) than in the tensor's other windows ` +
    `(${pyStr(expected)}); z ${pyStr(z)}`,
    { tensor: name, window, bit: bestB, z, mean, expected, window_elements: S4W_WINDOW });
}

// --- S5 --------------------------------------------------------------------

/** Histogram of the highest changed mantissa bit (-1 .. mantBits-1, stored at
 * index bit+1) over the elements that differ, and how many differ. */
function changedBitHistogram(aU: Uint32Array | Uint16Array, bU: Uint32Array | Uint16Array,
                             mantBits: number): { changed: number; hist: number[] } {
  const mask = mantBits < 31 ? (1 << mantBits) - 1 : 0x7fffffff;
  const hist = new Array(mantBits + 1).fill(0);
  let changed = 0;
  for (let i = 0; i < aU.length; i++) {
    if (aU[i] === bU[i]) continue;
    changed++;
    const x = (aU[i]! ^ bU[i]!) & mask;
    hist[x === 0 ? 0 : 32 - Math.clz32(x)]++; // index = highest set bit + 1
  }
  return { changed, hist };
}

/** numpy.percentile (linear) of the values a histogram describes; the same
 * arithmetic as percentile() on the sorted values, so the result is identical. */
function percentileFromHistogram(hist: number[], p: number): number {
  const L = hist.reduce((x, y) => x + y, 0);
  const at = (k: number) => { // k-th smallest value (0-based)
    let c = 0;
    for (let i = 0; i < hist.length; i++) { c += hist[i]!; if (k < c) return i - 1; }
    return hist.length - 2;
  };
  if (L === 1) return at(0);
  const rank = (p / 100) * (L - 1);
  const lo = Math.floor(rank), frac = rank - lo;
  return lo + 1 < L ? at(lo) + frac * (at(lo + 1) - at(lo)) : at(lo);
}

/** S5 for one tensor and its base counterpart: a finding, or null. */
export function s5Tensor(name: string, a: RawFloat, b: RawFloat): Finding | null {
  if (a.dtype !== b.dtype || a.shape.length !== b.shape.length || a.u.length !== b.u.length) return null;
  const mantBits = FLOAT_LAYOUT[a.dtype]!.mant;
  const { changed, hist } = changedBitHistogram(a.u, b.u, mantBits);
  if (changed < Math.max(1, S5_MIN_CHANGED)) return null;
  const cutoff = mantBits - S5_MARGIN;
  if (S5_RULE === "s5c") {
    let j = 0;
    for (let i = 1; i <= cutoff + 1; i++) if (hist[i]! > hist[j]!) j = i;  // bits -1 .. cutoff, first max
    const share = hist[j]! / changed;
    const threshold = 0.5 - 1.5 / Math.sqrt(changed);
    if (share < threshold) return null;
    const extra: Record<string, number> = {};
    if (S5_ABOVE_MAX !== null) {
      let up = 0;
      for (let i = j + 1; i < hist.length; i++) up += hist[i]!;
      const above = up / changed;
      if (above > S5_ABOVE_MAX) return null;
      extra.above = round(above, 4);
    }
    return finding("S5", "high", "low-bit-only-change",
      `tensor ${pyRepr(name)} differs from the base model in ${changed} elements, and in ` +
      `${pyStr(round(100 * share, 1))}% of them the highest changed bit is bit ${j - 1} of ${mantBits}: ` +
      `a payload in the low bits stops at one bit in about half its elements; training does not`,
      { tensor: name, changed, bit: j - 1, share: round(share, 4), threshold: round(threshold, 4), cutoff, mantissa_bits: mantBits, ...extra });
  }
  const reach = percentileFromHistogram(hist, S5_PERCENTILE);
  if (reach <= cutoff) {
    return finding("S5", "high", "low-bit-only-change",
      `tensor ${pyRepr(name)} differs from the base model in ${changed} elements, ` +
      `but the changes reach only bit ${Math.trunc(reach)} of ${mantBits} ` +
      `(fine-tuning reaches the top bit) - a low-bit payload, not training`,
      { tensor: name, changed, highest_changed_bit_p99: round(reach, 1), cutoff, mantissa_bits: mantBits });
  }
  return null;
}

// --- entry point -----------------------------------------------------------

/** Float tensors that S3/S4/S5 read, in file order. A later tensor with the same
 * name replaces an earlier one, as a Python dict does. */
function floatTensors(mf: ModelFile) {
  const byName = new Map<string, (typeof mf.tensors)[number]>();
  for (const t of mf.tensors) if (t.dtype in FLOAT_LAYOUT && t.offset !== null && t.nbytes) byName.set(t.name, t);
  return byName;
}

function load(t: { dtype: string; shape: number[]; offset: number | null; nbytes: number }, r: Reader): RawFloat | null {
  const bytes = r.slice(t.offset!, t.offset! + t.nbytes);
  return bytes.byteLength === t.nbytes ? RawFloat.fromBytes(t.dtype, t.shape, bytes) : null;
}

/** Every float tensor in memory at once. Used only by small callers (tests). */
export function loadFloats(mf: ModelFile, r: Reader): RawFloats {
  const out: RawFloats = {};
  for (const [name, t] of floatTensors(mf)) {
    const rf = load(t, r);
    if (rf) out[name] = rf;
  }
  return out;
}

// --- aligning tensor names with the base (weights.align) -------------------

const LEGACY_NAMES: Record<string, string> = { gamma: "weight", beta: "bias" };
const MAX_DROP = 2;

function canonName(name: string, drop: number): string {
  const parts = name.split(".").slice(drop);
  const last = parts[parts.length - 1];
  if (parts.length && last !== undefined && last in LEGACY_NAMES) parts[parts.length - 1] = LEGACY_NAMES[last]!;
  return parts.join(".");
}

/** Pair each model tensor with the base tensor it fine-tunes; see weights.align. */
export function align(model: Map<string, string>, base: Map<string, string>): { pairs: Map<string, string>; dm: number; db: number } {
  let best = { pairs: new Map<string, string>(), dm: 0, db: 0 };
  for (let dm = 0; dm <= MAX_DROP; dm++) {
    for (let db = 0; db <= MAX_DROP; db++) {
      const bkeys = new Map<string, string[]>();
      for (const b of base.keys()) { const k = canonName(b, db); (bkeys.get(k) ?? bkeys.set(k, []).get(k)!).push(b); }
      const mkeys = new Map<string, string[]>();
      for (const m of model.keys()) { const k = canonName(m, dm); (mkeys.get(k) ?? mkeys.set(k, []).get(k)!).push(m); }
      const pairs = new Map<string, string>();
      for (const [key, ms] of mkeys) {
        const bs = bkeys.get(key);
        if (key && ms.length === 1 && bs && bs.length === 1 && model.get(ms[0]!) === base.get(bs[0]!)) pairs.set(ms[0]!, bs[0]!);
      }
      if (pairs.size > best.pairs.size) best = { pairs, dm, db };
    }
  }
  return best;
}

const sig = (t: { dtype: string; shape: number[] }) => `${t.dtype}|${t.shape.join(",")}`;

/** A tensor's values as float64 numbers (every F32, F16 and BF16 value is exact in a double). */
function values(rf: RawFloat): Float64Array {
  const out = new Float64Array(rf.u.length);
  if (rf.dtype === "F32") {
    const f = new Float32Array(rf.u.buffer, rf.u.byteOffset, rf.u.length);
    for (let i = 0; i < f.length; i++) out[i] = f[i]!;
  } else if (rf.dtype === "BF16") {
    const w = new Uint32Array(1), f = new Float32Array(w.buffer);
    for (let i = 0; i < rf.u.length; i++) { w[0] = rf.u[i]! << 16; out[i] = f[0]!; }
  } else {
    for (let i = 0; i < rf.u.length; i++) {
      const h = rf.u[i]!, e = (h >> 10) & 31, m = h & 1023, sgn = h & 0x8000 ? -1 : 1;
      out[i] = e === 0 ? sgn * m * 2 ** -24 : e === 31 ? (m ? NaN : sgn * Infinity) : sgn * (1024 + m) * 2 ** (e - 25);
    }
  }
  return out;
}

/** [cosine rounded to 4 decimals (null if a norm is 0), identical] for a pair S5 compares, else null.
 * Sums run left to right, as weights._closeness's cumsum does, so the two builds agree. */
function closeness(a: RawFloat, b: RawFloat): [number | null, boolean] | null {
  if (a.dtype !== b.dtype || a.u.length !== b.u.length || a.u.length < MIN_ELEMENTS) return null;
  let same = true;
  for (let i = 0; i < a.u.length; i++) if (a.u[i] !== b.u[i]) { same = false; break; }
  if (same) return [1.0, true];
  const x = values(a), y = values(b);
  let xx = 0, yy = 0, xy = 0;
  for (let i = 0; i < x.length; i++) {
    const p = x[i]!, q = y[i]!;
    if (!Number.isFinite(p) || !Number.isFinite(q)) continue;
    xx += p * p; yy += q * q; xy += p * q;
  }
  if (xx === 0 || yy === 0) return [null, false];
  return [round(xy / Math.sqrt(xx * yy), 4), false];
}

function closenessFindings(rows: Array<[number | null, boolean]>): Finding[] {
  const cos = rows.map(([c]) => c).filter((c): c is number => c !== null);
  if (!S5_BASE_CHECK || cos.length === 0) return [];
  const med = round(median(cos), 4);
  const same = rows.filter(([, i]) => i).length;
  const detail = { median_cosine: med, compared: cos.length, identical: same, close_min: S5_CLOSE_MIN };
  const out = [finding("S5", "info", "base-closeness",
    `median cosine with the base over ${cos.length} aligned tensors: ${pyStr(med)}; ${same} identical`, detail)];
  if (med < S5_CLOSE_MIN) {
    out.push(finding("S5", "medium", "base-not-close",
      `the base is not a close relative of this model (median cosine ${pyStr(med)} < ${pyStr(S5_CLOSE_MIN)}): ` +
      `S5 can only see a payload in the ${same} tensor(s) identical to it`, detail));
  }
  return out;
}

/** S3, S4 and (with a base) S5, reading one tensor at a time: peak memory is
 * the largest tensor (and its base counterpart), not the file. */
export function analyseWeights(mf: ModelFile, r: Reader, base?: BaseModel): Finding[] {
  const sums = new Map<string, TensorSummary>();
  const baseTensors = base ? floatTensors(base.mf) : null;
  const s5: Array<[string, Finding]> = [];
  const close: Array<[number | null, boolean]> = [];
  const windowed: Finding[] = [];
  let shared = 0;
  const infos = floatTensors(mf);
  let pairing = { pairs: new Map<string, string>(), dm: 0, db: 0 };
  if (baseTensors) {
    pairing = align(new Map([...infos].map(([n, t]) => [n, sig(t)])), new Map([...baseTensors].map(([n, t]) => [n, sig(t)])));
  }
  for (const [name, t] of infos) {
    const rf = load(t, r);
    if (!rf) continue;
    sums.set(name, { name, dtype: rf.dtype, ndim: rf.shape.length, ...summarize(rf, S4_BITS[rf.dtype]!) });
    if (S4W) { const w = s4Windowed(name, rf); if (w) windowed.push(w); }
    const bt = baseTensors?.get(pairing.pairs.get(name) ?? name);
    if (bt && base) {
      const brf = load(bt, base.r);
      if (brf) {
        shared++;
        const f = s5Tensor(name, rf, brf);
        if (f) s5.push([name, f]);
        if (S5_BASE_CHECK) { const c = closeness(rf, brf); if (c) close.push(c); }
      }
    }
  }
  if (sums.size === 0) return [];
  const findings = [...s3PrecisionHistory(sums), ...s4BitPlanes(sums), ...windowed];
  const renamed = [...pairing.pairs].filter(([m, b]) => m !== b).length;
  if (renamed) {
    findings.push(finding("S5", "info", "aligned-names",
      `paired ${pairing.pairs.size} tensors with the base by name after renaming ${renamed} ` +
      `(dropped ${pairing.dm} leading name part(s) here and ${pairing.db} in the base, gamma/beta read as weight/bias)`,
      { paired: pairing.pairs.size, renamed, dropped_model: pairing.dm, dropped_base: pairing.db }));
  }
  if (base) {
    if (shared === 0) {
      findings.push(finding("S5", "info", "no-shared-tensors", "the base model shares no tensor names with this model"));
    } else {
      findings.push(...s5.sort((x, y) => (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0)).map(([, f]) => f),
                    ...closenessFindings(close));
    }
  }
  return findings;
}
