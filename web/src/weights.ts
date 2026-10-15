// S3/S4/S5 weight statistics. Mirrors stowaway/weights.py. All arithmetic is on
// IEEE-754 doubles with the same operations as numpy here, and bit-plane means
// are exact integer counts, so the findings match the Python core byte for byte.
import { Reader } from "./bytes.js";
import { Finding, finding, ModelFile } from "./model.js";
import { FLOAT_LAYOUT, RawFloat, bitMeans, informativeIndices } from "./floatbits.js";
import { pyRepr, pyStr } from "./pyrepr.js";

const MIN_ELEMENTS = 256;
const FP16_EXACT_LOW = 13;
const BF16_EXACT_LOW = 16;
const SIGNATURE_MIN_ELEMENTS = 4096;
const Z_FLAG = 6.0;
const S5_MARGIN = 3;
const S5_PERCENTILE = 99.0;
const S4_BITS: Record<string, number> = { F32: 8, F16: 3, BF16: 2 }; // round(mant*8/23)

export type RawFloats = Record<string, RawFloat>;

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

function percentile(xs: number[], p: number): number {
  const a = [...xs].sort((q, r) => q - r);
  const L = a.length;
  if (L === 1) return a[0]!;
  const rank = (p / 100) * (L - 1);
  const lo = Math.floor(rank), frac = rank - lo;
  return lo + 1 < L ? a[lo]! + frac * (a[lo + 1]! - a[lo]!) : a[lo]!;
}

export function role(name: string): string {
  return name.replace(/\d+/g, "#");
}

// --- S3 --------------------------------------------------------------------

export function s3PrecisionHistory(floats: RawFloats): Finding[] {
  const findings: Finding[] = [];
  const f32 = Object.entries(floats).filter(([, t]) => t.dtype === "F32");
  if (f32.length === 0) return findings;

  const profiles: Record<string, number> = {};
  for (const [name, t] of f32) {
    const info = informativeIndices(t);
    if (info.length < MIN_ELEMENTS) continue;
    let orBits = 0;
    for (const i of info) orBits = (orBits | t.u[i]!) >>> 0;
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

export function s4BitPlanes(floats: RawFloats): Finding[] {
  const findings: Finding[] = [];
  const byDtype = new Map<string, Stat[]>();
  for (const name of Object.keys(floats).sort()) {
    if (name.toLowerCase().includes("scale")) continue;
    const t = floats[name]!;
    const info = informativeIndices(t);
    if (info.length < MIN_ELEMENTS) continue;
    const kb = S4_BITS[t.dtype]!;
    (byDtype.get(t.dtype) ?? byDtype.set(t.dtype, []).get(t.dtype)!).push({ name, n: info.length, m: bitMeans(t, info, kb) });
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
    const key = (name: string) => `${role(name)}|${floats[name]!.shape.length}`;
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

// --- S5 --------------------------------------------------------------------

function highestChangedBits(aU: Uint32Array | Uint16Array, bU: Uint32Array | Uint16Array,
                            idx: number[], mantBits: number): number[] {
  const mask = mantBits < 31 ? (1 << mantBits) - 1 : 0x7fffffff;
  const out: number[] = [];
  for (const i of idx) {
    const x = (aU[i]! ^ bU[i]!) & mask;
    let hi = -1;
    for (let b = 0; b < mantBits; b++) if ((x >>> b) & 1) hi = b;
    out.push(hi);
  }
  return out;
}

export function s5Differential(floats: RawFloats, base: RawFloats): Finding[] {
  const findings: Finding[] = [];
  const shared = Object.keys(floats).filter((n) => n in base).sort();
  if (shared.length === 0) {
    findings.push(finding("S5", "info", "no-shared-tensors", "the base model shares no tensor names with this model"));
    return findings;
  }
  for (const name of shared) {
    const a = floats[name]!, b = base[name]!;
    if (a.dtype !== b.dtype || a.shape.length !== b.shape.length || a.u.length !== b.u.length) continue;
    const changedIdx: number[] = [];
    for (let i = 0; i < a.u.length; i++) if (a.u[i] !== b.u[i]) changedIdx.push(i);
    if (changedIdx.length < MIN_ELEMENTS) continue;
    const mantBits = FLOAT_LAYOUT[a.dtype]!.mant;
    const cutoff = mantBits - S5_MARGIN;
    const hi = highestChangedBits(a.u, b.u, changedIdx, mantBits);
    const reach = percentile(hi, S5_PERCENTILE);
    if (reach <= cutoff) {
      findings.push(finding("S5", "high", "low-bit-only-change",
        `tensor ${pyRepr(name)} differs from the base model in ${changedIdx.length} elements, ` +
        `but the changes reach only bit ${Math.trunc(reach)} of ${mantBits} ` +
        `(fine-tuning reaches the top bit) - a low-bit payload, not training`,
        { tensor: name, changed: changedIdx.length, highest_changed_bit_p99: round(reach, 1), cutoff, mantissa_bits: mantBits }));
    }
  }
  return findings;
}

// --- entry point -----------------------------------------------------------

export function loadFloats(mf: ModelFile, r: Reader): RawFloats {
  const out: RawFloats = {};
  for (const t of mf.tensors) {
    if (t.dtype in FLOAT_LAYOUT && t.offset !== null && t.nbytes) {
      const bytes = r.slice(t.offset, t.offset + t.nbytes);
      if (bytes.byteLength === t.nbytes) out[t.name] = RawFloat.fromBytes(t.dtype, t.shape, bytes);
    }
  }
  return out;
}

export function analyseWeights(mf: ModelFile, r: Reader, base?: RawFloats): Finding[] {
  const floats = loadFloats(mf, r);
  if (Object.keys(floats).length === 0) return [];
  const findings = [...s3PrecisionHistory(floats), ...s4BitPlanes(floats)];
  if (base) findings.push(...s5Differential(floats, base));
  return findings;
}

