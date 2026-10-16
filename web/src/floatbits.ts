// Float tensors as raw bit patterns, mirroring stowaway/bits.py. Weights are
// held as their unsigned integer view (Uint32 for F32, Uint16 for F16/BF16), so
// bit statistics are computed exactly — bit-plane means are integer counts over
// element count, which is identical in CPython and V8 (no float summation drift).

export const FLOAT_LAYOUT: Record<string, { bytes: number; mant: number; exp: number }> = {
  F32: { bytes: 4, mant: 23, exp: 8 },
  F16: { bytes: 2, mant: 10, exp: 5 },
  BF16: { bytes: 2, mant: 7, exp: 8 },
};

const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export class RawFloat {
  readonly dtype: string;
  readonly shape: number[];
  readonly u: Uint32Array | Uint16Array; // unsigned view of each element

  constructor(dtype: string, shape: number[], u: Uint32Array | Uint16Array) {
    this.dtype = dtype; this.shape = shape; this.u = u;
  }

  static fromBytes(dtype: string, shape: number[], bytes: Uint8Array): RawFloat {
    const { bytes: isz } = FLOAT_LAYOUT[dtype]!;
    const n = Math.floor(bytes.byteLength / isz);
    // Copy into an aligned buffer (the slice may be unaligned). Model files are
    // little-endian; so is every platform a browser runs on, which the check
    // below confirms, so the copy can be viewed directly.
    const buf = bytes.slice(0, n * isz);
    if (LITTLE_ENDIAN) {
      return new RawFloat(dtype, shape, isz === 4 ? new Uint32Array(buf.buffer, 0, n) : new Uint16Array(buf.buffer, 0, n));
    }
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if (isz === 4) {
      const out = new Uint32Array(n);
      for (let i = 0; i < n; i++) out[i] = dv.getUint32(i * 4, true);
      return new RawFloat(dtype, shape, out);
    }
    const out = new Uint16Array(n);
    for (let i = 0; i < n; i++) out[i] = dv.getUint16(i * 2, true);
    return new RawFloat(dtype, shape, out);
  }

  get size(): number { return this.u.length; }
}

// Elements whose mantissa says something: not zero, subnormal, inf/nan, or an
// exact power of two. Mirrors bits.informative.
export function isInformative(v: number, mant: number, exp: number): boolean {
  const expMax = (1 << exp) - 1;
  const mantMask = mant < 31 ? (1 << mant) - 1 : 0x7fffffff;
  const e = (v >>> mant) & expMax;
  return e !== 0 && e !== expMax && (v & mantMask) !== 0;
}

/** What S3 and S4 need from one tensor, in one pass and without an index array:
 * how many elements are informative, the OR of their bits (S3), and how many of
 * them have each of the low `nbits` planes set (S4). Counts are exact integers,
 * so the means equal numpy's. */
export interface BitSummary { n: number; orBits: number; counts: number[] }

export function summarize(rf: RawFloat, nbits: number): BitSummary {
  const { mant, exp } = FLOAT_LAYOUT[rf.dtype]!;
  const expMax = (1 << exp) - 1;
  const mantMask = mant < 31 ? (1 << mant) - 1 : 0x7fffffff;
  const counts = new Array(nbits).fill(0);
  const u = rf.u;
  let n = 0, orBits = 0;
  for (let i = 0; i < u.length; i++) {
    const v = u[i]!;
    const e = (v >>> mant) & expMax;
    if (e === 0 || e === expMax || (v & mantMask) === 0) continue;
    n++;
    orBits = (orBits | v) >>> 0;
    for (let b = 0; b < nbits; b++) counts[b] += (v >>> b) & 1;
  }
  return { n, orBits, counts };
}
