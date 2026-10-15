// Float tensors as raw bit patterns, mirroring stowaway/bits.py. Weights are
// held as their unsigned integer view (Uint32 for F32, Uint16 for F16/BF16), so
// bit statistics are computed exactly — bit-plane means are integer counts over
// element count, which is identical in CPython and V8 (no float summation drift).

export const FLOAT_LAYOUT: Record<string, { bytes: number; mant: number; exp: number }> = {
  F32: { bytes: 4, mant: 23, exp: 8 },
  F16: { bytes: 2, mant: 10, exp: 5 },
  BF16: { bytes: 2, mant: 7, exp: 8 },
};

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
    // Copy into an aligned buffer (the slice may be unaligned).
    const buf = bytes.slice(0, n * isz);
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

// Indices of elements whose mantissa says something: not zero, subnormal,
// inf/nan, or an exact power of two. Mirrors bits.informative.
export function informativeIndices(rf: RawFloat): number[] {
  const { mant, exp } = FLOAT_LAYOUT[rf.dtype]!;
  const expMax = (1 << exp) - 1;
  const mantMask = mant < 31 ? (1 << mant) - 1 : 0x7fffffff;
  const out: number[] = [];
  const u = rf.u;
  for (let i = 0; i < u.length; i++) {
    const v = u[i]!;
    const e = (v >>> mant) & expMax;
    const m = v & mantMask;
    if (e !== 0 && e !== expMax && m !== 0) out.push(i);
  }
  return out;
}

// Fraction of ones in each of the low `nbits` planes, over the given elements.
// Exact: an integer count divided by the element count.
export function bitMeans(rf: RawFloat, indices: number[], nbits: number): number[] {
  const u = rf.u;
  const counts = new Array(nbits).fill(0);
  for (const i of indices) {
    const v = u[i]!;
    for (let b = 0; b < nbits; b++) counts[b] += (v >>> b) & 1;
  }
  const n = indices.length;
  return counts.map((c) => c / n);
}
