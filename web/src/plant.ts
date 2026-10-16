// Write bytes into the low k bits of a model's float weights: the same
// construction as stowaway/embed.py (4-byte little-endian length frame, the
// payload as a little-endian bit stream cut into k-bit chunks, written into bits
// [0, k) of every `spread`-th element of the chosen tensors, in order).
//
// It exists for the limits page: a reader plants random bytes in a model and
// watches which checks see them. Nothing here reads a payload back or runs one.
import type { ModelFile, TensorInfo } from "./model.js";
import { FLOAT_LAYOUT } from "./floatbits.js";

export const FRAME = 4;

export interface PlantOptions {
  k: number;          // low bits per element
  spread: number;     // write every spread-th element
  names: string[];    // target tensors, in write order
}

function targets(mf: ModelFile, names: string[]): TensorInfo[] {
  const by = new Map(mf.tensors.map((t) => [t.name, t]));
  return names.map((n) => by.get(n)!).filter((t) => t && t.dtype in FLOAT_LAYOUT && t.offset !== null);
}

function elements(t: TensorInfo): number {
  return t.nbytes / FLOAT_LAYOUT[t.dtype]!.bytes;
}

/** Payload bytes that fit after the frame (stowaway.embed.capacity). */
export function capacity(mf: ModelFile, o: PlantOptions): number {
  let slots = 0;
  for (const t of targets(mf, o.names)) slots += Math.ceil(elements(t) / o.spread);
  return Math.max(0, Math.floor((slots * o.k) / 8) - FRAME);
}

/** Returns a copy of `file` with `payload` written as described above. Throws if
 * it does not fit or k is out of range for a target's dtype. */
export function plant(file: Uint8Array, mf: ModelFile, payload: Uint8Array, o: PlantOptions): Uint8Array {
  const ts = targets(mf, o.names);
  for (const t of ts) if (o.k < 1 || o.k > FLOAT_LAYOUT[t.dtype]!.mant) throw new Error(`k must be 1..${FLOAT_LAYOUT[t.dtype]!.mant} for ${t.dtype}`);
  if (payload.length > capacity(mf, o)) throw new Error("payload does not fit in the chosen tensors");
  const framed = new Uint8Array(FRAME + payload.length);
  new DataView(framed.buffer).setUint32(0, payload.length, true);
  framed.set(payload, FRAME);
  const totalBits = framed.length * 8;
  const nChunks = Math.ceil(totalBits / o.k);
  const bit = (i: number) => (i < totalBits ? (framed[i >> 3]! >> (i & 7)) & 1 : 0);
  const out = file.slice();
  const dv = new DataView(out.buffer);
  const mask = o.k >= 32 ? 0xffffffff : (2 ** o.k - 1);
  let chunk = 0;
  for (const t of ts) {
    const isz = FLOAT_LAYOUT[t.dtype]!.bytes;
    for (let e = 0; e < elements(t) && chunk < nChunks; e += o.spread, chunk++) {
      let v = 0;
      for (let j = 0; j < o.k; j++) v += bit(chunk * o.k + j) * 2 ** j;
      const at = t.offset! + e * isz;
      if (isz === 4) dv.setUint32(at, ((dv.getUint32(at, true) & ~mask) | v) >>> 0, true);
      else dv.setUint16(at, (dv.getUint16(at, true) & ~mask & 0xffff) | v, true);
    }
    if (chunk >= nChunks) break;
  }
  return out;
}
