// Synchronous raw DEFLATE decoder (RFC 1951), for compressed zip entries.
// torch never compresses entries, so a compressed data.pkl is itself a finding;
// Python's zipfile still reads it, and so must the browser to stay at parity.
// The browser's DecompressionStream is async, and the parsers are synchronous.

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/** Canonical Huffman code: counts per length and symbols in code order. */
interface Huffman { counts: Uint16Array; symbols: Uint16Array }

function build(lengths: ArrayLike<number>): Huffman {
  const counts = new Uint16Array(16);
  for (let i = 0; i < lengths.length; i++) { const l = lengths[i]!; counts[l] = counts[l]! + 1; }
  counts[0] = 0;
  const offs = new Uint16Array(16);
  for (let i = 1; i < 16; i++) offs[i] = offs[i - 1]! + counts[i - 1]!;
  const symbols = new Uint16Array(lengths.length);
  for (let i = 0; i < lengths.length; i++) { const l = lengths[i]!; if (l) { symbols[offs[l]!] = i; offs[l] = offs[l]! + 1; } }
  return { counts, symbols };
}

class Bits {
  pos = 0; bit = 0; buf = 0; cnt = 0;
  constructor(readonly d: Uint8Array) {}
  need(n: number): number {
    while (this.cnt < n) {
      if (this.pos >= this.d.length) throw new Error("inflate: unexpected end of data");
      this.buf |= this.d[this.pos++]! << this.cnt; this.cnt += 8;
    }
    const v = this.buf & ((1 << n) - 1);
    this.buf >>>= n; this.cnt -= n;
    return v;
  }
  decode(h: Huffman): number {
    let code = 0, first = 0, index = 0;
    for (let len = 1; len < 16; len++) {
      code |= this.need(1);
      const count = h.counts[len]!;
      if (code - count < first) return h.symbols[index + (code - first)]!;
      index += count; first += count; first <<= 1; code <<= 1;
    }
    throw new Error("inflate: bad Huffman code");
  }
  align(): void { this.buf = 0; this.cnt = 0; }
}

const FIXED_LIT = build(Array.from({ length: 288 }, (_, i) => (i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8)));
const FIXED_DIST = build(new Array(30).fill(5));

/** Inflate raw DEFLATE data. `limit` bounds the output (a zip entry's declared
 * size), so a decompression bomb cannot exhaust memory. */
export function inflateRaw(data: Uint8Array, limit: number): Uint8Array {
  const out = new Uint8Array(limit);
  let o = 0;
  const s = new Bits(data);
  const put = (b: number) => { if (o >= limit) throw new Error("inflate: output larger than declared"); out[o++] = b; };
  for (;;) {
    const final = s.need(1), type = s.need(2);
    if (type === 0) {
      s.align();
      if (s.pos + 4 > data.length) throw new Error("inflate: truncated stored block");
      const len = data[s.pos]! | (data[s.pos + 1]! << 8);
      s.pos += 4;
      if (s.pos + len > data.length) throw new Error("inflate: truncated stored block");
      for (let i = 0; i < len; i++) put(data[s.pos + i]!);
      s.pos += len;
    } else if (type === 1 || type === 2) {
      let lit = FIXED_LIT, dist = FIXED_DIST;
      if (type === 2) {
        const hlit = s.need(5) + 257, hdist = s.need(5) + 1, hclen = s.need(4) + 4;
        const cl = new Uint8Array(19);
        for (let i = 0; i < hclen; i++) cl[CL_ORDER[i]!] = s.need(3);
        const clh = build(cl);
        const lens = new Uint8Array(hlit + hdist);
        for (let i = 0; i < hlit + hdist;) {
          const sym = s.decode(clh);
          if (sym < 16) { lens[i++] = sym; continue; }
          let rep = 0, val = 0;
          if (sym === 16) { if (i === 0) throw new Error("inflate: repeat with no previous length"); val = lens[i - 1]!; rep = 3 + s.need(2); }
          else if (sym === 17) rep = 3 + s.need(3);
          else rep = 11 + s.need(7);
          if (i + rep > hlit + hdist) throw new Error("inflate: too many lengths");
          while (rep--) lens[i++] = val;
        }
        lit = build(lens.subarray(0, hlit));
        dist = build(lens.subarray(hlit));
      }
      for (;;) {
        const sym = s.decode(lit);
        if (sym < 256) { put(sym); continue; }
        if (sym === 256) break;
        const li = sym - 257;
        if (li >= 29) throw new Error("inflate: bad length symbol");
        const len = LEN_BASE[li]! + s.need(LEN_EXTRA[li]!);
        const di = s.decode(dist);
        if (di >= 30) throw new Error("inflate: bad distance symbol");
        const d = DIST_BASE[di]! + s.need(DIST_EXTRA[di]!);
        if (d > o) throw new Error("inflate: distance too far back");
        for (let i = 0; i < len; i++) put(out[o - d]!);
      }
    } else {
      throw new Error("inflate: bad block type");
    }
    if (final) break;
  }
  return out.subarray(0, o);
}
