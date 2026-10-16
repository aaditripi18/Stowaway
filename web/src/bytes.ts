// Random access to a model file without holding it in memory.
//
// A Source hands out byte ranges synchronously. In tests and in Node it wraps a
// Uint8Array; in the browser, the scan runs in a worker where FileReaderSync
// reads any range of a File on demand, so a multi-gigabyte model is scanned a
// tensor at a time instead of being loaded whole.

export interface Source {
  readonly size: number;
  read(start: number, end: number): Uint8Array;
}

export class BytesSource implements Source {
  readonly size: number;
  constructor(readonly bytes: Uint8Array) { this.size = bytes.byteLength; }
  read(start: number, end: number): Uint8Array {
    return this.bytes.subarray(Math.max(0, start), Math.min(this.size, end));
  }
}

export class Reader {
  readonly size: number;

  constructor(readonly source: Source) {
    this.size = source.size;
  }

  static of(bytes: Uint8Array): Reader {
    return new Reader(new BytesSource(bytes));
  }

  /** Bytes [start, end), clipped to the file. */
  slice(start: number, end: number): Uint8Array {
    return this.source.read(start, end);
  }

  /** The first n bytes (or the whole file if shorter). */
  head(n: number): Uint8Array {
    return this.slice(0, Math.min(n, this.size));
  }

  private view(off: number, n: number): DataView {
    const b = this.slice(off, off + n);
    if (b.byteLength < n) throw new RangeError("read past end of file");
    return new DataView(b.buffer, b.byteOffset, b.byteLength);
  }

  u64(off: number): number {
    const v = this.view(off, 8).getBigUint64(0, true);
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("value exceeds safe integer range");
    return Number(v);
  }

  u32(off: number): number {
    return this.view(off, 4).getUint32(0, true);
  }

  u16(off: number): number {
    return this.view(off, 2).getUint16(0, true);
  }

  /** True if the bytes at `off` are exactly `sig`. */
  eq(off: number, sig: number[]): boolean {
    if (off < 0) return false;
    const b = this.slice(off, off + sig.length);
    if (b.byteLength < sig.length) return false;
    for (let i = 0; i < sig.length; i++) if (b[i] !== sig[i]) return false;
    return true;
  }
}

/** Thrown by a parser that ran past the end of a prefix of the file, so the
 * caller can retry with a longer prefix. */
export class NeedMore extends Error {
  constructor() { super("need more bytes"); }
}

/**
 * Run `parse` on a growing prefix of the file: 1 MiB, then 16x larger each
 * time, up to the whole file. `parse` throws NeedMore when it reads past the
 * prefix it was given; on the whole file NeedMore is not thrown (the parser
 * reports a truncated file itself).
 */
export function withPrefix<T>(r: Reader, parse: (prefix: Uint8Array, whole: boolean) => T): T {
  for (let n = 1 << 20; ; n *= 16) {
    const whole = n >= r.size;
    try {
      return parse(r.head(n), whole);
    } catch (e) {
      if (!(e instanceof NeedMore) || whole) throw e;
    }
  }
}

const decoder = new TextDecoder("utf-8", { fatal: true });

export function decodeUtf8(bytes: Uint8Array): string {
  return decoder.decode(bytes);
}
