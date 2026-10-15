// Little-endian reader over a byte buffer. The scanner reads slices; a browser
// File is read via file.arrayBuffer() or sliced for large files.

export class Reader {
  readonly bytes: Uint8Array;
  readonly view: DataView;
  readonly size: number;

  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.size = bytes.byteLength;
  }

  slice(start: number, end: number): Uint8Array {
    return this.bytes.subarray(start, end);
  }

  u64(off: number): number {
    const v = this.view.getBigUint64(off, true);
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("value exceeds safe integer range");
    return Number(v);
  }

  u32(off: number): number {
    return this.view.getUint32(off, true);
  }
}

const decoder = new TextDecoder("utf-8", { fatal: true });

export function decodeUtf8(bytes: Uint8Array): string {
  return decoder.decode(bytes);
}
