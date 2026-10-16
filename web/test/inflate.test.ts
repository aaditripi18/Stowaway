// The DEFLATE decoder against Node's zlib on stored, fixed and dynamic blocks.
import { deflateRawSync, constants } from "node:zlib";
import { describe, expect, it } from "vitest";
import { inflateRaw } from "../src/inflate.js";

function sample(n: number, seed: number): Uint8Array {
  const out = new Uint8Array(n);
  let x = seed;
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) >>> 0;
    out[i] = i % 7 === 0 ? (x >>> 24) : "stowaway pickle payload "[(i >> 2) % 24]!.charCodeAt(0);
  }
  return out;
}

describe("inflateRaw", () => {
  const cases: Array<[string, Uint8Array, object]> = [
    ["empty", new Uint8Array(0), {}],
    ["stored (level 0)", sample(70_000, 1), { level: 0 }],
    ["fixed Huffman", sample(200, 2), { strategy: constants.Z_FIXED }],
    ["dynamic Huffman", sample(300_000, 3), { level: 9 }],
    ["long repeats", new Uint8Array(100_000).fill(7), {}],
    ["random (incompressible)", Uint8Array.from({ length: 50_000 }, (_, i) => (i * 2654435761) >>> 23 & 255), {}],
  ];
  for (const [name, data, opts] of cases) {
    it(name, () => {
      const z = new Uint8Array(deflateRawSync(data, opts));
      expect(Buffer.from(inflateRaw(z, data.length))).toEqual(Buffer.from(data));
    });
  }
  it("refuses output beyond the declared size", () => {
    const z = new Uint8Array(deflateRawSync(new Uint8Array(10_000)));
    expect(() => inflateRaw(z, 9_999)).toThrow(/larger than declared/);
  });
  it("refuses truncated input", () => {
    const z = new Uint8Array(deflateRawSync(sample(10_000, 4)));
    expect(() => inflateRaw(z.subarray(0, z.length >> 1), 10_000)).toThrow();
  });
});
