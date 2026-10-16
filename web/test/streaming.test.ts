// Scanning through a Source must give the same report as scanning the bytes,
// and the incremental SHA-256 must equal Web Crypto's. The browser reads a File
// through exactly this interface (one range at a time, in a worker).
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { Reader, Source } from "../src/bytes.js";
import { canon } from "../src/canon.js";
import { scan } from "../src/report.js";
import { Sha256 } from "../src/sha256.js";

const FIX = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

/** Hands out copies (never views of one big buffer), and records the reads. */
class CountingSource implements Source {
  readonly size: number;
  largest = 0;
  reads = 0;
  constructor(private readonly bytes: Uint8Array) { this.size = bytes.byteLength; }
  read(start: number, end: number): Uint8Array {
    const out = this.bytes.slice(Math.max(0, start), Math.min(this.size, end));
    this.largest = Math.max(this.largest, out.byteLength);
    this.reads++;
    return out;
  }
}

function modelFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...modelFiles(p));
    else if (!name.endsWith(".json")) out.push(p);
  }
  return out;
}

async function hex(bytes: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

describe("incremental SHA-256", () => {
  for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, (1 << 20) + 3]) {
    it(`${n} bytes, fed in uneven chunks`, async () => {
      const data = new Uint8Array(n);
      for (let i = 0; i < n; i++) data[i] = (i * 2654435761) >>> 24;
      const h = new Sha256();
      for (let o = 0, step = 1; o < n; o += step, step = (step * 7) % 97 + 1) h.update(data.subarray(o, Math.min(n, o + step)));
      expect(h.hex()).toEqual(await hex(data));
    });
  }
});

describe("scanning through a Source equals scanning the bytes", () => {
  const files = modelFiles(FIX);
  it("has fixtures", () => expect(files.length).toBeGreaterThan(20));
  for (const path of files) {
    it(path.slice(FIX.length + 1).replace(/\\/g, "/"), async () => {
      const bytes = new Uint8Array(readFileSync(path));
      const src = new CountingSource(bytes);
      const viaSource = await scan(new Reader(src));
      expect(canon(viaSource)).toEqual(canon(await scan(bytes)));
      expect(src.reads).toBeGreaterThan(0);
    });
  }

  it("a large GGUF header is read in growing pieces, not as the whole file first", async () => {
    const bytes = new Uint8Array(readFileSync(join(FIX, "gguf", "big_header.gguf")));
    expect(bytes.byteLength).toBeGreaterThan(1 << 20);
    const src = new CountingSource(bytes);
    await scan(new Reader(src));
    // the first attempt reads 1 MiB, runs out, then reads more
    expect(src.reads).toBeGreaterThan(3);
  });
});
