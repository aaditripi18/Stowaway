// Opt-in: scan a multi-gigabyte model from disk through a Source, the way the
// browser worker reads a File, and require (1) the same report as the Python
// tool, byte for byte, and (2) bounded memory: no single read and no growth in
// resident memory anywhere near the size of the file.
//
//   python scripts/make_large_fixture.py data/large/big.safetensors --gb 3.2   (repo root)
//   STOWAWAY_LARGE_FILE=../data/large/big.safetensors npx vitest run test/large_file.test.ts
import { closeSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Reader, Source } from "../src/bytes.js";
import { canon } from "../src/canon.js";
import { scan } from "../src/report.js";

const PATH = process.env.STOWAWAY_LARGE_FILE;

class DiskSource implements Source {
  readonly size: number;
  largest = 0;
  constructor(private readonly fd: number) { this.size = fstatSync(fd).size; }
  read(start: number, end: number): Uint8Array {
    const s = Math.max(0, start), e = Math.min(this.size, end);
    const out = new Uint8Array(Math.max(0, e - s));
    let got = 0;
    while (got < out.length) got += readSync(this.fd, out, got, out.length - got, s + got);
    this.largest = Math.max(this.largest, out.length);
    return out;
  }
}

describe.runIf(PATH)("large file", () => {
  it("matches the Python report with bounded memory", async () => {
    const fd = openSync(PATH!, "r");
    const src = new DiskSource(fd);
    let peak = 0;
    const timer = setInterval(() => { peak = Math.max(peak, process.memoryUsage().rss); }, 50);
    const t0 = performance.now();
    const rep = await scan(new Reader(src));
    clearInterval(timer);
    peak = Math.max(peak, process.memoryUsage().rss);
    closeSync(fd);
    rep.file = PATH!.split(/[\\/]/).pop()!;
    const golden = readFileSync(PATH!.replace(/\.safetensors$/, ".json"), "utf-8").trimEnd();
    const gib = (n: number) => (n / 2 ** 30).toFixed(2);
    console.log(`size ${gib(src.size)} GiB, ${((performance.now() - t0) / 1000).toFixed(1)} s, ` +
                `largest read ${(src.largest / 2 ** 20).toFixed(0)} MiB, peak RSS ${gib(peak)} GiB`);
    expect(canon(rep)).toEqual(golden);
    expect(src.size).toBeGreaterThan(2 ** 31);       // past what one ArrayBuffer read used to need
    expect(src.largest).toBeLessThanOrEqual(64 << 20); // one tensor or one hash chunk at a time
    expect(peak).toBeLessThan(src.size / 4);
  }, 30 * 60 * 1000);
});
