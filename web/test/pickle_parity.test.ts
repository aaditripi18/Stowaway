// S1 parity: for each pickle fixture, the TypeScript analysePickles() must
// produce the same canonical findings as the Python pickle_scan.analyse().
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { analysePickles } from "../src/pickle_scan.js";
import { scan } from "../src/report.js";
import { canon } from "../src/canon.js";

const DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "pickle");
const names = readdirSync(DIR).filter((f) => f.endsWith(".pkl")).map((f) => f.replace(/\.pkl$/, ""));

describe("S1 pickle findings parity", () => {
  it("has fixtures", () => expect(names.length).toBeGreaterThan(0));
  for (const name of names) {
    it(name, () => {
      const bytes = new Uint8Array(readFileSync(join(DIR, `${name}.pkl`)));
      const golden = readFileSync(join(DIR, `${name}.findings.json`), "utf-8").trimEnd();
      expect(canon(analysePickles(name, bytes))).toEqual(golden);
    });
  }
});

// Full-file parity: format detection must route every pickle fixture, including
// protocol 0 (no 0x80 marker), to the pickle scanner exactly as Python does.
describe("pickle full-report parity", () => {
  for (const name of names) {
    it(name, async () => {
      const bytes = new Uint8Array(readFileSync(join(DIR, `${name}.pkl`)));
      const golden = readFileSync(join(DIR, `${name}.json`), "utf-8").trimEnd();
      const rep = await scan(bytes);
      rep.file = `${name}.pkl`; // Python uses the file name
      expect(canon(rep)).toEqual(golden);
    });
  }

  it("does not mistake plain text for an old pickle", async () => {
    const rep = await scan(new TextEncoder().encode("hello world, this is not a model\n"));
    expect(rep.format).toBe("unknown");
  });
});
