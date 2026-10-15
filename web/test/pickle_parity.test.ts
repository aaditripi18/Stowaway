// S1 parity: for each pickle fixture, the TypeScript analysePickles() must
// produce the same canonical findings as the Python pickle_scan.analyse().
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { analysePickles } from "../src/pickle_scan.js";
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
