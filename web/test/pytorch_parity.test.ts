// PyTorch zip parity: the TypeScript scanner must match the Python report byte
// for byte, including the storage/tensor mapping and the embedded-pickle S1 scan.
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { scan } from "../src/report.js";
import { canon } from "../src/canon.js";

const DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "pytorch");
const names = readdirSync(DIR).filter((f) => f.endsWith(".pt")).map((f) => f.replace(/\.pt$/, ""));

describe("PyTorch zip report parity", () => {
  it("has fixtures", () => expect(names.length).toBeGreaterThan(0));
  for (const name of names) {
    it(name, async () => {
      const bytes = new Uint8Array(readFileSync(join(DIR, `${name}.pt`)));
      const golden = readFileSync(join(DIR, `${name}.json`), "utf-8").trimEnd();
      const rep = await scan(bytes);
      rep.file = `${name}.pt`;
      expect(canon(rep)).toEqual(golden);
    });
  }
});
