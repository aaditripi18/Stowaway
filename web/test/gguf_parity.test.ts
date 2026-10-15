// GGUF parity: the TypeScript scanner must match the Python report byte for byte.
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { scan } from "../src/report.js";
import { canon } from "../src/canon.js";

const DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "gguf");
const names = readdirSync(DIR).filter((f) => f.endsWith(".gguf")).map((f) => f.replace(/\.gguf$/, ""));

describe("GGUF report parity", () => {
  it("has fixtures", () => expect(names.length).toBeGreaterThan(0));
  for (const name of names) {
    it(name, async () => {
      const bytes = new Uint8Array(readFileSync(join(DIR, `${name}.gguf`)));
      const golden = readFileSync(join(DIR, `${name}.json`), "utf-8").trimEnd();
      const rep = await scan(bytes);
      rep.file = `${name}.gguf`;
      expect(canon(rep)).toEqual(golden);
    });
  }
});
