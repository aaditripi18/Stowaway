// S5 parity: scanning the poisoned model with the real base model as the diff
// target must match the Python scan(poisoned, base=base) report byte for byte.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { scan, openBase } from "../src/report.js";
import { canon } from "../src/canon.js";

const DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "s5");

describe("S5 differential parity", () => {
  it("poisoned vs base", async () => {
    const baseBytes = new Uint8Array(readFileSync(join(DIR, "base.safetensors")));
    const base = openBase(baseBytes);
    const poisoned = new Uint8Array(readFileSync(join(DIR, "poisoned.safetensors")));
    const golden = readFileSync(join(DIR, "poisoned.json"), "utf-8").trimEnd();
    const rep = await scan(poisoned, base);
    rep.file = "poisoned.safetensors";
    expect(canon(rep)).toEqual(golden);
  });
});
