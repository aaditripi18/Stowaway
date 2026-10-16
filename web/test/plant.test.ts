// plant() must write exactly what Python's stowaway.embed writes: same frame,
// same bit order, same elements. The fixture was planted by Python.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { Reader } from "../src/bytes.js";
import { openModel, scan } from "../src/report.js";
import { capacity, plant } from "../src/plant.js";

const FIX = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

describe("plant() matches stowaway.embed", () => {
  const base = new Uint8Array(readFileSync(join(FIX, "s5", "base.safetensors")));
  const mf = openModel(Reader.of(base));
  const opts = { k: 5, spread: 3, names: ["layer.2.weight", "layer.0.weight"] };

  it("writes the same bytes as Python", () => {
    const payload = new Uint8Array(readFileSync(join(FIX, "plant", "payload.bin")));
    const expected = new Uint8Array(readFileSync(join(FIX, "plant", "planted_k5_s3.safetensors")));
    expect(Buffer.from(plant(base, mf, payload, opts))).toEqual(Buffer.from(expected));
  });

  it("capacity matches the Python formula and is enforced", () => {
    // 2 tensors x ceil(8192 / 3) elements x 5 bits / 8, minus the 4-byte frame
    expect(capacity(mf, opts)).toBe(Math.floor((2 * Math.ceil(8192 / 3) * 5) / 8) - 4);
    expect(() => plant(base, mf, new Uint8Array(capacity(mf, opts) + 1), opts)).toThrow(/does not fit/);
    expect(() => plant(base, mf, new Uint8Array(10), { ...opts, k: 24 })).toThrow(/k must be/);
  });

  it("leaves the source untouched and S5 sees the planted tensors", async () => {
    const copy = base.slice();
    const planted = plant(base, mf, new Uint8Array(1000).fill(0xa5), opts);
    expect(Buffer.from(base)).toEqual(Buffer.from(copy));
    const { openBase } = await import("../src/report.js");
    const rep = (await scan(planted, openBase(base))) as { findings: Array<{ check: string; detail: { tensor?: string } }> };
    const s5 = rep.findings.filter((f) => f.check === "S5").map((f) => f.detail.tensor);
    expect(s5).toContain("layer.2.weight");
  });
});
