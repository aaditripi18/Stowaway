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

  it("thin payload (fewer than 256 changes per tensor) vs base", async () => {
    const base = openBase(new Uint8Array(readFileSync(join(DIR, "base.safetensors"))));
    const model = new Uint8Array(readFileSync(join(DIR, "thin.safetensors")));
    const rep = await scan(model, base);
    rep.file = "thin.safetensors";
    expect(canon(rep)).toEqual(readFileSync(join(DIR, "thin.json"), "utf-8").trimEnd());
  });

  it("honest LayerNorm shift is cleared by S5d, the payload is not", async () => {
    const base = openBase(new Uint8Array(readFileSync(join(DIR, "base_gain.safetensors"))));
    const model = new Uint8Array(readFileSync(join(DIR, "gain.safetensors")));
    const rep = await scan(model, base);
    rep.file = "gain.safetensors";
    expect(canon(rep)).toEqual(readFileSync(join(DIR, "gain.json"), "utf-8").trimEnd());
    const findings = (JSON.parse(canon(rep)) as { findings: { check: string; severity: string; detail: { tensor?: string } }[] }).findings;
    const s5 = findings.filter((f) => f.check === "S5" && f.severity === "high").map((f) => f.detail.tensor);
    expect(s5).toEqual(["layer.1.weight"]);
  });

  it("renamed (wrapper prefix and gamma/beta names) vs base", async () => {
    const base = openBase(new Uint8Array(readFileSync(join(DIR, "base_ln.safetensors"))));
    const model = new Uint8Array(readFileSync(join(DIR, "renamed.safetensors")));
    const rep = await scan(model, base);
    rep.file = "renamed.safetensors";
    expect(canon(rep)).toEqual(readFileSync(join(DIR, "renamed.json"), "utf-8").trimEnd());
  });
});
