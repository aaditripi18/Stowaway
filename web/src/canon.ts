// Canonical JSON: the exact string the Python side also produces, so reports
// can be compared byte for byte. Keys are sorted; numbers that are integer-valued
// are written without a decimal point (so Python's 0.0 and JS's 0 agree); other
// numbers use the shortest round-tripping decimal (identical in CPython repr and
// V8 for the rounded values the report contains). Mirrors scripts' _canon in
// tests/parity.py.

export function canon(o: unknown): string {
  if (o === null || o === undefined) return "null";
  if (typeof o === "boolean") return o ? "true" : "false";
  if (typeof o === "number") {
    if (!Number.isFinite(o)) throw new Error(`non-finite number in report: ${o}`);
    if (Number.isInteger(o)) return String(o); // 0.0 -> "0", 4 -> "4"
    return String(o);
  }
  if (typeof o === "string") return JSON.stringify(o);
  if (Array.isArray(o)) return "[" + o.map(canon).join(",") + "]";
  if (typeof o === "object") {
    const keys = Object.keys(o as Record<string, unknown>).sort();
    return "{" + keys.map((k) => JSON.stringify(k) + ":" + canon((o as Record<string, unknown>)[k])).join(",") + "}";
  }
  throw new Error(`cannot canonicalise ${typeof o}`);
}
