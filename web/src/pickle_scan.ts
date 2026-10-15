// S1: pickle analysis without executing anything. Mirrors the import-detection
// half of stowaway/pickle_scan.py — the restricted unpickler there is only used
// to rebuild tensors for the PyTorch path (ported with that format); the findings
// come from the same opcode walk and the same classify()/normalize() tables.
import { Finding, finding, Severity } from "./model.js";
import tables from "./pickle_tables.json";

const SAFE = new Set(tables.SAFE);
const DANGEROUS = new Set(tables.DANGEROUS);
const TORCH_DTYPES = new Set(tables.TORCH_DTYPES);
const IMPORT_MAPPING = tables.IMPORT_MAPPING as Record<string, string>;
const NAME_MAPPING = tables.NAME_MAPPING as Record<string, string>;

export function normalize(module: string, name: string): [string, string] {
  const nm = NAME_MAPPING[`${module}|${name}`];
  if (nm !== undefined) { const [m, n] = nm.split("|"); return [m!, n!]; }
  return [IMPORT_MAPPING[module] ?? module, name];
}

export function classify(module: string, name: string): "safe" | "dangerous" | "unknown" {
  if (SAFE.has(`${module}|${name}`)) return "safe";
  if (module === "torch" && (name.endsWith("Storage") || TORCH_DTYPES.has(name))) return "safe";
  const top = module.split(".");
  for (let i = top.length; i > 0; i--) {
    if (DANGEROUS.has(`${top.slice(0, i).join(".")}|*`)) return "dangerous";
  }
  if (DANGEROUS.has(`${module}|${name}`)) return "dangerous";
  return "unknown";
}

// --- a minimal pickle opcode walker (pickletools.genops equivalent) ----------

interface Op { name: string; arg: unknown }

class PickleReader {
  pos = 0;
  constructor(readonly data: Uint8Array, readonly view: DataView) {}
  byte(): number { if (this.pos >= this.data.length) throw new Error("eof"); return this.data[this.pos++]!; }
  take(n: number): Uint8Array { if (this.pos + n > this.data.length) throw new Error("eof"); const s = this.data.subarray(this.pos, this.pos + n); this.pos += n; return s; }
  u1(): number { return this.byte(); }
  u2(): number { const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
  u4(): number { const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  i4(): number { const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
  u8(): number { const v = this.view.getBigUint64(this.pos, true); this.pos += 8; if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("huge"); return Number(v); }
  line(): string { const start = this.pos; while (this.pos < this.data.length && this.data[this.pos] !== 0x0a) this.pos++; if (this.pos >= this.data.length) throw new Error("eof"); const s = this.data.subarray(start, this.pos); this.pos++; return new TextDecoder("latin1").decode(s); }
}

const utf8 = new TextDecoder("utf-8");
const latin1 = new TextDecoder("latin1");

// opcode byte -> (reader) -> Op. Only opcodes that carry imports or strings need
// a meaningful arg; the rest just consume the right number of bytes to stay aligned.
function readOp(r: PickleReader): Op {
  const code = r.byte();
  const c = String.fromCharCode(code);
  switch (c) {
    // imports
    case "c": { const mod = r.line(), name = r.line(); return { name: "GLOBAL", arg: `${mod} ${name}` }; }
    case "i": { const mod = r.line(), name = r.line(); return { name: "INST", arg: `${mod} ${name}` }; }
    case "\x93": return { name: "STACK_GLOBAL", arg: null };
    // strings (push onto recent)
    case "\x8c": return { name: "SHORT_BINUNICODE", arg: utf8.decode(r.take(r.u1())) };
    case "X": return { name: "BINUNICODE", arg: utf8.decode(r.take(r.u4())) };
    case "\x8d": return { name: "BINUNICODE8", arg: utf8.decode(r.take(r.u8())) };
    case "V": return { name: "UNICODE", arg: r.line() };
    case "U": return { name: "SHORT_BINSTRING", arg: latin1.decode(r.take(r.u1())) };
    case "T": return { name: "BINSTRING", arg: latin1.decode(r.take(r.u4())) };
    case "S": return { name: "STRING", arg: stripQuotes(r.line()) };
    // memo
    case "g": return { name: "GET", arg: parseInt(r.line(), 10) };
    case "h": return { name: "BINGET", arg: r.u1() };
    case "j": return { name: "LONG_BINGET", arg: r.u4() };
    case "p": return { name: "PUT", arg: parseInt(r.line(), 10) };
    case "q": return { name: "BINPUT", arg: r.u1() };
    case "r": return { name: "LONG_BINPUT", arg: r.u4() };
    case "\x94": return { name: "MEMOIZE", arg: null };
    case ".": return { name: "STOP", arg: null };
    // everything else: consume the argument bytes to stay byte-aligned
    case "\x80": r.u1(); return op("PROTO");
    case "\x95": r.u8(); return op("FRAME");
    case "F": r.line(); return op("FLOAT");
    case "I": r.line(); return op("INT");
    case "L": r.line(); return op("LONG");
    case "P": r.line(); return op("PERSID");
    case "J": r.i4(); return op("BININT");
    case "K": r.u1(); return op("BININT1");
    case "M": r.u2(); return op("BININT2");
    case "\x8a": r.take(r.u1()); return op("LONG1");
    case "\x8b": r.take(r.i4()); return op("LONG4");
    case "B": r.take(r.u4()); return op("BINBYTES");
    case "C": r.take(r.u1()); return op("SHORT_BINBYTES");
    case "\x8e": r.take(r.u8()); return op("BINBYTES8");
    case "\x96": r.take(r.u8()); return op("BYTEARRAY8");
    case "\x82": r.u1(); return op("EXT1");
    case "\x83": r.u2(); return op("EXT2");
    case "\x84": r.i4(); return op("EXT4");
    case "G": r.take(8); return op("BINFLOAT");
    default: return op(code.toString()); // opcodes with no argument
  }
}
function op(name: string): Op { return { name, arg: null }; }
function stripQuotes(s: string): string {
  const t = s.trim();
  if (t.length >= 2 && (t[0] === "'" || t[0] === '"') && t[t.length - 1] === t[0]) return t.slice(1, -1);
  return t;
}

const STRING_OPS = new Set(["SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"]);

export function opcodeImports(data: Uint8Array): { found: Array<[string, string]>; err: string | null } {
  const found: Array<[string, string]> = [];
  const recent: unknown[] = [];
  const memo = new Map<number, unknown>();
  const r = new PickleReader(data, new DataView(data.buffer, data.byteOffset, data.byteLength));
  let err: string | null = null;
  try {
    for (;;) {
      const { name, arg } = readOp(r);
      if (name === "GLOBAL" || name === "INST") {
        const s = arg as string; const i = s.indexOf(" ");
        found.push(normalize(s.slice(0, i), s.slice(i + 1))); recent.push(null);
      } else if (STRING_OPS.has(name)) {
        recent.push(arg);
      } else if (name === "BINGET" || name === "LONG_BINGET" || name === "GET") {
        recent.push(memo.get(arg as number) ?? null);
      } else if (name === "BINPUT" || name === "LONG_BINPUT" || name === "PUT") {
        memo.set(arg as number, recent.length ? recent[recent.length - 1] : null);
      } else if (name === "MEMOIZE") {
        memo.set(memo.size, recent.length ? recent[recent.length - 1] : null);
      } else if (name === "STACK_GLOBAL") {
        const mod = recent.length >= 2 ? recent[recent.length - 2] : null;
        const nm = recent.length >= 1 ? recent[recent.length - 1] : null;
        found.push(normalize(String(mod), String(nm))); recent.push(null);
      } else if (name === "STOP") {
        break;
      } else {
        recent.push(null);
      }
    }
  } catch (e) {
    err = `${(e as Error).name || "Error"}: ${(e as Error).message}`;
  }
  return { found, err };
}

export function analysePickles(label: string, data: Uint8Array): Finding[] {
  const { found, err } = opcodeImports(data);
  const seen = new Map<string, [string, string]>();
  for (const [mod, nm] of found) seen.set(`${mod}|${nm}`, [mod, nm]); // ordered dedupe
  const findings: Finding[] = [];
  for (const [mod, nm] of seen.values()) {
    const kind = classify(mod, nm);
    if (kind === "safe") continue;
    const sev: Severity = kind === "dangerous" ? "critical" : "medium";
    findings.push(finding("S1", sev, `${kind}-import`,
      `${label}: imports ${mod}.${nm}` + (kind === "dangerous" ? " (code execution gadget)" : ""),
      { stream: label, module: mod, name: nm }));
  }
  if (err) {
    findings.push(finding("S1", "medium", "malformed-pickle", `${label}: pickle data does not parse cleanly`, { stream: label }));
  }
  return findings;
}
