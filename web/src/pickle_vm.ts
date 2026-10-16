// A restricted pickle virtual machine that rebuilds the object graph WITHOUT
// executing anything, mirroring the _SafeUnpickler in stowaway/pickle_scan.py.
// find_class never returns a real callable — only the stubs defined here — so a
// malicious pickle builds inert placeholders. Used by the PyTorch parser to find
// the tensors (storage keys, shapes, strides) a checkpoint declares.
import { normalize } from "./pickle_scan.js";

export class Stub {
  qualname: string;
  args: unknown[] = [];
  state: unknown = null;
  items = new Map<unknown, unknown>();
  constructor(qualname: string, args: unknown[] = []) { this.qualname = qualname; this.args = args; }
}
export class StorageType { constructor(readonly name: string) {} }
export class StorageRef {
  constructor(readonly key: string, readonly dtype: string, readonly numel: number, readonly location: string) {}
}
export class TensorRef {
  constructor(readonly storage: StorageRef | null, readonly offset: number,
              readonly shape: number[], readonly stride: number[]) {}
}

const STORAGE_DTYPES: Record<string, string> = {
  FloatStorage: "F32", DoubleStorage: "F64", HalfStorage: "F16", BFloat16Storage: "BF16",
  LongStorage: "I64", IntStorage: "I32", ShortStorage: "I16", CharStorage: "I8",
  ByteStorage: "U8", BoolStorage: "BOOL", ComplexFloatStorage: "C64",
  ComplexDoubleStorage: "C128", Float8_e4m3fnStorage: "F8_E4M3", Float8_e5m2Storage: "F8_E5M2",
};
function storageDtype(name: string): string { return STORAGE_DTYPES[name] ?? `torch.${name}`; }

function rebuildTensor(args: unknown[]): TensorRef {
  const [storage, storageOffset, size, stride] = args;
  const st = storage instanceof StorageRef ? storage : null;
  return new TensorRef(st, Number(storageOffset) || 0, toNums(size), toNums(stride));
}
function toNums(x: unknown): number[] {
  if (Array.isArray(x)) return x.map((v) => Number(v));
  if (x instanceof Stub) return [...x.items.values()].map((v) => Number(v));
  return [];
}

// Protocol-0 string arguments. UNICODE lines use raw-unicode-escape (\uXXXX and
// \UXXXXXXXX are the only escapes); STRING lines are a quoted Python repr.
function rawUnicodeEscape(s: string): string {
  return s.replace(/\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})/g, (_, a, b) => String.fromCodePoint(parseInt(a ?? b, 16)));
}
function unquote(s: string): string {
  const t = s.trim();
  if (t.length < 2 || (t[0] !== "'" && t[0] !== '"') || t[t.length - 1] !== t[0]) throw new Error("insecure string pickle");
  return t.slice(1, -1).replace(/\\(x[0-9a-fA-F]{2}|.)/g, (_, e: string) =>
    e[0] === "x" ? String.fromCharCode(parseInt(e.slice(1), 16)) : ({ n: "\n", t: "\t", r: "\r", "0": "\0" } as Record<string, string>)[e] ?? e);
}

/** A pickle LONG: signed little-endian two's complement. A plain number when it
 * fits exactly, else a BigInt (Python ints are unbounded; the legacy torch magic
 * number is 80 bits). */
function longLE(bytes: Uint8Array): number | bigint {
  let v = 0n;
  for (let i = bytes.length - 1; i >= 0; i--) v = (v << 8n) | BigInt(bytes[i]!);
  if (bytes.length && bytes[bytes.length - 1]! & 0x80) v -= 1n << BigInt(8 * bytes.length);
  return v >= BigInt(Number.MIN_SAFE_INTEGER) && v <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(v) : v;
}

const INT_OPS =new Set(["BININT", "BININT1", "BININT2", "LONG1", "INT"]);

export class SafeUnpickler {
  pos = 0;
  stack: unknown[] = [];
  marks: number[] = [];
  memo = new Map<number, unknown>();
  imports: Array<[string, string]> = [];
  private dec = new TextDecoder("utf-8");

  constructor(readonly data: Uint8Array, readonly dv: DataView) {}

  private u1() { return this.data[this.pos++]!; }
  private u2() { const v = this.dv.getUint16(this.pos, true); this.pos += 2; return v; }
  private u4() { const v = this.dv.getUint32(this.pos, true); this.pos += 4; return v; }
  private i4() { const v = this.dv.getInt32(this.pos, true); this.pos += 4; return v; }
  private u8() { const v = this.dv.getBigUint64(this.pos, true); this.pos += 8; return Number(v); }
  private take(n: number) { const s = this.data.subarray(this.pos, this.pos + n); this.pos += n; return s; }
  private line() { const s = this.pos; while (this.data[this.pos] !== 0x0a) { if (this.pos >= this.data.length) throw new Error("eof"); this.pos++; } const b = this.data.subarray(s, this.pos); this.pos++; return new TextDecoder("latin1").decode(b); }
  private popMark(): unknown[] { const m = this.marks.pop()!; const items = this.stack.slice(m); this.stack.length = m; return items; }

  findClass(module: string, name: string): unknown {
    [module, name] = normalize(module, name);
    this.imports.push([module, name]);
    const mn = `${module}.${name}`;
    if (mn === "torch._utils._rebuild_tensor" || mn === "torch._utils._rebuild_tensor_v2") return { _rebuild: "tensor" };
    if (mn === "torch._utils._rebuild_parameter" || mn === "torch._utils._rebuild_parameter_with_state") return { _rebuild: "parameter" };
    if (mn === "torch._tensor._rebuild_from_type_v2") return { _rebuild: "from_type" };
    if (mn === "collections.OrderedDict") return { _odict: true };
    if (module === "torch" && name.endsWith("Storage")) return new StorageType(name);
    return { _stub: mn };
  }

  load(): unknown {
    for (;;) {
      const op = this.u1();
      switch (op) {
        case 0x80: this.u1(); break;                              // PROTO
        case 0x95: this.u8(); break;                              // FRAME
        case 0x28: this.marks.push(this.stack.length); break;      // MARK '('
        case 0x2e: return this.stack.pop();                        // STOP '.'
        case 0x4e: this.stack.push(null); break;                   // NONE 'N'
        case 0x88: this.stack.push(true); break;                   // NEWTRUE
        case 0x89: this.stack.push(false); break;                  // NEWFALSE
        case 0x4a: this.stack.push(this.i4()); break;              // BININT 'J'
        case 0x4b: this.stack.push(this.u1()); break;              // BININT1 'K'
        case 0x4d: this.stack.push(this.u2()); break;              // BININT2 'M'
        case 0x8a: this.stack.push(longLE(this.take(this.u1()))); break; // LONG1
        case 0x8b: this.stack.push(longLE(this.take(this.u4()))); break; // LONG4
        case 0x47: this.stack.push(this.dv.getFloat64(this.pos, false)); this.pos += 8; break; // BINFLOAT 'G' (big-endian)
        case 0x8c: this.stack.push(this.dec.decode(this.take(this.u1()))); break; // SHORT_BINUNICODE
        case 0x58: this.stack.push(this.dec.decode(this.take(this.u4()))); break; // BINUNICODE 'X'
        case 0x55: this.stack.push(new TextDecoder("latin1").decode(this.take(this.u1()))); break; // SHORT_BINSTRING 'U'
        case 0x43: this.stack.push(this.take(this.u1())); break;   // SHORT_BINBYTES 'C'
        case 0x42: this.stack.push(this.take(this.u4())); break;   // BINBYTES 'B'
        case 0x7d: this.stack.push(new Map()); break;              // EMPTY_DICT '}'
        case 0x5d: this.stack.push([]); break;                     // EMPTY_LIST ']'
        case 0x29: this.stack.push([]); break;                     // EMPTY_TUPLE ')'
        case 0x85: { const a = this.stack.pop(); this.stack.push([a]); break; } // TUPLE1
        case 0x86: { const b = this.stack.pop(), a = this.stack.pop(); this.stack.push([a, b]); break; } // TUPLE2
        case 0x87: { const c = this.stack.pop(), b = this.stack.pop(), a = this.stack.pop(); this.stack.push([a, b, c]); break; } // TUPLE3
        case 0x74: this.stack.push(this.popMark()); break;         // TUPLE 't'
        case 0x71: this.memo.set(this.u1(), this.top()); break;    // BINPUT 'q'
        case 0x72: this.memo.set(this.u4(), this.top()); break;    // LONG_BINPUT 'r'
        case 0x94: this.memo.set(this.memo.size, this.top()); break; // MEMOIZE
        case 0x68: this.stack.push(this.memo.get(this.u1())); break;  // BINGET 'h'
        case 0x6a: this.stack.push(this.memo.get(this.u4())); break;  // LONG_BINGET 'j'
        case 0x63: { const mod = this.line(), name = this.line(); this.stack.push(this.findClass(mod, name)); break; } // GLOBAL 'c'
        case 0x93: { const name = this.stack.pop() as string, mod = this.stack.pop() as string; this.stack.push(this.findClass(mod, name)); break; } // STACK_GLOBAL
        case 0x51: { const pid = this.stack.pop(); this.stack.push(this.persistentLoad(pid)); break; } // BINPERSID 'Q'
        case 0x52: this.reduce(); break;                           // REDUCE 'R'
        case 0x81: this.newobj(); break;                           // NEWOBJ
        case 0x62: this.build(); break;                            // BUILD 'b'
        case 0x73: { const v = this.stack.pop(), k = this.stack.pop(); const d = this.top(); if (d instanceof Map) d.set(k, v); break; } // SETITEM 's'
        case 0x75: { const items = this.popMark(); const d = this.top(); if (d instanceof Map) for (let i = 0; i < items.length; i += 2) d.set(items[i], items[i + 1]); break; } // SETITEMS 'u'
        case 0x61: { const v = this.stack.pop(); const l = this.top(); if (Array.isArray(l)) l.push(v); break; } // APPEND 'a'
        case 0x65: { const items = this.popMark(); const l = this.top(); if (Array.isArray(l)) l.push(...items); break; } // APPENDS 'e'
        // --- protocols 0 and 1 (text opcodes), plus the rest of the stack ops ---
        case 0x30: this.stack.pop(); break;                        // POP '0'
        case 0x31: this.popMark(); break;                          // POP_MARK '1'
        case 0x32: this.stack.push(this.top()); break;             // DUP '2'
        case 0x64: { const items = this.popMark(); const d = new Map(); for (let i = 0; i < items.length; i += 2) d.set(items[i], items[i + 1]); this.stack.push(d); break; } // DICT 'd'
        case 0x6c: this.stack.push(this.popMark()); break;         // LIST 'l'
        case 0x70: this.memo.set(parseInt(this.line(), 10), this.top()); break; // PUT 'p'
        case 0x67: this.stack.push(this.memo.get(parseInt(this.line(), 10))); break; // GET 'g'
        case 0x49: { const s = this.line(); this.stack.push(s === "01" ? true : s === "00" ? false : parseInt(s, 10)); break; } // INT 'I'
        case 0x4c: this.stack.push(parseInt(this.line().replace(/L$/, ""), 10)); break; // LONG 'L'
        case 0x46: this.stack.push(parseFloat(this.line())); break; // FLOAT 'F'
        case 0x56: this.stack.push(rawUnicodeEscape(this.line())); break; // UNICODE 'V'
        case 0x53: this.stack.push(unquote(this.line())); break;   // STRING 'S'
        case 0x54: this.stack.push(new TextDecoder("latin1").decode(this.take(this.u4()))); break; // BINSTRING 'T'
        case 0x8d: this.stack.push(this.dec.decode(this.take(this.u8()))); break; // BINUNICODE8
        case 0x8e: this.stack.push(this.take(this.u8())); break;   // BINBYTES8
        case 0x8f: this.stack.push(new Set()); break;              // EMPTY_SET
        case 0x90: { const items = this.popMark(); const s = this.top(); if (s instanceof Set) for (const x of items) s.add(x); break; } // ADDITEMS
        case 0x91: this.stack.push(new Set(this.popMark())); break; // FROZENSET
        case 0x50: this.stack.push(this.persistentLoad(this.line())); break; // PERSID 'P'
        case 0x69: { const mod = this.line(), name = this.line(); const args = this.popMark(); this.stack.push(this.callReduce(this.findClass(mod, name), args)); break; } // INST 'i'
        case 0x6f: { const items = this.popMark(); const cls = items.shift(); this.stack.push(this.callReduce(cls, items)); break; } // OBJ 'o'
        case 0x92: { this.stack.pop(); const args = this.stack.pop(); const cls = this.stack.pop(); this.stack.push(this.callReduce(cls, Array.isArray(args) ? args : [])); break; } // NEWOBJ_EX
        default: throw new Error(`unsupported opcode 0x${op.toString(16)} at ${this.pos - 1}`);
      }
      void INT_OPS;
    }
  }

  private top(): unknown { return this.stack[this.stack.length - 1]; }

  private persistentLoad(pid: unknown): unknown {
    if (Array.isArray(pid) && pid.length >= 5 && pid[0] === "storage") {
      const st = pid[1];
      const dtype = st instanceof StorageType ? storageDtype(st.name) : "?";
      return new StorageRef(String(pid[2]), dtype, Number(pid[4]), String(pid[3]));
    }
    return new Stub("persid");
  }

  private reduce(): void {
    const args = this.stack.pop() as unknown[];
    const func = this.stack.pop();
    this.stack.push(this.callReduce(func, Array.isArray(args) ? args : []));
  }
  private callReduce(func: unknown, args: unknown[]): unknown {
    const f = func as { _rebuild?: string; _stub?: string; _odict?: boolean };
    if (f && f._rebuild === "tensor") return rebuildTensor(args);
    if (f && f._rebuild === "parameter") return args[0];
    if (f && f._rebuild === "from_type") { const inner = args[0], iargs = (args[1] as unknown[]) ?? []; return this.callReduce(inner, iargs); }
    if (f && f._odict) return new Map();
    return new Stub(f?._stub ?? "?", args);
  }
  private newobj(): void { const args = this.stack.pop(); const cls = this.stack.pop(); this.stack.push(this.callReduce(cls, Array.isArray(args) ? args : [])); }

  private build(): void {
    const state = this.stack.pop();
    const obj = this.top();
    if (obj instanceof Stub) obj.state = state;
    else if (obj instanceof Map && state instanceof Map) for (const [k, v] of state) obj.set(k, v);
  }
}

export function safeLoad(data: Uint8Array): { obj: unknown; error: string | null } {
  try {
    const u = new SafeUnpickler(data, new DataView(data.buffer, data.byteOffset, data.byteLength));
    return { obj: u.load(), error: null };
  } catch (e) {
    return { obj: null, error: `${(e as Error).name}: ${(e as Error).message}` };
  }
}

// Yield [name, TensorRef] for every tensor reachable from obj, mirroring
// pytorch.walk_tensors (dict/list/Stub traversal, OrderedDict is a Map here).
const SKIP_SEGMENTS = new Set(["_parameters", "_buffers", "_modules"]);
export function walkTensors(obj: unknown, path: string[] = [], seen = new Set<unknown>(), depth = 0): Array<[string, TensorRef]> {
  const out: Array<[string, TensorRef]> = [];
  if (depth > 64 || (obj !== null && typeof obj === "object" && seen.has(obj))) return out;
  if (obj instanceof TensorRef) { out.push([path.join(".") || "<root>", obj]); return out; }
  if (obj !== null && typeof obj === "object") seen.add(obj);
  if (obj instanceof Map) {
    for (const [k, v] of obj) out.push(...walkTensors(v, [...path, String(k)], seen, depth + 1));
  } else if (Array.isArray(obj)) {
    obj.forEach((v, i) => out.push(...walkTensors(v, [...path, String(i)], seen, depth + 1)));
  } else if (obj instanceof Stub) {
    for (const part of [obj.state, obj.items]) {
      if (part instanceof Map) for (const [k, v] of part) out.push(...walkTensors(v, SKIP_SEGMENTS.has(String(k)) ? path : [...path, String(k)], seen, depth + 1));
    }
    if (Array.isArray(obj.args)) out.push(...walkTensors(obj.args, path, seen, depth + 1));
  }
  return out;
}
