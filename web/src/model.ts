// Format-independent description of a model file. Mirrors stowaway/model.py.

export const COVERING_KINDS = new Set(["header", "tensor", "padding", "structure"]);
export const REGION_KINDS = new Set([...COVERING_KINDS, "container"]);
export const SEVERITIES = ["info", "low", "medium", "high", "critical"] as const;
export type Severity = (typeof SEVERITIES)[number];

// Bytes per element; matches ITEMSIZE in stowaway/model.py.
export const ITEMSIZE: Record<string, number> = {
  F64: 8, F32: 4, F16: 2, BF16: 2,
  I64: 8, I32: 4, I16: 2, I8: 1,
  U64: 8, U32: 4, U16: 2, U8: 1,
  BOOL: 1, F8_E4M3: 1, F8_E5M2: 1,
  C64: 8, C128: 16,
};

export interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
  offset: number | null; // absolute first byte, or null if not addressable
  nbytes: number;
}

export interface Region {
  start: number;
  end: number; // exclusive
  kind: string;
  label: string;
}

export interface Finding {
  check: string; // "S1", "S2", ... or "parse"
  severity: Severity;
  code: string;
  message: string;
  detail: Record<string, unknown>;
}

export interface ModelFile {
  format: string;
  size: number;
  tensors: TensorInfo[];
  regions: Region[];
  findings: Finding[];
  metadata: Record<string, unknown>;
  pickles: Array<[string, number, number]>; // (label, start, end)
  inlinePickles: Record<string, Uint8Array>;
}

export function newModelFile(format: string, size: number): ModelFile {
  return { format, size, tensors: [], regions: [], findings: [], metadata: {}, pickles: [], inlinePickles: {} };
}

export function region(start: number, end: number, kind: string, label = ""): Region {
  if (!REGION_KINDS.has(kind)) throw new Error(`unknown region kind ${kind}`);
  if (!(start >= 0 && start <= end)) throw new Error(`bad region [${start}, ${end})`);
  return { start, end, kind, label };
}

export function finding(check: string, severity: Severity, code: string, message: string,
                        detail: Record<string, unknown> = {}): Finding {
  if (!SEVERITIES.includes(severity)) throw new Error(`unknown severity ${severity}`);
  return { check, severity, code, message, detail };
}

export function prod(shape: number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}
