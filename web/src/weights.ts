// S3/S4/S5 weight statistics. Ported in a later commit. Until then the browser
// build runs only the structural checks (S1/S2); the report omits weight findings.
import { Finding, ModelFile } from "./model.js";
import { Reader } from "./bytes.js";

export type RawFloats = Record<string, unknown>;

export function analyseWeights(_mf: ModelFile, _r: Reader, _base?: RawFloats): Finding[] {
  return [];
}
