// S1 pickle analysis. Ported in a later commit; safetensors files carry no
// pickle streams, so this is unused until the PyTorch/pickle formats land.
import { Finding } from "./model.js";

export function analysePickles(_label: string, _data: Uint8Array): Finding[] {
  return [];
}
