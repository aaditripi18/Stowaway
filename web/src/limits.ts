// Controller for the limits page: load a model into the limits worker, plant
// random bytes with the reader's parameters, and show what each check saw.
/// <reference types="vite/client" />
import "./styles.css";
import LimitsWorker from "./limits_worker.ts?worker&inline";
import type { TensorRow, TryResult } from "./limits_worker.js";

declare global { interface Window { STOWAWAY_SAMPLE?: { meta: Record<string, unknown>; b64: string } } }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
const fmt = (n: number) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(n >= 10 << 20 ? 0 : 1)} MB` : n >= 1024 ? `${(n / 1024).toFixed(n >= 10240 ? 0 : 1)} KB` : `${n} B`);

const worker: Worker = new LimitsWorker();
let nextId = 0;
const waiting = new Map<number, { resolve: (m: any) => void; reject: (e: Error) => void; progress?: (m: any) => void }>();
worker.onmessage = (ev) => {
  const m = ev.data, w = waiting.get(m.id);
  if (!w) return;
  if (m.kind === "progress") { w.progress?.(m); return; }
  waiting.delete(m.id);
  if (m.kind === "error") w.reject(new Error(m.message)); else w.resolve(m);
};
worker.onerror = (ev) => { for (const [id, w] of waiting) { waiting.delete(id); w.reject(new Error(ev.message || "worker stopped")); } };
function ask(msg: Record<string, unknown>, transfer: Transferable[] = [], progress?: (m: any) => void): Promise<any> {
  const id = nextId++;
  return new Promise((resolve, reject) => { waiting.set(id, { resolve, reject, progress }); worker.postMessage({ id, ...msg }, transfer); });
}

let rows: TensorRow[] = [];
let modelName = "";
let lastPlanted: Uint8Array | null = null;
const maxMant = () => Math.min(...rows.map((r) => ({ F32: 23, F16: 10, BF16: 7 }[r.dtype] ?? 23)));

function sizeFromSlider(): number {
  // log scale from 64 B to 16 MB
  const t = Number($<HTMLInputElement>("size").value) / 100;
  return Math.round(64 * Math.pow((16 << 20) / 64, t));
}
function refreshLabels(): void {
  $("size-label").textContent = fmt(sizeFromSlider()) + " (capped at what fits)";
  $("k-label").textContent = $<HTMLInputElement>("k").value;
}

function fillTargets(): void {
  const sel = $<HTMLSelectElement>("target");
  const fair = rows.filter((r) => r.bit0 !== null && Math.abs(r.bit0 - 0.5) < 0.01).length;
  const roles = [...new Set(rows.map((r) => r.role))].filter((ro) => rows.filter((r) => r.role === ro).length >= 2);
  sel.innerHTML = [
    `<option value="all">all ${rows.length} float tensors</option>`,
    `<option value="fair">tensors whose low bit is already a fair coin (${fair})</option>`,
    `<option value="largest">only the largest tensor</option>`,
    ...roles.map((ro) => `<option value="role:${esc(ro)}">every ${esc(ro)} (${rows.filter((r) => r.role === ro).length})</option>`),
  ].join("");
}

async function load(bytes: Uint8Array, name: string): Promise<void> {
  $("model-status").textContent = `Reading ${name}…`;
  try {
    const m = await ask({ kind: "load", bytes, bf16: $<HTMLInputElement>("bf16").checked }, [bytes.buffer]);
    rows = m.rows; modelName = name;
    $("model-status").textContent = `${name}: ${rows.length} float tensors (${m.dtypes.join(", ")}). Clean scan: ${m.cleanVerdict}` +
      (m.cleanFlagged.length ? `, ${m.cleanFlagged.length} tensor(s) already flagged and not counted as catches.` : ".") +
      ($<HTMLInputElement>("bf16").checked ? " Rounded to bf16 precision first." : "");
    $<HTMLInputElement>("k").max = String(maxMant());
    fillTargets();
    for (const id of ["controls", "sweep-box"]) $(id).classList.remove("opacity-50", "pointer-events-none");
    $("result").classList.add("hidden"); $("download").classList.add("hidden"); $("sweep-grid").innerHTML = "";
  } catch (e) {
    $("model-status").textContent = `Could not load ${name}: ${(e as Error).message}`;
  }
}

function list(el: HTMLElement, findings: TryResult["single"]["findings"]): void {
  el.innerHTML = findings.slice(0, 6).map((f) => `<li>[${esc(f.check)}] ${esc(f.message)}</li>`).join("") +
    (findings.length > 6 ? `<li>… and ${findings.length - 6} more</li>` : "");
}
function verdict(el: HTMLElement, caught: boolean): void {
  el.textContent = caught ? "Caught" : "Missed";
  el.className = `font-headline-md text-headline-md ${caught ? "text-primary-container" : "text-outline"}`;
}

async function tryOnce(): Promise<void> {
  $("try-status").textContent = "Planting and scanning…";
  try {
    const opts = { size: sizeFromSlider(), k: Number($<HTMLInputElement>("k").value),
                   spread: Number($<HTMLSelectElement>("spread").value), target: $<HTMLSelectElement>("target").value };
    const m = await ask({ kind: "try", opts, keep: true });
    const r: TryResult = m.result;
    lastPlanted = m.bytes;
    verdict($("single-verdict"), r.single.caught); list($("single-list"), r.single.findings);
    verdict($("base-verdict"), r.base.caught); list($("base-list"), r.base.findings);
    $("result-note").textContent = `${fmt(r.size)} of random bytes in the low ${r.k} bit(s) of every ${r.spread === 1 ? "" : r.spread + "th "}weight of ` +
      `${r.written.length} tensor(s); ${fmt(r.capacity)} would fit with these settings.`;
    $("result").classList.remove("hidden"); $("download").classList.remove("hidden");
    $("try-status").textContent = "";
  } catch (e) {
    $("try-status").textContent = (e as Error).message;
  }
}

function download(): void {
  if (!lastPlanted) return;
  const url = URL.createObjectURL(new Blob([lastPlanted], { type: "application/octet-stream" }));
  const a = document.createElement("a");
  a.href = url; a.download = modelName.replace(/\.safetensors$/, "") + ".planted.safetensors";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const SIZES = [1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20];
async function sweep(): Promise<void> {
  const ks = [1, 4, 8, 16, 20].filter((k) => k <= maxMant());
  $("sweep-status").textContent = "Running…";
  try {
    const m = await ask({ kind: "sweep", sizes: SIZES, ks, spread: Number($<HTMLSelectElement>("spread").value),
                          target: $<HTMLSelectElement>("target").value },
                        [], (p) => { $("sweep-status").textContent = `${p.done} of ${p.total} copies scanned`; });
    const cell = (c: any, key: "single" | "base") => !c.fits ? `<td class="px-2 py-1 text-center text-outline">—</td>`
      : `<td class="px-2 py-1 text-center ${c[key] ? "text-primary-container" : "text-outline"}">${c[key] ? "caught" : "missed"}</td>`;
    const head = `<tr><th class="px-2 py-1 text-left">k ↓ · size →</th>${SIZES.map((s) => `<th class="px-2 py-1">${fmt(s)}</th>`).join("")}</tr>`;
    const body = ks.map((k) => {
      const cs = SIZES.map((s) => m.cells.find((c: any) => c.k === k && c.size === s));
      return `<tr><th class="px-2 py-1 text-left" rowspan="2">${k}</th>${cs.map((c) => cell(c, "single")).join("")}<td class="pl-3 text-outline">file alone</td></tr>` +
             `<tr>${cs.map((c) => cell(c, "base")).join("")}<td class="pl-3 text-outline">with base</td></tr>`;
    }).join("");
    $("sweep-grid").innerHTML = `<table class="font-code-sm text-code-sm border-collapse">${head}${body}</table>`;
    $("sweep-status").textContent = "";
  } catch (e) {
    $("sweep-status").textContent = (e as Error).message;
  }
}

function loadSample(): void {
  $("model-status").textContent = "Loading the sample (17 MB)…";
  const go = () => {
    const s = window.STOWAWAY_SAMPLE!;
    const bin = atob(s.b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    void load(bytes, "bert_uncased_L-2_H-128_A-2.safetensors");
  };
  if (window.STOWAWAY_SAMPLE) return go();
  const tag = document.createElement("script");
  tag.src = "samples/bert-L2-H128.js";
  tag.onload = go;
  tag.onerror = () => { $("model-status").textContent = "Could not load the sample script."; };
  document.head.appendChild(tag);
}

$("load-sample").addEventListener("click", loadSample);
$<HTMLInputElement>("own-file").addEventListener("change", async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) await load(new Uint8Array(await f.arrayBuffer()), f.name);
});
for (const id of ["size", "k"]) $(id).addEventListener("input", refreshLabels);
$("try").addEventListener("click", () => void tryOnce());
$("download").addEventListener("click", download);
$("sweep").addEventListener("click", () => void sweep());
refreshLabels();
