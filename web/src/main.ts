// Controller: read the dropped file, scan it in the page with the real scanner
// core, and render the report into the dashboard. No network, no fabricated data
// — every value shown comes from scan().
import "./styles.css";
import { scan } from "./report.js";
import type { Finding } from "./model.js";

type Report = {
  file: string; sha256: string; size: number; format: string;
  verdict: "clean" | "suspicious" | "dangerous";
  tensors: { count: number; bytes: number; dtypes: Record<string, number> };
  findings: Finding[];
};

// What each check does, in plain words — matching the Python package exactly.
const CHECKS: Record<string, { title: string; what: string }> = {
  S1: { title: "Pickle imports", what: "Unsafe code in pickle data, found without ever running it." },
  S2: { title: "Byte coverage", what: "Bytes that belong to no header, tensor or zero padding." },
  S3: { title: "Precision history", what: "Low bits that should be zero for the stored precision." },
  S4: { title: "Bit-plane statistics", what: "Low bits pulled toward a coin flip, versus peer layers." },
  S5: { title: "Base-model diff", what: "Changes confined to the low bits, versus the claimed base." },
};
// Checks this browser build runs on safetensors. S5 needs a base model to diff
// against (not yet collected by this UI); non-safetensors formats are read by the
// Python tool until their parsers are ported. The strip says so rather than
// pretending a check ran.
const BUILD_RUNS = new Set(["S1", "S2", "S3", "S4"]);

const SEV_CHIP: Record<string, string> = {
  critical: "bg-error/20 text-error",
  high: "bg-tertiary-container/20 text-tertiary-container",
  medium: "bg-surface-container text-tertiary-container",
  low: "bg-surface-container text-primary",
  info: "bg-surface-container text-on-surface-variant",
};
const SEV_RANK = ["info", "low", "medium", "high", "critical"];

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));

let lastReport: Report | null = null;
let lastBytes: Uint8Array | null = null;

function show(state: "empty" | "scanning" | "result"): void {
  for (const [id, want] of [["empty-state", "empty"], ["scanning-state", "scanning"], ["result-dashboard", "result"]] as const) {
    const el = $(id);
    el.classList.toggle("hidden", state !== want);
    el.classList.toggle("flex", state === want);
  }
}

function fmtSize(n: number): string {
  for (const [unit, div] of [["GB", 1 << 30], ["MB", 1 << 20], ["KB", 1 << 10]] as const) {
    if (n >= div) return `${(n / div).toFixed(2)} ${unit}`;
  }
  return `${n} B`;
}

async function handle(file: File): Promise<void> {
  $("scan-file-label").textContent = file.name;
  $("scan-percent").textContent = "";
  $<HTMLDivElement>("scan-progress-bar").style.width = "40%";
  show("scanning");
  // Let the spinner paint before the (synchronous) scan.
  await new Promise((r) => setTimeout(r, 30));
  const bytes = new Uint8Array(await file.arrayBuffer());
  $<HTMLDivElement>("scan-progress-bar").style.width = "80%";
  const rep = (await scan(bytes)) as unknown as Report;
  rep.file = file.name;
  lastReport = rep;
  lastBytes = bytes;
  render(rep);
  show("result");
}

function render(rep: Report): void {
  renderVerdict(rep);
  renderTiles(rep);
  renderChecks(rep);
  renderFindings(rep);
}

function renderVerdict(rep: Report): void {
  const banner = $("verdict-banner"), iconBox = $("verdict-icon-container"), icon = $("verdict-icon");
  const tag = $("verdict-tag"), head = $("verdict-headline"), sub = $("verdict-subtext");
  const base = "w-full rounded-xl p-space-lg flex flex-col md:flex-row items-start md:items-center justify-between gap-space-md transition-colors ";
  const flags = rep.findings.filter((f) => ["medium", "high", "critical"].includes(f.severity));
  if (rep.verdict === "clean") {
    banner.className = base + "bg-secondary/15";
    iconBox.className = "w-12 h-12 rounded-xl flex items-center justify-center shrink-0 bg-secondary/20 text-secondary";
    icon.textContent = "verified_user";
    tag.className = "font-label-xs text-label-xs uppercase px-2 py-0.5 rounded font-bold tracking-widest bg-secondary/20 text-secondary";
    tag.textContent = "VERDICT: CLEAN";
    head.textContent = "No payloads or unaccounted bytes found";
    sub.textContent = "The checks this build ran found nothing. A clean result is not a guarantee — see what each check covers.";
  } else if (rep.verdict === "suspicious") {
    banner.className = base + "bg-tertiary-container/15";
    iconBox.className = "w-12 h-12 rounded-xl flex items-center justify-center shrink-0 bg-tertiary-container/20 text-tertiary-container";
    icon.textContent = "warning";
    tag.className = "font-label-xs text-label-xs uppercase px-2 py-0.5 rounded font-bold tracking-widest bg-tertiary-container/20 text-tertiary-container";
    tag.textContent = "VERDICT: SUSPICIOUS";
    head.textContent = "Something here is worth a closer look";
    sub.textContent = `${flags.length} finding${flags.length === 1 ? "" : "s"} at medium severity or above.`;
  } else {
    banner.className = base + "bg-error-container/20";
    iconBox.className = "w-12 h-12 rounded-xl flex items-center justify-center shrink-0 bg-error-container text-on-error-container";
    icon.textContent = "gpp_bad";
    tag.className = "font-label-xs text-label-xs uppercase px-2 py-0.5 rounded font-bold tracking-widest bg-error/20 text-error";
    tag.textContent = "VERDICT: DANGEROUS";
    head.textContent = "This file carries something dangerous";
    sub.textContent = `${flags.length} finding${flags.length === 1 ? "" : "s"} at medium severity or above, including high-severity items.`;
  }
  $("verdict-timestamp").textContent = `${rep.format} • scanned in your browser`;
}

function renderTiles(rep: Report): void {
  $("tile-file").textContent = rep.file;
  $("tile-file").title = rep.file;
  $("tile-format").textContent = rep.format;
  const dt = Object.entries(rep.tensors.dtypes).map(([k, v]) => `${k}×${v}`).join("  ");
  $("tile-dtypes").textContent = dt || "—";
  $("tile-size").textContent = fmtSize(rep.size);
  $("tile-size-bytes").textContent = `${rep.size.toLocaleString()} B`;
  $("tile-tensors").textContent = `${rep.tensors.count} tensor${rep.tensors.count === 1 ? "" : "s"}`;
  $("tile-tensor-detail").textContent = rep.tensors.bytes ? `${fmtSize(rep.tensors.bytes)} of weights` : "";
  $("tile-sha").textContent = rep.sha256 ? rep.sha256.slice(0, 8) + "…" + rep.sha256.slice(-6) : "—";
}

function renderChecks(rep: Report): void {
  const list = $("check-list");
  list.innerHTML = "";
  const byCheck = new Map<string, Finding[]>();
  for (const f of rep.findings) (byCheck.get(f.check) ?? byCheck.set(f.check, []).get(f.check)!).push(f);
  let ran = 0;
  for (const id of ["S1", "S2", "S3", "S4", "S5"]) {
    const fs = (byCheck.get(id) ?? []).filter((f) => ["medium", "high", "critical"].includes(f.severity));
    let icon: string, color: string, label: string;
    if (id === "S5") {
      icon = "horizontal_rule"; color = "text-outline"; label = "NEEDS BASE";
    } else if (!BUILD_RUNS.has(id) || rep.format !== "safetensors") {
      icon = "horizontal_rule"; color = "text-outline"; label = "PYTHON TOOL";
    } else if (id === "S1" && rep.format === "safetensors") {
      icon = "check_circle"; color = "text-secondary"; label = "N/A"; ran++; // safetensors has no pickle data
    } else if (fs.length) {
      icon = "warning"; color = "text-tertiary-container"; label = `${fs.length} FINDING${fs.length === 1 ? "" : "S"}`; ran++;
    } else {
      icon = "check_circle"; color = "text-secondary"; label = "CLEAN"; ran++;
    }
    const row = document.createElement("div");
    row.className = "flex items-center justify-between p-2 rounded bg-surface-container";
    row.innerHTML = `<div class="flex items-center gap-2"><span class="material-symbols-outlined ${color} text-[16px]">${icon}</span><span class="text-on-surface">${id}: ${CHECKS[id]!.title}</span></div><span class="font-label-xs text-label-xs ${color} font-bold">${label}</span>`;
    list.appendChild(row);
  }
  $("checks-ran").textContent = `${ran} / 5 RAN HERE`;
  const legend = $("check-legend");
  legend.innerHTML = "";
  for (const id of ["S1", "S2", "S3", "S4", "S5"]) {
    const row = document.createElement("div");
    row.className = "flex gap-2";
    row.innerHTML = `<span class="text-primary font-bold shrink-0">${id}</span><span>${esc(CHECKS[id]!.what)}</span>`;
    legend.appendChild(row);
  }
}

function hexPreview(bytes: Uint8Array, start: number, length: number): string {
  const n = Math.min(length, 64);
  const slice = bytes.subarray(start, start + n);
  const lines: string[] = [];
  for (let i = 0; i < slice.length; i += 16) {
    const row = slice.subarray(i, i + 16);
    const hex = [...row].map((b) => b.toString(16).padStart(2, "0")).join(" ");
    const ascii = [...row].map((b) => (b >= 0x20 && b < 0x7f ? String.fromCharCode(b) : ".")).join("");
    const off = (start + i).toString(16).padStart(8, "0");
    lines.push(`<div><span class="text-outline">${off}:</span> ${hex.padEnd(47)}  <span class="text-primary-container">${esc(ascii)}</span></div>`);
  }
  return `<div class="bg-surface-container-lowest p-space-sm rounded overflow-x-auto font-code-inline text-[11px] leading-tight text-on-surface-variant mt-space-xs"><div class="text-outline mb-1 text-label-xs font-label-xs font-bold uppercase tracking-wider">First ${n} byte${n === 1 ? "" : "s"} at offset ${start}</div><div class="space-y-0.5 select-all">${lines.join("")}</div></div>`;
}

function detailRows(detail: Record<string, unknown>): string {
  const entries = Object.entries(detail).filter(([k]) => !["start", "end"].includes(k));
  if (!entries.length) return "";
  const cells = entries.map(([k, v]) =>
    `<div><span class="block text-label-xs font-label-xs text-outline uppercase">${esc(k)}</span><span class="text-on-surface font-semibold break-all">${esc(String(v))}</span></div>`).join("");
  return `<div class="grid grid-cols-2 sm:grid-cols-4 gap-space-xs bg-surface-container p-space-sm rounded font-code-inline text-code-inline text-on-surface-variant">${cells}</div>`;
}

function renderFindings(rep: Report): void {
  const list = $("findings-list");
  list.innerHTML = "";
  const findings = [...rep.findings].sort((a, b) => SEV_RANK.indexOf(b.severity) - SEV_RANK.indexOf(a.severity));
  const clean = $("clean-reassurance");
  const actionable = findings.filter((f) => f.severity !== "info");
  if (actionable.length === 0) {
    clean.classList.remove("hidden"); clean.classList.add("flex");
    $("clean-reassurance-text").textContent =
      "Every byte is accounted for and no finding was raised by the checks this build ran. See the panel for what is and isn't covered.";
  } else {
    clean.classList.add("hidden"); clean.classList.remove("flex");
  }

  const counts: Record<string, number> = { all: findings.length };
  for (const s of SEV_RANK) counts[s] = findings.filter((f) => f.severity === s).length;
  renderFilters(counts);

  findings.forEach((f, i) => {
    const card = document.createElement("article");
    card.className = `finding-card ${f.severity} bg-surface-container-low rounded p-space-md transition-all flex flex-col gap-space-sm`;
    const bodyId = `finding-${i}-body`, iconId = `finding-${i}-icon`;
    const d = f.detail as Record<string, unknown>;
    const hex = typeof d.start === "number" && typeof d.bytes === "number" && lastBytes
      ? hexPreview(lastBytes, d.start as number, d.bytes as number) : "";
    card.innerHTML = `
      <div class="flex items-start justify-between cursor-pointer" data-toggle="${bodyId}">
        <div class="flex items-start gap-space-sm">
          <span class="px-2 py-0.5 rounded font-label-xs text-label-xs font-bold uppercase tracking-wider ${SEV_CHIP[f.severity]} shrink-0 mt-0.5">${esc(f.severity)} • ${esc(f.check)}</span>
          <div class="flex flex-col">
            <div class="flex items-center gap-2 flex-wrap"><span class="font-code-inline text-code-inline font-bold text-on-surface">[${esc(f.check)}: ${esc(f.code)}]</span></div>
            <span class="font-body-md text-body-md text-on-surface-variant mt-0.5">${esc(f.message)}</span>
          </div>
        </div>
        <button aria-label="Expand" class="text-on-surface-variant hover:text-on-surface p-1" type="button">
          <span class="material-symbols-outlined text-[20px] transition-transform duration-200" id="${iconId}">expand_more</span>
        </button>
      </div>
      <div class="flex-col gap-space-sm pt-space-xs hidden" id="${bodyId}">${detailRows(d)}${hex}</div>`;
    list.appendChild(card);
  });

  list.querySelectorAll<HTMLElement>("[data-toggle]").forEach((el) => {
    el.addEventListener("click", () => {
      const body = $(el.dataset.toggle!);
      const icon = body.previousElementSibling ? null : null;
      const open = body.classList.toggle("hidden");
      body.classList.toggle("flex", !open);
      const ic = el.querySelector(".material-symbols-outlined") as HTMLElement | null;
      if (ic) ic.style.transform = open ? "rotate(-90deg)" : "rotate(0deg)";
    });
  });
}

function renderFilters(counts: Record<string, number>): void {
  const bar = $("filter-buttons");
  bar.innerHTML = "";
  const make = (key: string, label: string) => {
    const b = document.createElement("button");
    const active = key === "all";
    b.className = `filter-btn px-2.5 py-1 rounded font-label-xs text-label-xs ${active ? "font-bold bg-primary-container text-on-primary" : "text-on-surface-variant hover:bg-surface-container"}`;
    b.textContent = `${label} (${counts[key] ?? 0})`;
    b.addEventListener("click", () => applyFilter(key, b));
    return b;
  };
  bar.appendChild(make("all", "All"));
  for (const s of ["critical", "high", "medium", "low", "info"]) if (counts[s]) bar.appendChild(make(s, s[0]!.toUpperCase() + s.slice(1)));
}

function applyFilter(key: string, btn: HTMLElement): void {
  $("filter-buttons").querySelectorAll(".filter-btn").forEach((b) =>
    (b as HTMLElement).className = "filter-btn px-2.5 py-1 rounded font-label-xs text-label-xs text-on-surface-variant hover:bg-surface-container");
  btn.className = "filter-btn px-2.5 py-1 rounded font-label-xs text-label-xs font-bold bg-primary-container text-on-primary";
  $("findings-list").querySelectorAll<HTMLElement>(".finding-card").forEach((card) =>
    card.classList.toggle("hidden", !(key === "all" || card.classList.contains(key))));
}

function search(query: string): void {
  const q = query.toLowerCase();
  $("findings-list").querySelectorAll<HTMLElement>(".finding-card").forEach((card) =>
    card.classList.toggle("hidden", !card.innerText.toLowerCase().includes(q)));
}

function toast(text: string): void {
  const t = $("toast");
  $("toast-text").textContent = text;
  t.classList.remove("translate-y-16", "opacity-0");
  t.classList.add("translate-y-0", "opacity-100");
  setTimeout(() => { t.classList.add("translate-y-16", "opacity-0"); t.classList.remove("translate-y-0", "opacity-100"); }, 2000);
}

// --- wiring ----------------------------------------------------------------
const input = $<HTMLInputElement>("file-input");
$("drop-zone").addEventListener("click", () => input.click());
$("browse-btn").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
input.addEventListener("change", () => { if (input.files?.[0]) void handle(input.files[0]); });

const drop = $("drop-zone");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("bg-surface-container"); });
drop.addEventListener("dragleave", () => drop.classList.remove("bg-surface-container"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("bg-surface-container");
  const f = e.dataTransfer?.files?.[0]; if (f) void handle(f);
});

$("rescan-btn").addEventListener("click", () => { input.value = ""; show("empty"); });
$("findings-search").addEventListener("keyup", (e) => search((e.target as HTMLInputElement).value));
$("copy-hash").addEventListener("click", async () => {
  if (lastReport?.sha256) { await navigator.clipboard.writeText(lastReport.sha256); toast("SHA-256 copied"); }
});
$("export-btn").addEventListener("click", () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `${lastReport.file}.stowaway.json`; a.click();
  URL.revokeObjectURL(url);
});

show("empty");
