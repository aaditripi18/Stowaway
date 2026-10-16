// Controller: read the dropped file, scan it in the page with the real scanner
// core, and render the report into the dashboard. No network, no fabricated data
// — every value shown comes from scan().
import "./styles.css";
import { scan, loadBaseFloats } from "./report.js";
import { copyText } from "./clipboard.js";
import type { Finding } from "./model.js";
import type { RawFloats } from "./weights.js";

type Report = {
  file: string; sha256: string; size: number; format: string;
  verdict: "clean" | "suspicious" | "dangerous";
  tensors: { count: number; bytes: number; dtypes: Record<string, number> };
  findings: Finding[];
};

// What each check does, in plain words — matching the Python package exactly.
const CHECKS: Record<string, { title: string; what: string }> = {
  S1: { title: "Pickle code", what: "Unsafe code in pickle data, found without ever running it." },
  S2: { title: "Unaccounted bytes", what: "Bytes that belong to no header, tensor or zero padding." },
  S3: { title: "Precision", what: "Low bits that should be zero for the stored precision." },
  S4: { title: "Bit statistics", what: "Low bits pulled toward a coin flip, versus peer layers." },
  S5: { title: "Base-model diff", what: "Changes confined to the low bits, versus the claimed base." },
};
const PARSED_FORMATS = ["safetensors", "gguf", "pytorch-zip", "torchscript", "pickle"];
const PICKLE_FORMATS = ["pytorch-zip", "torchscript", "pickle"];
const FLAGGED = ["medium", "high", "critical"];

const SEV_CHIP: Record<string, string> = {
  critical: "bg-error/20 text-error border-error/30",
  high: "bg-error/15 text-error border-error/25",
  medium: "bg-warn/15 text-warn border-warn/30",
  low: "bg-primary-container/15 text-primary-container border-primary-container/30",
  info: "bg-surface-container-high text-outline border-outline-variant/50",
};
const SEV_RANK = ["info", "low", "medium", "high", "critical"];

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

let lastReport: Report | null = null;
let lastBytes: Uint8Array | null = null;
let baseFloats: RawFloats | null = null;
let baseName: string | null = null;

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
  $("scan-percent").textContent = fmtSize(file.size);
  $<HTMLDivElement>("scan-progress-bar").style.width = "40%";
  show("scanning");
  // Let the spinner paint before the (synchronous) scan.
  await new Promise((r) => setTimeout(r, 30));
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    $<HTMLDivElement>("scan-progress-bar").style.width = "80%";
    const rep = (await scan(bytes, baseFloats ?? undefined)) as unknown as Report;
    rep.file = file.name;
    lastReport = rep;
    lastBytes = bytes;
    render(rep);
    show("result");
    window.scrollTo({ top: 0 });
  } catch (err) {
    // A file too large for memory, or a parser bug: say so rather than spin forever.
    show("empty");
    toast(`Could not scan ${file.name}: ${err instanceof Error ? err.message : String(err)}`, false);
  } finally {
    input.value = "";
  }
}

async function attachBase(file: File): Promise<void> {
  try {
    baseFloats = loadBaseFloats(new Uint8Array(await file.arrayBuffer()));
  } catch {
    baseFloats = null; baseName = null;
    updateBaseUi();
    toast("Could not read that base model", false);
    return;
  } finally {
    baseInput.value = "";
  }
  baseName = file.name;
  updateBaseUi();
  toast(`Base model attached: ${file.name}`);
  // If a model was already scanned, re-run it with the base so S5 applies now.
  if (lastBytes) {
    const rep = (await scan(lastBytes, baseFloats)) as unknown as Report;
    rep.file = lastReport?.file ?? "";
    lastReport = rep;
    render(rep);
  }
}

function updateBaseUi(): void {
  const attached = baseName !== null;
  const slot = $("base-attached");
  slot.classList.toggle("hidden", !attached);
  slot.classList.toggle("flex", attached);
  $("base-name").textContent = baseName ?? "";
  $("base-name").title = baseName ?? "";
  $("base-slot-text").textContent = attached
    ? "Base model attached — scans now include the S5 bit-for-bit diff."
    : "Add the claimed base model to diff the weights bit for bit (the S5 check).";
  $("result-base-status").textContent = attached
    ? `Comparing against ${baseName}. S5 runs on every scan while this base is attached.`
    : "Add the claimed base model (e.g. the unmodified upstream weights) to run the S5 bit-for-bit diff.";
  $("result-base-btn-text").textContent = attached ? "Change base model" : "Attach base model";
  $("base-attach-text").textContent = attached ? "Change" : "Select base model";
  $("result-base-remove-btn").classList.toggle("hidden", !attached);
}

async function removeBase(e?: Event): Promise<void> {
  e?.stopPropagation();
  baseFloats = null; baseName = null;
  updateBaseUi();
  toast("Base model removed");
  if (lastBytes) { // re-run without the base so the result no longer includes S5
    const rep = (await scan(lastBytes)) as unknown as Report;
    rep.file = lastReport?.file ?? "";
    lastReport = rep;
    render(rep);
  }
}

function render(rep: Report): void {
  renderVerdict(rep);
  renderTiles(rep);
  renderChecks(rep);
  renderFindings(rep);
}

function renderVerdict(rep: Report): void {
  const banner = $("verdict-banner"), icon = $("verdict-icon");
  const tag = $("verdict-tag"), head = $("verdict-headline"), sub = $("verdict-subtext");
  const flags = rep.findings.filter((f) => FLAGGED.includes(f.severity));
  const look = {
    clean: { color: "ok", icon: "verified_user", head: "No payloads or unaccounted bytes found",
      sub: "The checks that ran found nothing. A clean result is not a guarantee — see what each check covers." },
    suspicious: { color: "warn", icon: "warning", head: "Something here is worth a closer look",
      sub: `${plural(flags.length, "finding")} at medium severity or above.` },
    dangerous: { color: "error", icon: "gpp_bad", head: "This file carries something dangerous",
      sub: `${plural(flags.length, "finding")} at medium severity or above, including high-severity items.` },
  }[rep.verdict];
  banner.className = `border-l-4 border-${look.color} bg-${look.color}/10 p-4 rounded-r`;
  icon.className = `material-symbols-outlined text-2xl shrink-0 text-${look.color}`;
  icon.textContent = look.icon;
  tag.className = `font-label-mono text-[10px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider bg-${look.color}/20 text-${look.color}`;
  tag.textContent = `Verdict: ${rep.verdict}`;
  head.textContent = look.head;
  sub.textContent = look.sub;
  $("verdict-timestamp").textContent = `${rep.format} · scanned in your browser`;
}

function renderTiles(rep: Report): void {
  $("tile-file").textContent = rep.file;
  $("tile-file").title = rep.file;
  $("tile-format").textContent = rep.format;
  const dt = Object.entries(rep.tensors.dtypes).map(([k, v]) => `${k}×${v}`).join("  ");
  $("tile-dtypes").textContent = dt || "—";
  $("tile-dtypes").title = dt;
  $("tile-size").textContent = fmtSize(rep.size);
  $("tile-size-bytes").textContent = `${rep.size.toLocaleString()} bytes`;
  $("tile-tensors").textContent = plural(rep.tensors.count, "tensor");
  $("tile-tensor-detail").textContent = rep.tensors.bytes ? `${fmtSize(rep.tensors.bytes)} of weights` : "";
  $("tile-sha").textContent = rep.sha256 ? rep.sha256.slice(0, 8) + "…" + rep.sha256.slice(-6) : "—";
  $("tile-sha").title = rep.sha256;
}

function renderChecks(rep: Report): void {
  const list = $("check-list");
  list.innerHTML = "";
  const byCheck = new Map<string, Finding[]>();
  for (const f of rep.findings) (byCheck.get(f.check) ?? byCheck.set(f.check, []).get(f.check)!).push(f);
  const parsed = PARSED_FORMATS.includes(rep.format);
  // Structural problems (duplicate keys, sizes that disagree, unknown format)
  // come from the parser, not S1–S5; show them so the verdict is explained.
  const structural = (byCheck.get("parse") ?? []).filter((f) => FLAGGED.includes(f.severity));
  if (structural.length) {
    const color = structural.some((f) => f.severity !== "medium") ? "error" : "warn";
    const row = document.createElement("div");
    row.className = "flex items-center justify-between gap-2 p-2 rounded bg-surface-container-lowest border border-outline-variant/40";
    row.innerHTML = `<span class="font-semibold text-on-surface">File structure</span>` +
      `<span class="font-label-mono text-[10px] px-2 py-0.5 rounded font-semibold uppercase whitespace-nowrap bg-${color}/15 text-${color} border border-${color}/30">${plural(structural.length, "finding")}</span>`;
    list.appendChild(row);
  }
  let ran = 0;
  for (const id of ["S1", "S2", "S3", "S4", "S5"]) {
    const fs = (byCheck.get(id) ?? []).filter((f) => FLAGGED.includes(f.severity));
    const dangerous = fs.some((f) => f.severity !== "medium");
    let color: string, label: string;
    if (!parsed) {
      color = "outline"; label = "Not run: format not read";
    } else if (id === "S5" && !baseFloats) {
      color = "outline"; label = "Needs a base model";
    } else if (id === "S1" && !PICKLE_FORMATS.includes(rep.format)) {
      color = "ok"; label = "N/A: no pickle data"; ran++;
    } else if (fs.length) {
      color = dangerous ? "error" : "warn"; label = plural(fs.length, "finding"); ran++;
    } else {
      color = "ok"; label = "Clean"; ran++;
    }
    const row = document.createElement("div");
    row.className = "flex items-center justify-between gap-2 p-2 rounded bg-surface-container-lowest border border-outline-variant/40";
    row.innerHTML = `<div class="flex flex-col min-w-0"><span class="font-semibold text-on-surface">${id} ${esc(CHECKS[id]!.title.toLowerCase())}</span></div>` +
      `<span class="font-label-mono text-[10px] px-2 py-0.5 rounded font-semibold uppercase whitespace-nowrap bg-${color}/15 text-${color} border border-${color}/30">${esc(label)}</span>`;
    list.appendChild(row);
  }
  $("checks-ran").textContent = `${ran} / 5 ran here`;
  const legend = $("check-legend");
  legend.innerHTML = "";
  for (const id of ["S1", "S2", "S3", "S4", "S5"]) {
    const p = document.createElement("p");
    p.innerHTML = `<strong class="text-on-surface-variant font-semibold">${id} (${esc(CHECKS[id]!.title)}):</strong> ${esc(CHECKS[id]!.what)}`;
    legend.appendChild(p);
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
    lines.push(`<span class="text-[#575e74]">${off}: </span><span class="text-primary-container">${hex.padEnd(47)}</span>  <span class="text-on-surface">${esc(ascii)}</span>`);
  }
  return `<div class="rounded border border-outline-variant/70 bg-[#0c1018] p-3 overflow-x-auto">` +
    `<div class="border-b border-outline-variant/40 pb-1.5 mb-2 font-label-mono text-[10px] text-outline uppercase tracking-wider">First ${plural(n, "byte")} at offset ${start} (0x${start.toString(16)})</div>` +
    `<pre class="font-code-sm text-code-sm leading-relaxed text-[#8c93a8]">${lines.join("\n")}</pre></div>`;
}

function detailRows(detail: Record<string, unknown>): string {
  const entries = Object.entries(detail).filter(([k]) => !["start", "end"].includes(k));
  if (!entries.length) return "";
  const cells = entries.map(([k, v]) =>
    `<div class="min-w-0"><span class="text-outline uppercase block">${esc(k)}</span><span class="text-on-surface font-semibold break-all">${esc(typeof v === "object" ? JSON.stringify(v) : String(v))}</span></div>`).join("");
  return `<div class="grid grid-cols-2 sm:grid-cols-3 gap-2 py-2 px-3 bg-surface-container rounded border border-outline-variant/50 font-label-mono text-[11px]">${cells}</div>`;
}

function renderFindings(rep: Report): void {
  const list = $("findings-list");
  list.innerHTML = "";
  ($("findings-search") as HTMLInputElement).value = "";
  const findings = [...rep.findings].sort((a, b) => SEV_RANK.indexOf(b.severity) - SEV_RANK.indexOf(a.severity));
  const flagged = findings.filter((f) => FLAGGED.includes(f.severity)).length;
  const count = $("findings-count");
  count.textContent = `${findings.length} total · ${flagged} flagged`;
  count.className = `font-label-mono text-[11px] px-2 py-0.5 rounded border ${flagged ? (rep.verdict === "dangerous" ? SEV_CHIP.critical : SEV_CHIP.medium) : SEV_CHIP.info}`;

  const clean = $("clean-reassurance");
  const actionable = findings.filter((f) => f.severity !== "info");
  clean.classList.toggle("hidden", actionable.length > 0);
  clean.classList.toggle("flex", actionable.length === 0);
  $("clean-reassurance-text").textContent = findings.length
    ? "Only informational notes were raised. See the panel for what each check covers."
    : "Every byte is accounted for and no check raised a finding. See the panel for what is and isn't covered.";

  const counts: Record<string, number> = { all: findings.length };
  for (const s of SEV_RANK) counts[s] = findings.filter((f) => f.severity === s).length;
  renderFilters(counts);

  findings.forEach((f, i) => {
    const card = document.createElement("article");
    card.className = `finding-card sev-${f.severity} bg-surface-container-low border border-outline-variant/80 rounded overflow-hidden`;
    card.dataset.severity = f.severity;
    const bodyId = `finding-${i}-body`;
    const d = f.detail as Record<string, unknown>;
    const hex = typeof d.start === "number" && typeof d.bytes === "number" && lastBytes
      ? hexPreview(lastBytes, d.start as number, d.bytes as number) : "";
    const body = detailRows(d) + hex;
    card.innerHTML = `
      <div class="p-4 bg-[#161a26] ${body ? "cursor-pointer" : ""}" ${body ? `data-toggle="${bodyId}"` : ""}>
        <div class="flex items-start justify-between gap-2">
          <div class="flex flex-wrap items-center gap-2 min-w-0">
            <span class="text-[11px] font-label-mono uppercase px-2 py-0.5 rounded font-semibold border ${SEV_CHIP[f.severity]}">${esc(f.severity)}</span>
            <span class="font-code-sm text-code-sm text-on-surface font-semibold break-all">[${esc(f.check)}: ${esc(f.code)}]</span>
          </div>
          ${body ? `<span class="material-symbols-outlined text-[20px] text-outline transition-transform duration-200 -rotate-90 shrink-0">expand_more</span>` : ""}
        </div>
        <p class="font-body-md text-body-md text-on-surface mt-2 break-words">${esc(f.message)}</p>
      </div>
      ${body ? `<div class="hidden p-4 bg-surface-container-lowest flex-col gap-3" id="${bodyId}">${body}</div>` : ""}`;
    list.appendChild(card);
  });

  list.querySelectorAll<HTMLElement>("[data-toggle]").forEach((el) => {
    el.addEventListener("click", () => {
      const body = $(el.dataset.toggle!);
      const closed = body.classList.toggle("hidden");
      body.classList.toggle("flex", !closed);
      const ic = el.querySelector(".material-symbols-outlined") as HTMLElement | null;
      ic?.classList.toggle("-rotate-90", closed);
    });
  });
}

let activeFilter = "all";

function renderFilters(counts: Record<string, number>): void {
  const bar = $("filter-buttons");
  bar.innerHTML = "";
  activeFilter = "all";
  const make = (key: string, label: string) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.filter = key;
    b.textContent = `${label} (${counts[key] ?? 0})`;
    b.addEventListener("click", () => { activeFilter = key; applyFilters(); });
    return b;
  };
  bar.appendChild(make("all", "All"));
  for (const s of ["critical", "high", "medium", "low", "info"]) if (counts[s]) bar.appendChild(make(s, s[0]!.toUpperCase() + s.slice(1)));
  applyFilters();
}

// Severity filter and text search combine: a card shows only if it passes both.
function applyFilters(): void {
  $("filter-buttons").querySelectorAll<HTMLElement>("button").forEach((b) => {
    const on = b.dataset.filter === activeFilter;
    b.className = `px-2.5 py-1 rounded font-label-mono text-[11px] uppercase border transition-colors ${on
      ? "bg-primary-container text-surface border-primary-container font-semibold"
      : "text-on-surface-variant border-outline-variant/60 hover:bg-surface-container"}`;
  });
  const q = ($("findings-search") as HTMLInputElement).value.toLowerCase();
  $("findings-list").querySelectorAll<HTMLElement>(".finding-card").forEach((card) => {
    const sevOk = activeFilter === "all" || card.dataset.severity === activeFilter;
    const textOk = !q || (card.textContent ?? "").toLowerCase().includes(q);
    card.classList.toggle("hidden", !(sevOk && textOk));
  });
}

let toastTimer: ReturnType<typeof setTimeout> | undefined;
function toast(text: string, ok = true): void {
  const t = $("toast");
  $("toast-text").textContent = text;
  const ic = t.querySelector(".material-symbols-outlined")!;
  ic.textContent = ok ? "check" : "error";
  ic.className = `material-symbols-outlined text-[18px] ${ok ? "text-ok" : "text-error"}`;
  t.classList.remove("translate-y-16", "opacity-0");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("translate-y-16", "opacity-0"), ok ? 2000 : 5000);
}

// --- wiring ----------------------------------------------------------------
const input = $<HTMLInputElement>("file-input");
const baseInput = $<HTMLInputElement>("base-input");

function dropTarget(el: HTMLElement, onFile: (f: File) => void, openPicker: () => void): void {
  el.addEventListener("click", openPicker);
  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("!border-primary-container", "!bg-panel-hover"); });
  el.addEventListener("dragleave", () => el.classList.remove("!border-primary-container", "!bg-panel-hover"));
  el.addEventListener("drop", (e) => {
    e.preventDefault(); el.classList.remove("!border-primary-container", "!bg-panel-hover");
    const f = e.dataTransfer?.files?.[0]; if (f) onFile(f);
  });
}

const openFile = (e?: Event) => { e?.stopPropagation(); input.click(); };
const openBase = (e?: Event) => { e?.stopPropagation(); baseInput.click(); };
dropTarget($("drop-zone"), (f) => void handle(f), () => input.click());
dropTarget($("base-model-drop"), (f) => void attachBase(f), () => baseInput.click());
$("browse-btn").addEventListener("click", openFile);
$("base-attach-btn").addEventListener("click", openBase);
$("result-base-btn").addEventListener("click", openBase);
$("base-remove-btn").addEventListener("click", (e) => void removeBase(e));
$("result-base-remove-btn").addEventListener("click", (e) => void removeBase(e));
// Inputs sit inside their drop zones: stop their clicks bubbling back to the zone.
input.addEventListener("click", (e) => e.stopPropagation());
baseInput.addEventListener("click", (e) => e.stopPropagation());
input.addEventListener("change", () => { if (input.files?.[0]) void handle(input.files[0]); });
baseInput.addEventListener("change", () => { if (baseInput.files?.[0]) void attachBase(baseInput.files[0]); });

// Don't let a file dropped beside the drop zone navigate the tab away to it.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

$("rescan-btn").addEventListener("click", () => { show("empty"); window.scrollTo({ top: 0 }); });
$("findings-search").addEventListener("input", applyFilters);
$("copy-hash").addEventListener("click", async () => {
  if (!lastReport?.sha256) return;
  if (await copyText(lastReport.sha256)) toast("SHA-256 copied");
  else toast("Clipboard unavailable", false);
});
$("export-btn").addEventListener("click", () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `${lastReport.file}.stowaway.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

show("empty");
