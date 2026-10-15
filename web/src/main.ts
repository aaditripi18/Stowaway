// Thin controller: read the dropped file, scan it in the page, show the report.
// The visual design is layered on top of this (see the Stitch UI); this file
// only wires the file input to the scanner core and never touches the network.
import { scan } from "./report.js";

const drop = document.getElementById("drop")!;
const input = document.getElementById("file") as HTMLInputElement;
const report = document.getElementById("report") as HTMLPreElement;

async function handle(file: File): Promise<void> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const rep = await scan(bytes);
  rep.file = file.name;
  report.hidden = false;
  report.textContent = JSON.stringify(rep, null, 2);
}

drop.addEventListener("click", () => input.click());
input.addEventListener("change", () => {
  if (input.files?.[0]) void handle(input.files[0]);
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const file = e.dataTransfer?.files?.[0];
  if (file) void handle(file);
});
