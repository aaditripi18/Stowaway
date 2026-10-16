// Landing page: the only behaviour is the copy button on the reproduce block.
// The page's CSP forbids inline handlers, so it is wired here.
import "./styles.css";
import { copyText } from "./clipboard.js";

const btn = document.getElementById("copy-btn");
btn?.addEventListener("click", async () => {
  const block = document.getElementById(btn.dataset.copyTarget ?? "");
  if (!block) return;
  // Copy just the commands: drop the comment lines and the "$ " prompts.
  const cmds = (block.textContent ?? "").split("\n")
    .filter((l) => l.startsWith("$ ")).map((l) => l.slice(2)).join("\n");
  const label = btn.querySelector("[data-copy-label]");
  const ok = await copyText(cmds);
  if (label) label.textContent = ok ? "copied" : "copy failed";
  setTimeout(() => { if (label) label.textContent = "copy"; }, 1600);
});
