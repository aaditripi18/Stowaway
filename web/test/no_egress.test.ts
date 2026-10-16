// The page technically cannot send the file anywhere. Two independent guards:
//   1. both pages declare a Content-Security-Policy with connect-src 'none',
//      which makes the browser block every outbound connection (fetch, XHR,
//      WebSocket, EventSource, sendBeacon).
//   2. the scanner's own source uses no network API at all, so there is nothing
//      for a mistake to send even before the CSP applies.
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";

const WEB = join(dirname(fileURLToPath(import.meta.url)), "..");

function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

function tsFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...tsFiles(p));
    else if (entry.name.endsWith(".ts")) out.push(p);
  }
  return out;
}

describe("the page cannot exfiltrate the file", () => {
  // Both pages ship the same policy; scan.html is the one that reads files.
  for (const page of ["index.html", "scan.html"]) {
    it(`${page} declares connect-src 'none' in the CSP`, () => {
      const html = readFileSync(join(WEB, page), "utf-8");
      expect(html).toMatch(/Content-Security-Policy/);
      expect(html).toMatch(/connect-src 'none'/);
      expect(html).toMatch(/default-src 'none'/);
    });

    it(`${page} loads nothing from an external origin (fonts, styles, scripts are self-hosted)`, () => {
      const html = readFileSync(join(WEB, page), "utf-8");
      // Plain links a person clicks (<a href>) are not loads; everything else is.
      const loads = html.replace(/<a\s[^>]*>/g, "");
      expect(loads).not.toMatch(/https?:\/\//);
    });
  }

  it("uses no network API anywhere in src/", () => {
    const forbidden = /\b(fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|importScripts)\b/;
    const offenders: string[] = [];
    for (const f of tsFiles(join(WEB, "src"))) {
      if (forbidden.test(stripComments(readFileSync(f, "utf-8")))) offenders.push(f);
    }
    expect(offenders).toEqual([]);
  });
});
