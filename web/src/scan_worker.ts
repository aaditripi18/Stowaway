// The scan runs here, off the page's main thread. FileReaderSync (workers only)
// reads any byte range of the dropped File on demand, so the scanner core sees a
// synchronous Source and a multi-gigabyte model is never loaded whole: peak
// memory is about one tensor (plus its base counterpart for S5).
//
// The worker is created from a blob: URL (see main.ts), so it runs under the
// page's Content-Security-Policy, including connect-src 'none'.
import { Reader, Source } from "./bytes.js";
import { openBase, scan } from "./report.js";

export type ScanRequest = { id: number; file: File; base: File | null };
export type ScanResponse =
  | { id: number; kind: "progress"; read: number; size: number }
  | { id: number; kind: "done"; report: Record<string, unknown>; previews: Array<[number, Uint8Array]> }
  | { id: number; kind: "error"; message: string };

// FileReaderSync exists only in workers, so the DOM lib does not declare it.
declare const FileReaderSync: { new (): { readAsArrayBuffer(b: Blob): ArrayBuffer } };

const PROGRESS_EVERY = 64 << 20;

class FileSource implements Source {
  readonly size: number;
  private readonly fr = new FileReaderSync();
  private read_ = 0;
  private lastReport = 0;
  constructor(private readonly file: File, private readonly onProgress: (n: number) => void) {
    this.size = file.size;
  }
  read(start: number, end: number): Uint8Array {
    const s = Math.max(0, start), e = Math.min(this.size, end);
    if (e <= s) return new Uint8Array(0);
    const out = new Uint8Array(this.fr.readAsArrayBuffer(this.file.slice(s, e)));
    this.read_ += out.byteLength;
    if (this.read_ - this.lastReport >= PROGRESS_EVERY) { this.lastReport = this.read_; this.onProgress(this.read_); }
    return out;
  }
}

const post = (m: ScanResponse) => (self as unknown as Worker).postMessage(m);

self.onmessage = async (ev: MessageEvent<ScanRequest>) => {
  const { id, file, base } = ev.data;
  try {
    const progress = (n: number) => post({ id, kind: "progress", read: n, size: file.size });
    const r = new Reader(new FileSource(file, progress));
    const baseModel = base ? openBase(new Reader(new FileSource(base, () => {}))) : undefined;
    const report = await scan(r, baseModel);
    // The page shows the first bytes of every finding that points at a range;
    // read just those, so the page never needs the file's contents.
    const previews: Array<[number, Uint8Array]> = [];
    for (const f of report.findings as Array<{ detail: Record<string, unknown> }>) {
      const { start, bytes } = f.detail;
      if (typeof start === "number" && typeof bytes === "number") {
        previews.push([start, r.slice(start, start + Math.min(bytes, 64))]);
      }
    }
    post({ id, kind: "done", report, previews });
  } catch (e) {
    post({ id, kind: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
