import { defineConfig } from "vite";
import { resolve } from "node:path";

// Static two-page build: the project page (index.html) and the scanner
// (scan.html). No server is needed to use the result: the output is a folder
// of files that runs from any static host.
export default defineConfig({
  base: "./",
  build: {
    target: "es2022", outDir: "dist", emptyOutDir: true,
    rollupOptions: { input: { index: resolve(__dirname, "index.html"), scan: resolve(__dirname, "scan.html") } },
  },
});
