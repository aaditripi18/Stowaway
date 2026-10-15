import { defineConfig } from "vite";

// Static single-page build. No dev server is needed to use the result: the
// output is a folder of files that runs from disk or any static host.
export default defineConfig({
  base: "./",
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
});
