// Ported from the Stitch design's inline tailwind config so the same class
// names work, compiled locally (no CDN) to stay inside the pages' CSP.
const MONO = ["JetBrains Mono", "ui-monospace", "monospace"];
const SANS = ["Hanken Grotesk", "ui-sans-serif", "system-ui", "sans-serif"];

export default {
  darkMode: "class",
  content: ["./index.html", "./scan.html", "./src/**/*.{ts,js}"],
  // main.ts builds verdict/check colour classes from a variable (bg-${color}/15),
  // which the content scan cannot see.
  safelist: [{ pattern: /^(bg|text|border)-(ok|warn|error|outline)(\/(10|15|20|30))?$/ }],
  theme: {
    extend: {
      colors: {
        surface: "#0f131c", background: "#0f131c", "surface-dim": "#0f131c",
        "surface-container-lowest": "#0a0e17", "surface-container-low": "#181b25",
        "surface-container": "#1c2029", "surface-container-high": "#262a34",
        "surface-container-highest": "#31353f", "surface-variant": "#31353f", "surface-bright": "#353943",
        "on-surface": "#dfe2ef", "on-background": "#dfe2ef", "on-surface-variant": "#bdc8d1",
        outline: "#87929a", "outline-variant": "#3e484f",
        primary: "#8ed5ff", "primary-container": "#38bdf8", "on-primary": "#00354a",
        secondary: "#bfc6dc", tertiary: "#ffbcb4",
        error: "#ffb4ab", "error-container": "#93000a", "on-error-container": "#ffdad6",
        // Verdict colours beyond the Stitch palette (clean / suspicious).
        ok: "#4ade80", warn: "#fbbf24",
        // Hairlines and panels used directly by the Stitch export.
        hair: "#23293b", panel: "#151924", "panel-line": "#272d3f", "panel-hover": "#1b202e",
      },
      borderRadius: { DEFAULT: "0.125rem", lg: "0.25rem", xl: "0.5rem", full: "0.75rem" },
      spacing: {
        "gutter-lg": "1.5rem", "space-md": "0.75rem", "margin-md": "1.5rem", margin: "1rem",
        "space-sm": "0.5rem", "margin-lg": "2rem", gutter: "1rem", "space-lg": "1.25rem",
        "space-xs": "0.25rem", "space-xl": "2rem",
      },
      fontFamily: {
        "code-md": MONO, "code-sm": MONO, "label-mono": MONO, mono: MONO,
        "headline-xl": SANS, "headline-lg": SANS, "headline-md": SANS, "headline-sm": SANS,
        "body-lg": SANS, "body-md": SANS, "body-sm": SANS,
      },
      fontSize: {
        "code-md": ["0.8125rem", { lineHeight: "1.35rem", fontWeight: "400" }],
        "code-sm": ["0.75rem", { lineHeight: "1.2rem", fontWeight: "400" }],
        "label-mono": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.04em", fontWeight: "500" }],
        "headline-xl": ["2.25rem", { lineHeight: "2.75rem", letterSpacing: "-0.02em", fontWeight: "600" }],
        "headline-lg": ["1.75rem", { lineHeight: "2.25rem", letterSpacing: "-0.015em", fontWeight: "600" }],
        "headline-md": ["1.25rem", { lineHeight: "1.75rem", letterSpacing: "-0.01em", fontWeight: "600" }],
        "headline-sm": ["1rem", { lineHeight: "1.5rem", letterSpacing: "-0.005em", fontWeight: "600" }],
        "body-lg": ["1rem", { lineHeight: "1.6rem", fontWeight: "400" }],
        "body-md": ["0.875rem", { lineHeight: "1.45rem", fontWeight: "400" }],
        "body-sm": ["0.75rem", { lineHeight: "1.25rem", fontWeight: "400" }],
      },
    },
  },
};
