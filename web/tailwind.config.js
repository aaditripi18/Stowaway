// Ported from the Stitch design's inline tailwind config so the same class
// names work, compiled locally (no CDN) to stay inside the pages' CSP.
const MONO = ["JetBrains Mono", "ui-monospace", "monospace"];
const SANS = ["Hanken Grotesk", "ui-sans-serif", "system-ui", "sans-serif"];

export default {
  darkMode: "class",
  content: ["./index.html", "./scan.html", "./not-a-coin.html", "./limits.html", "./src/**/*.{ts,js}"],
  // main.ts builds verdict/check colour classes from a variable (bg-${color}/15),
  // which the content scan cannot see.
  safelist: [{ pattern: /^(bg|text|border)-(ok|warn|error|outline)(\/(10|15|20|30))?$/ }],
  theme: {
    extend: {
      // every colour is a CSS variable: light in :root, dark under .dark (src/styles.css)
      colors: {
        "surface": "rgb(var(--c-surface) / <alpha-value>)",
        "background": "rgb(var(--c-background) / <alpha-value>)",
        "surface-dim": "rgb(var(--c-surface-dim) / <alpha-value>)",
        "surface-container-lowest": "rgb(var(--c-surface-container-lowest) / <alpha-value>)",
        "surface-container-low": "rgb(var(--c-surface-container-low) / <alpha-value>)",
        "surface-container": "rgb(var(--c-surface-container) / <alpha-value>)",
        "surface-container-high": "rgb(var(--c-surface-container-high) / <alpha-value>)",
        "surface-container-highest": "rgb(var(--c-surface-container-highest) / <alpha-value>)",
        "surface-variant": "rgb(var(--c-surface-variant) / <alpha-value>)",
        "surface-bright": "rgb(var(--c-surface-bright) / <alpha-value>)",
        "on-surface": "rgb(var(--c-on-surface) / <alpha-value>)",
        "on-background": "rgb(var(--c-on-background) / <alpha-value>)",
        "on-surface-variant": "rgb(var(--c-on-surface-variant) / <alpha-value>)",
        "outline": "rgb(var(--c-outline) / <alpha-value>)",
        "outline-variant": "rgb(var(--c-outline-variant) / <alpha-value>)",
        "primary": "rgb(var(--c-primary) / <alpha-value>)",
        "primary-container": "rgb(var(--c-primary-container) / <alpha-value>)",
        "on-primary": "rgb(var(--c-on-primary) / <alpha-value>)",
        "secondary": "rgb(var(--c-secondary) / <alpha-value>)",
        "tertiary": "rgb(var(--c-tertiary) / <alpha-value>)",
        "error": "rgb(var(--c-error) / <alpha-value>)",
        "error-container": "rgb(var(--c-error-container) / <alpha-value>)",
        "on-error-container": "rgb(var(--c-on-error-container) / <alpha-value>)",
        "ok": "rgb(var(--c-ok) / <alpha-value>)",
        "warn": "rgb(var(--c-warn) / <alpha-value>)",
        "hair": "rgb(var(--c-hair) / <alpha-value>)",
        "panel": "rgb(var(--c-panel) / <alpha-value>)",
        "panel-line": "rgb(var(--c-panel-line) / <alpha-value>)",
        "panel-hover": "rgb(var(--c-panel-hover) / <alpha-value>)",
        "sunk": "rgb(var(--c-sunk) / <alpha-value>)",
        "strip": "rgb(var(--c-strip) / <alpha-value>)",
        "panel-2": "rgb(var(--c-panel-2) / <alpha-value>)",
        "tip": "rgb(var(--c-tip) / <alpha-value>)",
        "chip": "rgb(var(--c-chip) / <alpha-value>)",
        "line-mid": "rgb(var(--c-line-mid) / <alpha-value>)",
        "line-strong": "rgb(var(--c-line-strong) / <alpha-value>)",
        "muted": "rgb(var(--c-muted) / <alpha-value>)",
        "faint": "rgb(var(--c-faint) / <alpha-value>)"
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
