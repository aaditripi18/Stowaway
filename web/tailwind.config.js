// Ported from the Stitch design's inline tailwind config so the same class
// names work, compiled locally (no CDN) to stay inside the page's CSP.
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,js}"],
  theme: {
    extend: {
      colors: {
        "surface-container-high": "#262a34", "on-primary-fixed": "#001e2c",
        "surface-variant": "#31353f", "primary-fixed-dim": "#7bd0ff", "inverse-primary": "#00668a",
        "on-secondary": "#003824", "on-tertiary-container": "#8c0028",
        "on-secondary-fixed-variant": "#005236", "on-primary-container": "#004965",
        error: "#ffb4ab", "on-primary": "#00354a", "outline-variant": "#3e484f",
        "on-error-container": "#ffdad6", "on-background": "#dfe2ef", "on-error": "#690005",
        secondary: "#4edea3", "tertiary-container": "#ff929a", "surface-container-low": "#181b25",
        "on-tertiary-fixed-variant": "#92002a", outline: "#87929a", "on-tertiary-fixed": "#40000d",
        "tertiary-fixed": "#ffdadb", "on-surface-variant": "#bdc8d1", surface: "#0f131c",
        background: "#0f131c", "surface-dim": "#0f131c", "inverse-surface": "#dfe2ef",
        primary: "#8ed5ff", "primary-container": "#38bdf8", "primary-fixed": "#c4e7ff",
        "secondary-fixed": "#6ffbbe", "surface-container-highest": "#31353f",
        "on-primary-fixed-variant": "#004c69", "on-surface": "#dfe2ef", "secondary-fixed-dim": "#4edea3",
        "surface-container": "#1c1f29", "inverse-on-surface": "#2c303a", tertiary: "#ffbcbf",
        "on-tertiary": "#67001b", "surface-bright": "#353943", "surface-container-lowest": "#0a0e17",
        "surface-tint": "#7bd0ff", "secondary-container": "#00a572", "on-secondary-fixed": "#002113",
        "tertiary-fixed-dim": "#ffb2b7", "error-container": "#93000a", "on-secondary-container": "#00311f",
      },
      borderRadius: { DEFAULT: "0.125rem", lg: "0.25rem", xl: "0.5rem", full: "0.75rem" },
      spacing: {
        "gutter-mobile": "0.75rem", "space-lg": "1.5rem", "space-md": "1rem", "space-xs": "0.25rem",
        "margin-mobile": "1rem", "space-xl": "2.5rem", gutter: "1rem", margin: "2rem", "space-sm": "0.5rem",
      },
      fontFamily: {
        "body-lg": ["JetBrains Mono", "ui-monospace", "monospace"],
        "headline-md": ["Hanken Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
        "body-md": ["JetBrains Mono", "ui-monospace", "monospace"],
        "label-xs": ["JetBrains Mono", "ui-monospace", "monospace"],
        "headline-xl": ["Hanken Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
        "label-md": ["JetBrains Mono", "ui-monospace", "monospace"],
        "code-inline": ["JetBrains Mono", "ui-monospace", "monospace"],
        "title-sm": ["Hanken Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      fontSize: {
        "body-lg": ["14px", { lineHeight: "22px", letterSpacing: "-0.01em", fontWeight: "400" }],
        "headline-md": ["20px", { lineHeight: "28px", letterSpacing: "-0.015em", fontWeight: "600" }],
        "body-md": ["13px", { lineHeight: "20px", letterSpacing: "0em", fontWeight: "400" }],
        "label-xs": ["10px", { lineHeight: "14px", letterSpacing: "0.06em", fontWeight: "600" }],
        "headline-xl": ["32px", { lineHeight: "40px", letterSpacing: "-0.025em", fontWeight: "600" }],
        "label-md": ["11px", { lineHeight: "16px", letterSpacing: "0.04em", fontWeight: "500" }],
        "code-inline": ["12px", { lineHeight: "18px", letterSpacing: "0em", fontWeight: "500" }],
        "title-sm": ["16px", { lineHeight: "24px", letterSpacing: "-0.01em", fontWeight: "600" }],
      },
    },
  },
};
