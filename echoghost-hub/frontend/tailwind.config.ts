import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "var(--color-bg)",
        surface: "var(--color-surface)",
        "surface-2": "var(--color-surface-2)",
        border: "var(--color-border)",
        primary: "var(--color-primary)",
        "primary-dim": "var(--color-primary-dim)",
        accent: "var(--color-accent)",
        "accent-dim": "var(--color-accent-dim)",
        success: "var(--color-success)",
        warning: "var(--color-warning)",
        danger: "var(--color-danger)",
        text: "var(--color-text)",
        "text-dim": "var(--color-text-dim)",
        "text-muted": "var(--color-text-muted)",
      },
    },
  },
  plugins: [],
};

export default config;
