import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Custom palette — bukan default Tailwind dirubah.
        // Signature: highlighter amber untuk citations + studio indigo untuk AI.
        canvas: "#FAFAF7",              // warm off-white (bukan #F4F1EA cream default)
        ink:    "#0F172A",              // slate-900 untuk teks utama
        muted:  "#64748B",              // slate-500 untuk teks sekunder
        line:   "#E2E8F0",              // slate-200 untuk border halus
        ai:     "#4338CA",              // indigo-700 — aksen pesan AI
        user:   "#F1F5F9",              // slate-100 — background pesan user
        marker: "#FDE68A",              // amber-200 — highlighter accent (signature!)
        marker2:"#FBBF24",              // amber-400 — accent stronger
        error:  "#DC2626",              // red-600
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        bubble: "1.25rem",  // pesan chat — sedikit lebih besar dari default
      },
      boxShadow: {
        soft:    "0 1px 2px 0 rgb(15 23 42 / 0.04), 0 1px 3px 0 rgb(15 23 42 / 0.06)",
        bubble:  "0 1px 2px 0 rgb(15 23 42 / 0.03)",
      },
      animation: {
        "pulse-dot": "pulse-dot 1.4s ease-in-out infinite",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 80%, 100%": { opacity: "0.3" },
          "40%":           { opacity: "1"   },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
