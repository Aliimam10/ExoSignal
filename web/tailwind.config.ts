import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07111f",
        panel: "#0c192b",
        line: "#22334c",
        mist: "#b6c7dd",
        cyan: "#66d6f2",
        violet: "#9c9aff",
      },
      boxShadow: { panel: "0 18px 50px rgba(0, 0, 0, .22)" },
      keyframes: { enter: { "0%": { opacity: "0", transform: "translateY(8px)" }, "100%": { opacity: "1", transform: "translateY(0)" } } },
      animation: { enter: "enter .5s ease-out both" },
    },
  },
  plugins: [],
};

export default config;
