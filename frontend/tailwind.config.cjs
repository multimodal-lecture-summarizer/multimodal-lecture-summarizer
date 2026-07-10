/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "primary": "#7C3AED",
        "on-primary": "#FFFFFF",
        "secondary": "#6366F1",
        "accent": "#EC4899",
        "background": "#FAF5FF",
        "foreground": "#0F172A",
        "muted": "#F7F3FD",
        "border-color": "#EFE7FC",
        "destructive": "#DC2626",
        "ring": "#7C3AED",
      },
      fontFamily: {
        "heading": ["Space Grotesk", "sans-serif"],
        "body": ["DM Sans", "sans-serif"],
      },
      borderRadius: {
        "DEFAULT": "4px",
        "md": "8px",
        "lg": "12px",
      }
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
  ],
}
