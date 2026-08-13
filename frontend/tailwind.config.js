/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        hud: {
          bg: '#140e06',
          panel: '#1c1409',
          border: '#4a3813',
          amber: '#efb027',
          'amber-hover': '#ffc107',
          'amber-dim': '#2e2308',
          'amber-dark': '#241a0a',
          text: '#e8ddc7',
          muted: '#a3893f',
          subtle: '#6b5a2e',
        }
      },
      fontFamily: {
        mono: ['Fira Code', 'JetBrains Mono', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}
