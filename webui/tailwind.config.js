/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        mars: {
          red: '#C1440E',
          orange: '#E85D04',
          rust: '#8B4513',
          sand: '#F4A460',
        },
        hud: {
          black: '#0a0a0a',
          glass: 'rgba(16, 24, 39, 0.7)',
          border: 'rgba(6, 182, 212, 0.3)',
          text: '#22d3ee', // cyan-400
        }
      },
      backgroundImage: {
        'grid-pattern': "linear-gradient(to right, rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.05) 1px, transparent 1px)",
      },
      backgroundSize: {
        'grid-pattern': '20px 20px',
      }
    },
  },
  plugins: [],
}
