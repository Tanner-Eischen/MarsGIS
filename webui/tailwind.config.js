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
          brown: '#9D4EDD',
        },
      },
    },
  },
  plugins: [],
}




