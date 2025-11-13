/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        nfl: {
          primary: '#013369',
          secondary: '#D50A0A',
          accent: '#FFB612',
        },
      },
    },
  },
  plugins: [],
}

