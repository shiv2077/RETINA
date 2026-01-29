/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // KU Leuven brand colors
        'kuleuven-blue': '#1D3557',
        'kuleuven-red': '#E63946',
        // Semantic colors for anomaly detection
        'normal': '#10B981',    // Green for normal samples
        'anomaly': '#EF4444',   // Red for anomalies
        'uncertain': '#F59E0B', // Amber for uncertain samples
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};
