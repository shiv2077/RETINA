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
        // KU Leuven brand
        kul: {
          blue:   '#003366',
          light:  '#1A5276',
          accent: '#0066CC',
          gold:   '#B5A642',
        },
        // Surface system — warm dark, not pure black
        surface: {
          base:        '#0C0C0E',
          raised:      '#141416',
          overlay:     '#1C1C1F',
          border:      '#2A2A2E',
          borderhover: '#3A3A3E',
        },
        // Glass system — for data panels only
        glass: {
          bg:          'rgba(255,255,255,0.04)',
          bghover:     'rgba(255,255,255,0.07)',
          border:      'rgba(255,255,255,0.10)',
          borderhover: 'rgba(255,255,255,0.18)',
        },
        // Semantic states
        state: {
          pass:        '#34D399',
          passSubtle:  'rgba(52,211,153,0.12)',
          alert:       '#F87171',
          alertSubtle: 'rgba(248,113,113,0.12)',
          warn:        '#FBBF24',
          warnSubtle:  'rgba(251,191,36,0.12)',
          info:        '#60A5FA',
        },
        // Text hierarchy
        text: {
          primary:   '#F5F5F7',
          secondary: '#A1A1A6',
          tertiary:  '#6E6E73',
          disabled:  '#3A3A3E',
        },
        // Compatibility tokens — keep for api.ts type references
        normal:          '#34D399',
        anomaly:         '#F87171',
        uncertain:       '#FBBF24',
        'kuleuven-blue': '#003366',
        'kuleuven-red':  '#E63946',
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['var(--font-jetbrains)', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
