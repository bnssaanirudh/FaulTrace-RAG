/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f4ff',
          100: '#dce7ff',
          200: '#bfd2ff',
          300: '#93b4fd',
          400: '#5f8efa',
          500: '#3b6ef8',
          600: '#2450ed',
          700: '#1c3cd9',
          800: '#1e33b0',
          900: '#1e2f8b',
          950: '#161f5c',
        },
        surface: {
          0: '#0a0e1a',
          1: '#0f1526',
          2: '#151c32',
          3: '#1c2540',
          4: '#232e4f',
          5: '#2c3a61',
        },
        accent: {
          gold: '#f5c842',
          emerald: '#10d98a',
          coral: '#ff5f5f',
          violet: '#a855f7',
          cyan: '#22d3ee',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'mesh-dark': 'radial-gradient(at 40% 20%, hsla(228,81%,20%,1) 0px, transparent 50%), radial-gradient(at 80% 0%, hsla(240,100%,18%,1) 0px, transparent 50%), radial-gradient(at 0% 50%, hsla(222,81%,14%,1) 0px, transparent 50%)',
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeIn: { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        slideUp: { '0%': { transform: 'translateY(8px)', opacity: '0' }, '100%': { transform: 'translateY(0)', opacity: '1' } },
        shimmer: { '0%': { backgroundPosition: '-1000px 0' }, '100%': { backgroundPosition: '1000px 0' } },
      },
      boxShadow: {
        'glow-brand': '0 0 24px 0 rgba(59, 110, 248, 0.25)',
        'glow-gold': '0 0 24px 0 rgba(245, 200, 66, 0.20)',
        'glow-emerald': '0 0 24px 0 rgba(16, 217, 138, 0.20)',
        'card': '0 1px 3px 0 rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)',
      },
    },
  },
  plugins: [],
};
