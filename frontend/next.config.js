/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React strict mode for development
  reactStrictMode: true,

  // Output standalone build for Docker
  output: 'standalone',

  // Environment variables
  env: {
    // Default API URL (can be overridden at runtime)
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },

  // Experimental features
  experimental: {
    // Enable server actions for form handling
    serverActions: {
      allowedOrigins: ['localhost:3000'],
    },
  },
};

module.exports = nextConfig;
