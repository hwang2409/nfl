/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    API_URL: process.env.API_URL || 'http://localhost:8000',
  },
  images: {
    domains: ['a.espncdn.com', 'static.www.nfl.com'],
    unoptimized: false,
  },
}

module.exports = nextConfig

