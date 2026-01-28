/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Use production API URL on Vercel, localhost for development
    const isProduction = process.env.VERCEL === '1' || process.env.NODE_ENV === 'production';
    const apiUrl = isProduction
      ? 'https://shofo-lens-api.onrender.com'
      : 'http://localhost:8000';

    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
