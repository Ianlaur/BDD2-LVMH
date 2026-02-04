// API configuration for connecting to the remote server
const API_CONFIG = {
  // Update this URL to point to your server Mac
  // Example: 'http://192.168.1.100:8000' or 'http://your-server.local:8000'
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
};

export default API_CONFIG;
