/**
 * Vercel Web Analytics Initialization
 * 
 * This file initializes Vercel Web Analytics for the TorchBlox application.
 * The analytics script is loaded from the Vercel CDN and automatically tracks page views.
 */

// Import and inject Vercel Analytics
// Using the inject approach for vanilla JavaScript applications
import { inject } from 'https://cdn.jsdelivr.net/npm/@vercel/analytics@2/+esm';

// Initialize analytics with auto mode detection
inject({
  mode: 'auto', // Automatically detects development vs production
  debug: false   // Set to true for debug logging in development
});
