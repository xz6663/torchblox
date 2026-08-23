/**
 * Vercel Web Analytics Configuration
 * 
 * This file provides optional configuration for Vercel Web Analytics.
 * The analytics script is loaded via CDN in app.html using:
 * <script defer src="https://cdn.vercel-insights.com/v1/script.js"></script>
 * 
 * For vanilla JavaScript projects, the recommended approach is to use the CDN
 * script tag directly rather than importing the package as a module.
 * 
 * Optional: You can customize analytics behavior using the global window.va function.
 * Documentation: https://vercel.com/docs/analytics/package
 */

// Optional: Initialize the queue for beforeSend hooks or custom configuration
window.va = window.va || function () {
  (window.vaq = window.vaq || []).push(arguments);
};

// Example: Filter out events from specific paths (uncomment to use)
// window.va('beforeSend', (event) => {
//   if (event.url.includes('/private') || event.url.includes('/admin')) {
//     return null; // Don't send these events
//   }
//   return event;
// });

// Example: Debug mode - events will be logged to console (uncomment to use)
// This is automatically enabled in development environments
// window.va('debug', true);
