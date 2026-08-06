// Temporary config for the DL 2.0 regression test.
//
// Deliberately does NOT reuse vite.config.js: @vitejs/plugin-react injects a
// fast-refresh preamble that expects a real browser page, and inside jsdom it
// throws "can't detect preamble" before any test runs. esbuild handles JSX here
// on its own, which is all a test run needs.
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [],
  esbuild: { jsx: "automatic" },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["dl1.regression.test.jsx"],
    testTimeout: 30000,
  },
});
