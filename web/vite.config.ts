import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = process.env.AUDREY_DEV_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  build: {
    emptyOutDir: true,
    manifest: true,
    outDir: "../src/audrey/static/app",
    rolldownOptions: {
      output: {
        codeSplitting: {
          includeDependenciesRecursively: false,
          groups: [
            {
              name: "assistant-ui",
              test: /node_modules[\\/]@assistant-ui[\\/]/,
            },
            {
              name: "ag-ui",
              test: /node_modules[\\/]@ag-ui[\\/]/,
            },
          ],
        },
      },
    },
    sourcemap: false,
  },
  server: {
    proxy: {
      "/api": backend,
      "/v1": backend,
    },
  },
});
