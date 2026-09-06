import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = process.env.AUDREY_DEV_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  base: "/",
  plugins: [react()],
  build: {
    emptyOutDir: true,
    manifest: true,
    outDir: "../src/audrey/static/app",
    sourcemap: false,
  },
  server: {
    proxy: {
      "/api": backend,
      "/v1": backend,
    },
  },
});
