import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward all /api and /static calls to FastAPI on :8000
      "/api":    { target: "http://localhost:8000", changeOrigin: true },
      "/static": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "../static/react",
    emptyOutDir: true,
  },
});
