import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// En desarrollo, las llamadas a /api van a FastAPI (uvicorn api.main:app --reload, puerto 8000)
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
