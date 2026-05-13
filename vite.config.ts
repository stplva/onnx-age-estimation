import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/onnx-age-estimation/",
  plugins: [react()],
});
