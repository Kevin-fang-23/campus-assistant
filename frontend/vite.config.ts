import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the FastAPI backend so we avoid CORS in dev.
// In production, serve the built `dist/` behind a reverse proxy at /api too,
// or set VITE_API_BASE to the backend origin.
// BACKEND_PORT lets the one-click launcher avoid an already-occupied port
// (e.g. the sandbox's mock_llm service on 8000) without editing this file.
const BACKEND_PORT = process.env.BACKEND_PORT || "8000";
// 后端 uvicorn 绑定 127.0.0.1 (IPv4)，代理也走 IPv4 避免 localhost 在 Windows 上解析成 ::1 连不上的问题
const BACKEND_TARGET = `http://127.0.0.1:${BACKEND_PORT}`;

export default defineConfig({
  plugins: [react()],
  server: {
    // 监听所有网卡的 IPv4，确保浏览器用 127.0.0.1 / localhost 都能打开
    host: "0.0.0.0",
    port: 5173,
    // 允许任意 Host 访问（否则 ngrok 等公网隧道的域名会被 Vite 的
    // DNS-rebinding 防护拦截，返回 403 Blocked request）
    allowedHosts: true,
    proxy: {
      "/api": {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      "/health": {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
    },
  },
});
