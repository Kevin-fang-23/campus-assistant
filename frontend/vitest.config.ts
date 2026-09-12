import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// 测试配置与构建配置分开：构建产物里不该混入测试相关设置，
// 而测试需要自己的环境与 setup。两者共用同一个 react 插件。
export default defineConfig({
  plugins: [react()],
  test: {
    // happy-dom 比 jsdom 轻，且不引入 jsdom 的 canvas 可选 peer
    //（后者会触发 npm arborist 的 peer 解析缺陷，见 package.json 的 overrides 说明）
    environment: "happy-dom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    // 每个用例后自动还原被 spy/mock 的原函数，避免用例间互相污染
    restoreMocks: true,
  },
});
