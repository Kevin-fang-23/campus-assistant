// 全局测试前置。
//
// 1) 注册 jest-dom 的断言扩展（toBeInTheDocument 等）。
//    该 import 同时带来 TS 类型增强，因此测试文件里无需再单独引入。
//
// 2) 每个用例后卸载组件。
//    必须显式注册：vitest 的 `globals` 关闭时，React Testing Library 检测不到
//    全局的 afterEach，其自动 cleanup 不会生效 —— 组件会在用例间累积，
//    表现为 "Found multiple elements with the role ..." 这类假失败。
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
