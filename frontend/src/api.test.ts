import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { askQA, deleteTask, listTasks, search } from "./api";

/** 构造一个最小可用的 Response 替身。 */
function mockResponse(body: unknown, init: { status?: number; statusText?: string; json?: boolean } = {}) {
  const status = init.status ?? 200;
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: init.statusText ?? "OK",
    json: init.json === false
      ? () => Promise.reject(new Error("not json"))
      : () => Promise.resolve(body),
  } as unknown as Response;
}

describe("api 请求层", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("search 以 POST 发送 query 与 top_k，并带上 JSON 头", async () => {
    fetchMock.mockResolvedValueOnce(mockResponse({ query: "调课", backend: "faiss", hits: [] }));

    const out = await search("调课", 3);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/search");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ query: "调课", top_k: 3 });
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(out.backend).toBe("faiss");
  });

  it("askQA 打到 /api/qa（问答与检索是两条独立接口）", async () => {
    fetchMock.mockResolvedValueOnce(
      mockResponse({ query: "q", answer: "a", citations: [], backend: "faiss", degraded: true })
    );

    await askQA("宿舍报修怎么申请");

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/qa");
    expect(JSON.parse(init.body).query).toBe("宿舍报修怎么申请");
  });

  it("非 2xx 且响应体带 detail 时，抛出 detail 原文（后端的中文提示要能直达用户）", async () => {
    fetchMock.mockResolvedValueOnce(
      mockResponse({ detail: "文件超过 20MB 限制" }, { status: 400, statusText: "Bad Request" })
    );

    await expect(search("x")).rejects.toThrow("文件超过 20MB 限制");
  });

  it("detail 为对象时序列化后抛出，而不是显示 [object Object]", async () => {
    fetchMock.mockResolvedValueOnce(
      mockResponse({ detail: { msg: "字段缺失" } }, { status: 422, statusText: "Unprocessable" })
    );

    await expect(search("x")).rejects.toThrow('{"msg":"字段缺失"}');
  });

  it("响应体不是 JSON 时回退为 status + statusText（不能因为解析失败而丢掉错误）", async () => {
    fetchMock.mockResolvedValueOnce(
      mockResponse(null, { status: 500, statusText: "Internal Server Error", json: false })
    );

    await expect(search("x")).rejects.toThrow("500 Internal Server Error");
  });

  it("204 返回 undefined 而不是尝试解析 JSON（DELETE 无响应体）", async () => {
    fetchMock.mockResolvedValueOnce(
      mockResponse(null, { status: 204, statusText: "No Content", json: false })
    );

    await expect(deleteTask(7)).resolves.toBeUndefined();
    expect(fetchMock.mock.calls[0][1].method).toBe("DELETE");
  });

  it("GET 类接口不传 method，且查询参数拼进 URL", async () => {
    fetchMock.mockResolvedValueOnce(mockResponse([]));

    await listTasks({ status: "todo" });

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toContain("/api/tasks");
    expect(path).toContain("status=todo");
    expect(init.method).toBeUndefined();
  });
});
