import type {
  Health,
  IngestResult,
  NoticeOut,
  QAOut,
  SearchOut,
  StatsOut,
  TaskCreateIn,
  TaskDetailOut,
  TaskOut,
  TaskUpdateIn,
  TextIngestIn,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";
const HEALTH = import.meta.env.VITE_API_BASE ? `${import.meta.env.VITE_API_BASE}/health` : "/health";

// ---- 超时上限 ----
// 取值原则：宽到不会误杀正常的慢请求（OCR / VLM / LLM 本就慢），又短到不会在
// CPU 被抢占（如屏幕共享时本机代理/后端被饿住）时让请求无限挂着、界面永远转圈。
export const DEFAULT_TIMEOUT_MS = 60_000; // 列表 / 检索 / 统计等轻量请求
export const INGEST_TIMEOUT_MS = 180_000; // 上传解析：OCR + VLM + LLM 抽取，链路最长
export const QA_TIMEOUT_MS = 90_000; // 检索增强问答：比检索多一次 LLM 生成

function timeoutMessage(ms: number): string {
  return `请求超时（${Math.round(ms / 1000)}s）：后端或代理繁忙，请重试`;
}

/**
 * 用 AbortController 给一段异步请求加超时。
 *
 * 为什么必须有：原生 fetch 不带超时，一旦请求被卡住（开发模式下 /api 要经 Vite
 * 代理转发，CPU 被抢占时代理会变慢甚至挂起），Promise 永远不 resolve，界面就永远
 * 停在「处理中…」。这里到点主动 abort，并把 AbortError 翻译成可读的中文超时提示，
 * 让上层能落到「超时，请重试」而不是无限等待。
 */
async function withTimeout<T>(timeoutMs: number, run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await run(controller.signal);
  } catch (e) {
    if ((e as Error)?.name === "AbortError") {
      throw new Error(timeoutMessage(timeoutMs));
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function req<T>(path: string, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  return withTimeout(timeoutMs, async (signal) => {
    const res = await fetch(path, {
      ...init,
      // signal 放在 ...init 之后：超时由本层统一管理，覆盖调用方可能传入的 signal
      signal,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  });
}

// ---------- documents ----------
export function ingestFile(file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);
  return withTimeout(INGEST_TIMEOUT_MS, async (signal) => {
    const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form, signal });
    if (!res.ok) {
      let detail = `${res.status}`;
      try {
        const b = await res.json();
        detail = b?.detail ?? detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return (await res.json()) as IngestResult;
  });
}

export function ingestText(payload: TextIngestIn): Promise<IngestResult> {
  return req<IngestResult>(
    `${BASE}/documents/text`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    INGEST_TIMEOUT_MS
  );
}

// ---------- notices ----------
export function listNotices(params: { category?: string; needs_review?: boolean } = {}): Promise<NoticeOut[]> {
  const q = new URLSearchParams();
  if (params.category) q.set("category", params.category);
  if (params.needs_review !== undefined) q.set("needs_review", String(params.needs_review));
  const qs = q.toString();
  return req<NoticeOut[]>(`${BASE}/notices${qs ? `?${qs}` : ""}`);
}

export function updateNotice(
  id: number,
  patch: Partial<{
    category: string;
    // title 是 NOT NULL 列：前端清空标题时不提交该键（undefined），不能传 null
    title: string;
    summary: string | null;
    issuer: string | null;
    location: string | null;
    course: string | null;
    event_time: string | null;
    deadline: string | null;
    contacts: string[];
    tags: string[];
    reviewed: boolean;
  }>,
  regenerate = true
): Promise<NoticeOut> {
  return req<NoticeOut>(`${BASE}/notices/${id}?regenerate=${regenerate}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

// ---------- tasks ----------
export interface TaskListParams {
  status?: string;
  category?: string;
  overdue?: boolean;
  due_before?: string;
}
export function listTasks(params: TaskListParams = {}): Promise<TaskOut[]> {
  const q = new URLSearchParams();
  if (params.status) q.set("status", params.status);
  if (params.category) q.set("category", params.category);
  if (params.overdue !== undefined) q.set("overdue", String(params.overdue));
  if (params.due_before) q.set("due_before", params.due_before);
  const qs = q.toString();
  return req<TaskOut[]>(`${BASE}/tasks${qs ? `?${qs}` : ""}`);
}

export function getTask(id: number): Promise<TaskDetailOut> {
  return req<TaskDetailOut>(`${BASE}/tasks/${id}`);
}

export function createTask(payload: TaskCreateIn): Promise<TaskOut> {
  return req<TaskOut>(`${BASE}/tasks`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateTask(id: number, patch: TaskUpdateIn): Promise<TaskDetailOut> {
  return req<TaskDetailOut>(`${BASE}/tasks/${id}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function deleteTask(id: number): Promise<void> {
  return req<void>(`${BASE}/tasks/${id}`, { method: "DELETE" });
}

// ---------- search ----------
export function search(query: string, top_k = 5): Promise<SearchOut> {
  return req<SearchOut>(`${BASE}/search`, {
    method: "POST",
    body: JSON.stringify({ query, top_k }),
  });
}

// ---------- qa（检索增强问答）----------
/**
 * 检索增强问答：返回生成式答案 + 引用片段。
 *
 * 注意：后端在 LLM 不可用时会自行降级为抽取式回答（HTTP 仍为 200，
 * degraded=true），因此前端**不需要**为 LLM 故障单独兜底，
 * 只需按 degraded 展示不同提示即可。
 */
export function askQA(query: string, top_k = 5): Promise<QAOut> {
  return req<QAOut>(
    `${BASE}/qa`,
    {
      method: "POST",
      body: JSON.stringify({ query, top_k }),
    },
    QA_TIMEOUT_MS
  );
}

// ---------- meta ----------
export function stats(): Promise<StatsOut> {
  return req<StatsOut>(`${BASE}/tasks/stats`);
}

export function health(): Promise<Health> {
  return req<Health>(HEALTH);
}
