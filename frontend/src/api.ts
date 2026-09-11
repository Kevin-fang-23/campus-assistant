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

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
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
}

// ---------- documents ----------
export function ingestFile(file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${BASE}/documents/upload`, { method: "POST", body: form }).then(async (res) => {
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
  return req<IngestResult>(`${BASE}/documents/text`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
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
    title: string;
    summary: string;
    issuer: string;
    location: string;
    course: string;
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
  return req<QAOut>(`${BASE}/qa`, {
    method: "POST",
    body: JSON.stringify({ query, top_k }),
  });
}

// ---------- meta ----------
export function stats(): Promise<StatsOut> {
  return req<StatsOut>(`${BASE}/tasks/stats`);
}

export function health(): Promise<Health> {
  return req<Health>(HEALTH);
}
