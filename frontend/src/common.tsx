import type { Category, TaskStatus } from "./types";

export const CATEGORY_LABELS: Record<string, string> = {
  course_notice: "课程通知",
  activity_poster: "活动海报",
  homework: "作业要求",
  repair: "报修材料",
  other: "其他",
};

export const CATEGORY_COLORS: Record<string, string> = {
  course_notice: "#2563eb",
  activity_poster: "#db2777",
  homework: "#d97706",
  repair: "#0d9488",
  other: "#6b7280",
};

export const CATEGORIES: Category[] = [
  "course_notice",
  "activity_poster",
  "homework",
  "repair",
  "other",
];

export const STATUS_LABELS: Record<TaskStatus, string> = {
  todo: "待办",
  doing: "进行中",
  done: "已完成",
  archived: "已归档",
};

export const STATUS_ORDER: TaskStatus[] = ["todo", "doing", "done", "archived"];

export const STATUS_COLORS: Record<TaskStatus, string> = {
  todo: "#2563eb",
  doing: "#d97706",
  done: "#16a34a",
  archived: "#6b7280",
};

export const PRIORITY_LABELS: Record<number, string> = { 1: "高", 2: "中", 3: "低" };
export const PRIORITY_COLORS: Record<number, string> = {
  1: "#dc2626",
  2: "#d97706",
  3: "#6b7280",
};

export function categoryLabel(c: string): string {
  return CATEGORY_LABELS[c] ?? c;
}
export function statusLabel(s: string): string {
  return STATUS_LABELS[s as TaskStatus] ?? s;
}

/** Format an ISO datetime (backend returns naive local time) for display. */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const time = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return `${date} ${time}`;
}

/** Format a date-only value (datetime-local input needs yyyy-MM-ddTHH:mm). */
export function toLocalInputValue(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
    d.getHours()
  )}:${pad(d.getMinutes())}`;
}

/**
 * 把 datetime-local 的值转成后端要的「naive 本地墙钟」字符串。
 *
 * ⚠️ 不要用 `new Date(v).toISOString()` —— 它有两个叠加的坑：
 *   1. 产出 `...Z`（带时区偏移），后端解析成 tz-aware 时间；而项目内部全程是
 *      naive 本地时间，两者相减会抛 TypeError，表现为「保存并重置待办」稳定 500。
 *   2. 它把本地墙钟换算成 UTC，于是 GMT+8 下用户填的 16:12 会变成 08:12，
 *      即使不崩也会静默偏移一个时区 —— 比崩溃更糟，因为它不报错。
 *
 * 后端约定见 `backend/app/services/datetime_utils.to_naive_local`。
 */
export function toApiDateTime(localValue: string | null | undefined): string | null {
  const v = (localValue ?? "").trim();
  if (!v) return null;
  // datetime-local 给的是 `yyyy-MM-ddTHH:mm`（16 位），补秒即可；
  // 刻意不追加 Z、不做时区换算，保持 naive 语义。
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(v) ? `${v}:00` : v;
}

export function isOverdue(due_at: string | null | undefined, status?: string): boolean {
  if (!due_at) return false;
  if (status && (status === "done" || status === "archived")) return false;
  const d = new Date(due_at);
  return !isNaN(d.getTime()) && d.getTime() < Date.now();
}

export function confidenceColor(c: number): string {
  if (c >= 0.8) return "#16a34a";
  if (c >= 0.55) return "#d97706";
  return "#dc2626";
}
