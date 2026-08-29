// Mirrors the FastAPI backend schemas (app/schemas.py). Keep field names in sync.

export type Category =
  | "course_notice"
  | "activity_poster"
  | "homework"
  | "repair"
  | "other";

export type TaskStatus = "todo" | "doing" | "done" | "archived";

export interface TaskEventOut {
  id: number;
  from_status: string | null;
  to_status: string;
  note: string | null;
  created_at: string;
}

export interface TaskOut {
  id: number;
  notice_id: number | null;
  title: string;
  detail: string | null;
  category: string;
  due_at: string | null;
  remind_at: string | null;
  priority: number;
  status: string;
  source: string;
  needs_review: boolean;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface TaskDetailOut extends TaskOut {
  events: TaskEventOut[];
}

export interface NoticeOut {
  id: number;
  document_id: number;
  category: string;
  title: string;
  summary: string | null;
  issuer: string | null;
  location: string | null;
  course: string | null;
  event_time: string | null;
  deadline: string | null;
  contacts: string[];
  tags: string[];
  extra: Record<string, unknown>;
  confidence: number;
  needs_review: boolean;
  reviewed: boolean;
  duplicate_of_id: number | null;
  dedup_score: number | null;
  created_at: string;
  tasks: TaskOut[];
}

export interface DocumentOut {
  id: number;
  filename: string;
  kind: string;
  mime_type: string | null;
  sha256: string;
  size_bytes: number;
  status: string;
  raw_text: string | null;
  parse_meta: Record<string, unknown>;
  error: string | null;
  created_at: string;
}

export interface IngestResult {
  document: DocumentOut;
  notice: NoticeOut | null;
  tasks: TaskOut[];
  duplicate: boolean;
  duplicate_of_id: number | null;
  dedup_score: number | null;
  needs_review: boolean;
  trace: Record<string, unknown>[];
  warnings: string[];
}

export interface TextIngestIn {
  content: string;
  filename?: string | null;
}

export interface TaskCreateIn {
  title: string;
  detail?: string | null;
  category: string;
  due_at?: string | null;
  remind_at?: string | null;
  priority?: number;
  notice_id?: number | null;
}

export interface TaskUpdateIn {
  title?: string;
  detail?: string | null;
  due_at?: string | null;
  remind_at?: string | null;
  priority?: number;
  status?: TaskStatus;
  note?: string;
}

export interface SearchHit {
  notice_id: number;
  score: number;
  title: string;
  category: string;
  summary: string | null;
  deadline: string | null;
}

export interface SearchOut {
  query: string;
  backend: string;
  hits: SearchHit[];
}

export interface StatsOut {
  documents: number;
  notices: number;
  tasks_total: number;
  tasks_todo: number;
  tasks_doing: number;
  tasks_done: number;
  tasks_overdue: number;
  needs_review: number;
  by_category: Record<string, number>;
}

export interface Health {
  status: string;
  app: string;
  pipeline_engine: string;
  vlm: { provider: string; model: string; mock: boolean };
  ocr: string;
  vector: { backend: string; indexed: number };
  database: string;
}
