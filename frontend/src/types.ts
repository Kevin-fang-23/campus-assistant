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
  /** 归一化后的混合检索相关性 [0,1]（向量 + BM25 加权 RRF），非概率。 */
  score: number;
  title: string;
  category: string;
  summary: string | null;
  /**
   * 按当前查询选出的原文句级片段。与 summary 的分工：
   * summary 是抽取阶段的概括、与查询无关；snippet 直接回答"为什么这条被召回"。
   * 可选 —— 无原文时后端留空，前端回退展示 summary。
   */
  snippet?: string | null;
  deadline: string | null;
  /** 命中来源：both（两路都召回）/ vector（仅语义）/ bm25（仅关键词）。 */
  match?: "both" | "vector" | "bm25";
  score_vector?: number | null;
  score_bm25?: number | null;
}

export interface SearchOut {
  query: string;
  backend: string;
  hits: SearchHit[];
  /**
   * 是否「**因相关性阈值过滤而导致结果为空**」。
   *
   * 刻意收窄为只表示"被清空"，而不是"有任意一条被丢弃"：
   * 候选池本就故意多召回，几乎总有边缘候选被丢，若那种情况也置 true，
   * 这个标志位就恒为 true、无法用来区分下面两种空态：
   *   true  → 「未找到相关内容」（有候选，但都判为不相关）
   *   false → 「没有匹配的历史通知」（库里确实没有）
   * 阈值关闭（SEARCH_MIN_BIGRAM_OVERLAP=0）时恒为 false。
   */
  filtered?: boolean;
}

/** 问答引用片段：与 AnswerOut.answer 中的 [编号] 一一对应（1-based）。 */
export interface QACitationOut {
  notice_id: number;
  title: string;
  snippet: string;
  score: number;
}

/**
 * POST /api/qa 响应。
 * degraded=true 表示生成式回答不可用（未配置 Key / 服务端强制 mock / LLM 调用失败），
 * 此时 answer 退化为 top1 引用片段，仍可直接展示。
 */
export interface QAOut {
  query: string;
  answer: string;
  citations: QACitationOut[];
  backend: string;
  degraded: boolean;
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
  vector: {
    backend: string;
    indexed: number;
    /** 配置的后端（如 dashscope）。 */
    configured_embedding?: string;
    /** 实际生效的后端：上游不可用时回退为 local_hash，与 configured 不同即表示已降级。 */
    active_embedding?: string;
    embedding_degraded?: boolean;
  };
  database: string;
}
