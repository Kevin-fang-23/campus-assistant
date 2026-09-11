from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import Category, TaskStatus


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- 抽取结果（LangGraph 内部契约，同时用于人工修正入参）----------
class ExtractedNotice(BaseModel):
    """各类型通知的统一结构化 schema。"""

    category: str = Category.OTHER
    title: str = "未命名通知"
    summary: str | None = None
    issuer: str | None = None
    location: str | None = None
    course: str | None = None
    event_time: datetime | None = None
    deadline: datetime | None = None
    contacts: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        return v if v in Category.ALL else Category.OTHER

    @field_validator("title")
    @classmethod
    def _trim_title(cls, v: str) -> str:
        v = (v or "").strip().replace("\n", " ")
        return (v[:120] or "未命名通知")

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v or 0.0)))


class TaskDraft(BaseModel):
    title: str
    detail: str | None = None
    category: str = Category.OTHER
    due_at: datetime | None = None
    remind_at: datetime | None = None
    priority: int = 2


# ---------- 输出 ----------
class TaskEventOut(ORMModel):
    id: int
    from_status: str | None
    to_status: str
    note: str | None
    created_at: datetime


class TaskOut(ORMModel):
    id: int
    notice_id: int | None
    title: str
    detail: str | None
    category: str
    due_at: datetime | None
    remind_at: datetime | None
    priority: int
    status: str
    source: str
    needs_review: bool
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class TaskDetailOut(TaskOut):
    events: list[TaskEventOut] = []


class NoticeOut(ORMModel):
    id: int
    document_id: int
    category: str
    title: str
    summary: str | None
    issuer: str | None
    location: str | None
    course: str | None
    event_time: datetime | None
    deadline: datetime | None
    contacts: list[str]
    tags: list[str]
    extra: dict[str, Any]
    confidence: float
    needs_review: bool
    reviewed: bool
    duplicate_of_id: int | None
    dedup_score: float | None
    created_at: datetime
    tasks: list[TaskOut] = []


class DocumentOut(ORMModel):
    id: int
    filename: str
    kind: str
    mime_type: str | None
    sha256: str
    size_bytes: int
    status: str
    raw_text: str | None
    parse_meta: dict[str, Any]
    error: str | None
    created_at: datetime


class IngestResult(BaseModel):
    document: DocumentOut
    notice: NoticeOut | None = None
    tasks: list[TaskOut] = []
    duplicate: bool = False
    duplicate_of_id: int | None = None
    needs_review: bool = False
    trace: list[dict[str, Any]] = Field(default_factory=list)  # LangGraph 各节点执行轨迹
    warnings: list[str] = Field(default_factory=list)


# ---------- 入参 ----------
class TextIngestIn(BaseModel):
    content: str = Field(min_length=1)
    filename: str | None = None


class NoticeUpdateIn(BaseModel):
    """人工复核修正。"""

    category: str | None = None
    title: str | None = None
    summary: str | None = None
    issuer: str | None = None
    location: str | None = None
    course: str | None = None
    event_time: datetime | None = None
    deadline: datetime | None = None
    contacts: list[str] | None = None
    tags: list[str] | None = None
    reviewed: bool | None = None


class TaskCreateIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    detail: str | None = None
    category: str = Category.OTHER
    due_at: datetime | None = None
    remind_at: datetime | None = None
    priority: int = Field(default=2, ge=1, le=3)
    notice_id: int | None = None


class TaskUpdateIn(BaseModel):
    title: str | None = None
    detail: str | None = None
    due_at: datetime | None = None
    remind_at: datetime | None = None
    priority: int | None = Field(default=None, ge=1, le=3)
    status: Literal["todo", "doing", "done", "archived"] | None = None
    note: str | None = None

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: str | None) -> str | None:
        if v is not None and v not in TaskStatus.ALL:
            raise ValueError("非法状态")
        return v


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    notice_id: int
    # score 是**归一化后的混合检索相关性**（向量 + BM25 加权 RRF），
    # 取值 [0,1] 但语义是"相对相关性"，不是概率。
    score: float
    title: str
    category: str
    summary: str | None = None
    deadline: datetime | None = None
    # ---- 混合检索的可解释性字段（向量路/BM25 路各自的表现）----
    match: str = "vector"                 # both | vector | bm25
    score_vector: float | None = None     # 余弦相似度
    score_bm25: float | None = None       # 原始 BM25 分


class SearchOut(BaseModel):
    query: str
    backend: str
    hits: list[SearchHit]


class QAIn(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class CitationOut(BaseModel):
    """回答所依据的通知片段，供前端引用溯源高亮。"""

    notice_id: int
    title: str
    snippet: str
    score: float


class AnswerOut(BaseModel):
    """RAG 问答结果。degraded=True 表示未走 LLM（无 Key / LLM 失败 / 无召回），
    answer 为抽取式回答；degraded=False 表示 answer 由 LLM 基于 citations 生成。"""

    query: str
    answer: str
    citations: list[CitationOut]
    backend: str
    degraded: bool


class StatsOut(BaseModel):
    documents: int
    notices: int
    tasks_total: int
    tasks_todo: int
    tasks_doing: int
    tasks_done: int
    tasks_overdue: int
    needs_review: int
    by_category: dict[str, int]
