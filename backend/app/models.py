from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---- 取值常量（用 String 存储，避免 PG enum 迁移负担）----
class DocKind:
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"


class DocStatus:
    UPLOADED = "uploaded"
    PARSED = "parsed"
    FAILED = "failed"


class Category:
    COURSE_NOTICE = "course_notice"   # 课程/教务通知
    ACTIVITY = "activity_poster"      # 活动海报
    HOMEWORK = "homework"             # 作业要求
    REPAIR = "repair"                 # 报修材料
    OTHER = "other"

    ALL = (COURSE_NOTICE, ACTIVITY, HOMEWORK, REPAIR, OTHER)
    LABELS = {
        COURSE_NOTICE: "课程通知",
        ACTIVITY: "活动海报",
        HOMEWORK: "作业要求",
        REPAIR: "报修材料",
        OTHER: "其他",
    }


class TaskStatus:
    TODO = "todo"
    DOING = "doing"
    DONE = "done"
    ARCHIVED = "archived"

    ALL = (TODO, DOING, DONE, ARCHIVED)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(16), index=True)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    storage_path: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default=DocStatus.UPLOADED, index=True)
    raw_text: Mapped[str | None] = mapped_column(Text)
    parse_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    notices: Mapped[list["Notice"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Notice(Base):
    """一份文档抽取出的结构化通知。"""

    __tablename__ = "notices"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(Text)
    issuer: Mapped[str | None] = mapped_column(String(128))
    location: Mapped[str | None] = mapped_column(String(255))
    course: Mapped[str | None] = mapped_column(String(128))
    event_time: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    deadline: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    contacts: Mapped[list] = mapped_column(JSON, default=list)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    duplicate_of_id: Mapped[int | None] = mapped_column(
        ForeignKey("notices.id", ondelete="SET NULL"), nullable=True
    )
    dedup_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    document: Mapped[Document] = relationship(back_populates="notices")
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="notice", cascade="all, delete-orphan"
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    notice_id: Mapped[int | None] = mapped_column(
        ForeignKey("notices.id", ondelete="CASCADE"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(255))
    detail: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32), index=True)
    due_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    remind_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    priority: Mapped[int] = mapped_column(Integer, default=2)  # 1 高 / 2 中 / 3 低
    status: Mapped[str] = mapped_column(String(16), default=TaskStatus.TODO, index=True)
    source: Mapped[str] = mapped_column(String(16), default="auto")  # auto | manual
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    notice: Mapped[Notice | None] = relationship(back_populates="tasks")
    events: Mapped[list["TaskEvent"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="TaskEvent.id"
    )


Index("ix_tasks_status_due", Task.status, Task.due_at)


class TaskEvent(Base):
    """任务状态流转审计，用于任务追踪时间线。"""

    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str] = mapped_column(String(16))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    task: Mapped[Task] = relationship(back_populates="events")


class NoticeEmbedding(Base):
    """向量落库，进程启动时重建 FAISS 索引，避免索引文件与 DB 不一致。"""

    __tablename__ = "notice_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    notice_id: Mapped[int] = mapped_column(
        ForeignKey("notices.id", ondelete="CASCADE"), unique=True, index=True
    )
    dim: Mapped[int] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
