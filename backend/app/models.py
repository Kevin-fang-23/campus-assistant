from __future__ import annotations

from datetime import datetime

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


def local_now() -> datetime:
    """审计时间戳统一用「naive 本地墙钟」。

    与业务时间（deadline / event_time / due_at）保持同一时钟约定 ——
    全项目的时间语义是本地墙钟（见 services/datetime_utils.to_naive_local 的
    说明：相对时间解析、截止补 23:59 都以 `datetime.now()` 为基准）。

    历史实现是 `local_now()`（naive UTC），与业务时间的本地时钟**混在同一张表里**：
    `tasks.py` 的逾期/即将到期判断用 `datetime.now()`（本地）比对 `due_at`，
    而 `completed_at` 却写入 UTC —— 非 UTC 时区的服务器上两套时间相差一个时区偏移，
    一旦拿审计时间与业务时间做比较（或前端展示 created_at）就会系统性偏差。
    统一为本地时钟后，全库时间可直接相互比较。
    """
    return datetime.now()


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, index=True)

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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, onupdate=local_now)

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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, onupdate=local_now)

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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now)

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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now)
