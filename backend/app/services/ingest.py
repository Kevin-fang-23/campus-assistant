"""入库编排：文件落地 → 解析 → LangGraph → 持久化通知/待办/向量。"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..graph.pipeline import run_pipeline
from ..models import Document, DocKind, DocStatus, Notice, Task, TaskEvent, TaskStatus, utcnow
from ..parsers import detect_kind, parse_bytes
from ..schemas import DocumentOut, IngestResult, NoticeOut, TaskOut
from .hybrid import index_notice
from .qa_cache import invalidate_qa_cache

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _save_file(data: bytes, filename: str, sha: str) -> Path:
    safe = Path(filename).name.replace(" ", "_")[:120] or "upload.bin"
    target = Path(settings.storage_dir) / f"{sha[:12]}_{safe}"
    if not target.exists():
        target.write_bytes(data)
    return target


def _task_out(task: Task) -> TaskOut:
    return TaskOut.model_validate(task)


def create_tasks(db: Session, notice: Notice, drafts: list[dict[str, Any]]) -> list[Task]:
    tasks: list[Task] = []
    for d in drafts:
        task = Task(
            notice_id=notice.id,
            title=d["title"],
            detail=d.get("detail"),
            category=d.get("category") or notice.category,
            due_at=d.get("due_at"),
            remind_at=d.get("remind_at"),
            priority=int(d.get("priority") or 2),
            status=TaskStatus.TODO,
            source="auto",
            needs_review=notice.needs_review,
        )
        db.add(task)
        db.flush()
        db.add(TaskEvent(task_id=task.id, from_status=None, to_status=TaskStatus.TODO,
                         note="由通知自动生成"))
        tasks.append(task)
    return tasks


def ingest_bytes(
    db: Session,
    data: bytes,
    filename: str,
    mime: str | None = None,
    base_time: datetime | None = None,
) -> IngestResult:
    if not data:
        raise ValueError("上传内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 限制")

    sha = hashlib.sha256(data).hexdigest()
    kind = detect_kind(filename, mime)
    warnings: list[str] = []

    # 内容级幂等：同一份材料重复上传直接复用已有结果
    existing = db.execute(
        select(Document).where(Document.sha256 == sha, Document.status == DocStatus.PARSED)
    ).scalars().first()
    if existing:
        notice = existing.notices[0] if existing.notices else None
        return IngestResult(
            document=DocumentOut.model_validate(existing),
            notice=NoticeOut.model_validate(notice) if notice else None,
            tasks=[_task_out(t) for t in (notice.tasks if notice else [])],
            duplicate=True,
            duplicate_of_id=notice.id if notice else None,
            needs_review=bool(notice and notice.needs_review),
            warnings=["该文件已处理过（sha256 命中），直接返回已有结果"],
            trace=[{"node": "cache", "ms": 0.0, "summary": "命中文件级去重"}],
        )

    path = _save_file(data, filename, sha) if kind != DocKind.TEXT else None
    doc = Document(
        filename=Path(filename).name or "upload",
        kind=kind,
        mime_type=mime,
        sha256=sha,
        size_bytes=len(data),
        storage_path=str(path) if path else None,
        status=DocStatus.UPLOADED,
        parse_meta={},
    )
    db.add(doc)
    db.flush()

    parsed = parse_bytes(data, filename, mime)
    doc.raw_text = parsed.text
    doc.parse_meta = {"kind": parsed.kind, "pages": parsed.pages, **parsed.meta}
    warnings.extend(parsed.warnings)

    if not parsed.text.strip() and not parsed.images:
        doc.status = DocStatus.FAILED
        doc.error = "; ".join(parsed.warnings) or "未解析出任何内容"
        db.commit()
        return IngestResult(
            document=DocumentOut.model_validate(doc), warnings=warnings,
            trace=[{"node": "parse", "ms": 0.0, "summary": doc.error}],
        )

    state = run_pipeline(
        document_id=doc.id,
        raw_text=parsed.text,
        kind=parsed.kind,
        images=parsed.images,
        base_time=base_time,
    )
    payload = state.get("notice") or {}
    notice = Notice(
        document_id=doc.id,
        category=payload.get("category") or "other",
        title=payload.get("title") or "未命名通知",
        summary=payload.get("summary"),
        issuer=payload.get("issuer"),
        location=payload.get("location"),
        course=payload.get("course"),
        event_time=payload.get("event_time"),
        deadline=payload.get("deadline"),
        contacts=payload.get("contacts") or [],
        tags=payload.get("tags") or [],
        extra=payload.get("extra") or {},
        confidence=float(payload.get("confidence") or 0.0),
        needs_review=bool(state.get("needs_review")),
        duplicate_of_id=state.get("duplicate_of_id"),
        dedup_score=state.get("dedup_score"),
    )
    db.add(notice)
    db.flush()

    tasks = create_tasks(db, notice, state.get("task_drafts") or [])
    index_notice(db, notice)

    doc.status = DocStatus.PARSED
    db.commit()
    db.refresh(notice)

    # 新通知已可被检索 → 作废 /api/qa 的旧答案。
    # 为什么必须放在 commit 之后：缓存失效若在事务提交前发生，而提交随后失败，
    # 就会出现「缓存已清空、库里却没这条通知」——查询白跑一次，虽不致错但无意义。
    # 反过来的顺序（先提交后失效）最坏只是多清几个 key，是安全方向。
    invalidate_qa_cache(f"新通知入库 notice_id={notice.id}")

    warnings.extend(state.get("warnings") or [])
    return IngestResult(
        document=DocumentOut.model_validate(doc),
        notice=NoticeOut.model_validate(notice),
        tasks=[_task_out(t) for t in tasks],
        duplicate=bool(state.get("duplicate_of_id")),
        duplicate_of_id=state.get("duplicate_of_id"),
        needs_review=bool(state.get("needs_review")),
        trace=list(state.get("trace") or []),
        warnings=warnings,
    )


def ingest_text(
    db: Session, content: str, filename: str | None = None, base_time: datetime | None = None
) -> IngestResult:
    name = filename or f"paste-{utcnow():%Y%m%d%H%M%S}.txt"
    if not name.lower().endswith((".txt", ".md")):
        name += ".txt"
    return ingest_bytes(db, content.encode("utf-8"), name, mime="text/plain", base_time=base_time)


def regenerate_tasks(db: Session, notice: Notice, base_time: datetime | None = None) -> list[Task]:
    """人工修正通知后重算待办：保留已完成任务，重建未完成的自动任务。"""
    from ..graph.nodes import todo_gen_node

    for task in list(notice.tasks):
        if task.source == "auto" and task.status in (TaskStatus.TODO, TaskStatus.DOING):
            db.delete(task)
    db.flush()

    state = {
        "notice": {
            "category": notice.category, "title": notice.title, "summary": notice.summary,
            "course": notice.course, "location": notice.location,
            "event_time": notice.event_time, "deadline": notice.deadline,
            "contacts": notice.contacts or [], "tags": notice.tags or [],
        },
        "needs_review": notice.needs_review,
        "base_time": base_time or datetime.now(),
        "trace": [],
    }
    drafts = todo_gen_node(state).get("task_drafts") or []  # type: ignore[arg-type]
    tasks = create_tasks(db, notice, drafts)
    index_notice(db, notice)
    db.commit()
    # 通知内容被人工修正 → 旧答案里的标题/时间/地点可能已经不对了
    invalidate_qa_cache(f"通知人工修正 notice_id={notice.id}")
    return tasks
