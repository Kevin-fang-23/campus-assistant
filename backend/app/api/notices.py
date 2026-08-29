from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Category, Notice
from ..schemas import NoticeOut, NoticeUpdateIn, TaskOut
from ..services.ingest import regenerate_tasks

router = APIRouter(prefix="/api/notices", tags=["notices"])


@router.get("", response_model=list[NoticeOut], summary="通知列表（支持按类别/复核状态过滤）")
def list_notices(
    category: str | None = Query(None),
    needs_review: bool | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[NoticeOut]:
    stmt = select(Notice).order_by(Notice.id.desc())
    if category:
        if category not in Category.ALL:
            raise HTTPException(status_code=400, detail="非法类别")
        stmt = stmt.where(Notice.category == category)
    if needs_review is not None:
        stmt = stmt.where(Notice.needs_review == needs_review)
    rows = db.execute(stmt.limit(limit).offset(offset)).scalars().all()
    return [NoticeOut.model_validate(r) for r in rows]


@router.get("/{notice_id}", response_model=NoticeOut, summary="通知详情")
def get_notice(notice_id: int, db: Session = Depends(get_db)) -> NoticeOut:
    notice = db.get(Notice, notice_id)
    if not notice:
        raise HTTPException(status_code=404, detail="通知不存在")
    return NoticeOut.model_validate(notice)


@router.patch("/{notice_id}", response_model=NoticeOut, summary="人工复核修正（自动重算待办）")
def update_notice(
    notice_id: int,
    payload: NoticeUpdateIn,
    regenerate: bool = Query(True, description="是否按修正后的信息重算待办"),
    db: Session = Depends(get_db),
) -> NoticeOut:
    notice = db.get(Notice, notice_id)
    if not notice:
        raise HTTPException(status_code=404, detail="通知不存在")

    data = payload.model_dump(exclude_unset=True)
    if "category" in data and data["category"] not in Category.ALL:
        raise HTTPException(status_code=400, detail="非法类别")
    for key, value in data.items():
        setattr(notice, key, value)
    if data and "reviewed" not in data:
        notice.reviewed = True
    if notice.reviewed:
        notice.needs_review = False
        notice.confidence = max(notice.confidence, 0.95)
    db.flush()

    if regenerate:
        regenerate_tasks(db, notice)
    else:
        db.commit()
    db.refresh(notice)
    return NoticeOut.model_validate(notice)


@router.post("/{notice_id}/regenerate-tasks", response_model=list[TaskOut], summary="重算待办")
def regen(notice_id: int, db: Session = Depends(get_db)) -> list[TaskOut]:
    notice = db.get(Notice, notice_id)
    if not notice:
        raise HTTPException(status_code=404, detail="通知不存在")
    tasks = regenerate_tasks(db, notice)
    return [TaskOut.model_validate(t) for t in tasks]
