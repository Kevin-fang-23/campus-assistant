from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Category, Document, Notice, Task, TaskEvent, TaskStatus, utcnow
from ..schemas import StatsOut, TaskCreateIn, TaskDetailOut, TaskOut, TaskUpdateIn

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("", response_model=list[TaskOut], summary="任务列表（看板数据源）")
def list_tasks(
    status: str | None = Query(None),
    category: str | None = Query(None),
    overdue: bool | None = Query(None, description="只看已逾期"),
    due_before: datetime | None = Query(None),
    limit: int = Query(200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[TaskOut]:
    stmt = select(Task)
    if status:
        if status not in TaskStatus.ALL:
            raise HTTPException(status_code=400, detail="非法状态")
        stmt = stmt.where(Task.status == status)
    if category:
        if category not in Category.ALL:
            raise HTTPException(status_code=400, detail="非法类别")
        stmt = stmt.where(Task.category == category)
    if due_before:
        stmt = stmt.where(Task.due_at.is_not(None), Task.due_at <= due_before)
    if overdue:
        stmt = stmt.where(
            Task.due_at.is_not(None),
            Task.due_at < datetime.now(),
            Task.status.in_((TaskStatus.TODO, TaskStatus.DOING)),
        )
    stmt = stmt.order_by(
        Task.status.asc(), Task.due_at.is_(None).asc(), Task.due_at.asc(), Task.priority.asc()
    ).limit(limit)
    return [TaskOut.model_validate(t) for t in db.execute(stmt).scalars().all()]


@router.get("/upcoming", response_model=list[TaskOut], summary="未来 N 天待办（提醒用）")
def upcoming(days: int = Query(7, ge=1, le=60), db: Session = Depends(get_db)) -> list[TaskOut]:
    now = datetime.now()
    stmt = (
        select(Task)
        .where(
            Task.status.in_((TaskStatus.TODO, TaskStatus.DOING)),
            Task.due_at.is_not(None),
            Task.due_at <= now + timedelta(days=days),
        )
        .order_by(Task.due_at.asc())
    )
    return [TaskOut.model_validate(t) for t in db.execute(stmt).scalars().all()]


@router.get("/stats", response_model=StatsOut, summary="仪表盘统计")
def stats(db: Session = Depends(get_db)) -> StatsOut:
    now = datetime.now()
    counts = dict(
        db.execute(select(Task.status, func.count()).group_by(Task.status)).all()
    )
    by_category = dict(
        db.execute(select(Notice.category, func.count()).group_by(Notice.category)).all()
    )
    overdue = db.execute(
        select(func.count()).select_from(Task).where(
            Task.due_at.is_not(None), Task.due_at < now,
            Task.status.in_((TaskStatus.TODO, TaskStatus.DOING)),
        )
    ).scalar_one()
    return StatsOut(
        documents=db.execute(select(func.count()).select_from(Document)).scalar_one(),
        notices=db.execute(select(func.count()).select_from(Notice)).scalar_one(),
        tasks_total=sum(counts.values()),
        tasks_todo=counts.get(TaskStatus.TODO, 0),
        tasks_doing=counts.get(TaskStatus.DOING, 0),
        tasks_done=counts.get(TaskStatus.DONE, 0),
        tasks_overdue=overdue,
        needs_review=db.execute(
            select(func.count()).select_from(Notice).where(Notice.needs_review.is_(True))
        ).scalar_one(),
        by_category={Category.LABELS.get(k, k): v for k, v in by_category.items()},
    )


@router.get("/{task_id}", response_model=TaskDetailOut, summary="任务详情（含状态流转时间线）")
def get_task(task_id: int, db: Session = Depends(get_db)) -> TaskDetailOut:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskDetailOut.model_validate(task)


@router.post("", response_model=TaskOut, status_code=201, summary="手动新建待办")
def create_task(payload: TaskCreateIn, db: Session = Depends(get_db)) -> TaskOut:
    if payload.category not in Category.ALL:
        raise HTTPException(status_code=400, detail="非法类别")
    if payload.notice_id and not db.get(Notice, payload.notice_id):
        raise HTTPException(status_code=404, detail="关联通知不存在")
    task = Task(**payload.model_dump(), status=TaskStatus.TODO, source="manual")
    db.add(task)
    db.flush()
    db.add(TaskEvent(task_id=task.id, to_status=TaskStatus.TODO, note="手动创建"))
    db.commit()
    db.refresh(task)
    return TaskOut.model_validate(task)


@router.patch("/{task_id}", response_model=TaskDetailOut, summary="更新任务/流转状态")
def update_task(task_id: int, payload: TaskUpdateIn, db: Session = Depends(get_db)) -> TaskDetailOut:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    data = payload.model_dump(exclude_unset=True)
    note = data.pop("note", None)
    new_status = data.pop("status", None)
    for key, value in data.items():
        setattr(task, key, value)

    if new_status and new_status != task.status:
        old = task.status
        task.status = new_status
        task.completed_at = utcnow() if new_status == TaskStatus.DONE else None
        db.flush()
        db.add(TaskEvent(task_id=task.id, from_status=old, to_status=new_status, note=note))
    elif note:
        db.flush()
        db.add(TaskEvent(task_id=task.id, from_status=task.status, to_status=task.status, note=note))

    db.commit()
    db.refresh(task)
    return TaskDetailOut.model_validate(task)


@router.delete("/{task_id}", status_code=204, summary="删除任务")
def delete_task(task_id: int, db: Session = Depends(get_db)) -> None:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    db.delete(task)
    db.commit()
