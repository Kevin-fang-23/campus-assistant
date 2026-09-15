"""LangGraph 节点实现。每个节点是纯函数（不碰数据库），便于单独测试。"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from ..config import settings
from ..models import Category
from ..providers.vlm import get_vlm
from ..schemas import ExtractedNotice, TaskDraft
from ..services import rule_extract
from ..services.datetime_utils import to_naive_local
from ..services.vector_store import get_store
from .state import PipelineState

logger = logging.getLogger(__name__)


def _trace(state: PipelineState, node: str, t0: float, summary: str, **extra: Any) -> list[dict]:
    entry = {"node": node, "ms": round((time.perf_counter() - t0) * 1000, 1), "summary": summary}
    entry.update(extra)
    return [*state.get("trace", []), entry]


def _warn(state: PipelineState, *msgs: str) -> list[str]:
    return [*state.get("warnings", []), *[m for m in msgs if m]]


# ---------------------------------------------------------------- 1. 分类
def classify_node(state: PipelineState) -> dict[str, Any]:
    t0 = time.perf_counter()
    text = state.get("raw_text", "") or ""
    images = state.get("images") or []
    vlm = get_vlm()

    if not text.strip() and not images:
        return {
            "category": Category.OTHER,
            "category_confidence": 0.0,
            "classify_meta": {"provider": "none"},
            "warnings": _warn(state, "无可用文本内容，无法分类"),
            "trace": _trace(state, "classify", t0, "空输入，标记为 other"),
        }

    category, confidence, meta = vlm.classify(text, images)
    return {
        "category": category,
        "category_confidence": confidence,
        "classify_meta": meta,
        "trace": _trace(
            state, "classify", t0,
            f"{Category.LABELS.get(category, category)} (conf={confidence:.2f}, {meta.get('provider')})",
        ),
    }


# ---------------------------------------------------------------- 2. 抽取
def extract_node(state: PipelineState) -> dict[str, Any]:
    t0 = time.perf_counter()
    text = state.get("raw_text", "") or ""
    category = state.get("category", Category.OTHER)
    # base_time 必须透传到抽取层，否则「明天/本周五」会按 datetime.now() 解析，
    # 同一份样本在不同日期跑出不同结果（评测不可复现）。
    base = state.get("base_time")
    try:
        data = get_vlm().extract(text, category, state.get("images") or [], base=base)
    except Exception as exc:
        logger.exception("抽取节点异常，降级为规则抽取")
        data = rule_extract.extract(text, category, base=base)
        data["extra"]["error"] = str(exc)
    filled = [k for k in ("title", "location", "issuer", "course", "event_time", "deadline")
              if data.get(k)]
    return {
        "extracted": data,
        "trace": _trace(state, "extract", t0, f"命中字段 {len(filled)}/6: {'/'.join(filled) or '无'}"),
    }


# ---------------------------------------------------------------- 3. 校验
def validate_node(state: PipelineState) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw = dict(state.get("extracted") or {})
    base = state.get("base_time") or datetime.now()
    reasons: list[str] = []

    raw.setdefault("category", state.get("category", Category.OTHER))
    try:
        notice = ExtractedNotice(**raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("抽取结果不合 schema，走最小可用兜底: %s", exc)
        notice = ExtractedNotice(
            category=state.get("category", Category.OTHER),
            title=(raw.get("title") or "未命名通知"),
            summary=(state.get("raw_text") or "")[:140],
            confidence=0.2,
        )
        reasons.append("结构化校验失败，已降级")

    # 置信度融合：分类置信度 + 抽取置信度
    cls_conf = float(state.get("category_confidence") or 0.0)
    notice.confidence = round(min(1.0, 0.5 * cls_conf + 0.5 * notice.confidence), 3)

    # 时间合理性
    if notice.deadline and notice.deadline < base - timedelta(days=1):
        reasons.append(f"截止时间 {notice.deadline:%Y-%m-%d %H:%M} 已过期，请确认")
    if notice.event_time and notice.deadline and notice.deadline < notice.event_time:
        if notice.category in (Category.ACTIVITY, Category.COURSE_NOTICE):
            pass  # 报名截止早于活动时间是合理的
        else:
            reasons.append("截止时间早于活动时间，存在冲突")
    if notice.event_time and notice.event_time > base + timedelta(days=365):
        reasons.append("时间超出一年，可能识别错误")
        notice.event_time = None

    # 关键字段缺失
    if notice.category in (Category.HOMEWORK,) and not notice.deadline:
        reasons.append("作业类未识别到截止时间")
    if notice.category in (Category.COURSE_NOTICE, Category.ACTIVITY) and not notice.event_time:
        reasons.append("未识别到具体时间")
    if len(notice.title) < 4:
        reasons.append("标题过短")
    if notice.category == Category.OTHER:
        reasons.append("未能归类到已知场景")

    needs_review = notice.confidence < settings.review_confidence_threshold or bool(reasons)
    payload = notice.model_dump()
    payload["extra"] = {**(payload.get("extra") or {}), "review_reasons": reasons}
    return {
        "notice": payload,
        "needs_review": needs_review,
        "review_reasons": reasons,
        "trace": _trace(
            state, "validate", t0,
            f"conf={notice.confidence:.2f}, needs_review={needs_review}"
            + (f", 原因: {'; '.join(reasons[:2])}" if reasons else ""),
        ),
    }


# ---------------------------------------------------------------- 4. 去重
def dedup_node(state: PipelineState) -> dict[str, Any]:
    t0 = time.perf_counter()
    notice = state.get("notice") or {}
    store = get_store()
    text = "\n".join(
        str(notice.get(k) or "") for k in ("title", "summary", "course", "location")
    ).strip()
    if not text:
        return {
            "duplicate_of_id": None,
            "dedup_score": 0.0,
            "trace": _trace(state, "dedup", t0, "无文本，跳过去重"),
        }
    vector = store.embed(text)
    dup_id, score = store.find_duplicate(vector)
    return {
        "duplicate_of_id": dup_id,
        "dedup_score": round(score, 4),
        "notice": {**notice, "extra": {**(notice.get("extra") or {}),
                                       "dedup_vector_dim": int(vector.shape[0])}},
        "trace": _trace(
            state, "dedup", t0,
            f"最相似 {score:.3f} → {'判定重复 #' + str(dup_id) if dup_id else '非重复'}"
            f" ({store.backend}, 库内 {store.size} 条)",
        ),
    }


# ---------------------------------------------------------------- 5. 待办生成
def _mk(title: str, category: str, due: datetime | None, priority: int,
        detail: str | None = None) -> dict[str, Any]:
    remind = None
    if due:
        remind = due - timedelta(hours=settings.remind_lead_hours)
    return TaskDraft(
        title=title[:200], detail=detail, category=category, due_at=due,
        remind_at=remind, priority=priority,
    ).model_dump()


def _priority_by_due(due: datetime | None, base: datetime) -> int:
    if not due:
        return 3
    # 纵深防御：入口（schemas 的 NaiveLocalDateTime）已做归一化，这里再兜一次。
    # 因为 due 可能直接来自数据库或历史数据，一旦某个是 tz-aware，
    # `due - base` 会抛 TypeError，让整条「保存并重置待办」以 500 收场 ——
    # 而这类错误只在真实浏览器提交（带 Z）时才出现，单测很难覆盖。
    hours = (to_naive_local(due) - to_naive_local(base)).total_seconds() / 3600  # type: ignore[operator]
    if hours <= 48:
        return 1
    if hours <= 168:
        return 2
    return 3


def todo_gen_node(state: PipelineState) -> dict[str, Any]:
    t0 = time.perf_counter()
    n = state.get("notice") or {}
    base = state.get("base_time") or datetime.now()
    category = n.get("category", Category.OTHER)
    title = n.get("title") or "未命名通知"
    deadline = n.get("deadline")
    event = n.get("event_time")
    course = n.get("course")
    location = n.get("location")
    detail_bits = [b for b in (n.get("summary"), f"地点：{location}" if location else None,
                               f"课程：{course}" if course else None,
                               f"联系方式：{'、'.join(n.get('contacts') or [])}"
                               if n.get("contacts") else None) if b]
    detail = "\n".join(detail_bits) or None
    drafts: list[dict[str, Any]] = []

    if category == Category.HOMEWORK:
        due = deadline or event
        drafts.append(_mk(f"提交作业：{course + ' - ' if course else ''}{title}", category, due,
                          _priority_by_due(due, base), detail))
        if due and (due - base) > timedelta(days=3):
            drafts.append(_mk(f"开始动手：{title}", category, due - timedelta(days=3), 3,
                              "预留时间开工，避免 DDL 前赶工"))
    elif category == Category.COURSE_NOTICE:
        due = event or deadline
        drafts.append(_mk(f"课程安排：{title}", category, due, _priority_by_due(due, base), detail))
        tags = " ".join(n.get("tags") or []) + " " + title
        if any(k in tags for k in ("考试", "期末", "测验")) and due:
            drafts.append(_mk(f"复习备考：{course or title}", category, due - timedelta(days=3), 2,
                              "考前三天启动复习"))
    elif category == Category.ACTIVITY:
        if deadline:
            drafts.append(_mk(f"报名截止：{title}", category, deadline,
                              _priority_by_due(deadline, base), detail))
        if event:
            drafts.append(_mk(f"参加活动：{title}", category, event,
                              _priority_by_due(event, base), detail))
        if not drafts:
            drafts.append(_mk(f"关注活动：{title}", category, None, 3, detail))
    elif category == Category.REPAIR:
        submit_due = deadline or event or base + timedelta(hours=12)
        drafts.append(_mk(f"提交/跟进报修：{title}", category, submit_due,
                          _priority_by_due(submit_due, base), detail))
        drafts.append(_mk(f"确认维修结果：{title}", category, submit_due + timedelta(days=2), 3,
                          "维修完成后确认并关闭工单"))
    else:
        due = deadline or event
        drafts.append(_mk(f"待确认：{title}", category, due, _priority_by_due(due, base), detail))

    if state.get("needs_review"):
        for d in drafts:
            d["detail"] = "\n".join(
                filter(None, [d.get("detail"), "⚠ 自动抽取置信度较低，建议人工核对时间与地点"])
            )

    return {
        "task_drafts": drafts,
        "trace": _trace(state, "todo_gen", t0,
                        f"生成 {len(drafts)} 条待办: " + " / ".join(d["title"][:18] for d in drafts)),
    }


def skip_todo_node(state: PipelineState) -> dict[str, Any]:
    t0 = time.perf_counter()
    return {
        "task_drafts": [],
        "trace": _trace(state, "skip_todo", t0,
                        f"命中重复通知 #{state.get('duplicate_of_id')}，不再生成待办"),
        "warnings": _warn(state, f"与已有通知 #{state.get('duplicate_of_id')} 高度相似，已跳过待办生成"),
    }


def route_after_dedup(state: PipelineState) -> str:
    """条件边：重复通知不生成待办。"""
    return "skip_todo" if state.get("duplicate_of_id") else "todo_gen"
