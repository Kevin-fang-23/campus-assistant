from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """LangGraph 全局状态。节点只读写自己负责的键，便于单独测试。"""

    # 输入
    document_id: int
    raw_text: str
    kind: str
    images: list[tuple[str, bytes]]
    base_time: datetime

    # classify
    category: str
    category_confidence: float
    classify_meta: dict[str, Any]

    # extract
    extracted: dict[str, Any]

    # validate
    notice: dict[str, Any]
    needs_review: bool
    review_reasons: list[str]

    # dedup
    duplicate_of_id: int | None
    dedup_score: float

    # todo_gen
    task_drafts: list[dict[str, Any]]

    # 通用
    trace: list[dict[str, Any]]
    warnings: list[str]
