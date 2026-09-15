"""LangGraph 流程编排：classify → extract → validate → dedup →(条件边) todo_gen / skip_todo。

未安装 langgraph 时自动退化为等价的顺序执行，保证部署环境缺包也能跑。
"""
from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import Any

from .nodes import (
    classify_node,
    dedup_node,
    extract_node,
    route_after_dedup,
    skip_todo_node,
    todo_gen_node,
    validate_node,
)
from .state import PipelineState

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from langgraph.graph import END, StateGraph

    HAS_LANGGRAPH = True
except Exception:  # noqa: BLE001  pragma: no cover
    HAS_LANGGRAPH = False

NODES = {
    "classify": classify_node,
    "extract": extract_node,
    "validate": validate_node,
    "dedup": dedup_node,
    "todo_gen": todo_gen_node,
    "skip_todo": skip_todo_node,
}


@lru_cache
def get_graph():  # pragma: no cover - 依赖 langgraph
    graph = StateGraph(PipelineState)
    for name, fn in NODES.items():
        graph.add_node(name, fn)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "dedup")
    graph.add_conditional_edges(
        "dedup", route_after_dedup, {"todo_gen": "todo_gen", "skip_todo": "skip_todo"}
    )
    graph.add_edge("todo_gen", END)
    graph.add_edge("skip_todo", END)
    return graph.compile()


def _run_sequential(state: PipelineState) -> PipelineState:
    cur: dict[str, Any] = dict(state)
    for name in ("classify", "extract", "validate", "dedup"):
        cur.update(NODES[name](cur))  # type: ignore[arg-type]
    cur.update(NODES[route_after_dedup(cur)](cur))  # type: ignore[arg-type]
    return cur  # type: ignore[return-value]


def run_pipeline(
    *,
    document_id: int,
    raw_text: str,
    kind: str,
    images: list[tuple[str, bytes]] | None = None,
    base_time: datetime | None = None,
) -> PipelineState:
    state: PipelineState = {
        "document_id": document_id,
        "raw_text": raw_text or "",
        "kind": kind,
        "images": images or [],
        "base_time": base_time or datetime.now(),
        "trace": [],
        "warnings": [],
    }
    if HAS_LANGGRAPH:
        try:
            return get_graph().invoke(state)  # type: ignore[return-value]
        except Exception as exc:
            logger.exception("LangGraph 执行失败，退化为顺序执行: %s", exc)
    return _run_sequential(state)


def engine_name() -> str:
    return "langgraph" if HAS_LANGGRAPH else "sequential"
