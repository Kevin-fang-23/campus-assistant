"""通知正文解析：问答与检索共用的文本来源。

## 为什么单独抽一层

`/api/qa` 与 `/api/search` 都需要同一条规则 —— **原文优先，缺原文回退摘要/标题**。
若两处各写一份，引用片段就可能基于不同文本，出现
「问答里能选到答案句、检索页却只显示摘要」的不一致，排查起来很费劲。

本模块只负责「取哪份文本」（批量）；「怎么选片段」是 `snippet.py` 的
纯函数（select_snippet / select_context），由 API 层直接调用。

## 两个上限的区别

- `SNIPPET_MAX_CHARS`：**给人看**的片段（问答引用 / 检索结果卡片），要求短而准；
- `CONTEXT_MAX_CHARS`：**给模型看**的上下文，要求不漏关键信息。

## 为什么只有批量版本

单条版（逐通知 db.get）在 top_k=5 的问答路径会发 ~10 次查询
（Notice + Document 各 5 次），且没有任何调用方需要单条语义；
批量版固定 2 次查询，语义不变：有原文用原文，否则回退 summary/标题。
"""
from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Document, Notice

# 引用片段字符上限：问答引用与检索结果卡片共用，保证两处展示长度一致
SNIPPET_MAX_CHARS = 120
# 喂给 LLM 的每篇通知上下文上限
CONTEXT_MAX_CHARS = 400


def resolve_notice_texts(db: Session, notices: Sequence[Notice]) -> dict[int, str]:
    """批量解析通知文本：notice_id → 原文（缺原文回退摘要/标题）。

    始终返回非空字符串：标题是必填字段，因此最差情况也有内容可展示。
    """
    doc_ids = [n.document_id for n in notices if n.document_id]
    raws: dict[int, str] = {}
    if doc_ids:
        docs = db.execute(select(Document).where(Document.id.in_(doc_ids))).scalars().all()
        raws = {d.id: (d.raw_text or "") for d in docs}
    out: dict[int, str] = {}
    for n in notices:
        raw = raws.get(n.document_id) if n.document_id else None
        out[n.id] = raw if raw else (n.summary or n.title)
    return out
