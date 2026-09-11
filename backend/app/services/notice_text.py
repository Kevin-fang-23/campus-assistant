"""通知正文解析与句级片段：问答与检索共用的文本来源。

## 为什么单独抽一层

`/api/qa` 与 `/api/search` 都需要同一条规则 —— **原文优先，缺原文回退摘要/标题**。
若两处各写一份，引用片段就可能基于不同文本，出现
「问答里能选到答案句、检索页却只显示摘要」的不一致，排查起来很费劲。

这里把「取哪份文本」与「怎么选片段」收口成两个函数，
`snippet.py` 保持纯函数、不引入模型依赖。

## 两个上限的区别

- `SNIPPET_MAX_CHARS`：**给人看**的片段（问答引用 / 检索结果卡片），要求短而准；
- `CONTEXT_MAX_CHARS`：**给模型看**的上下文，要求不漏关键信息。
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import Document, Notice
from .snippet import select_context, select_snippet

# 引用片段字符上限：问答引用与检索结果卡片共用，保证两处展示长度一致
SNIPPET_MAX_CHARS = 120
# 喂给 LLM 的每篇通知上下文上限
CONTEXT_MAX_CHARS = 400


def resolve_notice_text(db: Session, notice: Notice) -> str:
    """优先用文档原文（信息最全），缺原文时回退摘要/标题。

    始终返回非空字符串：标题是必填字段，因此最差情况也有内容可展示。
    """
    if notice.document_id:
        document = db.get(Document, notice.document_id)
        if document is not None and document.raw_text:
            return document.raw_text
    return notice.summary or notice.title


def notice_snippet(
    db: Session,
    notice: Notice,
    query: str,
    *,
    max_chars: int = SNIPPET_MAX_CHARS,
) -> str:
    """为通知选出与 query 最相关的句级片段（保留原文换行排版）。"""
    return select_snippet(
        resolve_notice_text(db, notice),
        query,
        title=notice.title,
        max_chars=max_chars,
    )


def notice_context(
    db: Session,
    notice: Notice,
    query: str,
    *,
    max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """为通知选出喂给 LLM 的上下文（超预算时优先保留相关句）。"""
    return select_context(
        resolve_notice_text(db, notice),
        query,
        title=notice.title,
        max_chars=max_chars,
    )
