"""检索增强问答（RAG）接口：POST /api/qa。

主路径：向量检索召回 → 拼 [编号] context → OpenAICompatClient 生成答案。
降级路径（无 Key / QA_PROVIDER=mock / LLM 调用失败）：返回 top1 摘要的
抽取式回答，degraded=True——前端展示逻辑不变，只是答案非生成式。

与既有架构的对应关系：
- LLM 调用完全复用 providers/llm_client（鉴权/超时/重试/错误分级），
  不另起 httpx 客户端；
- 失败降级不抛 500，与 extract 节点「VLM 失败 → 规则抽取」同一模式。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Document, Notice
from ..providers.llm_client import LLMError, build_client_from_settings
from ..schemas import AnswerOut, CitationOut, QAIn
from ..services.vector_store import get_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])

_SYSTEM_PROMPT = "你是严谨的校园事务助手，只依据给定材料回答，不编造信息。"

_USER_PROMPT = """请依据下面的通知片段回答问题，引用来源用 [编号] 标注（如 [1]）；
片段中没有的信息不要编造，直接说明未找到。

通知片段：
{context}

问题：{query}
"""

# 单条通知进 context 的长度上限：太长既拖慢响应又稀释关键信息
_CONTEXT_PER_NOTICE = 400
# 引用片段（snippet）长度：S9 前端引用溯源高亮用
_SNIPPET_LIMIT = 120


def _build_llm_client():
    """工厂单独成函数，便于测试注入 MockTransport，不在业务代码里 if transport。"""
    return build_client_from_settings()


def _raw_or_summary(notice: Notice, db: Document | None) -> str:
    """优先用文档原文（信息最全），缺原文时回退摘要/标题。"""
    if db is not None and db.raw_text:
        return db.raw_text
    return notice.summary or notice.title


@router.post("", response_model=AnswerOut, summary="检索增强问答（RAG）：答案 + 引用片段")
def ask(payload: QAIn, db: Session = Depends(get_db)) -> AnswerOut:
    store = get_store()
    pairs = store.search(payload.query, payload.top_k)

    citations: list[CitationOut] = []
    sources: list[str] = []  # 与 citations 同序，用于拼 context
    for notice_id, score in pairs:
        notice = db.get(Notice, notice_id)
        if not notice:
            continue
        document = db.get(Document, notice.document_id)
        text = _raw_or_summary(notice, document)
        citations.append(
            CitationOut(
                notice_id=notice.id,
                title=notice.title,
                snippet=text[:_SNIPPET_LIMIT].strip(),
                score=round(score, 4),
            )
        )
        sources.append(text[:_CONTEXT_PER_NOTICE].strip())

    if not citations:
        return AnswerOut(
            query=payload.query,
            answer="知识库中暂时没有与该问题相关的通知。",
            citations=[],
            backend=store.backend,
            degraded=True,
        )

    # ---- LLM 生成路径：有 Key 且未被 QA_PROVIDER=mock 强制降级 ----
    if settings.dashscope_api_key and settings.qa_provider != "mock":
        context = "\n".join(
            f"[{i}] {c.title}\n{src}" for i, (c, src) in enumerate(zip(citations, sources), 1)
        )
        try:
            with _build_llm_client() as client:
                answer = client.chat(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _USER_PROMPT.format(
                            context=context, query=payload.query)},
                    ],
                    model=settings.qa_model,
                    max_tokens=512,
                    temperature=0.2,
                )
            return AnswerOut(
                query=payload.query,
                answer=answer.strip(),
                citations=citations,
                backend=store.backend,
                degraded=False,
            )
        except LLMError as exc:
            logger.warning("QA 生成失败，降级为抽取式回答：%s", exc)

    # ---- 降级路径：top1 摘要作为抽取式回答 ----
    return AnswerOut(
        query=payload.query,
        answer=citations[0].snippet,
        citations=citations,
        backend=store.backend,
        degraded=True,
    )
