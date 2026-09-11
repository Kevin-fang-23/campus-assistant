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
from ..models import Notice
from ..providers.llm_client import (
    LLMError,
    OpenAICompatClient,
    ResponseFormatError,
    build_client_from_settings,
)
from ..schemas import AnswerOut, CitationOut, QAIn
from ..services.hybrid import hybrid_search
from ..services.notice_text import (
    CONTEXT_MAX_CHARS,
    SNIPPET_MAX_CHARS,
    notice_context,
    notice_snippet,
)
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
# （上下限常量统一由 services/notice_text 定义，问答与检索共用同一份）
_CONTEXT_PER_NOTICE = CONTEXT_MAX_CHARS
# 引用片段（snippet）长度：S9 前端引用溯源高亮用
_SNIPPET_LIMIT = SNIPPET_MAX_CHARS


def _build_llm_client() -> OpenAICompatClient:
    """工厂单独成函数，便于测试注入 MockTransport，不在业务代码里 if transport。

    返回的是**进程级共享客户端**（连接池复用），因此调用方**不得**关闭它 ——
    见 ask() 中不使用 `with` 的原因说明。
    """
    return build_client_from_settings()


@router.post("", response_model=AnswerOut, summary="检索增强问答（RAG）：答案 + 引用片段")
def ask(payload: QAIn, db: Session = Depends(get_db)) -> AnswerOut:
    store = get_store()
    # 混合检索（向量 + BM25 加权融合）：问答的召回质量直接决定回答质量，
    # 精确串（课程名/房间号/手机号）靠 BM25 兜住，语义相近靠向量兜住。
    hits = hybrid_search(payload.query, payload.top_k)

    citations: list[CitationOut] = []
    sources: list[str] = []  # 与 citations 同序，用于拼 context
    for hit in hits:
        notice = db.get(Notice, hit.notice_id)
        if not notice:
            continue
        citations.append(
            CitationOut(
                notice_id=notice.id,
                title=notice.title,
                # 句级选片：挑出与问题最相关的句子，而不是从字中间硬切。
                # 硬切会切出残句（"…提交至学习通。截止时"），溯源展示不可读。
                snippet=notice_snippet(db, notice, payload.query, max_chars=_SNIPPET_LIMIT),
                score=round(hit.score, 4),
            )
        )
        # 上下文同样按句选：通知超长时优先保留相关句子。
        # 原实现取前 400 字，若关键信息（截止时间/地点）落在其后，
        # 模型看不到就只能回答"未找到"，而库里其实有答案。
        sources.append(
            notice_context(db, notice, payload.query, max_chars=_CONTEXT_PER_NOTICE)
        )

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
            # 刻意**不用** `with`：客户端由进程共享（连接复用），
            # `with` 会在请求结束时 close() 掉连接池，下一个请求又得重做
            # DNS+TCP+TLS 握手 —— 那正是本次改造要消除的开销。
            # 生命周期交给应用启动/关闭钩子（见 main.py 的 lifespan）。
            client = _build_llm_client()
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
            # 双保险：客户端已校验过内容非空，这里再挡一次空答案，
            # 避免"200 但 answer 为空串"这种前端无法展示的响应流出去
            if not answer or not answer.strip():
                logger.warning("QA 生成返回空内容，降级为抽取式回答")
                raise ResponseFormatError("模型返回空内容")
            return AnswerOut(
                query=payload.query,
                answer=answer.strip(),
                citations=citations,
                backend=store.backend,
                degraded=False,
            )
        except LLMError as exc:
            logger.warning("QA 生成失败，降级为抽取式回答：%s", exc)
        except Exception as exc:  # noqa: BLE001
            # 生成阶段的任何意外（含未预料的第三方异常）都不应让用户拿到 500：
            # /api/qa 的检索结果本身有效，降级为抽取式回答仍可用。
            logger.exception("QA 生成出现未预期异常，降级为抽取式回答：%s", exc)

    # ---- 降级路径：top1 摘要作为抽取式回答 ----
    return AnswerOut(
        query=payload.query,
        answer=citations[0].snippet,
        citations=citations,
        backend=store.backend,
        degraded=True,
    )
