"""检索增强问答（RAG）接口：POST /api/qa。

主路径：向量检索召回 → 拼 [编号] context → OpenAICompatClient 生成答案。
降级路径（无 Key / QA_PROVIDER=mock / LLM 调用失败）：返回 top1 摘要的
抽取式回答，degraded=True——前端展示逻辑不变，只是答案非生成式。

与既有架构的对应关系：
- LLM 调用完全复用 providers/llm_client（鉴权/超时/重试/错误分级），
  不另起 httpx 客户端；
- 失败降级不抛 500，与 extract 节点「VLM 失败 → 规则抽取」同一模式。

## 缓存层（qa_cache）

任何路径都会缓存最终 `AnswerOut`（LLM 成功 / LLM 降级 / 空召回），
缓存命中时**直接复用**：不再发起 embedding、也不再走 LLM。
key = (normalize(query), top_k)；TTL 与 max_size 可通过 .env 配置；
关闭缓存只需 `QA_CACHE_ENABLED=false`。
详见 services/qa_cache.py 模块 docstring。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select
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
    resolve_notice_texts,
)
from ..services.qa_cache import cache_key, get_cache
from ..services.snippet import select_context, select_snippet
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


def _build_answer(payload: QAIn, db: Session) -> AnswerOut:
    """构造 AnswerOut 的全部业务逻辑：检索 → 生成/降级。

    拆出内部函数的原因：让 `ask()` 入口的「查缓存 → 走业务 → 写缓存」
    三步结构清晰可见，业务逻辑本身不再关心缓存细节。
    单元测试也可以直接喂 QAIn 来验证。
    """
    store = get_store()
    # 混合检索（向量 + BM25 加权融合）：问答的召回质量直接决定回答质量，
    # 精确串（课程名/房间号/手机号）靠 BM25 兜住，语义相近靠向量兜住。
    hits = hybrid_search(payload.query, payload.top_k)

    # 批量取通知与文本：逐条 db.get 是 N+1（top_k=5 时 ~10 次查询），
    # 批量后固定 2 次；snippet/context 的选片逻辑不变（select_snippet/context 纯函数）。
    notices: dict[int, Notice] = {}
    if hits:
        rows = (
            db.execute(select(Notice).where(Notice.id.in_([h.notice_id for h in hits])))
            .scalars()
            .all()
        )
        notices = {n.id: n for n in rows}
    texts = resolve_notice_texts(db, list(notices.values()))

    citations: list[CitationOut] = []
    sources: list[str] = []  # 与 citations 同序，用于拼 context
    for hit in hits:
        notice = notices.get(hit.notice_id)
        if not notice:
            continue
        text = texts[notice.id]
        citations.append(
            CitationOut(
                notice_id=notice.id,
                title=notice.title,
                # 句级选片：挑出与问题最相关的句子，而不是从字中间硬切。
                # 硬切会切出残句（"…提交至学习通。截止时"），溯源展示不可读。
                snippet=select_snippet(
                    text, payload.query, title=notice.title, max_chars=_SNIPPET_LIMIT
                ),
                score=round(hit.score, 4),
            )
        )
        # 上下文同样按句选：通知超长时优先保留相关句子。
        # 原实现取前 400 字，若关键信息（截止时间/地点）落在其后，
        # 模型看不到就只能回答"未找到"，而库里其实有答案。
        sources.append(
            select_context(
                text, payload.query, title=notice.title, max_chars=_CONTEXT_PER_NOTICE
            )
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


@router.post("", response_model=AnswerOut, summary="检索增强问答（RAG）：答案 + 引用片段")
def ask(payload: QAIn, db: Session = Depends(get_db)) -> AnswerOut:
    """缓存 → 业务 → 回写 三段式。

    **为什么缓存命中要 `model_copy(update={"cache_hit": True})`**：
    Pydantic 对象是 immutable-friendly，缓存里的值是 cache_hit=False 的副本
    （写缓存前已显式重置），命中时再复制一份标 True。前端 / 调用方只看
    cache_hit 字段就能判断本次是否真正调用了 LLM，便于演示时可见化。
    """
    # ---- 缓存读：命中直接返回，不再发起 LLM/embedding ----
    if settings.qa_cache_enabled:
        key = cache_key(payload.query, payload.top_k)
        cached = get_cache().get(key)
        if cached is not None:
            # 深拷贝：避免外部修改 cache_hit 影响后续命中（缓存里应保持 False）
            return cached.model_copy(update={"cache_hit": True})

    # ---- 业务：检索 + 生成/降级 ----
    answer = _build_answer(payload, db)

    # ---- 缓存写：把 cache_hit 重置为 False 再存；下次命中时再标 True ----
    if settings.qa_cache_enabled:
        answer_to_cache = answer.model_copy(update={"cache_hit": False})
        get_cache().set(key, answer_to_cache)

    return answer
