from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Notice
from ..schemas import SearchHit, SearchIn, SearchOut
from ..services.hybrid import hybrid_search_filtered
from ..services.notice_text import SNIPPET_MAX_CHARS, resolve_notice_texts
from ..services.snippet import select_snippet
from ..services.vector_store import get_store

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchOut, summary="语义检索历史通知（向量 + BM25 混合）")
def semantic_search(payload: SearchIn, db: Session = Depends(get_db)) -> SearchOut:
    store = get_store()
    # 混合检索：向量（语义）与 BM25（字面精确）加权 RRF 融合。
    # 返回的 match / score_vector / score_bm25 让"这条是靠哪一路命中的"可被看见，
    # 便于演示与排查（例如精确房间号只有 BM25 能命中）。
    #
    # 融合后再过一道**相关性阈值**：候选与查询必须有至少 N 个共享二字词，
    # 否则判为无关并丢弃、置 filtered=True。没有这道判定时，
    # 问"今天天气怎么样"也会一本正经地返回 3 条通知，像是胡答。
    raw_hits, filtered = hybrid_search_filtered(db, payload.query, payload.top_k)

    # 批量取通知与文本（逐条 db.get 是 N+1，top_k=5 时 ~10 次查询 → 固定 2 次）
    notices: dict[int, Notice] = {}
    if raw_hits:
        rows = (
            db.execute(select(Notice).where(Notice.id.in_([h.notice_id for h in raw_hits])))
            .scalars()
            .all()
        )
        notices = {n.id: n for n in rows}
    texts = resolve_notice_texts(db, list(notices.values()))

    hits: list[SearchHit] = []
    for hit in raw_hits:
        notice = notices.get(hit.notice_id)
        if not notice:
            continue
        # 句级选片：按查询挑出原文里最相关的句子，与 /api/qa 的引用同一套逻辑。
        # 原实现只给 summary，那是抽取阶段的概括、与查询无关 ——
        # 用户搜"活动在哪举办"，卡片却显示"为提升动手能力举办本次工作坊"，答非所问。
        snippet = select_snippet(
            texts[notice.id], payload.query, title=notice.title, max_chars=SNIPPET_MAX_CHARS
        )
        hits.append(
            SearchHit(
                notice_id=notice.id,
                score=round(hit.score, 4),
                title=notice.title,
                category=notice.category,
                summary=notice.summary,
                snippet=snippet,
                deadline=notice.deadline,
                match=hit.match,
                score_vector=round(hit.score_vector, 4) if hit.score_vector is not None else None,
                score_bm25=round(hit.score_bm25, 4) if hit.score_bm25 is not None else None,
            )
        )
    return SearchOut(
        query=payload.query, backend=store.backend, hits=hits, filtered=filtered
    )
