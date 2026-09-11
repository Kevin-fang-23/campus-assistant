from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Notice
from ..schemas import SearchHit, SearchIn, SearchOut
from ..services.hybrid import hybrid_search
from ..services.notice_text import notice_snippet
from ..services.vector_store import get_store

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchOut, summary="语义检索历史通知（向量 + BM25 混合）")
def semantic_search(payload: SearchIn, db: Session = Depends(get_db)) -> SearchOut:
    store = get_store()
    # 混合检索：向量（语义）与 BM25（字面精确）加权 RRF 融合。
    # 返回的 match / score_vector / score_bm25 让"这条是靠哪一路命中的"可被看见，
    # 便于演示与排查（例如精确房间号只有 BM25 能命中）。
    hits: list[SearchHit] = []
    for hit in hybrid_search(payload.query, payload.top_k):
        notice = db.get(Notice, hit.notice_id)
        if not notice:
            continue
        hits.append(
            SearchHit(
                notice_id=notice.id,
                score=round(hit.score, 4),
                title=notice.title,
                category=notice.category,
                summary=notice.summary,
                # 句级选片：按查询挑出原文里最相关的句子，与 /api/qa 的引用同一套逻辑。
                # 原实现只给 summary，那是抽取阶段的概括、与查询无关 ——
                # 用户搜"活动在哪举办"，卡片却显示"为提升动手能力举办本次工作坊"，答非所问。
                snippet=notice_snippet(db, notice, payload.query),
                deadline=notice.deadline,
                match=hit.match,
                score_vector=round(hit.score_vector, 4) if hit.score_vector is not None else None,
                score_bm25=round(hit.score_bm25, 4) if hit.score_bm25 is not None else None,
            )
        )
    return SearchOut(query=payload.query, backend=store.backend, hits=hits)
