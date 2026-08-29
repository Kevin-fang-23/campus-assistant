from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Notice
from ..schemas import SearchHit, SearchIn, SearchOut
from ..services.vector_store import get_store

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchOut, summary="语义检索历史通知（FAISS）")
def semantic_search(payload: SearchIn, db: Session = Depends(get_db)) -> SearchOut:
    store = get_store()
    pairs = store.search(payload.query, payload.top_k)
    hits: list[SearchHit] = []
    for notice_id, score in pairs:
        notice = db.get(Notice, notice_id)
        if not notice:
            continue
        hits.append(
            SearchHit(
                notice_id=notice.id,
                score=round(score, 4),
                title=notice.title,
                category=notice.category,
                summary=notice.summary,
                deadline=notice.deadline,
            )
        )
    return SearchOut(query=payload.query, backend=store.backend, hits=hits)
