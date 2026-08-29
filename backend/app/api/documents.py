from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Document
from ..schemas import DocumentOut, IngestResult, TextIngestIn
from ..services.ingest import ingest_bytes, ingest_text

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=IngestResult, summary="上传图片/PDF/文本并跑完整链路")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)) -> IngestResult:
    data = await file.read()
    try:
        return ingest_bytes(db, data, file.filename or "upload.bin", mime=file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("处理上传文件失败")
        raise HTTPException(status_code=500, detail=f"处理失败：{exc}") from exc


@router.post("/text", response_model=IngestResult, summary="直接粘贴文本处理")
def upload_text(payload: TextIngestIn, db: Session = Depends(get_db)) -> IngestResult:
    try:
        return ingest_text(db, payload.content, payload.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("处理文本失败")
        raise HTTPException(status_code=500, detail=f"处理失败：{exc}") from exc


@router.get("", response_model=list[DocumentOut], summary="文档列表")
def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[DocumentOut]:
    rows = db.execute(
        select(Document).order_by(Document.id.desc()).limit(limit).offset(offset)
    ).scalars().all()
    return [DocumentOut.model_validate(r) for r in rows]


@router.get("/{doc_id}", response_model=DocumentOut, summary="文档详情")
def get_document(doc_id: int, db: Session = Depends(get_db)) -> DocumentOut:
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentOut.model_validate(doc)
