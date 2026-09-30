from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..db import get_db
from ..models import Document
from ..schemas import DocumentOut, IngestResult, TextIngestIn
from ..services.ingest import MAX_UPLOAD_BYTES, ingest_bytes, ingest_text

router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# 分块读取大小：内存占用被钉在「上传上限 + 一个分块」，而不是整个请求体
_READ_CHUNK = 1024 * 1024


@router.post("/upload", response_model=IngestResult, summary="上传图片/PDF/文本并跑完整链路")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)) -> IngestResult:
    # 边读边计量，而不是 `await file.read()` 一次读完再检查：
    # 后者会把整个请求体读进内存后才轮到 ingest_bytes 的大小检查，
    # 单个超大请求（如数 GB）先撑内存再被拒。分块读取、超限立即 413。
    # ingest_bytes 内部的同款检查保留作纵深防御（/text 等其他入口仍靠它）。
    data = bytearray()
    while chunk := await file.read(_READ_CHUNK):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 限制",
            )
    try:
        # ingest_bytes 是同步阻塞的（文件 I/O / 数据库 / VLM / OCR），
        # 直接在 async 端点里调会卡住事件循环，期间所有其它请求（含 /health）
        # 都得等它跑完。丢进线程池执行，事件循环保持响应。
        return await run_in_threadpool(
            ingest_bytes, db, bytes(data), file.filename or "upload.bin", mime=file.content_type
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("处理上传文件失败")
        # 对外只给固定文案：原始异常文本可能含文件路径 / SQL 片段等内部信息，
        # 公网部署时不应返回给任意访问者；完整异常已由上面的日志记录。
        raise HTTPException(status_code=500, detail="处理失败，请稍后重试或联系管理员") from exc


@router.post("/text", response_model=IngestResult, summary="直接粘贴文本处理")
def upload_text(payload: TextIngestIn, db: Session = Depends(get_db)) -> IngestResult:
    try:
        return ingest_text(db, payload.content, payload.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("处理文本失败")
        raise HTTPException(status_code=500, detail="处理失败，请稍后重试或联系管理员") from exc


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
