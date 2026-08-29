from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError:  # pragma: no cover - only for static analysis / minimal envs
    class _Router:
        def add_api_route(self, *args, **kwargs):
            return None

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            self.router = _Router()

        def add_middleware(self, *args, **kwargs):
            return None

        def include_router(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class CORSMiddleware:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    FastAPI = _FastAPI

from .api import documents, notices, search, tasks
from .config import settings
from .db import SessionLocal, init_db
from .graph.pipeline import engine_name
from .providers.ocr import get_ocr
from .providers.vlm import get_vlm
from .services.vector_store import get_store

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: Any):
    init_db()
    with SessionLocal() as db:
        get_store().load_from_db(db)
    logger.info(
        "启动完成 | 流程引擎=%s | VLM=%s | OCR=%s | 向量=%s",
        engine_name(), get_vlm().name, get_ocr().name, get_store().backend,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="多模态校园通知识别 → 关键信息抽取 → 待办生成 → 任务追踪",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(notices.router)
app.include_router(tasks.router)
app.include_router(search.router)


@app.get("/health", tags=["meta"], summary="健康检查与运行时能力")
def health() -> dict:
    store = get_store()
    return {
        "status": "ok",
        "app": settings.app_name,
        "pipeline_engine": engine_name(),
        "vlm": {"provider": get_vlm().name, "model": settings.vlm_model, "mock": get_vlm().is_mock},
        "ocr": get_ocr().name,
        "vector": {"backend": store.backend, "indexed": store.size},
        "database": settings.database_url.split("://", 1)[0],
    }


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": settings.app_name, "docs": "/docs", "health": "/health"}
