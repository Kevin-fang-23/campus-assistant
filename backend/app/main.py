from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
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

    class StaticFiles:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

    FastAPI = _FastAPI

from .api import documents, notices, qa, search, tasks
from .config import settings
from .db import SessionLocal, init_db
from .graph.pipeline import engine_name
from .middleware import RateLimitMiddleware
from .providers.ocr import get_ocr
from .providers.llm_client import close_shared_client, get_shared_client
from .providers.vlm import get_vlm
from .services.bm25 import get_bm25_index
from .services.hybrid import rebuild_bm25
from .services.vector_store import get_store

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: Any):
    init_db()
    # 预热共享 LLM 客户端：把首次 DNS+TCP+TLS 握手的代价挪到启动阶段，
    # 而不是让第一个真实用户的请求承担（演示现场尤其在意首问延迟）。
    get_shared_client()
    with SessionLocal() as db:
        store = get_store()
        store.load_from_db(db)
        # 库内向量与当前后端维度不符（provider 切换或运行时回退的历史遗留）时
        # 自动重算：否则索引与查询向量不可比，检索会静默返回 0 条。
        if store.skipped_mismatched and settings.reindex_on_dim_mismatch:
            logger.warning(
                "检测到 %d 条历史向量与当前后端维度不符，自动重建全部向量…",
                store.skipped_mismatched,
            )
            store.reindex(db)
        # BM25 索引：与向量索引用同一份文本、同一批通知，启动时一次性构建
        rebuild_bm25(db)
    logger.info(
        "启动完成 | 流程引擎=%s | VLM=%s | OCR=%s | 向量=%s | BM25=%d 条%s | embedding=%s%s",
        engine_name(), get_vlm().name, get_ocr().name, get_store().backend,
        get_bm25_index().size,
        "（混合检索）" if settings.hybrid_enabled else "（未启用）",
        get_store().embedder.active_name,
        "（已降级）" if get_store().embedder.degraded else "",
    )
    yield
    # 关停：释放共享连接池，避免 uvicorn --reload 反复重启时残留连接
    close_shared_client()
    logger.info("共享 LLM 客户端已关闭")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="多模态校园通知识别 → 关键信息抽取 → 待办生成 → 任务追踪",
    lifespan=lifespan,
)

# 限流必须先于 CORS 注册：Starlette「后添加者在外层」，这样 CORS 处于外层、
# 限流处于内层，429 响应才会带上跨域头（否则前端只看到 CORS 报错而非可读的 429）。
app.add_middleware(RateLimitMiddleware)

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
app.include_router(qa.router)


@app.get("/health", tags=["meta"], summary="健康检查与运行时能力")
def health() -> dict:
    store = get_store()
    embedder = store.embedder
    return {
        "status": "ok",
        "app": settings.app_name,
        "pipeline_engine": engine_name(),
        "vlm": {"provider": get_vlm().name, "model": settings.vlm_model, "mock": get_vlm().is_mock},
        "ocr": get_ocr().name,
        "vector": {
            "backend": store.backend,
            "indexed": store.size,
            # 配置的后端 vs 实际生效的后端：两者不一致即说明发生了运行时回退，
            # 回退是粘性的（本进程内不再重试上游），重启服务即可恢复。
            "configured_embedding": embedder.name,
            "active_embedding": embedder.active_name,
            "embedding_degraded": embedder.degraded,
        },
        "database": settings.database_url.split("://", 1)[0],
    }


# ---- 生产模式：若前端已构建（frontend/dist），由后端直接托管静态页面 ----
# 这样公网只需映射后端一个端口（8000），页面与 /api 同源、加载快
# （构建产物是少量打包文件，不像 vite dev 有几百个小请求）。
# 本地开发不受影响：仍可用 vite dev（5173）做热更新。
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
