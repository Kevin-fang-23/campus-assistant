"""HTTP 中间件：把 services/rate_limit 的判定接到请求链上。

放在独立模块而非 services/ 下，是为了保持 services 层框架无关
（该层现有代码不引入 fastapi / starlette）。

## 中间件栈（main.py 中按"后加在外层"的洋葱模型）

  CORSMiddleware → QaCacheMiddleware → RateLimitMiddleware → endpoint

- CORSMiddleware 在最外层：处理跨域响应头；
- QaCacheMiddleware 在 CORS 内、限流外：拦截 POST /api/qa 缓存命中，
  命中直接返回（**绕过限流**），不命中透传；
- RateLimitMiddleware 在 QaCache 内：缓存未命中才会计费。
"""
from __future__ import annotations

import json
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import settings
from .schemas import QAIn
from .services.qa_cache import cache_key, get_cache
from .services.rate_limit import Verdict, client_key, get_limiter

logger = logging.getLogger(__name__)

# 拒绝原因文案，与项目既有错误响应结构保持一致（{"detail": "..."}）
_MESSAGES = {
    "ip_minute": "请求过于频繁，请稍后重试。",
    "ip_day": "今日该接口的个人配额已用完，请明天再试。",
    "global_day": "今日该接口的总额度已用完，请明天再试。",
}


def _message(verdict: Verdict) -> str:
    return _MESSAGES.get(verdict.scope or "", "请求被限流。")


class QaCacheMiddleware:
    """在限流外层拦截 `POST /api/qa` 的缓存命中。

    ## 为什么不在 `ask()` endpoint 内查缓存

    FastAPI 中间件按洋葱顺序执行，后注册的在外层。当前栈
    （`main.py`）若仅含 CORS + RateLimit，请求必然先过限流再进
    endpoint；缓存命中在 endpoint 内部发生 = 已经挤占了一次限流额度
    —— 而缓存命中**不消耗上游资源**，让它跟真实调用共享同一额度是错的。

    把缓存检查提到 RateLimitMiddleware 之外，缓存命中直接 JSONResponse
    返回，不计限流。

    ## 注册顺序

    必须在 `RateLimitMiddleware` **之后** `add_middleware`，否则它会落在
    限流内层，失去绕过限流的意义。

    ## 实现要点

    1. 只拦截 `POST /api/qa`，其它路径透传；
    2. 自己 `read_body` 后必须**重构 receive**，把 body 重新喂给下一层
       （Starlette 的 `receive` 是单次消费的 async generator）；
    3. 命中时构造 `JSONResponse`，响应头加 `X-Cache: HIT` 便于调试；
    4. 解析失败透传（让 FastAPI endpoint 返回 422），不要在本中间件
       模拟 FastAPI 的错误格式 —— 容易漂移。
    """

    # 仅这一个路径。其它接口的缓存策略各自决定。
    _QA_PATH = "/api/qa"

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # 不是 HTTP / 不是 POST / 不是 /api/qa → 透传
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("method") != "POST" or scope.get("path") != self._QA_PATH:
            await self.app(scope, receive, send)
            return
        # 关闭开关时也透传（避免行为漂移：双保险，ask() 内也检查一次）
        if not settings.qa_cache_enabled:
            await self.app(scope, receive, send)
            return

        # 1. 读 body（必须先消费 receive 才能解析）
        body = await self._read_body(receive)

        # 2. 解析 QAIn：失败透传，由 FastAPI endpoint 返回 422
        try:
            payload = json.loads(body or b"{}")
            qa_in = QAIn.model_validate(payload)
        except (json.JSONDecodeError, ValueError):
            await self._forward_with_body(scope, receive, send, body)
            return

        # 3. 查缓存：命中直接返回，绕过限流
        key = cache_key(qa_in.query, qa_in.top_k)
        cached = get_cache().get(key)
        if cached is not None:
            hit = cached.model_copy(update={"cache_hit": True})
            response = JSONResponse(content=hit.model_dump(mode="json"))
            response.headers["X-Cache"] = "HIT"
            await response(scope, receive, send)
            logger.debug("QaCacheMiddleware HIT path=%s key=%s", self._QA_PATH, key)
            return

        # 4. 不命中：body 重构 receive，透传给限流 → endpoint
        await self._forward_with_body(scope, receive, send, body)

    # ------------------------------------------------------------------
    # 内部：读 body / 重构 receive
    # ------------------------------------------------------------------
    async def _read_body(self, receive: Receive) -> bytes:
        """单次消费 receive，拼出完整 body。

        必须拿到完整的 body 才能解析 QAIn；Starlette 会把 body 切成
        多个 chunk，靠 `more_body` 标记判断是否结束。
        """
        body = b""
        while True:
            msg = await receive()
            mtype = msg.get("type")
            if mtype == "http.request":
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
            elif mtype == "http.disconnect":
                break
        return body

    async def _forward_with_body(
        self, scope: Scope, original_receive: Receive, send: Send, body: bytes
    ) -> None:
        """把 body 重新喂给下一层（receive 是单次消费的）。"""
        sent = False

        async def replay_receive() -> dict:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件。

    ⚠️ 注册顺序：必须在 CORSMiddleware **之前** add_middleware。
    Starlette 中「后添加的中间件在外层」，把 CORS 放在外层、限流放在内层，
    429 响应才会带上跨域头；否则前端拿到的是 CORS 报错而不是可读的 429。
    """

    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path
        ip = client_key(
            forwarded_for=request.headers.get("x-forwarded-for"),
            real_ip=request.headers.get("x-real-ip"),
            host=request.client.host if request.client else None,
            trust_proxy=settings.rate_limit_trust_proxy,
        )
        verdict = get_limiter().check(path, ip)

        if verdict.allowed:
            response = await call_next(request)
            if verdict.remaining >= 0:
                response.headers["X-RateLimit-Remaining"] = str(verdict.remaining)
            return response

        logger.warning(
            "限流拦截 | %s %s | ip=%s | 层级=%s | 建议重试=%ss",
            request.method, path, ip, verdict.scope, verdict.retry_after,
        )
        return JSONResponse(
            status_code=429,
            content={"detail": _message(verdict)},
            headers={
                "Retry-After": str(verdict.retry_after),
                "X-RateLimit-Scope": verdict.scope or "",
                "X-RateLimit-Remaining": "0",
            },
        )
