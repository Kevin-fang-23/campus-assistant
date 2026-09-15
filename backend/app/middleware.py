"""HTTP 中间件：把 services/rate_limit 的判定接到请求链上。

放在独立模块而非 services/ 下，是为了保持 services 层框架无关
（该层现有代码不引入 fastapi / starlette）。

## 中间件栈（main.py 中按"后加在外层"的洋葱模型）

  CORSMiddleware → QaCacheMiddleware → RateLimitMiddleware → endpoint

- CORSMiddleware 在最外层：处理跨域响应头；
- QaCacheMiddleware 在 CORS 内、限流外：拦截 POST /api/qa 的**精确 key** 缓存命中，
  命中直接返回（**绕过限流**），不命中透传；
  （语义近邻命中不在这里做 —— 它要算查询向量，是阻塞调用且消耗上游资源，
   见下面 QaCacheMiddleware 的"为何语义路径不在本中间件"。）
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
from .services.qa_cache import get_cache
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

## 为何语义路径不在本中间件里

本中间件只处理**精确 key** 命中。语义近邻匹配（改写问法复用，见
`services/qa_cache.py` 的 `find_similar`）刻意放在 endpoint（`api/qa.py`）内：

1. **不能在这里发阻塞调用**：`__call__` 是 async 的，而 embedding 是同步
   网络调用。在事件循环里直接跑它会把整个进程的其他请求一起卡住；
   同步端点跑在线程池，天然适合做这件事（要用 `run_in_threadpool` 也能绕，
   但这会引入"命中判定发生在限流之外"的第二个问题）。
2. **语义命中本就该计入限流**：既有约定"缓存命中不入限流"的理由是
   命中不消耗上游资源 —— 精确命中确实如此（零成本），但语义命中**要算一次
   embedding**，是真实的上游开销。把它留在限流内层，规则才自洽。
"""

    # 仅这一个路径。其它接口的缓存策略各自决定。
    _QA_PATH = "/api/qa"

    # body 大小上限：/api/qa 的合法载荷只有 {query, top_k}，几十字节量级；
    # 1MB 已留足余量。没有上限时，攻击者可以发送超大 body 耗尽进程内存
    # （_read_body 必须读完整个 body 才能解析）。超限直接 413，不再继续读。
    _MAX_BODY_BYTES = 1024 * 1024

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

        # 1. 读 body（必须先消费 receive 才能解析）；超限返回 None
        body = await self._read_body(receive)
        if body is None:
            response = JSONResponse(
                status_code=413,
                content={"detail": "请求体过大。"},
            )
            await response(scope, receive, send)
            return

        # 2. 解析 QAIn：失败透传，由 FastAPI endpoint 返回 422
        try:
            payload = json.loads(body or b"{}")
            qa_in = QAIn.model_validate(payload)
        except (json.JSONDecodeError, ValueError):
            await self._forward_with_body(scope, receive, send, body)
            return

        # 3. 查缓存：命中直接返回，绕过限流
        #    只做**精确 key** 查找：语义近邻匹配要算查询向量（阻塞的 embedding
        #    调用），在 async 中间件里发它会把事件循环卡住、拖慢所有并发请求；
        #    而且语义命中消耗一次 embedding，按"限流是上游护栏"的约定本就该
        #    照常计入限流。因此语义路径放在 endpoint 内（同步端点跑线程池），
        #    详见 api/qa.py 的模块说明。
        cached = get_cache().lookup_exact(qa_in.query, qa_in.top_k)
        if cached is not None:
            hit = cached.answer.model_copy(update={"cache_hit": True})
            response = JSONResponse(content=hit.model_dump(mode="json"))
            response.headers["X-Cache"] = "HIT"
            await response(scope, receive, send)
            logger.debug("QaCacheMiddleware HIT path=%s q=%s", self._QA_PATH, qa_in.query)
            return

        # 4. 不命中：body 重构 receive，透传给限流 → endpoint
        await self._forward_with_body(scope, receive, send, body)

    # ------------------------------------------------------------------
    # 内部：读 body / 重构 receive
    # ------------------------------------------------------------------
    async def _read_body(self, receive: Receive) -> bytes | None:
        """单次消费 receive，拼出完整 body；超过 `_MAX_BODY_BYTES` 返回 None。

        必须拿到完整的 body 才能解析 QAIn；Starlette 会把 body 切成
        多个 chunk，靠 `more_body` 标记判断是否结束。

        超限处理：一旦累计字节数超过上限立即返回 None（调用方回 413），
        并继续把剩余 chunk 消费掉 —— 若提前停止读取，部分服务器/客户端
        会因请求未被完整消费而挂起或报错。
        """
        body = b""
        oversized = False
        while True:
            msg = await receive()
            mtype = msg.get("type")
            if mtype == "http.request":
                if not oversized:
                    body += msg.get("body", b"")
                    if len(body) > self._MAX_BODY_BYTES:
                        oversized = True
                        body = b""  # 释放已累积内容，防内存被撑大
                if not msg.get("more_body", False):
                    break
            elif mtype == "http.disconnect":
                break
        return None if oversized else body

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
