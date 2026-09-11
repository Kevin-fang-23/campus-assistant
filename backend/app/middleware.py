"""HTTP 中间件：把 services/rate_limit 的判定接到请求链上。

放在独立模块而非 services/ 下，是为了保持 services 层框架无关
（该层现有代码不引入 fastapi / starlette）。
"""
from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import settings
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
