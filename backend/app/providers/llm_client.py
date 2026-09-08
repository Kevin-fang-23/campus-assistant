"""统一的 OpenAI 兼容 API 客户端。

集中处理四件事，避免每个 provider 各写一遍：
1. 鉴权头构造（Bearer，密钥只存在于请求头，不进日志与异常文本）
2. 超时（连接/读/写统一取自 settings.vlm_timeout）
3. 错误分级（哪些该重试、哪些不该重试）
4. 重试（指数退避 + 抖动，429 优先遵循 Retry-After）

设计约束：
- 密钥绝不出现在异常信息、日志或 repr 中
- 4xx 客户端错误（401/403/400/404/422）**不重试**——重试无意义且浪费额度
- 429/5xx/网络类错误才重试
- sleep 可注入，便于单元测试无需真实等待
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Mapping

import httpx

logger = logging.getLogger(__name__)

# 不重试：请求本身有问题，重试只会得到同样结果
_NO_RETRY_STATUS = (400, 401, 403, 404, 405, 422)
# 重试：限流或上游故障
_RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)


class LLMError(Exception):
    """所有 LLM 调用异常的基类。"""

    retryable = False

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(LLMError):
    """401/403：密钥无效、欠费或权限不足。不可重试。"""


class BadRequestError(LLMError):
    """400/404/422：请求参数或模型名有问题。不可重试。"""


class RateLimitError(LLMError):
    """429：限流。可重试。"""

    retryable = True


class ServerError(LLMError):
    """5xx：上游故障。可重试。"""

    retryable = True


class TransportError(LLMError):
    """超时/连接失败/读超时。可重试。"""

    retryable = True


class ResponseFormatError(LLMError):
    """响应结构不符合预期（缺 choices / data）。不可重试。"""


class OpenAICompatClient:
    """OpenAI 兼容 /chat/completions、/embeddings 的薄封装。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff_base: float = 0.8,
        backoff_max: float = 8.0,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise AuthError("缺少 API Key：请在 .env 中配置 ALIYUN_API_KEY")
        self._api_key = api_key
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._sleep = sleep_fn
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=self._build_headers(api_key),
            transport=transport,
        )

    # ---------------- 内部 ----------------
    @staticmethod
    def _build_headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "campus-assistant/0.1",
        }

    def __repr__(self) -> str:  # 防止密钥随对象打印泄漏
        return f"OpenAICompatClient(base_url={self._client.base_url!s}, retries={self._max_retries})"

    def _classify(self, exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return TransportError(f"请求超时: {exc.__class__.__name__}")
        if isinstance(exc, httpx.TransportError):
            return TransportError(f"网络异常: {exc.__class__.__name__}")
        return TransportError(f"未知异常: {exc.__class__.__name__}")

    def _raise_for_status(self, resp: httpx.Response) -> None:
        code = resp.status_code
        if code == 200:
            return
        # 只截取响应体片段，且不含请求头（避免把密钥回显出来）
        detail = resp.text[:200].replace("\n", " ")
        if code in (401, 403):
            raise AuthError(f"鉴权失败或被拒绝（HTTP {code}）：{detail}", code)
        if code in _NO_RETRY_STATUS:
            raise BadRequestError(f"请求被拒绝（HTTP {code}）：{detail}", code)
        if code == 429:
            raise RateLimitError(f"触发限流（HTTP 429）：{detail}", code)
        raise ServerError(f"上游服务错误（HTTP {code}）：{detail}", code)

    def _backoff(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self._backoff_max)
            except ValueError:
                pass
        raw = min(self._backoff_max, self._backoff_base * (2**attempt))
        return raw * (0.5 + random.random() * 0.5)  # 抖动，避免多实例同时重试

    def post_json(self, path: str, payload: Mapping[str, Any]) -> dict:
        """带重试的 POST，返回已解析的 JSON dict。"""
        last: LLMError | None = None
        for attempt in range(self._max_retries + 1):
            resp: httpx.Response | None = None
            try:
                resp = self._client.post(path, json=dict(payload))
                self._raise_for_status(resp)
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                err = self._classify(exc)
                last = err
                if not err.retryable or attempt >= self._max_retries:
                    raise err from None
                retry_after = resp.headers.get("Retry-After") if resp is not None else None
                wait = self._backoff(attempt, retry_after)
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次），%.2fs 后重试：%s",
                    attempt + 1, self._max_retries + 1, wait, err,
                )
                self._sleep(wait)
        raise last or LLMError("LLM 调用失败")

    # ---------------- 对外 ----------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> str:
        data = self.post_json(
            "/chat/completions",
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ResponseFormatError(f"响应结构异常：{str(data)[:200]}") from exc

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        data = self.post_json("/embeddings", {"model": model, "input": texts})
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise ResponseFormatError(f"响应结构异常：{str(data)[:200]}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_client_from_settings(
    *, transport: httpx.BaseTransport | None = None, sleep_fn: Callable[[float], None] = time.sleep
) -> OpenAICompatClient:
    """按当前配置构造客户端（VLM 与 Embedding 共用同一套鉴权与重试策略）。"""
    from ..config import settings

    return OpenAICompatClient(
        base_url=settings.vlm_base_url,
        api_key=settings.dashscope_api_key or settings.aliyun_api_key,
        timeout=settings.vlm_timeout,
        max_retries=settings.vlm_max_retries,
        backoff_base=settings.vlm_backoff_base,
        backoff_max=settings.vlm_backoff_max,
        transport=transport,
        sleep_fn=sleep_fn,
    )
