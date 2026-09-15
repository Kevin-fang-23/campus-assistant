"""统一 API 客户端的单元测试：鉴权头、错误分级、重试次数、密钥不泄漏。

用 httpx.MockTransport 拦截请求，sleep 注入为记录器，全程无真实网络调用、无真实等待。
"""
from __future__ import annotations

import httpx
import pytest

from app.providers import llm_client
from app.providers.llm_client import (
    AuthError,
    BadRequestError,
    OpenAICompatClient,
    ResponseFormatError,
    ServerError,
    TransportError,
)

KEY = "sk-test-only-fake-key"
OK_CHAT = {"choices": [{"message": {"content": "hello"}}]}


def make_client(handler, *, max_retries: int = 2) -> tuple[OpenAICompatClient, list[float]]:
    waits: list[float] = []
    client = OpenAICompatClient(
        base_url="https://example.invalid/v1",
        api_key=KEY,
        timeout=5.0,
        max_retries=max_retries,
        transport=httpx.MockTransport(handler),
        sleep_fn=waits.append,
    )
    return client, waits


def test_auth_header_is_bearer() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=OK_CHAT)

    client, _ = make_client(handler)
    assert client.chat([{"role": "user", "content": "hi"}], model="m")
    assert seen[0].headers["Authorization"] == f"Bearer {KEY}"
    assert seen[0].headers["Content-Type"] == "application/json"


def test_secret_never_leaks_in_repr_or_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    client, _ = make_client(handler)
    with pytest.raises(AuthError):
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert KEY not in repr(client)


def test_missing_key_raises_auth_error() -> None:
    with pytest.raises(AuthError):
        OpenAICompatClient(base_url="https://example.invalid/v1", api_key="")


def test_auth_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="forbidden")

    client, waits = make_client(handler, max_retries=3)
    with pytest.raises(AuthError):
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert calls["n"] == 1, "鉴权失败不应重试"
    assert waits == []


def test_bad_request_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad model")

    client, _ = make_client(handler, max_retries=3)
    with pytest.raises(BadRequestError):
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert calls["n"] == 1


def test_rate_limit_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "1"})
        return httpx.Response(200, json=OK_CHAT)

    client, waits = make_client(handler, max_retries=2)
    assert client.chat([{"role": "user", "content": "hi"}], model="m") == "hello"
    assert calls["n"] == 2
    assert waits == [1.0], "429 应优先遵循 Retry-After"


def test_server_error_exhausts_retries(monkeypatch) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    # 固定抖动因子为下限，使退避可断言（0.8*2^n * 0.5）
    monkeypatch.setattr(llm_client.random, "random", lambda: 0.0)
    client, waits = make_client(handler, max_retries=2)
    with pytest.raises(ServerError):
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert calls["n"] == 3, "max_retries=2 应共尝试 3 次"
    assert waits == [0.4, 0.8], "退避应指数递增"


def test_timeout_is_retried_as_transport_error() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("timed out")

    client, _ = make_client(handler, max_retries=1)
    with pytest.raises(TransportError):
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert calls["n"] == 2


def test_malformed_response_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"unexpected": True})

    client, _ = make_client(handler, max_retries=3)
    with pytest.raises(ResponseFormatError):
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert calls["n"] == 1, "响应结构错误重试无意义"


def test_embed_parses_vectors() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.0, 3.0, 4.0]}]})

    client, _ = make_client(handler)
    assert client.embed(["a"], model="text-embedding-v4") == [[0.0, 3.0, 4.0]]
