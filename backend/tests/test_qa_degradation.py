"""「已有 API Key 但 LLM 调用抛错」场景下的降级测试。

背景：`/api/qa` 有两条路径，此前只有 `QA_PROVIDER=mock`（未进 LLM 分支）
和 LLM 成功两条用例，**没覆盖「进了 LLM 分支但调用抛错」**——而这恰恰是
演示现场最容易翻车的地方（网络抖动、额度耗尽、上游 5xx）。

本文件要证明的契约（不可放宽）：
1. 无论 LLM 抛哪类错误，HTTP 状态码始终 200（用户侧核心流程不中断）；
2. `degraded=True`，前端据此展示"抽取式回答"标识；
3. `citations` / `query` / `backend` 结构与成功路径完全一致，只是
   `answer` 退化为 top1 摘要；
4. 降级只发生在 LLM 层，检索层（向量召回）结果不受影响。

实现方式：monkeypatch `qa_module._build_llm_client`，返回一个把网络层换成
`httpx.MockTransport` 的真实 `OpenAICompatClient`——即「复用同一个客户端，
只换传输层」，不需要为测试新增任何生产代码分支。
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import qa as qa_module
from app.main import app
from app.providers.llm_client import (
    AuthError,
    LLMError,
    OpenAICompatClient,
    RateLimitError,
    ResponseFormatError,
    ServerError,
    TransportError,
)

# 与 test_api.py 保持同一路径语义：用真实查得到的校园通知做检索底料
QUERY = "操作系统作业什么时候截止"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# 工具：把各种「故障」包装成一个可注入的 fake 客户端
# --------------------------------------------------------------------------
def _client_with_handler(handler, *, max_retries: int = 0) -> OpenAICompatClient:
    """构造真实客户端 + MockTransport。

    max_retries=0：测试要验证的是「失败后降级」，不是重试次数；
    重试策略由 test_llm_client.py 单独覆盖，这里让失败一次即返回，
    既加快用例又避免退避 sleep 干扰断言。
    """
    return OpenAICompatClient(
        base_url="https://mock.test/v1",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        max_retries=max_retries,
        sleep_fn=lambda _s: None,
    )


def _install(monkeypatch, handler, *, max_retries: int = 0) -> dict:
    """注入 fake 客户端并强制开启 LLM 分支，返回可观测的计数器。"""
    calls: dict = {"count": 0}

    def _wrapped(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return handler(request)

    monkeypatch.setattr(
        qa_module, "_build_llm_client", lambda: _client_with_handler(_wrapped, max_retries=max_retries)
    )
    # 有 Key + 非 mock ⇒ 一定会进入 LLM 生成路径
    monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
    monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")
    return calls


def _assert_degraded_body(body: dict, *, expect_answer: bool = True) -> None:
    """降级响应的结构契约：与成功路径同构，仅 degraded/answer 不同。"""
    assert body["query"] == QUERY
    assert body["degraded"] is True, "LLM 故障时必须标记为降级"
    assert isinstance(body["answer"], str)
    if expect_answer:
        assert body["answer"].strip(), "降级也要给出可用回答（top1 摘要）"
    assert body["citations"], "检索层不受 LLM 故障影响，引用列表必须非空"
    for c in body["citations"]:
        assert c["notice_id"] and c["title"] and c["snippet"]
        assert c["score"] > 0
    assert body["backend"]


# --------------------------------------------------------------------------
# 1. LLM 服务超时
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc,label",
    [
        (httpx.ReadTimeout("read timed out"), "读超时"),
        (httpx.ConnectTimeout("connect timed out"), "连接超时"),
        (httpx.WriteTimeout("write timed out"), "写超时"),
    ],
)
def test_llm_timeout_degrades(client: TestClient, monkeypatch, exc: Exception, label: str) -> None:
    """超时（读/连接/写）→ 200 + degraded，用户看到的是抽取式回答而非 500。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    calls = _install(monkeypatch, handler)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})

    assert resp.status_code == 200, f"{label} 不应把 500 抛给用户：{resp.text}"
    _assert_degraded_body(resp.json())
    assert calls["count"] == 1, "max_retries=0 时只应尝试一次"


def test_llm_connect_error_degrades(client: TestClient, monkeypatch) -> None:
    """连接层失败（如 DNS 解析不了、端口不可达）同样降级。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install(monkeypatch, handler)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert resp.status_code == 200
    _assert_degraded_body(resp.json())


# --------------------------------------------------------------------------
# 2. 返回格式异常
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload,label",
    [
        ({}, "空对象"),
        ({"choices": []}, "choices 为空数组"),
        ({"choices": [{}]}, "choice 缺 message"),
        ({"choices": [{"message": {}}]}, "message 缺 content"),
        ({"choices": [{"message": {"content": None}}]}, "content 为 null"),
        ({"choices": "not-a-list"}, "choices 类型错误"),
        ({"error": {"message": "quota exceeded"}}, "只返回 error 字段"),
    ],
)
def test_llm_malformed_response_degrades(
    client: TestClient, monkeypatch, payload: dict, label: str
) -> None:
    """HTTP 200 但响应结构不对 → ResponseFormatError → 降级。

    这类故障最隐蔽：状态码是 200，若不校验结构会把 None/KeyError
    一路带到响应体，前端拿到空答案还以为模型"没话说"。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    _install(monkeypatch, handler)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})

    assert resp.status_code == 200, f"{label} 不应 500：{resp.text}"
    _assert_degraded_body(resp.json())


def test_llm_non_json_body_degrades(client: TestClient, monkeypatch) -> None:
    """上游返回 HTML 错误页（网关劫持）→ 解析失败 → 降级。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway</html>")

    _install(monkeypatch, handler)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert resp.status_code == 200
    _assert_degraded_body(resp.json())


# --------------------------------------------------------------------------
# 3. 接口限流（上游 429）与鉴权/参数错误
# --------------------------------------------------------------------------
def test_llm_rate_limited_degrades(client: TestClient, monkeypatch) -> None:
    """上游 429（百炼额度用尽）→ 重试耗尽后降级，而不是把 429 透传给用户。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Requests rate limit exceeded"}},
            headers={"Retry-After": "1"},
        )

    # max_retries=1：验证「重试一次仍 429 后再降级」这条完整链路
    calls = _install(monkeypatch, handler, max_retries=1)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})

    assert resp.status_code == 200
    _assert_degraded_body(resp.json())
    assert calls["count"] == 2, "限流是可重试错误：应重试 1 次后放弃（共 2 次请求）"


def test_llm_server_error_degrades(client: TestClient, monkeypatch) -> None:
    """上游 500/503 → 可重试 → 重试耗尽后降级。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "service unavailable"}})

    calls = _install(monkeypatch, handler, max_retries=1)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert resp.status_code == 200
    _assert_degraded_body(resp.json())
    assert calls["count"] == 2


@pytest.mark.parametrize("code", [401, 403])
def test_llm_auth_error_degrades_without_retry(
    client: TestClient, monkeypatch, code: int
) -> None:
    """Key 失效/欠费（401/403）→ 不可重试，一次即降级（不能白耗额度）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"error": {"message": "invalid api key"}})

    calls = _install(monkeypatch, handler, max_retries=2)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})

    assert resp.status_code == 200
    _assert_degraded_body(resp.json())
    assert calls["count"] == 1, "鉴权错误不可重试，只应请求一次"


def test_llm_bad_request_degrades_without_retry(client: TestClient, monkeypatch) -> None:
    """模型名写错（400/404）→ 不可重试，一次即降级。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    calls = _install(monkeypatch, handler, max_retries=2)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert resp.status_code == 200
    _assert_degraded_body(resp.json())
    assert calls["count"] == 1


# --------------------------------------------------------------------------
# 4. 未知运行时错误
# --------------------------------------------------------------------------
def test_unknown_runtime_error_degrades(client: TestClient, monkeypatch) -> None:
    """未知异常（既非 HTTP 错误也非超时，如解析层 RuntimeError）。

    llm_client._classify 把所有非 HTTP 异常兜底成 TransportError，
    因此这类"没预料到的"故障同样被 LLMError 捕获 → 降级。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("unexpected internal failure")

    _install(monkeypatch, handler)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert resp.status_code == 200
    _assert_degraded_body(resp.json())


def test_error_classification_contract() -> None:
    """错误分级契约：哪些该重试、哪些不该。

    这是降级链能"不卡死也不早退"的前提——重试策略一旦被改错，
    要么白等重试（400 重试无意义），要么过早放弃（429 其实可恢复）。
    """
    assert LLMError("x").retryable is False
    assert AuthError("x", 401).retryable is False
    assert ResponseFormatError("x").retryable is False
    assert RateLimitError("x", 429).retryable is True
    assert ServerError("x", 503).retryable is True
    assert TransportError("x").retryable is True
    # 全部是 LLMError 子类 ⇒ qa.py 单个 except 即可覆盖
    for exc in (AuthError, RateLimitError, ServerError, TransportError, ResponseFormatError):
        assert issubclass(exc, LLMError)


def test_client_catches_bare_exception_from_handler() -> None:
    """直接验证客户端层：handler 抛任意异常都会被归成 LLMError 而非裸冒泡。

    这是 qa.py `except LLMError` 能兜住"未知错误"的根本原因，
    单独断言以免有人日后把 _classify 的兜底分支删掉。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise ValueError("boom")

    client_obj = _client_with_handler(handler)
    try:
        with pytest.raises(LLMError) as excinfo:
            client_obj.chat([{"role": "user", "content": "hi"}], model="qwen-plus")
        assert isinstance(excinfo.value, TransportError)
    finally:
        client_obj.close()


# --------------------------------------------------------------------------
# 4b. 回归：「content 为 null / 空串」曾导致 500
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "content,label",
    [
        (None, "content 为 null（安全策略拦截）"),
        ("", "content 为空串"),
        ("   \n\t ", "content 仅空白字符"),
        (12345, "content 为数字"),
        (["a", "b"], "content 为数组"),
    ],
)
def test_null_or_blank_content_degrades_not_500(
    client: TestClient, monkeypatch, content, label: str
) -> None:
    """回归用例：HTTP 200 且键存在，但 content 是 null/空/非字符串。

    修复前：`data["choices"][0]["message"]["content"]` 取到 None 不抛 KeyError，
    逃过格式校验 → `answer.strip()` 触发 AttributeError → **500 暴露给用户**。
    这是「LLM 故障中断核心流程」最直接的反例，必须锁死。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    _install(monkeypatch, handler)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})

    assert resp.status_code == 200, f"{label} 修复前会 500：{resp.text}"
    body = resp.json()
    _assert_degraded_body(body)
    assert isinstance(body["answer"], str) and body["answer"].strip()


def test_client_rejects_null_content_directly() -> None:
    """客户端层单测：null content 必须抛 ResponseFormatError，而不是返回 None。

    与上面的端到端用例互为保障——即使有人改了 qa.py 的兜底，
    客户端层也不会把 None 交出去。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    client_obj = _client_with_handler(handler)
    try:
        with pytest.raises(ResponseFormatError):
            client_obj.chat([{"role": "user", "content": "hi"}], model="qwen-plus")
    finally:
        client_obj.close()


def test_empty_answer_degrades(client: TestClient, monkeypatch) -> None:
    """模型"成功返回"但答案为空串时，也走降级而非给前端一个空答案。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    _install(monkeypatch, handler)
    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    _assert_degraded_body(body)
    assert body["degraded"] is True


def test_unexpected_exception_in_generation_degrades(client: TestClient, monkeypatch) -> None:
    """生成阶段抛出非 LLMError 异常（如客户端构造失败）也不应 500。

    llm_client 之外的因素（工厂函数、settings 读取、第三方库）出错时，
    检索结果依然有效，降级为抽取式回答远好于给用户 500。
    """

    def boom_factory():
        raise KeyError("unexpected config error")

    monkeypatch.setattr(qa_module, "_build_llm_client", boom_factory)
    monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
    monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")

    resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert resp.status_code == 200, f"非 LLMError 异常不应 500：{resp.text}"
    _assert_degraded_body(resp.json())


# --------------------------------------------------------------------------
# 5. 降级不污染检索层 / 不因 LLM 故障中断核心流程
# --------------------------------------------------------------------------
def test_degraded_answer_equals_top1_snippet(client: TestClient, monkeypatch) -> None:
    """降级回答 == top1 引用片段，保证前端"答案↔来源"仍能对上。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, handler)
    body = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    assert body["degraded"] is True
    assert body["answer"] == body["citations"][0]["snippet"]


def test_degradation_keeps_same_citations_as_success_path(
    client: TestClient, monkeypatch
) -> None:
    """同一 query 下，LLM 成功与 LLM 失败召回的通知集合必须一致。

    证明降级只在生成层发生，检索层（向量召回 + 排序）完全不受影响。
    """

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "生成答案 [1]"}}]})

    _install(monkeypatch, ok)
    success = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()

    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "boom"}})

    _install(monkeypatch, fail)
    degraded = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()

    assert success["degraded"] is False and degraded["degraded"] is True
    # 召回顺序一致（分数末位有浮点抖动，用近似比较而非逐位相等）
    assert [c["notice_id"] for c in success["citations"]] == [
        c["notice_id"] for c in degraded["citations"]
    ]
    for s, d in zip(success["citations"], degraded["citations"]):
        assert s["score"] == pytest.approx(d["score"], abs=1e-3)
    # 引用片段也不该被 LLM 故障改变
    assert [c["snippet"] for c in success["citations"]] == [
        c["snippet"] for c in degraded["citations"]
    ]


def test_empty_knowledge_base_not_affected_by_llm_failure(
    client: TestClient, monkeypatch
) -> None:
    """检索为空时不调用 LLM，直接返回"未找到"，且请求计数器为 0。

    该分支与 LLM 故障无关，但必须一起断言：否则一旦有人把 LLM 调用
    挪到检索判空之前，就会对着空 context 白烧一次额度。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install(monkeypatch, handler)
    resp = client.post(
        "/api/qa", json={"query": "zzz_绝对召回不到的查询_zzz", "top_k": 1}
    )
    assert resp.status_code == 200
    body = resp.json()
    if not body["citations"]:
        assert body["degraded"] is True
        assert "没有" in body["answer"] or "未找到" in body["answer"]
        assert calls["count"] == 0, "无引用时不应发起 LLM 调用"


def test_llm_failure_never_surfaces_500(client: TestClient, monkeypatch) -> None:
    """遍历各类故障，统一断言"永不 500"——用户侧核心流程不中断。"""

    def make(code_or_exc):
        def handler(request: httpx.Request) -> httpx.Response:
            if isinstance(code_or_exc, Exception):
                raise code_or_exc
            return httpx.Response(code_or_exc, text="upstream failure")

        return handler

    cases = [
        make(httpx.ReadTimeout("timeout")),
        make(httpx.ConnectError("refused")),
        make(RuntimeError("unknown")),
        make(500),
        make(502),
        make(503),
        make(429),
        make(401),
        make(400),
        make(404),
    ]
    for handler in cases:
        _install(monkeypatch, handler)
        resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
        assert resp.status_code == 200, f"{handler} 触发了非 200：{resp.status_code}"
        assert resp.json()["degraded"] is True
