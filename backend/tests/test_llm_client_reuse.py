"""连接复用（建议 4）的验证。

改造前：`/api/qa` 每次请求都新建 httpx.Client，请求结束即 close()
→ 每个请求重做一次 DNS+TCP+TLS 握手。

改造后：进程级共享客户端 + keep-alive 连接池。

验证策略：**起一个真实的本地 OpenAI 兼容服务器，统计它接受了多少条 TCP 连接**。
这是对"连接复用"最诚实的度量 —— 不依赖任何 mock 传输层，
因为 mock 传输层恰好会掩盖真实的连接建立行为。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from app.api import qa as qa_module
from app.config import settings
from app.main import app
from app.providers import llm_client as llm
from app.providers.embedding import DashScopeEmbedding, get_embedding
from app.providers.llm_client import (
    OpenAICompatClient,
    build_client_from_settings,
    close_shared_client,
    get_shared_client,
)
from app.services import vector_store as vs_module
from app.services.vector_store import VectorStore

_EMBED_DIM = 1024


class _MockLLMServer:
    """最小 OpenAI 兼容服务：支持 /v1/embeddings 与 /v1/chat/completions。

    用 HTTP/1.1 + keep-alive，并统计 accept 的 TCP 连接数 ——
    连接复用生效时，多条请求应共用同一条连接。
    """

    def __init__(self) -> None:
        self.connections = 0
        self.requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # 允许 keep-alive

            def setup(self) -> None:  # 每接受一条 TCP 连接调用一次
                outer.connections += 1
                super().setup()

            def log_message(self, *args) -> None:  # 静音访问日志
                return

            def _send(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                outer.requests += 1
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = {}

                if self.path.endswith("/embeddings"):
                    inputs = payload.get("input") or [""]
                    if isinstance(inputs, str):
                        inputs = [inputs]
                    self._send(
                        {"data": [{"embedding": [0.01] * _EMBED_DIM} for _ in inputs]}
                    )
                elif self.path.endswith("/chat/completions"):
                    self._send(
                        {"choices": [{"message": {"content": "根据[1]，答案如下。"}}]}
                    )
                else:
                    self._send({"error": {"message": "not found"}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def mock_llm():
    server = _MockLLMServer()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(autouse=True)
def _clean_shared():
    """用例前后重置共享实例，避免用例间互相污染。"""
    close_shared_client()
    yield
    close_shared_client()


@pytest.fixture
def wired(mock_llm, monkeypatch):
    """把整个应用指向本地 mock 服务器，并准备好 store 与 QA 开关。

    用 monkeypatch 而非直接赋值 settings，保证用例结束后配置自动还原。
    """
    monkeypatch.setattr(settings, "vlm_base_url", mock_llm.base_url)
    monkeypatch.setattr(settings, "dashscope_api_key", "test-key")
    monkeypatch.setattr(settings, "aliyun_api_key", "test-key")
    monkeypatch.setattr(settings, "vlm_max_retries", 0)
    # conftest 把 EMBEDDING_PROVIDER 固定为 local_hash（让绝大多数用例
    # 不依赖真实上游、结果可复现）。本组用例要验证的恰恰是 dashscope 这条路径
    # （共享客户端被关停后 provider 不得持有失效引用），因此需显式改回 dashscope。
    monkeypatch.setattr(settings, "embedding_provider", "dashscope")
    get_embedding.cache_clear()  # 让 embedding 单例按新 base_url 重建

    # store 里换成指向 mock 服务器的 DashScopeEmbedding
    monkeypatch.setattr(
        vs_module, "_store", VectorStore(embedder=DashScopeEmbedding())
    )
    monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
    monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")
    return mock_llm


# --------------------------------------------------------------------------
# 1. 单例语义（单元层）
# --------------------------------------------------------------------------
def test_shared_client_is_singleton(wired) -> None:
    assert get_shared_client() is get_shared_client(), "共享客户端必须是同一个实例"


def test_default_factory_returns_shared_client(wired) -> None:
    """默认参数下 build_client_from_settings 返回共享实例。

    embedding / vlm 都走这个入口，因此它们无需改一行代码，
    也会自动复用同一个连接池。
    """
    assert build_client_from_settings() is get_shared_client()


def test_explicit_transport_gets_independent_client(wired) -> None:
    """显式传 transport 时必须构造独立实例，绝不污染共享实例。

    否则某个测试用例注入的 MockTransport 会泄漏给其它用例，
    造成"单独跑通过、一起跑失败"的诡异现象。
    """
    import httpx

    shared = get_shared_client()
    independent = build_client_from_settings(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    assert independent is not shared
    assert build_client_from_settings() is shared, "共享实例不应被覆盖"


def test_shared_client_rebuilds_after_close(wired) -> None:
    """自愈：共享客户端被关掉后，下次获取应重建而非返回废池。

    防呆措施 —— 万一有人误用 `with get_shared_client() as c:`，
    进程不应带着一个已关闭的连接池继续跑。
    """
    first = get_shared_client()
    first.close()
    assert first.is_closed is True

    second = get_shared_client()
    assert second is not first, "应重建而非返回已关闭的旧实例"
    assert second.is_closed is False


# --------------------------------------------------------------------------
# 2. 硬证明：N 次请求只建立 1 条 TCP 连接
# --------------------------------------------------------------------------
def test_many_requests_share_one_tcp_connection(wired) -> None:
    """核心断言：一整轮「入库 + 多次问答」只应建立 1 条 TCP 连接。

    这是连接复用的直接度量。改造前每请求新建客户端、请求结束即 close，
    连接数会随请求数线性增长；改造后 embedding 与 QA 共用同一个池，全程 1 条。
    """
    with TestClient(app) as c:
        c.post(
            "/api/documents/text",
            json={
                "content": "《编译原理》课程设计答辩\n时间：12月3日 14:00\n",
                "filename": "reuse.txt",
            },
        )
        after_ingest = wired.connections

        # 连续 5 次问答：每次 1 次 query embedding + 1 次 chat
        for i in range(5):
            resp = c.post("/api/qa", json={"query": f"答辩时间 {i}", "top_k": 3})
            assert resp.status_code == 200
            assert resp.json()["degraded"] is False, "mock 服务器应返回正常答案"

    assert wired.connections == 1, (
        f"整轮只应建立 1 条 TCP 连接（keep-alive 复用），实际 {wired.connections} 条 —— "
        "说明客户端或连接池被反复重建"
    )
    assert wired.connections == after_ingest, "问答阶段不应新建任何连接"
    assert wired.requests >= 6, f"应有入库与问答的多次请求，实际 {wired.requests}"


def test_request_does_not_close_shared_client(wired) -> None:
    """请求结束后共享客户端必须仍可用（证明 qa.py 没有把它 close 掉）。

    这是本次改造最容易踩的坑：残留的 `with client` 会在请求结束时关闭共享连接池，
    导致每个请求实际都在重建连接 —— 改动形同虚设，且连接数照旧线性增长。
    """
    with TestClient(app) as c:
        assert c.post("/api/qa", json={"query": "答辩", "top_k": 3}).status_code == 200
        assert get_shared_client().is_closed is False, "请求结束后共享客户端被关闭了"
        assert c.post("/api/qa", json={"query": "通知", "top_k": 3}).status_code == 200
    assert wired.connections == 1


# --------------------------------------------------------------------------
# 3. 与降级链衔接：上游故障时仍复用同一客户端
# --------------------------------------------------------------------------
def test_server_error_degrades_without_rebuilding_client(wired, monkeypatch) -> None:
    """上游 5xx 触发降级时，不应因此重建客户端。

    降级发生在客户端**内部**（重试耗尽 → 抛 LLMError → 路由层兜底），
    连接池本身无需重建；若这里连接数暴涨，说明降级路径泄漏了客户端。
    """
    def failing_chat(self, *args, **kwargs):
        raise llm.ServerError("模拟上游 503", 503)

    monkeypatch.setattr(OpenAICompatClient, "chat", failing_chat)

    with TestClient(app) as c:
        for _ in range(3):
            body = c.post("/api/qa", json={"query": "答辩时间", "top_k": 3}).json()
            assert body["degraded"] is True, "上游故障应降级"
            assert body["answer"].strip(), "降级仍应给出可用回答"

    assert wired.connections == 1, (
        f"降级路径不应重建客户端，实际 {wired.connections} 条连接"
    )


# --------------------------------------------------------------------------
# 4. 生命周期：shutdown 释放
# --------------------------------------------------------------------------
def test_shutdown_releases_shared_client(wired) -> None:
    """应用关停后共享实例应被释放，避免 --reload 反复重启残留连接。"""
    with TestClient(app):
        client = get_shared_client()
        assert client.is_closed is False
    assert client.is_closed is True, "关停时应关闭共享连接池"

    rebuilt = get_shared_client()
    assert rebuilt.is_closed is False, "关停后再获取应能重建"


# --------------------------------------------------------------------------
# 5. 回归：共享客户端被关停后，provider 不得持有失效引用
#    这是「连接复用」与「embedding 回退/向量索引」两处改动最容易撞车的地方
# --------------------------------------------------------------------------
def test_embedder_survives_shared_client_close(wired) -> None:
    """共享客户端被关停后，embedder 仍应正常工作（不得降级到 local_hash）。

    背景：provider 若在 __init__ 里把客户端存成普通属性，共享实例一旦被
    关停/重建，它手里就是一份失效引用 —— 表现为 embedding 持续失败、
    粘性降级到 local_hash(256 维)，与已建索引(1024 维)不符，**检索静默返回 0 条**。
    这类故障在集成测试里表现为"sematic_search 突然召回为空"，极难定位。

    修法：provider 通过 SharedClientRef 描述符**按需解析**共享客户端。
    """
    embedder = get_embedding()
    assert embedder.active_name == "dashscope", "前置条件：应使用 dashscope"

    # 模拟应用关停（或测试夹具重置）关闭共享客户端
    close_shared_client()

    vec = embedder.embed(["答辩时间"])
    assert vec.shape[1] == _EMBED_DIM, f"应仍产出 {_EMBED_DIM} 维向量，实际 {vec.shape}"
    assert embedder.degraded is False, "共享客户端重建后不应残留降级状态"
    assert embedder.active_name == "dashscope", "不应降级到 local_hash"


def test_search_still_works_after_shared_client_close(wired) -> None:
    """端到端：关停共享客户端后，检索仍能正常召回（防"静默返空"）。"""
    with TestClient(app) as c:
        c.post(
            "/api/documents/text",
            json={"content": "《数字逻辑》实验验收\n时间：12月10日 13:30\n", "filename": "survive.txt"},
        )
        before = c.post("/api/search", json={"query": "实验验收", "top_k": 3}).json()
        assert before["hits"], "前置条件：应能召回"

    # 模拟服务重启（关停 → 下次请求自动重建）
    close_shared_client()

    with TestClient(app) as c:
        after = c.post("/api/search", json={"query": "实验验收", "top_k": 3}).json()
        assert after["hits"], "关停重建后检索不应静默返回空"
        assert get_embedding().degraded is False


# --------------------------------------------------------------------------
# 5. 启动健壮性：预热失败不得阻断启动
# --------------------------------------------------------------------------
def test_app_starts_when_prewarm_fails(monkeypatch) -> None:
    """回归：LLM 客户端预热失败时，应用仍须正常启动。

    背景（真实回归）：lifespan 里曾**无条件** `get_shared_client()` 做预热，
    而未配置 API Key 时构造函数会抛 AuthError —— 导致应用根本起不来。
    后果不只是少个优化：所有依赖 TestClient 启动应用的用例会**整体 error**，
    且违背项目「无 Key 也能跑通全链路（问答走抽取式降级）」的既定前提。

    这里把预热打成必然失败，断言应用仍可启动且 /health 正常。
    """
    from app.providers.llm_client import AuthError

    def _boom():
        raise AuthError("缺少 API Key：请在 .env 中配置 ALIYUN_API_KEY")

    # main.py 直接 import 了该名字，需打在 main 模块上
    monkeypatch.setattr("app.main.get_shared_client", _boom)

    with TestClient(app) as c:  # 不应抛异常
        resp = c.get("/health")
        assert resp.status_code == 200, "预热失败不应阻断应用启动"
        assert resp.json()["status"] == "ok"

    # 预热失败后，无 Key 路径下的问答应走抽取式降级而非 500
    with TestClient(app) as c:
        monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "")
        resp = c.post("/api/qa", json={"query": "随便问一句", "top_k": 1})
        assert resp.status_code == 200, f"无 Key 时问答不应报错：{resp.text[:200]}"
