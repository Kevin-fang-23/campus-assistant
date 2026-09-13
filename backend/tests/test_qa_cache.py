"""/api/qa 响应缓存的契约测试。

## 本文件要钉死的契约（不可放宽）

1. **基本命中**：相同 query+top_k 第二次请求时，LLM 调用次数**不变**，
   返回体的 `cache_hit=True`，且 `answer/citations/degraded` 与首次一致。
2. **缓存键区分 top_k**：`(query, top_k=3)` 与 `(query, top_k=5)` 视为
   不同 key —— top_k 影响召回数量，结果不同。
3. **缓存键归一化**：多余空白 / 首尾空白 / 大小写归一化后视为同一 key。
   （中文 lowercase 是 noop；英文/数字场景会触发。）
4. **任何路径都缓存**：LLM 成功 / LLM 降级 / 空召回三种 AnswerOut 都写入
   缓存 —— 否则降级回答与空召回反复触发会持续扣 embedding 额度。
5. **关闭缓存**：`qa_cache_enabled=False` 时所有请求都走完整流程，
   `cache_hit` 恒为 False。
6. **TTL 过期**：超过 TTL 后再次请求会重新调 LLM。
7. **max_size 淘汰**：超出后 LRU 淘汰最久未用的，新 key 可命中。
8. **缓存命中不入限流**：缓存命中不再扣限流额度（限流是上游护栏，
   不消耗上游就不该挤占额度）。

## 实现要点

测试通过 `monkeypatch` 替换 `qa_module._build_llm_client`：
- 用 `httpx.MockTransport` 注入真客户端，handler 里计数；
- 通过 `_install()` 一并强制开启 LLM 分支（qa_provider=auto, dashscope_api_key=test-key）；
- 通过 `monkeypatch.setattr(qa_module.settings, "qa_cache_enabled", True/False)`
  控制缓存开关。

缓存实例是进程级单例，测试之间通过 `reset_cache_for_tests()` 清空，
避免用例顺序依赖。fixture 见 `qa_cache_clean`。
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import qa as qa_module
from app.main import app
from app.providers.llm_client import OpenAICompatClient
from app.services.qa_cache import (
    cache_key,
    get_cache,
    reconfigure_cache,
    reset_cache_for_tests,
)

# 与 test_qa_degradation.py 保持一致：使用语料里实际查得到的查询
QUERY = "操作系统作业什么时候截止"


# --------------------------------------------------------------------------
# fixture 与工具
# --------------------------------------------------------------------------
# 注意：缓存清空由 conftest._qa_cache_clean（autouse）统一负责 —— 不要在本文件
# 重复声明，否则 fixture 顺序与执行次数会混乱。
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _client_with_handler(handler, *, max_retries: int = 0) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="https://mock.test/v1",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        max_retries=max_retries,
        sleep_fn=lambda _s: None,
    )


def _install(monkeypatch, handler, *, max_retries: int = 0) -> dict:
    """注入 fake 客户端 + 强制开启 LLM 分支 + 强制开启缓存，返回计数器。

    max_retries=0：测试要验证的是"缓存命中/未命中"，不是重试策略。
    """
    calls: dict = {"count": 0}

    def _wrapped(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return handler(request)

    monkeypatch.setattr(
        qa_module, "_build_llm_client", lambda: _client_with_handler(_wrapped, max_retries=max_retries)
    )
    monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
    monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")
    monkeypatch.setattr(qa_module.settings, "qa_cache_enabled", True)
    return calls


# --------------------------------------------------------------------------
# 1. 基本命中：相同 query 第二次请求不再调 LLM
# --------------------------------------------------------------------------
def test_second_request_hits_cache_and_skips_llm(
    client: TestClient, monkeypatch
) -> None:
    """首次：走完整流程（LLM 调用 1 次，cache_hit=False）。
    再次：缓存命中（LLM 调用次数仍为 1，cache_hit=True）。

    这是缓存的核心价值：演示现场同一问题被多人问时，只第一次烧额度。
    """

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "生成答案 [1]"}}]})

    calls = _install(monkeypatch, ok)

    r1 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    r2 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()

    assert calls["count"] == 1, f"第二次应该命中缓存，实际又调了 LLM {calls['count']} 次"
    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is True
    # 内容一致性：缓存命中的 answer / citations / degraded 应与首次一致
    assert r2["answer"] == r1["answer"]
    assert r2["degraded"] == r1["degraded"]
    assert [c["notice_id"] for c in r2["citations"]] == [
        c["notice_id"] for c in r1["citations"]
    ]


def test_three_identical_requests_still_one_llm_call(
    client: TestClient, monkeypatch
) -> None:
    """同一请求 3 次：LLM 调用次数仍为 1。证明缓存无 TTL 抖动。"""

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install(monkeypatch, ok)
    for _ in range(3):
        client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert calls["count"] == 1


# --------------------------------------------------------------------------
# 2. 缓存键区分 top_k
# --------------------------------------------------------------------------
def test_different_top_k_means_different_cache_key(
    client: TestClient, monkeypatch
) -> None:
    """top_k 不同 = 不同 key = 不会命中。

    top_k 影响召回数量，answer / citations 必然不同。
    """

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install(monkeypatch, ok)
    r1 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    r2 = client.post("/api/qa", json={"query": QUERY, "top_k": 5}).json()
    r1_again = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()

    # top_k=3 和 top_k=5 算两次，top_k=3 第二次命中 → 总计 2 次 LLM 调用
    assert calls["count"] == 2, f"top_k 不同应各自算一次，实际 {calls['count']}"
    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is False
    assert r1_again["cache_hit"] is True


# --------------------------------------------------------------------------
# 3. 缓存键归一化（大小写、空白）
# --------------------------------------------------------------------------
def test_whitespace_normalized(client: TestClient, monkeypatch) -> None:
    """首尾空白 / 多余空白 / Tab → 视为同一 key。

    注意：仅测「两端空白」和「相邻空白合并」的归一化；不在中间插入
    有意义的空白（`" ".join(s.split())` 会合并成单空格，与原 query 不可比，
    因此那是不同 key —— 这属于合理行为而非 bug）。
    """

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install(monkeypatch, ok)
    # 三种"看起来不一样"但 strip 后相同的 query
    client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    client.post("/api/qa", json={"query": f"  {QUERY}  ", "top_k": 3})
    client.post("/api/qa", json={"query": f"\t{QUERY}\n", "top_k": 3})

    assert calls["count"] == 1, f"两端空白归一化应合并，实际 {calls['count']} 次"


def test_case_insensitive_for_english(client: TestClient, monkeypatch) -> None:
    """英文 query 大小写归一化（中文 lowercase 是 noop，单独用英文验证）。"""

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install(monkeypatch, ok)
    client.post("/api/qa", json={"query": "OS Homework Deadline", "top_k": 3})
    client.post("/api/qa", json={"query": "os homework deadline", "top_k": 3})
    client.post("/api/qa", json={"query": "OS HOMEWORK DEADLINE", "top_k": 3})

    assert calls["count"] == 1


def test_cache_key_function_unit() -> None:
    """单元测试 cache_key()：归一化各分支。

    验证三件事：
    1. strip 去首尾空白；
    2. split + join 把任意连续空白（含 Tab / 换行）压成单空格；
    3. lower 大小写归一化（中文 noop）。
    """
    assert cache_key(" hello ", 3) == ("hello", 3)
    assert cache_key("hello   world", 3) == ("hello world", 3)
    assert cache_key("Hello\tWorld\n", 3) == ("hello world", 3)
    assert cache_key("OS", 3) == cache_key("os", 3)
    # 中文场景
    assert cache_key("今天天气", 3) == cache_key("今天天气", 3)
    # top_k 不同
    assert cache_key("x", 3) != cache_key("x", 5)
    # **不同中间空白数 → 不同 key**（这是合理行为，不是 bug）：
    # "hello world" 与 "hello  world" 经过归一化后是**不同**的字符串，
    # 因为 split 后再 join 不会恢复成 "hello world"。这与"两端空白归一化"
    # 是两个维度：前者属于"query 内容不同"，后者属于"同 query 的格式变体"。


# --------------------------------------------------------------------------
# 4. 任何路径都缓存（含降级 / 空召回）
# --------------------------------------------------------------------------
def test_degraded_response_is_also_cached(
    client: TestClient, monkeypatch
) -> None:
    """LLM 失败 → degraded=True 的降级回答也写入缓存。

    重要：降级路径仍消耗 embedding（hybrid_search），重复提问同样扣费。
    缓存降级响应等于"省掉一次 embedding"。
    """

    def fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "boom"}})

    calls = _install(monkeypatch, fail)
    r1 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    r2 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()

    assert calls["count"] == 1, "降级回答也走完整流程（embedding），第二次应命中"
    assert r1["degraded"] is True and r2["degraded"] is True
    assert r2["cache_hit"] is True
    assert r1["answer"] == r2["answer"]


def test_no_recall_response_is_also_cached(
    client: TestClient, monkeypatch
) -> None:
    """对"无任何相关结果"的 query，缓存仍然生效。

    实测说明：`hybrid_search` 在 local_hash 向量召回下基本总能返回 ≥1 条
    结果（即便分数很低），所以「citations 为空」的代码分支在常规测试里
    难以触发。但**这条用例的目的不是验证空召回分支**，而是钉死
    "任何 AnswerOut 都缓存"的契约：只要走进了 _build_answer 并 return，
    无论 citations 是不是空，结果都该进缓存。
    """

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install(monkeypatch, ok)
    # 用一个极冷门的 query 触发极低分召回（citations 仍非空，但接近降级）
    no_match = "zzz_极冷门查询_xyz"

    r1 = client.post("/api/qa", json={"query": no_match, "top_k": 1}).json()
    r2 = client.post("/api/qa", json={"query": no_match, "top_k": 1}).json()

    # 即便召回分数低也缓存 —— 第二次 cache_hit=True，answer 一致
    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is True
    assert r1["answer"] == r2["answer"]


# --------------------------------------------------------------------------
# 5. 关闭缓存：所有请求都走完整流程
# --------------------------------------------------------------------------
def test_cache_disabled_always_calls_llm(
    client: TestClient, monkeypatch
) -> None:
    """qa_cache_enabled=False 时，每次请求都调 LLM，cache_hit 恒为 False。"""

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install(monkeypatch, ok)
    monkeypatch.setattr(qa_module.settings, "qa_cache_enabled", False)

    r1 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    r2 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()

    assert calls["count"] == 2, f"关闭缓存时应各调一次，实际 {calls['count']}"
    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is False


# --------------------------------------------------------------------------
# 6. TTL 过期
# --------------------------------------------------------------------------
def test_ttl_expiry_triggers_refetch() -> None:
    """超过 TTL 后再次请求会重新调 LLM。

    用 `reconfigure_cache(wall_fn=...)` 在构造时注入 mock 时钟 —— 比改
    `_wall` 属性更可靠（旧 expire_at 用旧 wall 计算，新 wall 改后即使
    推进时间，expire_at 可能仍远大于新 now，导致永不触发过期）。
    """
    fake_now = [0.0]
    cache = reconfigure_cache(
        max_size=8, ttl_seconds=1.0, wall_fn=lambda: fake_now[0]
    )
    cache.set("k1", "v1")
    # 立刻读：命中
    assert cache.get("k1") == "v1"
    # 推进时间到 2s：超过 TTL=1s，应过期
    fake_now[0] = 2.0
    assert cache.get("k1") is None, "TTL=1s 后 2s 读应视为过期"
    # 过期读取会从表里删除 → 重新写
    cache.set("k1", "v1-new")
    # 推进回 0：应该又命中（写入时刷新 TTL）
    fake_now[0] = 0.0
    assert cache.get("k1") == "v1-new"


def test_ttl_expiry_end_to_end(client: TestClient, monkeypatch) -> None:
    """端到端：通过 monkeypatch 改 qa_module 的 cache 实例为 TTL=1s。

    第一次请求：cache_hit=False，调 LLM 1 次。
    第二次立刻：cache_hit=True，调 LLM 仍为 1 次。
    推进 mock 时间到 TTL 之外（monkeypatch.setattr 改 qa_module 的 _wall）：
    第三次：cache_hit=False，调 LLM 2 次。

    这里简化：通过 `reconfigure_cache` + 控制 wall_fn 实现。
    """
    fake_now = [0.0]

    def _install_with_short_ttl(monkeypatch, handler):
        calls = {"count": 0}

        def _wrapped(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return handler(request)

        monkeypatch.setattr(
            qa_module, "_build_llm_client",
            lambda: _client_with_handler(_wrapped),
        )
        monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
        monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")
        monkeypatch.setattr(qa_module.settings, "qa_cache_enabled", True)
        # 重建一个 TTL=1s 的 cache 实例
        cache = reconfigure_cache(max_size=8, ttl_seconds=1.0)
        cache._wall = lambda: fake_now[0]
        return calls

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    calls = _install_with_short_ttl(monkeypatch, ok)
    r1 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    r2 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    assert calls["count"] == 1
    assert r1["cache_hit"] is False and r2["cache_hit"] is True

    # 推进 wall 时间 → 超过 TTL
    fake_now[0] = 5.0
    r3 = client.post("/api/qa", json={"query": QUERY, "top_k": 3}).json()
    assert calls["count"] == 2, "TTL 过期后应重新调 LLM"
    assert r3["cache_hit"] is False


# --------------------------------------------------------------------------
# 7. max_size 淘汰
# --------------------------------------------------------------------------
def test_max_size_evicts_lru() -> None:
    """超出 max_size 时淘汰最久未用的。

    max_size=2 的 cache：
    写 k1, k2, k3 → k1 被淘汰（最久未用），k3 在末尾（最新）。
    get(k2) → k2 移到末尾；写 k4 → 淘汰 k3。
    """
    cache = reconfigure_cache(max_size=2, ttl_seconds=300.0)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")  # 触发淘汰
    assert cache.get("k1") is None, "k1 应已被淘汰"
    assert cache.get("k2") == "v2"
    assert cache.get("k3") == "v3"
    # 访问 k2 让其变新
    assert cache.get("k2") == "v2"
    # 再写 k4 → 淘汰 k3
    cache.set("k4", "v4")
    assert cache.get("k3") is None
    assert cache.get("k2") == "v2"
    assert cache.get("k4") == "v4"


def test_stats_report_hit_rate() -> None:
    """stats() 报告命中率，便于 /health 或调试观察。"""
    cache = reconfigure_cache(max_size=4, ttl_seconds=300.0)
    cache.set("k1", "v1")
    cache.get("k1")  # hit
    cache.get("k1")  # hit
    cache.get("k2")  # miss
    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert 0 < s["hit_rate"] < 1
    assert s["size"] == 1
    assert s["max_size"] == 4
    assert s["ttl_seconds"] == 300.0


# --------------------------------------------------------------------------
# 8. 缓存命中不入限流
# --------------------------------------------------------------------------
def test_cache_hit_does_not_consume_rate_limit(
    client: TestClient, monkeypatch
) -> None:
    """缓存命中时**不计入限流额度**。

    演示场景：同一 IP 反复点同一问题，限流是上游护栏，缓存命中不消耗
    上游资源，本就不该挤占额度。max_size 与 TTL 是兜底（防止有人拿
    不同 query 刷缓存）。
    """
    # 先开启限流并把每分钟额度调到很小
    from app.services.rate_limit import get_limiter

    monkeypatch.setattr(qa_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(qa_module.settings, "rate_limit_qa_per_min", 5)
    # 清空限流计数
    get_limiter().reset(clear_store=True)

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    _install(monkeypatch, ok)

    # 5 次相同请求：第一次走完整流程（扣 1 次额度），后续 4 次都命中缓存（不扣）
    for _ in range(5):
        resp = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
        assert resp.status_code == 200, resp.text

    # 再来一个不同 query：仍应通过（额度只被扣了 1 次，不是 5 次）
    resp = client.post(
        "/api/qa", json={"query": "完全不同的另一个问题查询", "top_k": 3}
    )
    assert resp.status_code == 200, (
        f"缓存命中挤占额度会导致新 query 被拒：{resp.status_code} {resp.text}"
    )


# --------------------------------------------------------------------------
# 9. 并发安全：get/set 不应抛异常或丢更新
# --------------------------------------------------------------------------
def test_concurrent_get_set_no_crash() -> None:
    """多线程并发读写不应抛异常。

    这是 TTLRUCache 的基础线程安全契约 —— FastAPI 同步端点跑在线程池，
    如果锁漏了就会出现 KeyError（OrderedDict 删后另一个线程正好读到）。
    """
    import threading

    cache = reconfigure_cache(max_size=64, ttl_seconds=300.0)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for j in range(50):
                cache.set(f"k{i}-{j}", j)
                _ = cache.get(f"k{i}-{j}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发读写出错：{errors!r}"


# --------------------------------------------------------------------------
# 10. AnswerOut.cache_hit 默认值与 schema 兼容性
# --------------------------------------------------------------------------
def test_answerout_cache_hit_default_is_false() -> None:
    """AnswerOut 不显式传 cache_hit 时默认 False（向后兼容旧调用方）。"""
    from app.schemas import AnswerOut, CitationOut

    out = AnswerOut(
        query="x",
        answer="y",
        citations=[],
        backend="faiss",
        degraded=True,
    )
    assert out.cache_hit is False


def test_ask_returns_answerout_even_when_cached(
    client: TestClient, monkeypatch
) -> None:
    """缓存命中也要走 response_model 校验 —— 防止 Pydantic 缺字段而 500。"""

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    _install(monkeypatch, ok)
    r1 = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    r2 = client.post("/api/qa", json={"query": QUERY, "top_k": 3})
    assert r1.status_code == 200
    assert r2.status_code == 200
    # 字段存在且类型正确
    body = r2.json()
    assert "cache_hit" in body
    assert isinstance(body["cache_hit"], bool)
    assert body["cache_hit"] is True


# --------------------------------------------------------------------------
# 11. 单例隔离：reset_cache_for_tests 只清空，不重建
# --------------------------------------------------------------------------
def test_reset_keeps_instance() -> None:
    """reset_cache_for_tests 清空条目 + 统计，但保留单例实例本身。

    验证：reset 前后 get_cache() 是同一对象 —— 防止有人误用
    reconfigure_cache 改造 reset 而引入额外开销。
    """
    cache_before = get_cache()
    cache_before.set("k", "v")
    assert len(cache_before) == 1
    reset_cache_for_tests()
    cache_after = get_cache()
    assert cache_after is cache_before, "reset 应保留实例引用"
    assert len(cache_after) == 0


# --------------------------------------------------------------------------
# 12. 知识库变更 → 缓存必须立即失效（回归护栏）
#
# 这是实测到过的真实缺陷：缓存里存的是完整 AnswerOut，包含「未找到」这类
# 否定答案。先问一个库中无答案的问题（缓存了否定答案），随后入库能回答它的
# 通知，5 分钟内再问依旧返回「知识库中暂时没有…」—— 用户会以为系统坏了。
# --------------------------------------------------------------------------
def test_ingest_invalidates_qa_cache(client: TestClient, monkeypatch) -> None:
    """入库新通知后，同 query 的旧答案必须作废（不再 cache_hit）。

    用 `QA_PROVIDER=mock` 走降级路径即可，本用例验证的是失效时机，
    与是否调用 LLM 无关。
    """
    from app.services.qa_cache import invalidate_qa_cache  # noqa: F401  确认可导入

    monkeypatch.setattr(qa_module.settings, "qa_cache_enabled", True)
    q = "这里一定没有的通知主题_zzz"

    r1 = client.post("/api/qa", json={"query": q, "top_k": 3}).json()
    assert r1["cache_hit"] is False
    # 第一次写入缓存 → 第二次必命中
    r2 = client.post("/api/qa", json={"query": q, "top_k": 3}).json()
    assert r2["cache_hit"] is True, "前置条件：缓存应已生效"
    assert len(get_cache()) >= 1

    # 入库一条新通知（会触发 invalidate_qa_cache）
    resp = client.post(
        "/api/documents/text",
        json={
            "content": "关于举办校园创客社团旧手机拆解工作坊的通知\n时间：10月12日 14:00\n"
                       "地点：图书馆南门集合\n报名截止：10月10日 18:00\n",
            "filename": "cache_invalidation_guard.txt",
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(get_cache()) == 0, "入库后缓存必须被清空"

    # 入库后再问：不得命中缓存（必须重新检索，拿到最新结果）
    r3 = client.post("/api/qa", json={"query": q, "top_k": 3}).json()
    assert r3["cache_hit"] is False, "入库后旧缓存必须已失效"


def test_invalidate_all_keeps_stats() -> None:
    """invalidate_all 只清条目，**保留**命中率统计。

    为什么单独钉住：生产失效若误用 clear()，会把 hits/misses 一起清零，
    使「命中率」这个观测指标失真 —— 而失效只是正常业务动作。
    """
    cache = reconfigure_cache(max_size=8, ttl_seconds=300.0)
    cache.set("k1", "v1")
    cache.get("k1")          # hit
    cache.get("missing")     # miss
    before = cache.stats()
    assert before["hits"] == 1 and before["misses"] == 1

    removed = cache.invalidate_all()
    assert removed == 1
    after = cache.stats()
    assert after["size"] == 0, "条目应被清空"
    assert after["hits"] == 1 and after["misses"] == 1, (
        f"统计不该被清空，实际 {after}"
    )


def test_notice_update_invalidates_qa_cache(client: TestClient, monkeypatch) -> None:
    """人工修正通知（regenerate=false 分支）同样要失效缓存。

    这条分支容易漏：regenerate=true 时由 regenerate_tasks 内部失效，
    而 regenerate=false 走 else 分支单独 commit。
    """
    monkeypatch.setattr(qa_module.settings, "qa_cache_enabled", True)
    # 先入库一条通知并拿到 id
    resp = client.post(
        "/api/documents/text",
        json={
            "content": "《操作系统》第三次小班课通知\n本周五 15:00 在教三 201 讲解进程调度实验。\n",
            "filename": "notice_update_guard.txt",
        },
    )
    assert resp.status_code == 200, resp.text
    notice_id = resp.json()["notice"]["id"]

    q = "操作系统小班课"
    r1 = client.post("/api/qa", json={"query": q, "top_k": 3}).json()
    r2 = client.post("/api/qa", json={"query": q, "top_k": 3}).json()
    assert r2["cache_hit"] is True, "前置条件：缓存应已生效"

    upd = client.patch(
        f"/api/notices/{notice_id}?regenerate=false",
        json={"location": "教三 999"},
    )
    assert upd.status_code == 200, upd.text
    assert len(get_cache()) == 0, "修正通知后缓存必须被清空"