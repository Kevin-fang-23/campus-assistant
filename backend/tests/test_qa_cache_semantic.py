"""/api/qa 缓存「语义相似匹配」的契约测试。

## 背景

字面归一化识别不了「同一意图的不同问法」：
`作业什么时候截止` 与 `作业截止时间是什么` 是两个不同的 key，
会各走一次完整流程（各烧一次 LLM+embedding），而答案是同一份。
本文件钉住新加的语义近邻判定路径。

## 本文件要钉死的契约（不可放宽）

1. **改写问法命中**：先问 A，再用改写问法 B 问，**不再调 LLM**，
   `cache_hit=True`，且响应头 `X-Cache: HIT_SEMANTIC` 与相似度可见。
2. **阈值 0 = 关闭**：完全退回字面归一化（B 必须重新走完整流程），
   保证可随时回滚。
3. **只比对同 top_k 的条目**：`top_k` 不同的问法不得互相命中
   （否则用户拿到的引用列表与他请求的参数不符）。
4. **精确 key 优先**：同一问法重复问走零成本的精确路径（`X-Cache: HIT`），
   不会为了语义匹配先算一次向量。
5. **维度守卫**：与查询向量维度不同的缓存条目不得参与比对 ——
   切换 embedding 后端后缓存里会短暂留着旧维度的向量，
   跨维度算出的"余弦"无意义，若被判命中就是拿错答案。
6. **过期条目不参与语义匹配**：TTL 到期后不能再被近邻命中。
7. **语义命中仍计入限流**：它消耗一次 embedding（真实上游开销），
   与"精确命中零成本故不入限流"是两条不同的规则。
8. **向量复用**：为语义探测算出的查询向量必须透传给检索层，
   同一请求不得重复调用 embedding。

## 实现要点

- 集成用例沿用 test_qa_cache.py 的手法：monkeypatch `_build_llm_client`
  注入 `httpx.MockTransport` 并计数，`_install()` 一并开启 LLM 分支与缓存。
- 单元用例直接操作 `QaSemanticCache`（`reconfigure_cache` 拿到干净实例），
  用构造好的向量精确控制相似度，不依赖 embedding 后端的数值。
- 缓存单例的清空由 `conftest._qa_cache_clean`（autouse）统一负责，
  本文件**不要**重复声明该 fixture。
"""
from __future__ import annotations

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import qa as qa_module
from app.main import app
from app.providers.llm_client import OpenAICompatClient
from app.services import hybrid as hybrid_module
from app.services.qa_cache import (
    get_cache,
    reconfigure_cache,
)

# 需求里给出的原例：这是本功能存在的理由。
QUERY_A = "作业什么时候截止"
QUERY_B = "作业截止时间是什么"

# 生产默认阈值（dashscope 上标定为 0.86）。local_hash 下该值远高于实测相似度，
# 所以集成用例显式下调阈值 —— 用例要验的是**机制**，不是某个后端的数值；
# 真机上的阈值标定由 eval/run_cache_threshold_eval.py 负责。
LOW_THRESHOLD = 0.5


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


def _install(monkeypatch, *, threshold: float = LOW_THRESHOLD) -> dict:
    """注入 fake LLM 客户端 + 开启缓存 + 设定语义阈值，返回调用计数器。"""

    def _ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "生成答案 [1]"}}]}
        )

    calls: dict = {"count": 0}

    def _wrapped(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _ok(request)

    monkeypatch.setattr(
        qa_module, "_build_llm_client", lambda: _client_with_handler(_wrapped)
    )
    monkeypatch.setattr(qa_module.settings, "qa_provider", "auto")
    monkeypatch.setattr(qa_module.settings, "dashscope_api_key", "test-key")
    monkeypatch.setattr(qa_module.settings, "qa_cache_enabled", True)
    monkeypatch.setattr(
        qa_module.settings, "qa_cache_semantic_threshold", threshold
    )
    return calls


def _similarity(a: str, b: str) -> float:
    """用当前（测试环境为 local_hash）后端算两句的余弦相似度。

    刻意在用例里现算而不是抄一个魔数：这样"前提条件"是自证的 ——
    若 embedding 实现变了、相似度掉到阈值以下，用例会以
    "前置条件不成立"失败，而不是含糊地报"没命中缓存"。
    """
    from app.services.vector_store import get_store

    store = get_store()
    va, vb = store.embed(a), store.embed(b)
    va = va / np.linalg.norm(va)
    vb = vb / np.linalg.norm(vb)
    return float(np.dot(va, vb))


# --------------------------------------------------------------------------
# 1. 核心收益：改写问法命中，不再调 LLM
# --------------------------------------------------------------------------
def test_paraphrase_hits_semantic_cache_and_skips_llm(
    client: TestClient, monkeypatch
) -> None:
    """先问 A，再用改写问法 B 提问 → 第二次不调 LLM，cache_hit=True。

    这是整个功能的存在理由：演示现场招聘官用不同措辞问同一件事时，
    只该烧一次额度。
    """
    sim = _similarity(QUERY_A, QUERY_B)
    assert sim >= LOW_THRESHOLD, (
        f"前置条件不成立：两句在 local_hash 下相似度仅 {sim:.3f} < 阈值 "
        f"{LOW_THRESHOLD}，本用例无法验证语义命中"
    )
    calls = _install(monkeypatch)

    r1 = client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})
    r2 = client.post("/api/qa", json={"query": QUERY_B, "top_k": 3})

    assert r1.status_code == 200 and r2.status_code == 200
    assert calls["count"] == 1, (
        f"改写问法应复用缓存，实际调了 LLM {calls['count']} 次"
    )
    body1, body2 = r1.json(), r2.json()
    assert body1["cache_hit"] is False
    assert body2["cache_hit"] is True
    # 复用的是**同一份答案与引用**
    assert body2["answer"] == body1["answer"]
    assert [c["notice_id"] for c in body2["citations"]] == [
        c["notice_id"] for c in body1["citations"]
    ]
    # 可观测性：能看出走的是语义路径而不是精确路径
    assert r2.headers.get("X-Cache") == "HIT_SEMANTIC"
    assert float(r2.headers["X-Cache-Similarity"]) >= LOW_THRESHOLD


def test_exact_hit_wins_and_reports_exact(
    client: TestClient, monkeypatch
) -> None:
    """同一问法重复问 → 走精确路径（X-Cache: HIT），不是语义路径。

    顺序很重要：精确命中零成本，语义命中要算一次向量。
    把精确放在前面，才能让最常见的场景（同一问题被反复点）保持最快。
    """
    calls = _install(monkeypatch)
    client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})
    r2 = client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})

    assert calls["count"] == 1
    assert r2.json()["cache_hit"] is True
    assert r2.headers.get("X-Cache") == "HIT"


# --------------------------------------------------------------------------
# 2. 阈值 0 = 关闭（可回滚）
# --------------------------------------------------------------------------
def test_threshold_zero_disables_semantic_match(
    client: TestClient, monkeypatch
) -> None:
    """阈值 0 时完全退回字面归一化：改写问法必须重新走完整流程。

    这是回滚开关 —— 语义匹配若在生产上表现异常，把阈值配 0 即可回到
    改造前的行为，不需要改代码或回滚发版。
    """
    calls = _install(monkeypatch, threshold=0.0)
    client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})
    r2 = client.post("/api/qa", json={"query": QUERY_B, "top_k": 3})

    assert calls["count"] == 2, "关闭语义匹配后改写问法应各走一次"
    assert r2.json()["cache_hit"] is False
    # 关闭时不该宣称任何缓存路径命中
    assert r2.headers.get("X-Cache") is None


# --------------------------------------------------------------------------
# 3. 只比对同 top_k
# --------------------------------------------------------------------------
def test_semantic_match_requires_same_top_k(
    client: TestClient, monkeypatch
) -> None:
    """top_k 不同 → 不得语义命中。

    top_k 影响召回条数与 citations，命中了就会给用户一份与他请求参数
    不符的引用列表 —— 与精确 key 把 top_k 纳入 key 的理由完全一致。
    """
    _install(monkeypatch)
    client.post("/api/qa", json={"query": QUERY_A, "top_k": 5})
    r2 = client.post("/api/qa", json={"query": QUERY_B, "top_k": 3})

    assert r2.json()["cache_hit"] is False
    assert r2.headers.get("X-Cache") is None


# --------------------------------------------------------------------------
# 4/5/6. 单元级：阈值边界、维度守卫、过期条目
# --------------------------------------------------------------------------
def test_find_similar_honours_threshold_boundary() -> None:
    """相似度跨过阈值时才命中。

    刻意不测"恰好等于阈值"：`cos = 0.8` 在 float32 下未必精确等于 0.8，
    卡等值断言会变成随机红灯。两侧各留一点余量，测的是**比较方向**。
    """
    cache = reconfigure_cache(max_size=8, ttl_seconds=300.0)
    # a 为 4 维单位向量；b 与之点积 = 0.8（0.8²+0.6²=1，本身已是单位向量）
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
    cache.put_answer(QUERY_A, 3, "answer-A", vector=a)

    hit = cache.find_similar(QUERY_B, 3, b, threshold=0.79)
    assert hit is not None, "0.8 > 0.79 应命中"
    assert abs(hit.similarity - 0.8) < 1e-5

    miss = cache.find_similar(QUERY_B, 3, b, threshold=0.81)
    assert miss is None, "0.8 < 0.81 不应命中"


def test_find_similar_scores_cosine_not_raw_dot() -> None:
    """未归一化的向量也要按**余弦**算，不能退化成点积。

    条目内的向量在 put 时就已归一化，但查询向量来自调用方（可能未归一化）。
    若不归一化就点积，长向量会被系统性抬高分数、误判命中。
    """
    cache = reconfigure_cache(max_size=8, ttl_seconds=300.0)
    cache.put_answer(QUERY_A, 3, "answer-A", vector=np.array([1.0, 0.0, 0.0, 0.0]))

    # 与 a 同方向但长度 10：余弦应为 1.0 而非 10.0
    scale = 10.0
    hit = cache.find_similar(
        QUERY_B, 3, np.array([scale, 0.0, 0.0, 0.0]), threshold=0.99
    )
    assert hit is not None
    assert abs(hit.similarity - 1.0) < 1e-6, f"应按余弦得 1.0，实际 {hit.similarity}"

    # 反方向：余弦 -1，任何正阈值都不该命中
    assert (
        cache.find_similar(QUERY_B, 3, np.array([-scale, 0, 0, 0]), threshold=0.1)
        is None
    )


def test_find_similar_skips_dimension_mismatch() -> None:
    """维度不符的条目必须被跳过，不得参与比对。

    实测场景：EMBEDDING_PROVIDER 从 dashscope(1024) 切到 local_hash(256)
    （或运行期粘性降级）后，缓存里会短暂留着旧维度的向量。
    跨维度算余弦是无意义的数字，一旦被判命中就是**拿错答案**。
    """
    cache = reconfigure_cache(max_size=8, ttl_seconds=300.0)
    cache.put_answer(QUERY_A, 3, "answer-old", vector=np.ones(1024, dtype=np.float32))

    # 查询向量是 256 维（local_hash）→ 那条 1024 维条目不可比，必须不命中
    hit = cache.find_similar(QUERY_A, 3, np.ones(256, dtype=np.float32), threshold=0.01)
    assert hit is None, "跨维度条目不得被命中"

    # 同维度时（且方向相同 → 余弦 1.0）应正常命中，证明上面的 None 是维度守卫
    # 造成的，而不是查找逻辑整体失效
    hit_same_dim = cache.find_similar(
        QUERY_A, 3, np.ones(1024, dtype=np.float32), threshold=0.99
    )
    assert hit_same_dim is not None


def test_expired_entry_is_not_semantically_matched() -> None:
    """TTL 过期的条目不得被语义命中（与精确路径的过期语义一致）。"""
    fake_now = [0.0]
    cache = reconfigure_cache(
        max_size=8, ttl_seconds=1.0, wall_fn=lambda: fake_now[0]
    )
    vec = np.array([1.0, 0.0], dtype=np.float32)
    cache.put_answer(QUERY_A, 3, "answer-A", vector=vec)

    # TTL 内：命中
    assert cache.find_similar(QUERY_B, 3, vec, threshold=0.9) is not None

    # 推进到 TTL 之外：不得再命中
    fake_now[0] = 2.0
    assert cache.find_similar(QUERY_B, 3, vec, threshold=0.9) is None


def test_entries_without_vector_are_skipped() -> None:
    """未带向量写入的条目（语义关闭时写入 / 向量计算失败）只服务精确命中。"""
    cache = reconfigure_cache(max_size=8, ttl_seconds=300.0)
    cache.put_answer(QUERY_A, 3, "answer-A", vector=None)
    vec = np.array([1.0, 0.0], dtype=np.float32)

    assert cache.find_similar(QUERY_A, 3, vec, threshold=0.1) is None
    # 精确命中仍然可用
    assert cache.lookup_exact(QUERY_A, 3) is not None


# --------------------------------------------------------------------------
# 7. 语义命中仍计入限流（与精确命中的规则不同）
# --------------------------------------------------------------------------
def test_semantic_hit_still_consumes_rate_limit(
    client: TestClient, monkeypatch
) -> None:
    """语义命中**要**扣限流额度，精确命中不扣。

    既有约定"缓存命中不入限流"的理由是命中不消耗上游资源 ——
    精确命中确实零成本；但语义命中要算一次查询向量（真实 embedding 调用），
    所以它走限流内层、照常计费。这条差异是刻意设计的，不是遗漏。
    """
    from app.services.rate_limit import get_limiter

    monkeypatch.setattr(qa_module.settings, "rate_limit_enabled", True)
    # 额度 2：首次查询 1 次 + 语义命中 1 次 = 刚好用完；第 3 次必须被拒
    monkeypatch.setattr(qa_module.settings, "rate_limit_qa_per_min", 2)
    get_limiter().reset(clear_store=True)
    _install(monkeypatch)

    r1 = client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})
    r2 = client.post("/api/qa", json={"query": QUERY_B, "top_k": 3})
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert r2.headers.get("X-Cache") == "HIT_SEMANTIC"

    # 额度已被语义命中用掉 1 次 → 再来一个不同 query 应被限流拦下
    r3 = client.post("/api/qa", json={"query": "又一个不同的问题", "top_k": 3})
    assert r3.status_code == 429, (
        f"语义命中应计入限流；若不计入，这里会返回 {r3.status_code}"
    )


def test_exact_hit_does_not_consume_rate_limit(
    client: TestClient, monkeypatch
) -> None:
    """对照：精确命中反复问不扣额度（回归护栏，防止改造把它弄丢）。"""
    from app.services.rate_limit import get_limiter

    monkeypatch.setattr(qa_module.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(qa_module.settings, "rate_limit_qa_per_min", 2)
    get_limiter().reset(clear_store=True)
    _install(monkeypatch)

    for _ in range(5):
        resp = client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})
        assert resp.status_code == 200, resp.text


# --------------------------------------------------------------------------
# 8. 向量复用：同一请求不得重复调用 embedding
# --------------------------------------------------------------------------
def test_hybrid_search_reuses_provided_query_vector(monkeypatch) -> None:
    """传入 query_vector 时，hybrid_search **不得**再调 embed。

    这条是"语义匹配不增加额外成本"的支点：为缓存探测算出的向量必须
    一路带到检索层。若这里退化成各算一次，语义匹配就变成了
    "命中省一次 LLM、未命中多花一次 embedding"，收益被抵消一半。
    """
    calls: dict = {"embed": 0, "vectors": []}

    class _StubStore:
        def embed(self, text: str) -> np.ndarray:
            calls["embed"] += 1
            return np.ones(4, dtype=np.float32)

        def search_vector(self, vec, top_k, exclude=None):
            calls["vectors"].append(np.asarray(vec))
            return [(1, 0.9)]

    monkeypatch.setattr(hybrid_module, "get_store", lambda: _StubStore())
    provided = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    hybrid_module.hybrid_search("测试查询", 3, query_vector=provided)

    assert calls["embed"] == 0, (
        f"传入了 query_vector 就不该再调用 embed，实际调用 {calls['embed']} 次"
    )
    assert len(calls["vectors"]) == 1
    assert np.allclose(calls["vectors"][0], provided), "检索应使用传入的那个向量"

    # 不传时仍自行计算（保持既有行为，改造不能把默认路径弄丢）
    hybrid_module.hybrid_search("测试查询", 3)
    assert calls["embed"] == 1


# --------------------------------------------------------------------------
# 观测：stats 能区分精确命中与语义命中
# --------------------------------------------------------------------------
def test_stats_report_semantic_hits_separately(
    client: TestClient, monkeypatch
) -> None:
    """/health 上的统计要能把"语义层救回多少次"单独看见。

    否则无法判断这个阈值配得值不值 —— 精确命中率涨了也可能只是同一问法
    被重复问，与语义匹配无关。"""
    _install(monkeypatch)
    client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})  # 未命中
    client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})  # 精确命中
    client.post("/api/qa", json={"query": QUERY_B, "top_k": 3})  # 语义命中

    s = get_cache().stats()
    assert s["semantic_lookups"] >= 1, "精确未命中后应做过语义探测"
    assert s["semantic_hits"] == 1, f"应记录 1 次语义命中，实际 {s['semantic_hits']}"
    assert s["hits"] >= 1, "精确命中仍单独计数"


def test_cache_carries_vector_into_entry(client: TestClient, monkeypatch) -> None:
    """写缓存时向量确实随条目落库（否则语义匹配永远无条目可比）。"""
    _install(monkeypatch)
    client.post("/api/qa", json={"query": QUERY_A, "top_k": 3})

    entry = get_cache().lookup_exact(QUERY_A, 3)
    assert entry is not None
    assert entry.vector is not None, "启用语义匹配时条目必须带向量"
    # 存的是**已归一化**的单位向量（查找路径借此把余弦退化为点积）
    assert abs(float(np.linalg.norm(entry.vector)) - 1.0) < 1e-5
    assert entry.top_k == 3
    assert entry.query == QUERY_A
