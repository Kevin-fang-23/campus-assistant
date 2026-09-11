"""限流功能验证。

分两类：
- 单元级：直接驱动 RateLimiter，注入假时钟 → 阈值、回填、重置、剪枝全部确定性可测
- 接口级：经 TestClient 发真实请求 → 验证 429 状态码、响应头、错误结构、CORS 兼容性、豁免路径

时间注入是重点：用真实 sleep 测回填会让测试变慢且不稳，注入假时钟后
「1 分钟后回填」「次日重置」可在毫秒内确定性验证。
"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import rate_limit as rl
from app.services.rate_limit import RateLimiter, client_key, reset_limiter, tier_for


class FakeClock:
    """单调钟与墙钟一起前进，模拟真实时间流逝。"""

    def __init__(self, start: datetime | None = None) -> None:
        self.wall = (start or datetime(2026, 9, 11, 12, 0, 0)).timestamp()
        self.mono = 1000.0

    def now_fn(self) -> float:
        return self.mono

    def wall_fn(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


@pytest.fixture(autouse=True)
def _clean_limiter():
    """每个用例前后都清空全局计数器，避免相互污染。"""
    reset_limiter()
    yield
    reset_limiter()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _limits(monkeypatch, *, per_min=0, per_ip_day=0, per_day=0, tier="qa"):
    """给某一层级设置阈值；其余层级置 0（不启用）。"""
    monkeypatch.setattr(settings, "rate_limit_qa_per_min", per_min if tier == "qa" else 0)
    monkeypatch.setattr(settings, "rate_limit_qa_per_ip_day", per_ip_day if tier == "qa" else 0)
    monkeypatch.setattr(settings, "rate_limit_qa_per_day", per_day if tier == "qa" else 0)
    monkeypatch.setattr(settings, "rate_limit_ingest_per_min", per_min if tier == "ingest" else 0)
    monkeypatch.setattr(settings, "rate_limit_ingest_per_ip_day", per_ip_day if tier == "ingest" else 0)
    monkeypatch.setattr(settings, "rate_limit_ingest_per_day", per_day if tier == "ingest" else 0)
    monkeypatch.setattr(settings, "rate_limit_default_per_min", per_min if tier == "default" else 0)


# --------------------------- 层级归属 ---------------------------

@pytest.mark.parametrize("path,expected", [
    ("/api/qa", "qa"),
    ("/api/documents/text", "ingest"),
    ("/api/documents/upload", "ingest"),
    ("/api/notices", "default"),
    ("/api/tasks/1", "default"),
    ("/api/search", "default"),
    ("/health", None),                     # 监控探针不能被限流
    ("/docs", None),
    ("/openapi.json", None),
    ("/assets/index-abc.js", None),        # 静态资源：一次页面加载几十个请求
    ("/", None),
])
def test_tier_mapping(path: str, expected: str | None) -> None:
    assert tier_for(path) == expected


# --------------------------- L1 令牌桶 ---------------------------

def test_token_bucket_allows_quota_then_blocks(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=3)

    assert [limiter.check("/api/qa", "1.1.1.1").allowed for _ in range(3)] == [True] * 3

    blocked = limiter.check("/api/qa", "1.1.1.1")
    assert blocked.allowed is False
    assert blocked.scope == "ip_minute"
    assert blocked.retry_after >= 1, "必须给出 Retry-After，否则客户端只能盲目重试"


def test_token_bucket_refills_over_time(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=60)          # 1 个/秒

    for _ in range(60):
        assert limiter.check("/api/qa", "1.1.1.1").allowed
    assert limiter.check("/api/qa", "1.1.1.1").allowed is False

    clock.advance(5)                          # 5 秒 → 回填 5 个
    assert [limiter.check("/api/qa", "1.1.1.1").allowed for _ in range(5)] == [True] * 5
    assert limiter.check("/api/qa", "1.1.1.1").allowed is False


def test_token_bucket_is_per_ip(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=2)

    for _ in range(2):
        assert limiter.check("/api/qa", "1.1.1.1").allowed
    assert limiter.check("/api/qa", "1.1.1.1").allowed is False
    # 另一个来源不受影响——限流不能变成「一人超限全体连坐」
    assert limiter.check("/api/qa", "2.2.2.2").allowed is True


# --------------------------- L2 每 IP 每日 ---------------------------

def test_per_ip_daily_cap_blocks_even_when_rate_is_loose(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=1000, per_ip_day=5)

    for _ in range(5):
        assert limiter.check("/api/qa", "1.1.1.1").allowed

    blocked = limiter.check("/api/qa", "1.1.1.1")
    assert blocked.allowed is False
    assert blocked.scope == "ip_day"
    assert blocked.retry_after > 3600, "日额度应提示到次日零点，而非几秒后"


def test_per_ip_daily_cap_is_per_ip(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=1000, per_ip_day=2)

    for _ in range(2):
        assert limiter.check("/api/qa", "1.1.1.1").allowed
    assert limiter.check("/api/qa", "1.1.1.1").allowed is False
    assert limiter.check("/api/qa", "2.2.2.2").allowed is True


# --------------------------- L3 全局每日（资金护栏）---------------------------

def test_global_daily_cap_bounds_total_across_ips(monkeypatch) -> None:
    """总额度上限必须跨 IP 生效——这是唯一能兜住实际支出的那一层。"""
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=1000, per_ip_day=100, per_day=3)

    for i in range(3):
        assert limiter.check("/api/qa", f"10.0.0.{i}").allowed, "换 IP 也应被总额度拦住"

    blocked = limiter.check("/api/qa", "10.0.0.99")
    assert blocked.allowed is False
    assert blocked.scope == "global_day"


def test_daily_counters_reset_on_next_day(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=1000, per_ip_day=2, per_day=2)

    assert limiter.check("/api/qa", "1.1.1.1").allowed
    assert limiter.check("/api/qa", "1.1.1.1").allowed
    assert limiter.check("/api/qa", "1.1.1.1").allowed is False
    assert limiter.check("/api/qa", "9.9.9.9").allowed is False    # 全局额度已尽

    clock.advance(13 * 3600)                                       # 跨到次日
    assert limiter.check("/api/qa", "1.1.1.1").allowed is True
    assert limiter.check("/api/qa", "9.9.9.9").allowed is True


# --------------------------- 边界与健壮性 ---------------------------

def test_all_layers_zero_means_unlimited(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=0, per_ip_day=0, per_day=0)

    assert all(limiter.check("/api/qa", "1.1.1.1").allowed for _ in range(500))


def test_non_api_path_never_counted(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=1)

    assert all(limiter.check("/health", "1.1.1.1").allowed for _ in range(50))


def test_bucket_map_is_pruned(monkeypatch) -> None:
    """伪造 IP 轮换不能把内存撑爆。"""
    clock = FakeClock()
    limiter = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    _limits(monkeypatch, per_min=5)

    for i in range(1200):                       # 制造大量不同来源
        limiter.check("/api/qa", f"203.0.{i // 256}.{i % 256}")
    assert len(limiter._buckets) <= rl._MAX_BUCKETS

    clock.advance(300)                          # 超过空闲阈值
    limiter.check("/api/qa", "203.0.0.1")       # 触发清理
    assert len(limiter._buckets) < 1200


def test_disabled_flag_short_circuits(client: TestClient, monkeypatch) -> None:
    """总开关关闭时即使阈值极小也应全部放行（离线/本地环境用）。

    注意：开关判定在中间件层，不在 RateLimiter 内部——所以这里走接口级验证。
    """
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    _limits(monkeypatch, per_min=1)
    reset_limiter()

    statuses = {
        client.post("/api/qa", json=_qa_payload(), headers={"X-Forwarded-For": "198.51.100.3"}).status_code
        for _ in range(5)
    }
    assert statuses == {200}


# --------------------------- 来源识别 ---------------------------

def test_client_key_ignores_forged_header_when_not_trusting_proxy() -> None:
    key = client_key(forwarded_for="9.9.9.9", real_ip=None, host="1.1.1.1", trust_proxy=False)
    assert key == "1.1.1.1", "直连时信任 XFF 等于把限流开关交给攻击者"


def test_client_key_prefers_first_xff_when_trusting_proxy() -> None:
    key = client_key(
        forwarded_for="203.0.113.7, 10.0.0.1", real_ip=None,
        host="127.0.0.1", trust_proxy=True,
    )
    assert key == "203.0.113.7", "反代链路上取最左侧原始客户端地址"


def test_client_key_falls_back_to_real_ip_then_host() -> None:
    assert client_key(forwarded_for=None, real_ip="10.1.1.1", host="127.0.0.1",
                      trust_proxy=True) == "10.1.1.1"
    assert client_key(forwarded_for=None, real_ip=None, host=None,
                      trust_proxy=True) == "unknown"


# --------------------------- 接口级：429 与响应契约 ---------------------------

def _qa_payload():
    return {"query": "作业什么时候截止", "top_k": 1}


def test_qa_returns_429_with_standard_error_shape(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_trust_proxy", True)
    _limits(monkeypatch, per_min=2)
    reset_limiter()

    headers = {"X-Forwarded-For": "198.51.100.1"}
    assert client.post("/api/qa", json=_qa_payload(), headers=headers).status_code == 200
    assert client.post("/api/qa", json=_qa_payload(), headers=headers).status_code == 200

    resp = client.post("/api/qa", json=_qa_payload(), headers=headers)
    assert resp.status_code == 429
    # 错误结构沿用项目既有的 {"detail": "..."}，不引入第二套形状
    assert "detail" in resp.json()
    assert resp.headers["Retry-After"].isdigit()
    assert resp.headers["X-RateLimit-Scope"] == "ip_minute"
    assert resp.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limited_response_keeps_cors_headers(client: TestClient, monkeypatch) -> None:
    """429 必须带跨域头，否则前端只会看到 CORS 报错、拿不到可读原因。

    这依赖 main.py 中「限流先注册、CORS 后注册」的顺序，是顺序写反的守卫测试。
    """
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_trust_proxy", True)
    _limits(monkeypatch, per_min=1)
    reset_limiter()

    origin = settings.cors_origins[0]
    headers = {"X-Forwarded-For": "198.51.100.9", "Origin": origin}

    assert client.post("/api/qa", json=_qa_payload(), headers=headers).status_code == 200
    resp = client.post("/api/qa", json=_qa_payload(), headers=headers)
    assert resp.status_code == 429
    assert resp.headers.get("access-control-allow-origin") == origin


def test_health_is_exempt_from_limiting(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_default_per_min", 1)
    reset_limiter()

    statuses = {client.get("/health").status_code for _ in range(20)}
    assert statuses == {200}, "监控探针被限流会导致服务被误判为不可用"


def test_allowed_response_exposes_remaining_quota(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_trust_proxy", True)
    _limits(monkeypatch, per_min=5)
    reset_limiter()

    resp = client.post("/api/qa", json=_qa_payload(), headers={"X-Forwarded-For": "198.51.100.50"})
    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Remaining"] == "4"


def test_ingest_tier_is_limited_independently(client: TestClient, monkeypatch) -> None:
    """上传解析同样花钱，必须与 QA 分别计量（互不占用对方额度）。"""
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_trust_proxy", True)
    _limits(monkeypatch, tier="ingest", per_min=1)
    reset_limiter()

    headers = {"X-Forwarded-For": "198.51.100.77"}
    body = {"content": "《操作系统》作业截止 9月20日 22:00", "filename": "t.txt"}
    assert client.post("/api/documents/text", json=body, headers=headers).status_code == 200
    assert client.post("/api/documents/text", json=body, headers=headers).status_code == 429
    # QA 层级未被 ingest 的消耗影响
    assert client.post("/api/qa", json=_qa_payload(), headers=headers).status_code == 200
