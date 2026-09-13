"""限流计数持久化验证（清单 #8 多 worker 额度精确 / #9 重启不清零）。

分四层：
1. `DailyCounter` 单元级 —— 直查存储、跨实例一致、时钟回拨安全
2. `SqliteCountStore` —— 建表、WAL、原子自增、并发不丢更新
3. `FileCountStore` / `InMemoryCountStore` —— 另两种后端
4. `RateLimiter` 集成级 —— 这是清单 #8/#9 的直接验收

## 为什么这里有一条"反例"用例

`test_limiter_worker_quota_total_is_correct_across_workers` 钉住的是
**初版设计的失败模式**：早期实现用「内存缓存 + 异步刷盘」，
4 个 worker × 每日额度 40 实际放行了 60 次。
该用例是那次返工留下的防回归护栏，不要因为"看起来多余"删掉它。

## 三种 store 的分工（别再混用）

| 实现 | 有状态 | 跨进程 | 跨重启 | 用途 |
|---|---|---|---|---|
| `NullCountStore` | ✗ | — | — | 显式关闭日额度（等价于 L2/L3 停用） |
| `InMemoryCountStore` | ✓ | ✗ | ✗ | 默认值 / 单 worker |
| `SqliteCountStore` | ✓ | ✓ | ✓ | 生产（解决 #8 / #9） |
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from app.services.rate_limit import RateLimiter
from app.services.rate_limit_store import (
    DailyCounter,
    FileCountStore,
    InMemoryCountStore,
    NullCountStore,
    SqliteCountStore,
)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.wall = (start or datetime(2026, 9, 13, 12, 0, 0)).timestamp()
        self.mono = 1000.0

    def now_fn(self) -> float:
        return self.mono

    def wall_fn(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


@pytest.fixture
def engine(tmp_path: Path):
    """临时文件 SQLite（多连接可见，能验证真正的跨实例共享）。"""
    eng = create_engine(f"sqlite:///{(tmp_path / 'rl.db').as_posix()}", future=True)
    yield eng
    eng.dispose()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# ---------------------------------------------------------------------------
# 第 1 层：DailyCounter
# ---------------------------------------------------------------------------
def test_counter_defaults_to_zero(clock):
    c = DailyCounter(NullCountStore(), wall_fn=clock.wall_fn)
    assert c.used("global:qa", "2026-09-13") == 0


def test_counter_bump_then_used_reflects_store(engine, clock):
    c = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    day = "2026-09-13"
    assert c.bump("k", day) == 1
    assert c.bump("k", day) == 2
    assert c.used("k", day) == 2
    # 绕过对象直接查库，确认不是内存假象
    with engine.connect() as conn:
        v = conn.execute(
            text("SELECT value FROM rate_limit_counters WHERE day=:d AND key=:k"),
            {"d": day, "k": "k"},
        ).scalar()
    assert v == 2


def test_counter_no_memory_cache_sees_external_write(engine, clock):
    """关键：外部（另一 worker）写入后，本对象必须立刻看到。

    初版有内存缓存，这里会读到旧值 —— 正是超发的根因。
    """
    store = SqliteCountStore(engine)
    c = DailyCounter(store, wall_fn=clock.wall_fn)
    assert c.used("k", "2026-09-13") == 0
    store.add("2026-09-13", {"k": 5})          # 模拟另一 worker 记账
    assert c.used("k", "2026-09-13") == 5, "存在内存缓存 —— 多 worker 会超发"


def test_counter_survives_restart(engine, clock):
    """#9 核心：新实例（模拟重启）应读回旧计数。"""
    day = "2026-09-13"
    c1 = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    for _ in range(3):
        c1.bump("global:qa", day)
    c2 = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    assert c2.used("global:qa", day) == 3, "重启后计数丢失 —— #9 未修复"


def test_counter_isolates_keys(engine, clock):
    c = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    day = "2026-09-13"
    c.bump("a", day)
    c.bump("a", day)
    c.bump("b", day)
    assert c.used("a", day) == 2
    assert c.used("b", day) == 1


def test_counter_isolates_days(engine, clock):
    c = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    c.bump("k", "2026-09-13")
    c.bump("k", "2026-09-14")
    c.bump("k", "2026-09-14")
    assert c.used("k", "2026-09-13") == 1
    assert c.used("k", "2026-09-14") == 2


def test_counter_clock_rollback_keeps_count(engine, clock):
    """时钟回拨（D → D+1 → D）不得让当天额度被清零。

    攻击者可等一次 NTP 校时来绕过每日上限，所以清理必须保留昨天。
    """
    c = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    c.bump("k", "2026-09-13")
    c.bump("k", "2026-09-13")
    for d in ["2026-09-14", "2026-09-13", "2026-09-14", "2026-09-13"]:
        c.used("k", d)
    assert c.used("k", "2026-09-13") == 2, "时钟回拨导致当天计数被清"


def test_counter_purge_keeps_yesterday_and_today(engine, clock):
    """清理只删「老到确定过期」的数据，保留昨天与今天。

    注意 `used` 本身就会触发清理检查（纯只读场景不会走 bump，
    只在 bump 里挂清理的话读多写少的服务永远不清理旧数据）。
    """
    store = SqliteCountStore(engine)
    store.add("2026-09-01", {"k": 7})   # 老数据，应回收
    store.add("2026-09-12", {"k": 8})   # 昨天，保留（容忍回拨）
    store.add("2026-09-13", {"k": 9})   # 今天，保留
    c = DailyCounter(store, wall_fn=clock.wall_fn)
    c.used("k", "2026-09-13")           # 首次用到今天 → 触发一次清理
    assert store.load_day("2026-09-01") == {}
    assert store.load_day("2026-09-12") == {"k": 8}
    assert store.load_day("2026-09-13") == {"k": 9}


def test_counter_purge_runs_once_per_day(engine, clock, monkeypatch):
    """清理每天只做一次（否则每请求一次 DELETE，白白写盘）。

    混合 used / bump 两种调用路径，确认两条路径共用同一个
    「今天已清理过」标记，不会各自触发一次。
    """
    store = SqliteCountStore(engine)
    calls = []
    orig = store.purge_before

    def spy(day):  # noqa: ANN001
        calls.append(day)
        return orig(day)

    monkeypatch.setattr(store, "purge_before", spy)
    c = DailyCounter(store, wall_fn=clock.wall_fn)
    for _ in range(5):
        c.used("k", "2026-09-13")
    for _ in range(5):
        c.bump("k", "2026-09-13")
    assert len(calls) == 1, f"清理被调用 {len(calls)} 次，应只 1 次"


def test_counter_bump_on_old_day_is_not_purged_away(engine, clock):
    """给「旧日期」记账后必须读得回来。

    清理是 `purge_before(今天-1)`，若在写入**之前**清理，那么对
    `day='2026-09-01'` 的记账会被当成历史数据删掉 —— 表现为
    「bump 成功但读回 0」。次序必须是先写后清理。
    """
    c = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    assert c.bump("k", "2026-09-01") == 1
    assert c.used("k", "2026-09-01") == 1


def test_counter_tolerates_read_failure(clock):
    """读失败按 0 处理（放行）—— 可用性优先于护栏完整性，但要留 warning。"""

    class BrokenStore:
        def get(self, day, key):  # noqa: ANN001, ARG002
            raise RuntimeError("db down")

        def add(self, day, deltas):  # noqa: ANN001, ARG002
            raise RuntimeError("db down")

        def load_day(self, day):  # noqa: ANN001, ARG002
            raise RuntimeError("db down")

        def purge_before(self, day):  # noqa: ANN001, ARG002
            raise RuntimeError("db down")

        def clear(self):  # noqa: ANN201
            raise RuntimeError("db down")

    c = DailyCounter(BrokenStore(), wall_fn=clock.wall_fn)
    assert c.used("k", "2026-09-13") == 0     # 不抛
    assert c.bump("k", "2026-09-13") == 0     # 不抛


def test_counter_reset_without_clear_store_keeps_data(engine, clock):
    c = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    c.bump("k", "2026-09-13")
    c.reset()                                  # 默认不清存储
    assert c.used("k", "2026-09-13") == 1


def test_counter_reset_with_clear_store_wipes(engine, clock):
    c = DailyCounter(SqliteCountStore(engine), wall_fn=clock.wall_fn)
    c.bump("k", "2026-09-13")
    c.reset(clear_store=True)
    assert c.used("k", "2026-09-13") == 0


# ---------------------------------------------------------------------------
# 第 2 层：SqliteCountStore
# ---------------------------------------------------------------------------
def test_sqlite_store_creates_table(engine):
    SqliteCountStore(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='rate_limit_counters'")
        ).all()
    assert rows


def test_sqlite_store_enables_wal(engine):
    """WAL 是性能前提（实测 4071µs → 88.8µs），必须被自动开启。"""
    SqliteCountStore(engine)
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert str(mode).lower() == "wal", f"WAL 未开启（当前 {mode}），性能会差 46 倍"


def test_sqlite_store_get_returns_zero_for_missing(engine):
    assert SqliteCountStore(engine).get("2026-09-13", "nope") == 0


def test_sqlite_store_add_and_get(engine):
    s = SqliteCountStore(engine)
    s.add("2026-09-13", {"k": 3})
    assert s.get("2026-09-13", "k") == 3


def test_sqlite_store_upsert_increments(engine):
    s = SqliteCountStore(engine)
    s.add("2026-09-13", {"k": 1})
    s.add("2026-09-13", {"k": 2})
    assert s.get("2026-09-13", "k") == 3


def test_sqlite_store_load_day_isolates_days(engine):
    s = SqliteCountStore(engine)
    s.add("2026-09-13", {"k": 1})
    s.add("2026-09-14", {"k": 2})
    assert s.load_day("2026-09-13") == {"k": 1}
    assert s.load_day("2026-09-14") == {"k": 2}


def test_sqlite_store_ignores_nonpositive_delta(engine):
    s = SqliteCountStore(engine)
    s.add("2026-09-13", {"k": 0, "j": -5})
    assert s.load_day("2026-09-13") == {}


def test_sqlite_store_purge_before(engine):
    s = SqliteCountStore(engine)
    s.add("2026-09-01", {"a": 1})
    s.add("2026-09-13", {"b": 1})
    s.purge_before("2026-09-13")
    assert s.load_day("2026-09-01") == {}
    assert s.load_day("2026-09-13") == {"b": 1}


def test_sqlite_store_clear(engine):
    s = SqliteCountStore(engine)
    s.add("2026-09-13", {"a": 1})
    s.clear()
    assert s.load_day("2026-09-13") == {}


def test_sqlite_store_recreates_dropped_table(engine):
    """表被外部删除后应能自动重建（幂等建表）。"""
    s = SqliteCountStore(engine)
    s.add("2026-09-13", {"k": 1})
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE rate_limit_counters"))
    s2 = SqliteCountStore(engine)
    assert s2.load_day("2026-09-13") == {}


def test_sqlite_store_concurrent_increments_exact(engine):
    """并发自增不得丢更新 —— 这是「每请求读库记账」方案成立的前提。

    8 线程 × 50 次，总量必须精确等于 400。
    若实现是读-改-写而非数据库原子加，这里会丢更新。
    """
    store = SqliteCountStore(engine)
    day = "2026-09-13"
    n_threads, per_thread = 8, 50
    errors: list[Exception] = []

    def worker() -> None:
        try:
            local = SqliteCountStore(engine)   # 模拟各 worker 独立实例
            for _ in range(per_thread):
                local.add(day, {"global:qa": 1})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发写入报错：{errors}"
    assert store.get(day, "global:qa") == n_threads * per_thread


def test_sqlite_store_reserve_enforces_limit_exactly(engine):
    """`reserve` 是原子占位：并发下通过数必须**恰好**等于 limit。

    这是修掉「check-then-act 超发」的核心。原先用「先读 used 判、后写 bump」，
    8 线程并发抢 limit=50 实测出现过 51+ 次通过。
    """
    store = SqliteCountStore(engine)
    day, limit = "2026-09-13", 50
    n_threads, per_thread = 8, 20          # 160 次尝试抢 50 个名额
    granted = []
    lock = threading.Lock()

    def worker() -> None:
        local = SqliteCountStore(engine)
        for _ in range(per_thread):
            if local.reserve(day, "global:qa", limit) <= limit:
                with lock:
                    granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(granted) == limit, f"通过 {len(granted)} 次，应恰好 {limit}"
    assert store.get(day, "global:qa") == limit, "库中计数应恰为 limit（占位余量已撤销）"


def test_sqlite_store_reserve_rejects_and_does_not_inflate(engine):
    """超限的 reserve 必须撤销自己的增量，不能让计数虚高。

    否则计数会被撑到远大于 limit，且**再也降不回来**（下次判定永远为满）。
    """
    store = SqliteCountStore(engine)
    day, limit = "2026-09-13", 3
    for _ in range(3):
        assert store.reserve(day, "k", limit) <= limit
    assert store.get(day, "k") == 3
    for _ in range(10):                     # 再抢 10 次，全部应被拒
        assert store.reserve(day, "k", limit) > limit
    assert store.get(day, "k") == 3, "超限占位未被撤销 —— 计数虚高"


def test_sqlite_store_release_decrements_but_not_below_zero(engine):
    store = SqliteCountStore(engine)
    day = "2026-09-13"
    store.add(day, {"k": 2})
    store.release(day, "k")
    assert store.get(day, "k") == 1
    store.release(day, "k")
    store.release(day, "k")                 # 多退不该变负数
    assert store.get(day, "k") == 0


# ---------------------------------------------------------------------------
# 第 3 层：FileCountStore / InMemoryCountStore
# ---------------------------------------------------------------------------
def test_null_store_is_stateless(clock):
    """NullCountStore 无状态 —— 日额度永远用不完（等价于停用 L2/L3）。

    这不是缺陷，而是「显式关闭」的语义，所以要被钉住：
    如果哪天它变得有状态，本用例会红，迫使人复核默认值是否仍安全。
    """
    s = NullCountStore()
    s.add("2026-09-13", {"k": 1})
    assert s.get("2026-09-13", "k") == 0
    assert s.load_day("2026-09-13") == {}


def test_memory_store_roundtrip():
    s = InMemoryCountStore()
    s.add("2026-09-13", {"a": 1, "b": 2})
    s.add("2026-09-13", {"a": 3})
    assert s.load_day("2026-09-13") == {"a": 4, "b": 2}
    assert s.get("2026-09-13", "a") == 4


def test_memory_store_isolates_days():
    s = InMemoryCountStore()
    s.add("2026-09-13", {"k": 1})
    s.add("2026-09-14", {"k": 2})
    assert s.load_day("2026-09-13") == {"k": 1}
    assert s.load_day("2026-09-14") == {"k": 2}


def test_memory_store_ignores_nonpositive_delta():
    s = InMemoryCountStore()
    s.add("2026-09-13", {"k": 0, "j": -5})
    assert s.load_day("2026-09-13") == {}


def test_memory_store_purge_and_clear():
    s = InMemoryCountStore()
    s.add("2026-09-01", {"a": 1})
    s.add("2026-09-13", {"b": 1})
    s.purge_before("2026-09-13")
    assert s.load_day("2026-09-01") == {}
    assert s.load_day("2026-09-13") == {"b": 1}
    s.clear()
    assert s.load_day("2026-09-13") == {}


def test_memory_store_instances_are_independent():
    """进程内两个实例互不可见 —— 这正是「不持久化」的体现。

    与 SqliteCountStore 共享同一个文件形成对照。
    """
    InMemoryCountStore().add("2026-09-13", {"k": 5})
    assert InMemoryCountStore().get("2026-09-13", "k") == 0


def test_memory_store_concurrent_increments_exact():
    """内存实现也必须线程安全（FastAPI 同步端点跑在线程池里）。"""
    s = InMemoryCountStore()
    n_threads, per_thread = 8, 200
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(per_thread):
                s.add("2026-09-13", {"k": 1})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"并发写入报错：{errors}"
    assert s.get("2026-09-13", "k") == n_threads * per_thread


def test_memory_store_reserve_enforces_limit_exactly():
    """`reserve` 在内存实现下同样必须严格限流（锁内读-判-写不可拆分）。"""
    s = InMemoryCountStore()
    day, limit = "2026-09-13", 50
    n_threads, per_thread = 8, 20
    granted = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(per_thread):
            if s.reserve(day, "k", limit) <= limit:
                with lock:
                    granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == limit, f"通过 {len(granted)} 次，应恰好 {limit}"
    assert s.get(day, "k") == limit


def test_memory_store_reserve_rejects_without_inflating():
    s = InMemoryCountStore()
    day, limit = "2026-09-13", 2
    assert s.reserve(day, "k", limit) == 1
    assert s.reserve(day, "k", limit) == 2
    assert s.reserve(day, "k", limit) == 3    # 超限，不写入
    assert s.get(day, "k") == 2, "超限占位未撤销"


def test_memory_store_release():
    s = InMemoryCountStore()
    day = "2026-09-13"
    s.add(day, {"k": 2})
    s.release(day, "k")
    assert s.get(day, "k") == 1
    s.release(day, "k")
    s.release(day, "k")
    assert s.get(day, "k") == 0


def test_file_store_roundtrip(tmp_path: Path):
    p = tmp_path / "counts.json"
    s = FileCountStore(p)
    s.add("2026-09-13", {"a": 1, "b": 2})
    s.add("2026-09-13", {"a": 3})
    assert s.load_day("2026-09-13") == {"a": 4, "b": 2}
    assert s.get("2026-09-13", "a") == 4


def test_file_store_persists_across_instances(tmp_path: Path):
    p = tmp_path / "counts.json"
    FileCountStore(p).add("2026-09-13", {"a": 5})
    assert FileCountStore(p).get("2026-09-13", "a") == 5


def test_file_store_tolerates_corrupt_file(tmp_path: Path):
    p = tmp_path / "counts.json"
    p.write_text("{ this is not json", encoding="utf-8")
    s = FileCountStore(p)
    assert s.load_day("2026-09-13") == {}      # 告警而非抛异常
    s.add("2026-09-13", {"a": 1})
    assert s.get("2026-09-13", "a") == 1


def test_file_store_clear(tmp_path: Path):
    p = tmp_path / "counts.json"
    s = FileCountStore(p)
    s.add("2026-09-13", {"a": 1})
    s.clear()
    assert s.load_day("2026-09-13") == {}


def test_file_store_purge_before(tmp_path: Path):
    p = tmp_path / "counts.json"
    s = FileCountStore(p)
    s.add("2026-09-01", {"a": 1})
    s.add("2026-09-13", {"b": 1})
    s.purge_before("2026-09-13")
    assert s.load_day("2026-09-01") == {}
    assert s.load_day("2026-09-13") == {"b": 1}


# ---------------------------------------------------------------------------
# 第 4 层：RateLimiter 集成 —— 清单 #8/#9 的直接验收
# ---------------------------------------------------------------------------
def _patch_limits(monkeypatch, *, per_ip_day=0, per_day=0, per_min=0) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "rate_limit_qa_per_min", per_min)
    monkeypatch.setattr(settings, "rate_limit_qa_per_ip_day", per_ip_day)
    monkeypatch.setattr(settings, "rate_limit_qa_per_day", per_day)


def test_limiter_daily_quota_survives_restart(engine, clock, monkeypatch):
    """#9 直接验收：全局日额度跨"重启"仍生效。"""
    _patch_limits(monkeypatch, per_day=2)
    lim1 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim1.check("/api/qa", "1.1.1.1").allowed is True
    assert lim1.check("/api/qa", "1.1.1.1").allowed is True
    assert lim1.check("/api/qa", "1.1.1.1").allowed is False

    lim2 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    v = lim2.check("/api/qa", "2.2.2.2")   # 换 IP，但全局额度已耗尽
    assert v.allowed is False, "重启后全局日额度被清零 —— #9 未修复"
    assert v.scope == "global_day"


def test_limiter_per_ip_quota_survives_restart(engine, clock, monkeypatch):
    """#9 另一面：每 IP 日额度也跨重启保持。"""
    _patch_limits(monkeypatch, per_ip_day=1)
    lim1 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim1.check("/api/qa", "9.9.9.9").allowed is True
    assert lim1.check("/api/qa", "9.9.9.9").allowed is False

    lim2 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    v = lim2.check("/api/qa", "9.9.9.9")
    assert v.allowed is False
    assert v.scope == "ip_day"
    # 换 IP 应放行（证明是 IP 维度，不是全局误伤）
    assert lim2.check("/api/qa", "8.8.8.8").allowed is True


def test_limiter_worker_quota_total_is_correct_across_workers(engine, clock, monkeypatch):
    """#8 直接验收（防回归）：多 worker 下额度**总量**必须精确。

    初版「内存缓存 + 异步刷盘」在此失败：4 worker × 额度 40 实际放行 60 次。
    每个 worker 都凭自己那份滞后内存值认为"还没到"，集体超发。
    本用例是第一版返工的护栏，不要删。
    """
    _patch_limits(monkeypatch, per_day=40)
    workers = [
        RateLimiter(
            now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
        )
        for _ in range(4)
    ]
    allowed = 0
    for _ in range(20):
        for w in workers:
            if w.check("/api/qa", "1.1.1.1").allowed:
                allowed += 1
    assert allowed == 40, f"多 worker 超发：放行了 {allowed} 次，应恰好 40"
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT value FROM rate_limit_counters WHERE day=:d AND key=:k"),
            {"d": "2026-09-13", "k": "global:qa"},
        ).scalar()
    assert total == 40, f"库中总量错误：{total}，应为 40"


def test_limiter_two_workers_share_quota_immediately(engine, clock, monkeypatch):
    """跨 worker **立即**可见（无刷盘延迟）——这是新设计的核心改进。"""
    _patch_limits(monkeypatch, per_day=2)
    w1 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    w2 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert w1.check("/api/qa", "1.1.1.1").allowed is True
    assert w2.check("/api/qa", "2.2.2.2").allowed is True
    # w2 刚记的账，w1 必须立刻看到
    v = w1.check("/api/qa", "3.3.3.3")
    assert v.allowed is False, "跨 worker 存在可见性延迟 —— 会超发"
    assert v.scope == "global_day"


def test_limiter_concurrent_checks_never_exceed_quota(engine, clock, monkeypatch):
    """**并发 check 不得超发** —— check-then-act 竞态的护栏。

    真实踩过：4 个独立子进程并发抢额度 40 时偶发放行 41 次。
    根因是「先 used() 判定、后 bump() 记账」两步可交错：
    两个进程同时读到 39 → 都认定还有额度 → 各自占位成功。
    进程内的 threading.Lock 跨进程无效，必须把两步合并为原子 reserve。

    16 线程 × 15 次（共 240 次尝试）抢额度 30，通过数必须**恰好 30**。
    """
    _patch_limits(monkeypatch, per_day=30)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    day = "2026-09-13"
    n_threads, per_thread = 16, 15
    allowed = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(per_thread):
            if lim.check("/api/qa", f"10.0.{hash(threading.current_thread()) % 251}.1").allowed:
                with lock:
                    allowed.append(1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(allowed) == 30, f"并发下超发：放行 {len(allowed)} 次，应恰好 30"
    assert lim.used_today("global:qa") == 30
    with engine.connect() as conn:
        db_total = conn.execute(
            text("SELECT value FROM rate_limit_counters WHERE day=:d AND key=:k"),
            {"d": day, "k": "global:qa"},
        ).scalar()
    assert db_total == 30, f"库中计数 {db_total}，应恰为 30"


def test_limiter_ip_day_rejection_releases_global_quota(engine, clock, monkeypatch):
    """被 L2（每 IP 额度）拒绝时必须回退已占的 L3（全局）额度。

    否则每次「被每 IP 额度拦住」的请求都会永久吃掉一次全局额度 ——
    一个正常用户的正常拒绝会把全局护栏推向提前耗尽。
    """
    _patch_limits(monkeypatch, per_ip_day=1, per_day=100)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    # 同一 IP 连续 5 次：第 1 次放行，后 4 次应被 ip_day 拒绝
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    for _ in range(4):
        v = lim.check("/api/qa", "1.1.1.1")
        assert v.allowed is False
        assert v.scope == "ip_day"
    # 全局额度只该消耗 1（那 4 次拒绝必须已回退）
    assert lim.used_today("global:qa") == 1, "L2 拒绝未回退 L3 占位 —— 全局额度被白吃"
    assert lim.used_today("ip:qa:1.1.1.1") == 1


def test_limiter_global_day_rejection_does_not_consume_ip_quota(engine, clock, monkeypatch):
    """被 L3 拒绝时不该消耗 L2 —— 判定次序是 L3 先于 L2。"""
    _patch_limits(monkeypatch, per_ip_day=100, per_day=1)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    v = lim.check("/api/qa", "2.2.2.2")     # 换 IP，撞全局额度
    assert v.allowed is False
    assert v.scope == "global_day"
    assert lim.used_today("ip:qa:2.2.2.2") == 0, "L3 拒绝却占了 L2 额度"


def test_limiter_daily_resets_next_day(engine, clock, monkeypatch):
    """跨天后额度恢复（持久化不该把限制变成「永久」）。"""
    _patch_limits(monkeypatch, per_day=1)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    assert lim.check("/api/qa", "1.1.1.1").allowed is False
    clock.advance(86400)   # 次日
    lim2 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim2.check("/api/qa", "1.1.1.1").allowed is True


def test_limiter_minute_bucket_stays_in_memory(engine, clock, monkeypatch):
    """L1 每分钟仍是实例内存（不持久化）—— 有意的设计，不是遗漏。"""
    _patch_limits(monkeypatch, per_min=1)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    assert lim.check("/api/qa", "1.1.1.1").allowed is False
    lim2 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim2.check("/api/qa", "1.1.1.1").allowed is True   # 新实例桶是满的


def test_limiter_attach_store_switches_backend(engine, clock, monkeypatch):
    """attach_store 应真正替换后端（启动钩子用的就是它）。

    默认（内存）store 里的计数**不会**迁移过去 —— 这是预期行为：
    换后端等于换一份"真相"，而启动时内存里本来就没有历史。
    断言只针对换后端之后的计数。
    """
    _patch_limits(monkeypatch, per_day=5)
    lim = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)   # 默认内存 store
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    lim.attach_store(SqliteCountStore(engine))
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    with engine.connect() as conn:
        v = conn.execute(
            text("SELECT value FROM rate_limit_counters WHERE day=:d AND key=:k"),
            {"d": "2026-09-13", "k": "global:qa"},
        ).scalar()
    assert v == 1, "attach_store 后未落盘 —— 后端未真正替换"
    lim2 = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim2.used_today("global:qa") == 1


def test_limiter_attach_store_switches_from_null_to_sqlite(engine, clock, monkeypatch):
    """从 NullCountStore 换到 SQLite：换之前额度用不完、换之后开始累计。

    这条覆盖「显式关闭日额度」的部署路径（RATE_LIMIT_STORE=none 起步，
    后来改成 sqlite），确认切换后护栏立刻生效。
    """
    _patch_limits(monkeypatch, per_day=1)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=NullCountStore()
    )
    for _ in range(5):   # Null 无状态 → 额度用不完
        assert lim.check("/api/qa", "1.1.1.1").allowed is True
    assert lim.used_today("global:qa") == 0

    lim.attach_store(SqliteCountStore(engine))
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    assert lim.check("/api/qa", "1.1.1.1").allowed is False   # 额度开始生效
    assert lim.used_today("global:qa") == 1


def test_limiter_default_store_is_stateful_but_not_persistent(clock, monkeypatch):
    """默认后端：**有状态、不持久化**（保住「按天累计」语义，不碰数据库）。

    为什么默认不是 NullCountStore：Null 无状态会让日额度永远用不完，
    等于静默关掉 L2/L3 —— 本地开发时护栏形同虚设却无人察觉。
    默认 `InMemoryCountStore` 则与改造前的行为一致。
    """
    _patch_limits(monkeypatch, per_day=1)
    lim = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    # 额度真的会被用掉（这与 NullCountStore 的关键区别）
    assert lim.check("/api/qa", "1.1.1.1").allowed is False
    assert lim.used_today("global:qa") == 1

    # reset 后额度复位（测试可复现性）
    lim.reset()
    assert lim.check("/api/qa", "1.1.1.1").allowed is True

    # 默认后端不持久化：新实例看不到旧计数（与 SqliteCountStore 的关键区别）
    lim2 = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    assert lim2.used_today("global:qa") == 0


def test_limiter_reset_does_not_touch_other_instances_store(engine, clock, monkeypatch):
    """`reset()` 只能清**本实例**的存储，不得波及别的实例。

    早期实现把计数器缓存在模块级，导致测试的 `reset_limiter()` 会把
    `main.py` 注入给全局单例的真实 SQLite 每日计数一并清空 ——
    那是能直接抹掉资金护栏的缺陷。本用例是它的护栏。
    """
    _patch_limits(monkeypatch, per_day=5)
    persistent = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert persistent.check("/api/qa", "1.1.1.1").allowed is True

    # 另建一个用默认（内存）store 的实例并 reset 它
    scratch = RateLimiter(now_fn=clock.now_fn, wall_fn=clock.wall_fn)
    scratch.check("/api/qa", "9.9.9.9")
    scratch.reset()

    # 持久化实例的计数必须完好
    assert persistent.used_today("global:qa") == 1, "reset 误伤了其他实例的存储"


def test_limiter_rejected_request_does_not_consume_quota(engine, clock, monkeypatch):
    """被拒的请求不得消耗额度 ——「先判定后记账」次序仍须成立。"""
    _patch_limits(monkeypatch, per_day=1)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    assert lim.check("/api/qa", "1.1.1.1").allowed is False
    assert lim.check("/api/qa", "1.1.1.1").allowed is False
    assert lim.used_today("global:qa") == 1, "被拒的请求把额度用掉了"


def test_limiter_minute_rejection_does_not_consume_daily(engine, clock, monkeypatch):
    """L1 拒绝时不应记入 L2/L3（否则死亡螺旋：被拒还在扣额度）。"""
    _patch_limits(monkeypatch, per_min=1, per_day=10)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    assert lim.check("/api/qa", "1.1.1.1").allowed is False   # 撞 L1
    assert lim.used_today("global:qa") == 1, "L1 拒绝却记了日计数"


def test_limiter_non_api_path_unaffected(engine, clock, monkeypatch):
    _patch_limits(monkeypatch, per_day=1)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    for _ in range(5):
        assert lim.check("/health", "1.1.1.1").allowed is True
    assert lim.used_today("global:qa") == 0


def test_limiter_disabled_tier_skips_store(engine, clock, monkeypatch):
    _patch_limits(monkeypatch, per_day=0, per_ip_day=0, per_min=0)
    lim = RateLimiter(
        now_fn=clock.now_fn, wall_fn=clock.wall_fn, store=SqliteCountStore(engine)
    )
    assert lim.check("/api/qa", "1.1.1.1").allowed is True
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM rate_limit_counters")).scalar()
    assert n == 0
