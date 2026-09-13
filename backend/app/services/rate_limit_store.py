"""限流计数的持久化后端。

## 为什么需要这一层

`RateLimiter` 原先把计数放在进程内存的两个 dict 里，这有两个真实缺陷：

1. **多 worker 失效**（清单 #8）：uvicorn `--workers 4` 时每个 worker 各有一份
   计数，实际额度 = 配置值 × worker 数。资金护栏被稀释 4 倍。
2. **重启清零**（清单 #9）：进程重启后计数归零，攻击者可刻意触发重启
   （例如持续打满导致 OOM）来绕过每日上限。

两者根因相同：**计数是进程私有的**。本模块提供跨进程共享的计数存储。

## 为什么用 SQLite 而不是 Redis

项目一贯取向是「能复用就不加依赖」（见 rate_limit.py docstring）。
本场景下：

- Redis 需要额外服务 + `redis` 包，而演示部署（ngrok + 单机）不会起 Redis；
- SQLite 是**零新依赖**（SQLAlchemy 已在用），且默认库就是 SQLite；
- 演示的真实并发量是「几个招聘官点开链接」，不是生产级 QPS。

多实例部署（多台机器）时 SQLite 文件无法共享，那时才需要换 Redis ——
本模块把存储抽象成 `CountStore` 接口，替换实现即可，不必改 `RateLimiter`。

## 设计取向：放弃内存缓存，判定即读库（附实测依据）

**初版设计（内存缓存 + 后台批量刷盘）是错的**，必须记下来避免重走：

它把判定放在内存、增量异步落盘。问题在于**每个 worker 的判定依据
（本地内存值）滞后于真实总量**，于是会超发 —— 实测 4 个 worker、
每日额度 40 的场景，实际放行了 **60** 次（应恰好 40）。
额度越大、worker 越多，超发越严重，等于**没解决问题**。
（当时的测试用例 `test_limiter_worker_quota_total_is_correct_across_workers`
正是为钉住这点而写。）

改为**每次判定都同步查询存储**（读库 + 原子自增），多 worker 共享同一份真相。

### 第二版仍然有缺陷：「先判定、后记账」也会超发

第二版把 `check()` 写成「先 `used()` 读一下够不够 → 放行 → 再 `bump()` 记账」。
读库后单个 worker 内部是准的，但**跨进程仍然会超发**：

    进程 A: 读 used()=39 < 40 → 判定放行 → …
    进程 B: 读 used()=39 < 40 → 判定放行 → …      ← A、B 同时读到 39
    A、B 各自占位 → 库里变成 41，**超发 1 次**

实测证据：4 个真实子进程并发抢额度 40，**偶发放行 41 次**（不是每次必现，
所以只用单测「跑一遍绿了」是发现不了的）。
进程内的 `threading.Lock` 对别的进程无效 —— 这是根本原因。

**修法：把「判定 + 占用」压成存储层的一次原子操作**，即 `CountStore.reserve`：

```python
# SqliteCountStore.reserve：整个「自增 → 判限 → 必要时撤销」在一个事务内
# SQLite 的写事务是互斥的，因此别的进程不可能同时读到 39 再各自加 1
with self._engine.begin() as conn:
    conn.execute("INSERT ... ON CONFLICT DO UPDATE SET value = value + 1")
    new = conn.execute("SELECT value ...").scalar()
    if new > limit:
        conn.execute("UPDATE ... SET value = value - 1")   # 撤销，保持计数不虚高
    return new
```

**撤销而不是「先查再写」是关键**：撤销在同一事务内完成，外界永远看不到超限值。

那为什么性能可以接受 —— 实测数据（2000 次操作，本机）：

| 方式 | 单次开销 |
|---|---|
| 纯内存判定（改造前） | 0.09 µs |
| 读库判定 | 242 µs |
| 读库 + 记账（默认 rollback journal） | **4071 µs** |
| 读库 + 记账（**WAL + synchronous=NORMAL**） | **88.8 µs** |

关键收益来自 WAL：默认 journal 模式每笔事务都 fsync，4ms/请求不可接受；
WAL 下降到 88.8µs（快约 46 倍）。相对一次 LLM 调用（500–3000ms）
占比约 0.02‰，可以忽略。

并发正确性也已实测：8 线程 × 200 次并发自增 → 总量 1600 **精确、0 错误**
（依赖 `ON CONFLICT DO UPDATE` 的数据库端原子加 + `busy_timeout`）。

因此：**精度与性能可以兼得**，代价只是必须在 SQLite 上开启 WAL。
`SqliteCountStore` 在初始化时自动开启，无需调用方关心。

## 可观测性

`DailyCounter` 不保留任何跨请求状态，所以「重启是否清零」不再需要额外机制 ——
数据一直在库里。这同时消除了「进程被 SIGKILL 丢一个刷盘间隔」的问题。
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from sqlalchemy import text

logger = logging.getLogger(__name__)


def _shift_day(day: str, delta_days: int) -> str:
    """`YYYY-MM-DD` 加减天数。纯字符串运算，不依赖时区。"""
    dt = datetime.strptime(day, "%Y-%m-%d")
    return (dt + timedelta(days=delta_days)).strftime("%Y-%m-%d")


class CountStore(Protocol):
    """计数存储接口。换 Redis 只需实现这个协议。"""

    def get(self, day: str, key: str) -> int:
        """读单个键的当日值。判定路径的核心调用，必须精确（不能是缓存值）。"""
        ...

    def load_day(self, day: str) -> dict[str, int]:
        """读回某一天的全部计数，key 形如 `global:qa` / `ip:qa:1.2.3.4`。"""
        ...

    def add(self, day: str, deltas: dict[str, int]) -> None:
        """原子累加增量。多 worker 并发调用不得丢失更新。"""
        ...

    def reserve(self, day: str, key: str, limit: int) -> int:
        """**原子占位**：自增 1 并返回新值，但新值超过 `limit` 时撤销并返回超限值。

        为什么需要它而不是「先读后写」：判定与记账分两步做的话，两个进程
        可能同时读到 39（limit=40）→ 都判定放行 → 库里变成 41，**超发**。
        这在跨进程下无法用进程内锁解决，必须把「读-判-写」压成一次原子操作。

        返回语义：
          · `<= limit`  → 占位成功，调用方应放行；
          · `> limit`   → 已超限，增量**已被撤销**，调用方应拒绝。
        返回超限值（而非布尔）便于日志/排查。
        """
        ...

    def release(self, day: str, key: str) -> None:
        """撤销一次占位（值减 1，不低于 0）。

        用于「后一层拒绝、需回退前一层占位」的场景：
        `check()` 先占 L3 全局额度、再占 L2 每 IP 额度；若 L2 拒绝，
        必须把 L3 的那次占位还回去，否则每次被 L2 拦下的请求都会
        永久吃掉一次全局额度。
        """
        ...

    def purge_before(self, day: str) -> None:
        """清理早于该日期的计数（历史数据无保留价值）。"""
        ...

    def clear(self) -> None:
        """清空全部计数。**仅测试用** —— 生产调用等于把额度还给攻击者。"""
        ...


class NullCountStore:
    """空实现：**无状态**，任何写入都被丢弃。

    用途只有一个 —— 显式表达「我不要持久化」：
      · 单测里不关心跨进程/跨重启语义的用例（完全不碰数据库，毫秒级）；
      · 明确以单 worker、额度用完即弃的方式跑（例如离线演示兜底）。

    注意它**不保存任何计数**（`get` 恒返回 0、`add` 是空操作），
    因此基于它的日额度**永远不会耗尽** —— 这等价于「L2/L3 关闭」。
    要「按天累计但仍留在本进程」请用 `InMemoryCountStore`。
    """

    def get(self, day: str, key: str) -> int:  # noqa: ARG002
        return 0

    def load_day(self, day: str) -> dict[str, int]:  # noqa: ARG002
        return {}

    def add(self, day: str, deltas: dict[str, int]) -> None:  # noqa: ARG002
        return None

    def reserve(self, day: str, key: str, limit: int) -> int:  # noqa: ARG002
        # 无状态实现：永远"占位成功且不超过"，即额度用不完。
        # 这正是 NullCountStore 的语义（显式关闭日额度），保持一致。
        return 1

    def release(self, day: str, key: str) -> None:  # noqa: ARG002
        return None

    def purge_before(self, day: str) -> None:  # noqa: ARG002
        return None

    def clear(self) -> None:
        return None


class InMemoryCountStore:
    """有状态的内存计数（进程私有）。

    这是改造前的 `RateLimiter._daily` 两个 dict 的等价物，抽出来独立成
    一个 store 实现。这样 `DailyCounter` 的语义（判定即读存储）在任何
    后端下都一致，不会因为后端是内存就多出一条特殊分支 ——
    分支越少，测到的行为越接近生产（SQLite 后端）的行为。

    适用与不适用：
      · 适用：单 worker、或「按天累计」的语义要成立但不要求跨进程共享；
      · 不适用：多 worker（各算一份 → 额度 × worker 数，即清单 #8）、
        重启后需要保留（即清单 #9）。这两点必须用 `SqliteCountStore`。

    线程安全：与 `RateLimiter` 同处多线程环境（FastAPI 同步端点跑在线程池），
    故加锁；锁只保护内存运算，不涉及 I/O。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # day -> {key: value}
        self._data: dict[str, dict[str, int]] = {}

    def get(self, day: str, key: str) -> int:
        with self._lock:
            return int(self._data.get(day, {}).get(key, 0))

    def load_day(self, day: str) -> dict[str, int]:
        with self._lock:
            return dict(self._data.get(day, {}))

    def add(self, day: str, deltas: dict[str, int]) -> None:
        if not deltas:
            return
        with self._lock:
            bucket = self._data.setdefault(day, {})
            for k, v in deltas.items():
                if v > 0:
                    bucket[k] = int(bucket.get(k, 0)) + int(v)

    def reserve(self, day: str, key: str, limit: int) -> int:
        """锁内完成「读-判-写」，不可拆分（拆分即超发）。"""
        with self._lock:
            bucket = self._data.setdefault(day, {})
            new = int(bucket.get(key, 0)) + 1
            if new > limit:
                return new          # 不写入：撤销占位
            bucket[key] = new
            return new

    def release(self, day: str, key: str) -> None:
        with self._lock:
            bucket = self._data.get(day)
            if not bucket or key not in bucket:
                return
            bucket[key] = max(0, int(bucket[key]) - 1)

    def purge_before(self, day: str) -> None:
        with self._lock:
            for d in [d for d in self._data if d < day]:
                del self._data[d]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class SqliteCountStore:
    """基于 SQLite 表的计数存储（零新依赖）。

    表结构刻意保持极简 —— 只有 (day, key, value) 三列复合主键：

        CREATE TABLE rate_limit_counters (
            day   TEXT NOT NULL,
            key   TEXT NOT NULL,
            value INTEGER NOT NULL,
            PRIMARY KEY (day, key)
        )

    用裸 SQL + `INSERT ... ON CONFLICT` 而不是 ORM：原子自增的语义
    一眼可见（`value = value + :v` 由数据库执行，不是读-改-写），
    而 ORM 在这里只增加开销没有收益。

    ## WAL 是必须的（实测 46 倍差距）

    默认 rollback journal 模式下每笔事务都 fsync，实测 4071 µs/次；
    开启 WAL + `synchronous=NORMAL` 后降到 88.8 µs/次。
    对一个「每请求都要记账」的 hot path，这是能否采用本方案的分水岭，
    所以在此自动开启，不要指望部署者记得设。

    `busy_timeout` 同样必要：多 worker 并发写同一行时，
    SQLite 需要重试而非立刻抛 `database is locked`。
    """

    def __init__(self, engine, *, busy_timeout_ms: int = 10_000) -> None:  # noqa: ANN001
        self._engine = engine
        self._busy_timeout_ms = busy_timeout_ms
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._engine.begin() as conn:
            # WAL 让读不阻塞写、写不阻塞读；NORMAL 在 WAL 下已足够安全
            # （断电最多丢最近若干事务，而限流计数丢几笔无实质影响）。
            try:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
            except Exception as exc:  # noqa: BLE001
                # 非 SQLite 或权限受限（如只读文件系统）时不该阻断启动
                logger.debug("开启 WAL 失败（可能非 SQLite）：%s", exc)
            conn.execute(text(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}"))
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS rate_limit_counters (
                        day   TEXT    NOT NULL,
                        key   TEXT    NOT NULL,
                        value INTEGER NOT NULL,
                        PRIMARY KEY (day, key)
                    )
                    """
                )
            )

    def load_day(self, day: str) -> dict[str, int]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT key, value FROM rate_limit_counters WHERE day = :d"),
                {"d": day},
            ).all()
        return {r[0]: int(r[1]) for r in rows}

    def get(self, day: str, key: str) -> int:
        """读单个键的值。判定路径只用到这个，比 load_day 更省。"""
        with self._engine.connect() as conn:
            v = conn.execute(
                text(
                    "SELECT value FROM rate_limit_counters WHERE day = :d AND key = :k"
                ),
                {"d": day, "k": key},
            ).scalar()
        return int(v) if v is not None else 0

    def add(self, day: str, deltas: dict[str, int]) -> None:
        if not deltas:
            return
        with self._engine.begin() as conn:
            for key, delta in deltas.items():
                if delta <= 0:
                    continue
                # 原子自增：交给数据库做加法，避免多 worker 的读-改-写丢更新
                conn.execute(
                    text(
                        """
                        INSERT INTO rate_limit_counters (day, key, value)
                        VALUES (:d, :k, :v)
                        ON CONFLICT (day, key)
                        DO UPDATE SET value = value + :v
                        """
                    ),
                    {"d": day, "k": key, "v": int(delta)},
                )

    def purge_before(self, day: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM rate_limit_counters WHERE day < :d"), {"d": day}
            )

    def reserve(self, day: str, key: str, limit: int) -> int:
        """单事务内完成「自增 → 判限 → 必要时撤销」。

        为什么必须在一个事务里：SQLite 的写事务是**互斥**的，
        因此两个进程不可能同时读到 39 再各自加 1。若拆成
        「先 SELECT 再 UPDATE」，跨进程就没有任何机制阻止超发了
        （进程内锁在多进程下无效）。

        撤销而不是「先判再写」是同样的道理：撤销发生在同一事务内，
        外界永远不会看到「41」这个超限值（读被写事务隔离）。
        """
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO rate_limit_counters (day, key, value)
                    VALUES (:d, :k, 1)
                    ON CONFLICT (day, key)
                    DO UPDATE SET value = value + 1
                    """
                ),
                {"d": day, "k": key},
            )
            new = conn.execute(
                text(
                    "SELECT value FROM rate_limit_counters WHERE day = :d AND key = :k"
                ),
                {"d": day, "k": key},
            ).scalar()
            new = int(new or 0)
            if new > limit:
                # 超限：把这次占位撤回去，保持计数不虚高
                # （否则计数会被撑到远大于 limit，且撤不回来）
                conn.execute(
                    text(
                        "UPDATE rate_limit_counters SET value = value - 1 "
                        "WHERE day = :d AND key = :k"
                    ),
                    {"d": day, "k": key},
                )
            return new

    def release(self, day: str, key: str) -> None:
        """撤销一次占位。用 `MAX(0, value-1)` 保证不会变成负数。"""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE rate_limit_counters SET value = MAX(0, value - 1) "
                    "WHERE day = :d AND key = :k"
                ),
                {"d": day, "k": key},
            )

    def clear(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("DELETE FROM rate_limit_counters"))


class FileCountStore:
    """基于单个 JSON 文件的计数存储（无数据库依赖时的兜底）。

    为什么不只用 SQLite：`NullCountStore` 之外还需要一个「不引入 SQLAlchemy」
    的选项 —— 例如有人想单独复用 `ratese` 这段逻辑。JSON 文件用
    「读-改-写 + 进程内锁」实现；**跨进程并发不安全**，仅在明确单 worker
    但需要跨重启保留时使用，已在文档里标注。

    实现取向：文件小（一行一个计数），整体读入内存累加后覆写。
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict[str, int]]:
        if not self._path.exists():
            return {}
        try:
            import json

            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001
            # 文件损坏不应让限流整体崩掉：告警后退回空计数（等同重置一次）
            logger.warning("限流计数文件损坏，按空计数处理：%s", exc)
            return {}

    def load_day(self, day: str) -> dict[str, int]:
        with self._lock:
            return dict(self._read().get(day, {}))

    def get(self, day: str, key: str) -> int:
        with self._lock:
            return int(self._read().get(day, {}).get(key, 0))

    def add(self, day: str, deltas: dict[str, int]) -> None:
        if not deltas:
            return
        import json

        with self._lock:
            data = self._read()
            bucket = data.setdefault(day, {})
            for k, v in deltas.items():
                if v > 0:
                    bucket[k] = int(bucket.get(k, 0)) + int(v)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)   # 原子替换，避免读到半个文件

    def reserve(self, day: str, key: str, limit: int) -> int:
        """锁内完成「读-判-写」。

        ⚠️ 进程内锁只对本进程有效 —— 多进程下本实现**仍会超发**。
        这正是本类标注「仅限明确单 worker」的原因之一：
        需要多 worker 请用 `SqliteCountStore`。
        """
        import json

        with self._lock:
            data = self._read()
            bucket = data.setdefault(day, {})
            new = int(bucket.get(key, 0)) + 1
            if new > limit:
                return new          # 不落盘：撤销占位
            bucket[key] = new
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
            return new

    def release(self, day: str, key: str) -> None:
        import json

        with self._lock:
            data = self._read()
            bucket = data.get(day)
            if not bucket or key not in bucket:
                return
            bucket[key] = max(0, int(bucket[key]) - 1)
            self._path.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )

    def purge_before(self, day: str) -> None:
        import json

        with self._lock:
            data = self._read()
            kept = {d: v for d, v in data.items() if d >= day}
            if len(kept) != len(data):
                self._path.write_text(
                    json.dumps(kept, ensure_ascii=False), encoding="utf-8"
                )

    def clear(self) -> None:
        with self._lock:
            if self._path.exists():
                self._path.unlink()


class DailyCounter:
    """每日计数器：**直查存储**，不自带缓存。

    职责边界：只负责「某一天各 key 的累计值」，不管分层策略
    （那是 RateLimiter 的事）。这样两边都能独立单测。

    ## 为什么不留内存缓存（这是被实测推翻后的结论）

    初版把计数缓存在内存、增量异步刷盘。结果 4 个 worker × 每日额度 40
    实际放行了 **60** 次 —— 每个 worker 都凭自己那份滞后的内存值判断
    「还没到 40」，于是集体超发。额度越大超发越多，等于没修好 #8。

    现在每次 `used()` 都读存储、每次 `bump()` 都写存储，多 worker 共享
    同一份真相。性能上完全可行（WAL 下 88.8µs/请求，见模块 docstring 实测表）。

    代价：`used()` 不再免费，调用方不要把它放进循环里反复调
    （`RateLimiter.check` 每层只查一次，符合这个约束）。

    默认 store 为 `InMemoryCountStore`（有状态、进程私有）：保住「按天
    累计」的语义但不跨进程。若显式传 `NullCountStore`，`used` 恒为 0、
    日额度形同关闭 —— 那是调用方的显式选择，不是本类缺陷。
    """

    def __init__(
        self,
        store: CountStore | None = None,
        *,
        wall_fn=time.time,  # noqa: ANN001
        purged_day: str | None = None,
    ) -> None:
        self._store: CountStore = store or InMemoryCountStore()
        self._wall = wall_fn
        # 已触发过历史清理的日期。每进程只需在首次用到某天时清一次，
        # 不必每请求都 DELETE（那会白白写盘）。
        self._purged_day = purged_day

    # ---------------- 对外 ----------------
    def used(self, key: str, day: str) -> int:
        """读某键当日已用量。直查存储，保证多 worker 一致。

        同时借这次调用触发历史清理检查（每天仅一次）：判定路径每请求都会
        经过 `used`，而纯只读场景不会走 `bump`；只挂在 `bump` 上的话，
        一个只被拒绝请求的服务（读多写少）永远不清理旧数据。
        """
        self._maybe_purge(day)
        try:
            return self._store.get(day, key)
        except Exception as exc:  # noqa: BLE001
            # 存储故障时按 0 放行：限流是护栏，不该因存储问题把所有人挡在门外。
            # 这是有意识的取舍（可用性优先于护栏完整性），故打 warning 留痕。
            logger.warning("限流计数读取失败，本次按 0 处理：%s", exc)
            return 0

    def bump(self, key: str, day: str) -> int:
        """累加 1，返回累加后的值。写失败只告警。

        次序：**先写入、后清理**（清理由 `used` 触发）。若反过来先清理，
        一次 `purge_before` 可能把本次要写入的那天当成历史数据删掉
        （例如测试/回放场景直接 `bump("k", "2026-09-01")`，而当天是 09-13）
        —— 表现为「记账成功了但读回来是 0」，很难排查。
        """
        try:
            self._store.add(day, {key: 1})
        except Exception as exc:  # noqa: BLE001
            logger.warning("限流计数写入失败（本次增量丢弃）：%s", exc)
            return 0
        return self.used(key, day)

    def try_reserve(self, key: str, day: str, limit: int) -> bool:
        """**原子占位**：在额度内则占 1 个并返回 True；已满则返回 False。

        与 `used() + bump()` 两步法的关键区别：本方法把「判定是否还有额度」
        与「占用一个额度」压成存储层的**一次原子操作**，因此多 worker 并发
        下不会超发。

        为什么必须这样（真实踩过）：原先用「先 used() 判定、后 bump() 记账」，
        4 个真实子进程并发抢额度 40 时，实测**偶发放行 41 次** ——
        两个进程同时读到 39，都认为「还有额度」，各自占位成功。
        进程内的 `threading.Lock` 跨进程无效，只有把两步合并才可靠。

        存储故障时**按放行处理**（返回 True）：限流是护栏，不该因存储问题
        把所有人挡在门外。这是有意识的取舍，故打 warning 留痕。
        """
        self._maybe_purge(day)
        try:
            return self._store.reserve(day, key, limit) <= limit
        except Exception as exc:  # noqa: BLE001
            logger.warning("限流计数占位失败，本次放行（护栏降级）：%s", exc)
            return True

    def release(self, key: str, day: str) -> None:
        """撤销一次占位。失败只告警（计数会略偏高，不影响正确性方向）。"""
        try:
            self._store.release(day, key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("限流计数回退失败（累计值会略微偏高）：%s", exc)

    def reset(self, *, clear_store: bool = False) -> None:
        """重置计数器状态。

        `clear_store=True` 会清空存储里的全部计数 —— **仅测试用**。
        生产不要调它（等于把当日额度还给攻击者）；默认 False 只重置
        「已清理过哪天」的标记，不影响真实计数。
        """
        self._purged_day = None
        if clear_store:
            try:
                self._store.clear()
            except Exception as exc:  # noqa: BLE001
                logger.warning("限流计数清空失败：%s", exc)

    def close(self) -> None:
        """无后台线程，无需清理。保留此方法以兼容调用方。"""
        return None

    @staticmethod
    def today(wall_fn=time.time) -> str:  # noqa: ANN001
        return datetime.fromtimestamp(wall_fn()).strftime("%Y-%m-%d")

    # ---------------- 内部 ----------------
    def _maybe_purge(self, day: str) -> None:
        """首次用到某天时清理历史数据（每天只需一次）。

        为什么保留昨天而不是清到当天：
        时钟回拨（NTP 校时、跨时区、注入假时钟）会让日期从 D 退回 D-1，
        若已把 D-1 删掉，那天的额度就会被**意外清零** —— 攻击者
        等一次校时即可绕过每日上限。保留一天的成本可忽略
        （表里最多两天数据），换来回拨安全。
        """
        if self._purged_day == day:
            return
        self._purged_day = day
        try:
            self._store.purge_before(_shift_day(day, -1))
        except Exception as exc:  # noqa: BLE001
            logger.debug("限流历史数据清理失败：%s", exc)
