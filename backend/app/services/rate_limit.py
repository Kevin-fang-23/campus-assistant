"""请求限流策略与计数。

为什么需要（按优先级）：
1. **资金护栏**——演示链接经由 ngrok 公开后任何人可访问，而 /api/qa 与
   /api/documents 每次请求都会真实调用上游大模型（花钱）。被脚本刷时，
   免费额度会被抽干并自动转为按量扣费。
2. **可用性**——单进程 uvicorn，高频请求会拖垮所有访问者。
3. **公平性**——单个来源不应长时间独占成本端点。

三层策略（自上而下判定，任一超限即拒）：

| 层 | 作用域 | 结构 | 拦住什么 |
|----|--------|------|----------|
| L1 | 每 IP 每分钟 | 令牌桶（允许小突发） | 单点瞬时洪峰 |
| L2 | 每 IP 每日   | 固定窗口计数        | 单源长时间慢刷（防它吃光全局额度）|
| L3 | 全局每日     | 固定窗口计数        | 总支出上限（真正的资金护栏）|

为什么 L2 不可省：若只有 L3，一个 IP 花一小时就能把当日总额度耗尽，
之后真实访客全部撞 429——演示站会「看起来是坏的」。L2 让单源先撞墙。

为什么 L1 用令牌桶而非固定窗口：桶按分钟速率连续回填，允许「打开页面连问
3 个问题」这种正常突发，同时长期速率被钳制；固定窗口在边界会放过 2 倍流量。

已知局限（显式披露，不夸大）：
- L1 令牌桶在进程内存中，多 worker 部署时实际突发额度 = 单 worker 额度 ×
  worker 数。这是有意的：突发控制本就该按实例算，且令牌桶不适合共享
  （要共享就得每请求一次网络往返，代价大于收益）。若必须全局精确，换 Redis。
- L2/L3 的日计数**已持久化到 SQLite**（见 rate_limit_store.py），因此
  跨重启保留、多 worker 各写各的增量不丢 —— 总量精确（清单 #8/#9 已解决）。
- **仅限单机多 worker**。多台机器时 SQLite 文件无法共享，那时必须换
  Redis（实现 `rate_limit_store.CountStore` 协议即可，`RateLimiter` 不必改）。

多 worker 下的准确语义：
- 日计数（L2/L3）**总量精确**：日额度采用**原子占位**（`CountStore.reserve`，
  把「自增 + 判限 + 必要时撤销」压进一个数据库事务），因此各 worker 无法
  同时认定「还有额度」而集体超发；攻击者也**无法通过重启、或把自己分散到
  不同 worker 来放大每日总额度**；
- **无可见性延迟**：判定即读库、记账即写库。
  （两版被推翻的设计已记录在 rate_limit_store.py docstring：
   ①「内存缓存 + 异步刷盘」→ 4 worker × 额度 40 放行 60 次；
   ②「先 used() 判定、后 bump() 记账」→ 4 个真实子进程并发时偶发放行 41 次。）
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..config import settings
from .rate_limit_store import CountStore, DailyCounter, InMemoryCountStore

logger = logging.getLogger(__name__)

# 层级名
TIER_QA = "qa"
TIER_INGEST = "ingest"
TIER_DEFAULT = "default"

# 路径前缀 → 层级。顺序敏感：更具体的前缀必须排在前面。
_TIER_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("/api/qa", TIER_QA),
    ("/api/documents", TIER_INGEST),
)

# 令牌桶维护参数
_MAX_BUCKETS = 8192          # 桶数量硬上限，防伪造 IP 撑爆内存
_BUCKET_IDLE_SECONDS = 120.0  # 空闲超过该时长即可回收
_PRUNE_THRESHOLD = 1024      # 桶数低于该值时不触发清理扫描


@dataclass(frozen=True)
class Tier:
    name: str
    per_min: int
    per_ip_day: int
    per_day: int

    @property
    def disabled(self) -> bool:
        return self.per_min <= 0 and self.per_ip_day <= 0 and self.per_day <= 0


@dataclass(frozen=True)
class Verdict:
    """限流判定结果。scope 为 None 表示放行。"""

    allowed: bool
    scope: str | None
    retry_after: int   # 秒；仅在被拒时有意义
    remaining: int     # L1 剩余可用次数；-1 表示未参与判定


def tier_for(path: str) -> str | None:
    """返回该路径所属限流层级；None = 不受限（静态资源、/health、文档页）。

    只对 /api/* 生效：静态资源由 StaticFiles 挂载在 /，一个页面加载会产生
    几十个资源请求，若一并限流会直接把页面打挂。
    """
    if not path.startswith("/api"):
        return None
    for prefix, name in _TIER_BY_PREFIX:
        if path.startswith(prefix):
            return name
    return TIER_DEFAULT


def tier_config(name: str) -> Tier:
    """每次调用都从 settings 读取，便于测试用 monkeypatch 调整阈值。"""
    if name == TIER_QA:
        return Tier(name, settings.rate_limit_qa_per_min,
                    settings.rate_limit_qa_per_ip_day, settings.rate_limit_qa_per_day)
    if name == TIER_INGEST:
        return Tier(name, settings.rate_limit_ingest_per_min,
                    settings.rate_limit_ingest_per_ip_day, settings.rate_limit_ingest_per_day)
    return Tier(name, settings.rate_limit_default_per_min, 0, 0)


def client_key(
    *,
    forwarded_for: str | None,
    real_ip: str | None,
    host: str | None,
    trust_proxy: bool,
) -> str:
    """提取限流主体标识。

    trust_proxy 为 False 时只用 TCP 层源地址——X-Forwarded-For 是客户端
    可自由伪造的请求头，直连公网时信任它等于把限流开关交给攻击者。
    置于 ngrok / Nginx 之后时源地址恒为代理地址，才需要开启该开关。
    """
    if trust_proxy:
        if forwarded_for:
            first = forwarded_for.split(",")[0].strip()
            if first:
                return first
        if real_ip:
            return real_ip.strip()
    return host or "unknown"


class RateLimiter:
    """限流器。线程安全（FastAPI 同步端点跑在线程池里）。

    计数分层：
    - L1 令牌桶：进程内存（快，且突发控制按实例算本就合理）；
    - L2/L3 日计数：走 `DailyCounter`，直查共享存储
      → 跨重启保留、多 worker 额度**总量精确**（解决清单 #8 / #9）。

    存储后端由 `store` 注入，默认 `InMemoryCountStore`（**有状态但不持久化**）。

    为什么默认值不是 `NullCountStore`：Null 是无状态实现，日额度永远用不完，
    等于把 L2/L3 静默关掉 —— 作为默认值太危险（本地开发时护栏形同虚设而
    无人察觉）。`InMemoryCountStore` 保住了「按天累计」的语义，只是不跨
    进程/重启（与改造前一致），作为默认值不会让人误以为已有护栏。
    生产在 `main.py` 启动时注入 SQLite 后端，才真正解决 #8 / #9。

    ## 与全局单例的隔离（这是「不污染生产计数」的关键）

    `store` 是**实例私有**的，类级别不做任何缓存。因此：
      · 单测里 `RateLimiter(...)` 新建的实例天然与生产库隔离；
      · `reset()` 只清**本实例自己**的存储，不会去动 `main.py` 注入给
        全局单例的那个 SQLite 后端。
    早期版本把计数器缓存在模块级，导致测试的 `reset_limiter()` 会把真实
    数据库里的每日计数一并清空 —— 那是能直接抹掉资金护栏的缺陷。

    ## 锁的边界（重要）

    I/O 必须在锁外。日计数现在要查库（88.8µs/次），若在 `self._lock`
    内执行，所有请求会被串行化 —— 一个慢查询会拖住全部并发。
    因此把 `check()` 拆成两段：

      1. 锁外读日计数（L2/L3）并判定；
      2. 锁内取令牌（L1）、锁内记账。

    这样锁只保护纯内存操作（微秒级），与改造前的持锁时长相当。
    """

    def __init__(
        self,
        *,
        now_fn: Callable[[], float] = time.monotonic,
        wall_fn: Callable[[], float] = time.time,
        store: CountStore | None = None,
    ) -> None:
        self._now = now_fn
        self._wall = wall_fn
        self._lock = threading.Lock()
        # (tier, ip) -> [tokens, last_refill_ts]
        self._buckets: dict[tuple[str, str], list[float]] = {}
        # 日计数：直查存储（无内存缓存）
        self._store: CountStore = store or InMemoryCountStore()
        self._counter = DailyCounter(self._store, wall_fn=wall_fn)

    # ---------------- 存储注入 ----------------
    def attach_store(self, store: CountStore) -> None:
        """在应用启动时注入持久化后端。

        单独一个方法（而不是只走构造参数）是因为全局单例 `_limiter`
        在模块导入时就建好了，而那时数据库引擎可能还没就绪；
        启动钩子里再挂载更安全，也便于测试替换。

        只替换本实例的计数器，不影响其他实例（含测试里新建的）。
        """
        self._store = store
        self._counter = DailyCounter(store, wall_fn=self._wall)

    def close(self) -> None:
        """应用退出时调用。当前实现无后台线程，保留以兼容调用方。"""
        self._counter.close()

    def used_today(self, key: str) -> int:
        """读某计数键的当日用量（排查/运维用，也供测试断言）。

        key 形如 `global:qa` / `ip:qa:1.2.3.4`。这是只读查询，
        便于 `/health` 或运维脚本观察「额度用了多少」而不必猜。
        """
        day = datetime.fromtimestamp(self._wall()).strftime("%Y-%m-%d")
        return self._counter.used(key, day)

    # ---------------- 对外 ----------------
    def check(self, path: str, ip: str) -> Verdict:
        """三层判定。**日额度用原子占位，不是「先读后写」**。

        次序（每一步的位置都有原因）：

          1. 锁内取令牌桶（L1）——纯内存、微秒级，被拒时不消耗任何日额度；
          2. 锁外做日额度**原子占位**（L3 全局 → L2 每 IP）——
             这一步会读+写库，故绝不能持锁，否则慢查询会串行化全部请求。

        为什么日额度不能「先 used() 判定、再 bump() 记账」：
        跨进程下两个 worker 可能同时读到 39（额度 40）、都判定放行，
        最终放行 41 次 —— 实测真实子进程并发时**偶发复现**。
        进程内的 `threading.Lock` 对别的进程无效，唯一可靠的做法是把
        「判定 + 占用」压成存储层的一次原子操作（见 `CountStore.reserve`）。

        为什么 L1 排在日额度之前：L1 被拒的请求不该扣日额度。
        否则高频刷子会把日额度也刷掉，形成「被拒还在扣额度」的死亡螺旋。
        """
        tier_name = tier_for(path)
        if tier_name is None:
            return Verdict(True, None, 0, -1)
        cfg = tier_config(tier_name)
        if cfg.disabled:
            return Verdict(True, None, 0, -1)

        now, day = self._now(), self._today()

        # ---- 第一段（锁内，纯内存）：令牌桶。被拒则不触碰日额度 ----
        with self._lock:
            remaining = -1
            if cfg.per_min > 0:
                remaining, allowed, retry_after = self._take(tier_name, ip, cfg.per_min, now)
                if not allowed:
                    return Verdict(False, "ip_minute", retry_after, 0)
            self._prune(now, day)

        # ---- 第二段（锁外，会读写库）：日额度原子占位 ----
        # `and` 短路求值保证 try_reserve 只在该层启用时被调用（等价于嵌套 if）。
        global_key = f"global:{tier_name}"
        ip_key = f"ip:{tier_name}:{ip}"
        if cfg.per_day > 0 and not self._counter.try_reserve(global_key, day, cfg.per_day):
            return Verdict(False, "global_day", self._until_midnight(), 0)
        if cfg.per_ip_day > 0 and not self._counter.try_reserve(ip_key, day, cfg.per_ip_day):
            # 全局额度已占，此处退出时应把它还回去 —— 否则
            # 「被每 IP 额度拒绝」的请求会白吃掉一次全局额度。
            self._release(global_key, day, cfg.per_day > 0)
            return Verdict(False, "ip_day", self._until_midnight(), 0)

        return Verdict(True, None, 0, remaining)

    def _release(self, global_key: str, day: str, should_release: bool) -> None:
        """撤销一次已占用的全局额度（L2 拒绝时回退 L3 的占位）。

        为什么需要：L2 的占位发生在 L3 之后，若 L2 拒绝而 L3 的占位不回退，
        那么每次「被每 IP 额度拦住」的请求都会永久吃掉一次全局额度 ——
        一个正常用户的正常拒绝会把全局护栏推向提前耗尽。

        容错职责**只在 `DailyCounter.release()` 一层**：它内部已捕获存储异常
        并打 warning（「限流计数回退失败（累计值会略微偏高）」）。
        此处不再包 try/except —— 两层都兜底时，外层 except 是永不执行的死代码，
        而永不执行的代码意味着下次改动时没人会发现它已经坏了
        （P0-1 的 NameError 就是这么来的）。
        """
        if not should_release:
            return
        self._counter.release(global_key, day)

    def reset(self, *, clear_store: bool = True) -> None:
        """清空令牌桶与**本实例**的日计数。**仅供测试使用。**

        clear_store 默认为 True：测试的意图通常是「让限流回到初始状态」，
        而日计数现在会落到存储里，只清内存不足以复位。

        安全性边界：只作用于 `self._store`，而 store 是实例私有的
        （见类 docstring）。所以测试里对一个临时实例调 `reset()`，
        不会波及 `main.py` 注入给全局单例的 SQLite 后端。
        **生产代码不应调用本方法** —— 那等于把当日额度还给攻击者。
        """
        with self._lock:
            self._buckets.clear()
        if clear_store:
            self._counter.reset(clear_store=True)
        else:
            self._counter.reset(clear_store=False)

    # ---------------- 内部 ----------------
    def _take(self, tier: str, ip: str, per_min: int, now: float) -> tuple[int, bool, int]:
        """取一个令牌。返回 (剩余整数令牌, 是否放行, 建议重试秒数)。"""
        capacity = float(per_min)
        rate = per_min / 60.0                      # 每秒回填量
        k = (tier, ip)
        entry = self._buckets.get(k)
        if entry is None:
            tokens, last = capacity, now
        else:
            tokens, last = entry[0], entry[1]
            tokens = min(capacity, tokens + (now - last) * rate)   # 惰性回填
        if tokens >= 1.0:
            self._buckets[k] = [tokens - 1.0, now]
            return int(tokens - 1.0), True, 0
        need = 1.0 - tokens
        retry_after = max(1, int(need / rate + 0.999)) if rate > 0 else 60
        self._buckets[k] = [tokens, now]
        return 0, False, retry_after

    def _today(self) -> str:
        return datetime.fromtimestamp(self._wall()).strftime("%Y-%m-%d")

    def _until_midnight(self) -> int:
        now = datetime.fromtimestamp(self._wall())
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1, int((tomorrow - now).total_seconds()))

    def _prune(self, now: float, day: str) -> None:
        """回收空闲令牌桶。

        公网暴露下攻击者可伪造大量 X-Forwarded-For 制造无限多的桶，
        不清理会持续占内存；此处按空闲时长清理，并保留硬上限兜底。

        日计数的过期清理已移到 `DailyCounter._maybe_purge`（它知道存储后端，
        也能顺带清理库里的历史数据），这里不再重复处理。
        """
        _ = day   # 签名保留 day，便于将来需要时在此处做天级处理
        size = len(self._buckets)
        if size >= _PRUNE_THRESHOLD or size > _MAX_BUCKETS:
            idle_before = now - _BUCKET_IDLE_SECONDS
            for k in [k for k, e in self._buckets.items() if e[1] < idle_before]:
                del self._buckets[k]
            if len(self._buckets) > _MAX_BUCKETS:      # 仍然超限则按最久未用淘汰
                excess = len(self._buckets) - _MAX_BUCKETS
                for k in sorted(self._buckets, key=lambda x: self._buckets[x][1])[:excess]:
                    del self._buckets[k]


_limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    return _limiter


def reset_limiter() -> None:
    """测试辅助：清空**全局单例**的限流状态。

    ⚠️ 生产环境禁止调用。全局单例在生产里挂的是 SQLite 后端，
    清空它等于把当日额度全部还给攻击者。
    本函数存在只为让测试能复位经 TestClient 走中间件的那条路径。
    """
    _limiter.reset()
