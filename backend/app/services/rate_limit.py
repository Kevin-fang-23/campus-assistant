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

已知局限（显式披露）：计数在进程内存中，多 worker 部署时实际额度 =
单 worker 额度 × worker 数。演示场景是单进程 uvicorn，故适用；若将来扩
多 worker 或上多实例，需换成 Redis 计数或网关/Ingress 层限流。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from ..config import settings

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
    """进程内限流器。线程安全（FastAPI 同步端点跑在线程池里）。"""

    def __init__(
        self,
        *,
        now_fn: Callable[[], float] = time.monotonic,
        wall_fn: Callable[[], float] = time.time,
    ) -> None:
        self._now = now_fn
        self._wall = wall_fn
        self._lock = threading.Lock()
        # (tier, ip) -> [tokens, last_refill_ts]
        self._buckets: dict[tuple[str, str], list[float]] = {}
        # (bucket_key, day) -> 计数；bucket_key 形如 "global:qa" / "ip:qa:1.2.3.4"
        self._daily: dict[tuple[str, str], int] = {}

    # ---------------- 对外 ----------------
    def check(self, path: str, ip: str) -> Verdict:
        tier_name = tier_for(path)
        if tier_name is None:
            return Verdict(True, None, 0, -1)
        cfg = tier_config(tier_name)
        if cfg.disabled:
            return Verdict(True, None, 0, -1)

        now, day = self._now(), self._today()
        with self._lock:
            # 先自上而下查三层，全部通过后才记账——避免「已拒绝但把额度用掉了」
            if cfg.per_day > 0 and self._used(f"global:{tier_name}", day) >= cfg.per_day:
                return Verdict(False, "global_day", self._until_midnight(), 0)
            if cfg.per_ip_day > 0 and self._used(f"ip:{tier_name}:{ip}", day) >= cfg.per_ip_day:
                return Verdict(False, "ip_day", self._until_midnight(), 0)

            remaining = -1
            if cfg.per_min > 0:
                remaining, allowed, retry_after = self._take(tier_name, ip, cfg.per_min, now)
                if not allowed:
                    return Verdict(False, "ip_minute", retry_after, 0)

            if cfg.per_day > 0:
                self._bump(f"global:{tier_name}", day)
            if cfg.per_ip_day > 0:
                self._bump(f"ip:{tier_name}:{ip}", day)

            self._prune(now, day)
            return Verdict(True, None, 0, remaining)

    def reset(self) -> None:
        """清空所有计数。仅供测试使用。"""
        with self._lock:
            self._buckets.clear()
            self._daily.clear()

    # ---------------- 内部 ----------------
    def _used(self, key: str, day: str) -> int:
        return self._daily.get((key, day), 0)

    def _bump(self, key: str, day: str) -> None:
        k = (key, day)
        self._daily[k] = self._daily.get(k, 0) + 1

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
        """回收空闲令牌桶与过期日计数。

        公网暴露下攻击者可伪造大量 X-Forwarded-For 制造无限多的桶，
        不清理会持续占内存；此处按空闲时长清理，并保留硬上限兜底。
        """
        size = len(self._buckets)
        if size >= _PRUNE_THRESHOLD or size > _MAX_BUCKETS:
            idle_before = now - _BUCKET_IDLE_SECONDS
            for k in [k for k, e in self._buckets.items() if e[1] < idle_before]:
                del self._buckets[k]
            if len(self._buckets) > _MAX_BUCKETS:      # 仍然超限则按最久未用淘汰
                excess = len(self._buckets) - _MAX_BUCKETS
                for k in sorted(self._buckets, key=lambda x: self._buckets[x][1])[:excess]:
                    del self._buckets[k]
        if self._daily:
            for k in [k for k in self._daily if k[1] != day]:
                del self._daily[k]


_limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    return _limiter


def reset_limiter() -> None:
    """测试辅助：清空限流状态。"""
    _limiter.reset()
